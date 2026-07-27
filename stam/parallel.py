r"""Filling the machine: micro-batch autotuning and multi-GPU job execution.

Two separate concerns, both about not leaving hardware idle.

**Micro-batch autotuning.**  A probe's marginal cost per example falls as the
micro-batch grows -- until the GPU is saturated, after which it is flat.  Since the
per-anchor overhead :math:`\tau` is amortised over the whole draw, the right micro-batch
is the largest one that still fits, and finding it is a measurement, not a guess.
:func:`autotune_micro_batch` grows the batch geometrically, timing each size, and stops
at an out-of-memory error, a memory target, or a throughput plateau.  A side effect is
that PyTorch's caching allocator then holds the peak working set, so steady-state
utilisation reflects the tuned size.

**Multi-GPU execution.**  Every expensive experiment here is a list of independent
jobs -- (method, budget, seed) cells of a sweep, or blocks of a dense reference grid.
Workers are separate processes, one per device, each building its own model and its own
resident copy of the evaluation set; jobs are handed out through a queue so that
unequal job costs do not leave a device idle at the end.  A single-device or CPU-only
machine falls back to running the same jobs in-process, so results do not depend on how
many GPUs happen to be present.
"""

from __future__ import annotations

import dataclasses
import os
import queue
import time
import traceback
from typing import Any, Callable, Iterable, Sequence

import torch


def gpu_devices() -> list[int]:
    if not torch.cuda.is_available():
        return []
    visible = os.environ.get("STAM_DEVICES")
    if visible:
        return [int(x) for x in visible.split(",") if x.strip() != ""]
    return list(range(torch.cuda.device_count()))


def device_memory(index: int) -> tuple[int, int]:
    """``(free, total)`` bytes on a device."""
    free, total = torch.cuda.mem_get_info(index)
    return int(free), int(total)


@dataclasses.dataclass
class BatchTuning:
    micro_batch: int
    throughput: float             # examples per second at the chosen size
    peak_bytes: int
    curve: list[dict[str, float]]

    def to_dict(self) -> dict:
        return {
            "micro_batch": self.micro_batch,
            "examples_per_second": round(self.throughput, 1),
            "peak_bytes": self.peak_bytes,
            "peak_gib": round(self.peak_bytes / 2**30, 2),
            "curve": self.curve,
        }


def autotune_micro_batch(
    run: Callable[[int], None],
    device: torch.device | str,
    start: int = 64,
    maximum: int = 1 << 16,
    target_fraction: float = 0.78,
    plateau_tolerance: float = 0.04,
    repeats: int = 2,
) -> BatchTuning:
    r"""Find the largest useful micro-batch for ``run``.

    ``run(b)`` must perform one unit of the work being tuned at micro-batch ``b``.
    Sizes double until an out-of-memory error, ``target_fraction`` of device memory, or
    a throughput plateau (relative gain below ``plateau_tolerance``) is reached.  The
    plateau test matters: past saturation, a larger batch buys nothing but costs memory
    that the resident evaluation set and the trajectory buffers need.
    """
    dev = torch.device(device)
    curve: list[dict[str, float]] = []
    best_b, best_tp, best_peak = start, 0.0, 0
    b = start
    prev_tp = 0.0

    while b <= maximum:
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
            torch.cuda.synchronize(dev)
        try:
            run(b)  # warm-up (allocation, autotune of cuDNN algorithms)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            for _ in range(repeats):
                run(b)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            dt = (time.perf_counter() - t0) / repeats
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            break
        except RuntimeError as exc:  # pragma: no cover - driver-dependent phrasing
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            break

        tp = b / max(dt, 1e-9)
        peak = int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else 0
        curve.append({"micro_batch": b, "seconds": round(dt, 6), "throughput": round(tp, 1),
                      "peak_gib": round(peak / 2**30, 3)})
        if tp > best_tp:
            best_b, best_tp, best_peak = b, tp, peak

        if dev.type == "cuda":
            _, total = device_memory(dev.index or 0)
            if peak > target_fraction * total:
                break
        if prev_tp > 0 and (tp - prev_tp) / prev_tp < plateau_tolerance:
            break
        prev_tp = tp
        b *= 2

    return BatchTuning(micro_batch=best_b, throughput=best_tp, peak_bytes=best_peak, curve=curve)


# ---------------------------------------------------------------------------
# Multi-device job execution
# ---------------------------------------------------------------------------


def _worker(
    rank: int,
    device_index: int,
    setup: Callable[[dict, torch.device], Any],
    work: Callable[[Any, dict], dict],
    spec: dict,
    job_q: Any,
    out_q: Any,
) -> None:  # pragma: no cover - runs in a child process
    try:
        torch.cuda.set_device(device_index)
        device = torch.device(f"cuda:{device_index}")
        ctx = setup(spec, device)
        while True:
            try:
                job = job_q.get(timeout=1.0)
            except queue.Empty:
                break
            if job is None:
                break
            t0 = time.perf_counter()
            try:
                result = work(ctx, job)
                result.setdefault("_device", device_index)
                result.setdefault("_seconds", time.perf_counter() - t0)
                out_q.put(("ok", job, result))
            except Exception:
                out_q.put(("error", job, traceback.format_exc()))
    except Exception:
        out_q.put(("fatal", {"rank": rank}, traceback.format_exc()))
    finally:
        out_q.put(("done", {"rank": rank}, None))


def run_jobs(
    jobs: Sequence[dict],
    setup: Callable[[dict, torch.device], Any],
    work: Callable[[Any, dict], dict],
    spec: dict,
    devices: Iterable[int] | None = None,
    workers_per_device: int = 1,
    progress: Callable[[int, int, dict], None] | None = None,
) -> list[dict]:
    """Run ``jobs`` across the available GPUs, returning results in completion order.

    ``setup(spec, device)`` is called once per worker and returns whatever context
    ``work(ctx, job)`` needs.  ``work`` must return a JSON-serialisable dict.
    """
    devs = list(devices) if devices is not None else gpu_devices()
    jobs = list(jobs)
    if not jobs:
        return []

    if len(devs) <= 1:
        device = torch.device(f"cuda:{devs[0]}" if devs else "cpu")
        ctx = setup(spec, device)
        results = []
        for i, job in enumerate(jobs):
            t0 = time.perf_counter()
            r = work(ctx, job)
            r.setdefault("_device", devs[0] if devs else -1)
            r.setdefault("_seconds", time.perf_counter() - t0)
            results.append(r)
            if progress:
                progress(i + 1, len(jobs), r)
        return results

    import torch.multiprocessing as mp

    ctx_mp = mp.get_context("spawn")
    job_q: Any = ctx_mp.Queue()
    out_q: Any = ctx_mp.Queue()
    for job in jobs:
        job_q.put(job)

    procs = []
    n_workers = len(devs) * workers_per_device
    for rank in range(n_workers):
        p = ctx_mp.Process(
            target=_worker,
            args=(rank, devs[rank % len(devs)], setup, work, spec, job_q, out_q),
            daemon=False,
        )
        p.start()
        procs.append(p)

    results: list[dict] = []
    errors: list[str] = []
    finished = 0
    while finished < n_workers:
        kind, job, payload = out_q.get()
        if kind == "done":
            finished += 1
        elif kind == "ok":
            results.append(payload)
            if progress:
                progress(len(results), len(jobs), payload)
        else:
            errors.append(f"{kind} on job {job}:\n{payload}")

    for p in procs:
        p.join(timeout=60)
        if p.is_alive():  # pragma: no cover
            p.terminate()

    if errors:
        raise RuntimeError(f"{len(errors)} job(s) failed:\n\n" + "\n\n".join(errors[:3]))
    return results


def memory_report(devices: Iterable[int] | None = None) -> list[dict]:
    devs = list(devices) if devices is not None else gpu_devices()
    out = []
    for i in devs:
        free, total = device_memory(i)
        out.append(
            {
                "device": i,
                "name": torch.cuda.get_device_name(i),
                "total_gib": round(total / 2**30, 2),
                "used_gib": round((total - free) / 2**30, 2),
                "torch_reserved_gib": round(torch.cuda.memory_reserved(i) / 2**30, 2),
                "torch_allocated_gib": round(torch.cuda.memory_allocated(i) / 2**30, 2),
            }
        )
    return out


__all__ = [
    "gpu_devices", "device_memory", "BatchTuning", "autotune_micro_batch", "run_jobs",
    "memory_report",
]
