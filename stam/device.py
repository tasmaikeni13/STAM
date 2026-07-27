"""Runtime GPU capability detection and kernel dispatch policy.

Everything downstream of this module asks *the device* what it can do rather than
assuming a hardware generation.  The information comes from three sources, in order
of preference:

1. ``cudaDeviceProp`` read by the compiled C++/CUDA extension (authoritative: exposes
   clock rates, L2 size, bus width, cooperative-launch support).
2. ``torch.cuda.get_device_properties`` (always available when torch sees a GPU).
3. Static tables keyed by compute capability, for facts the driver does not report
   (tensor-core generations, cores/SM, fp64 throughput ratio).

A ``DevicePolicy`` bundles the derived decisions -- which reduction primitive, which
vector width, which accumulate dtype -- so kernels and the pure-PyTorch fallback make
the *same* numerical choices on a given machine.
"""

from __future__ import annotations

import dataclasses
import functools
import os
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Static per-architecture facts the driver does not report.
# Keyed by compute-capability major version; (major, minor) entries win.
# ---------------------------------------------------------------------------

# FP32 CUDA cores per SM.
_CORES_PER_SM: dict[tuple[int, int], int] = {
    (3, 0): 192, (3, 5): 192, (3, 7): 192,
    (5, 0): 128, (5, 2): 128, (5, 3): 128,
    (6, 0): 64, (6, 1): 128, (6, 2): 128,
    (7, 0): 64, (7, 2): 64, (7, 5): 64,
    (8, 0): 64, (8, 6): 128, (8, 7): 128, (8, 9): 128,
    (9, 0): 128,
    (10, 0): 128, (10, 3): 128, (12, 0): 128,
}

# FP64 throughput as a fraction of FP32.  Determines whether a float64 accumulator
# is affordable (data-centre parts) or ruinous (consumer/workstation parts).
_FP64_RATIO: dict[tuple[int, int], float] = {
    (6, 0): 1 / 2, (6, 1): 1 / 32, (6, 2): 1 / 32,
    (7, 0): 1 / 2, (7, 2): 1 / 2, (7, 5): 1 / 32,
    (8, 0): 1 / 2, (8, 6): 1 / 64, (8, 7): 1 / 64, (8, 9): 1 / 64,
    (9, 0): 1 / 2,
    (10, 0): 1 / 2, (12, 0): 1 / 64,
}

_ARCH_NAME: dict[int, str] = {
    3: "Kepler", 5: "Maxwell", 6: "Pascal", 7: "Volta/Turing",
    8: "Ampere/Ada", 9: "Hopper", 10: "Blackwell", 12: "Blackwell",
}


def _lookup(table: dict[tuple[int, int], Any], major: int, minor: int, default: Any) -> Any:
    if (major, minor) in table:
        return table[(major, minor)]
    same_major = [k for k in table if k[0] == major]
    if same_major:
        return table[max(same_major)]
    return default


@dataclasses.dataclass(frozen=True)
class DeviceCaps:
    """What a single device can actually do."""

    index: int
    name: str
    major: int
    minor: int
    total_memory: int
    multi_processor_count: int
    warp_size: int
    max_threads_per_block: int
    max_threads_per_sm: int
    shared_memory_per_block: int
    shared_memory_per_sm: int
    regs_per_block: int
    l2_cache_size: int
    memory_bus_width: int          # bits; 0 if unknown
    memory_clock_khz: int          # 0 if unknown
    clock_khz: int                 # 0 if unknown
    cooperative_launch: bool
    source: str                    # "cuda_ext" | "torch" | "cpu"

    # -- derived capability flags -------------------------------------------------
    @property
    def sm(self) -> int:
        return self.major * 10 + self.minor

    @property
    def arch_name(self) -> str:
        return _ARCH_NAME.get(self.major, f"sm_{self.sm}")

    @property
    def has_fp16_arith(self) -> bool:
        """Native half-precision arithmetic (not just storage)."""
        return self.sm >= 53

    @property
    def has_fp16_tensor_cores(self) -> bool:
        return self.sm >= 70

    @property
    def has_bf16(self) -> bool:
        """Native bfloat16 arithmetic/tensor cores.  Turing (sm_75) does *not* have it;
        torch will happily emulate, which is slower than fp32 and must not be chosen."""
        return self.sm >= 80

    @property
    def has_tf32(self) -> bool:
        return self.sm >= 80

    @property
    def has_fp8(self) -> bool:
        return self.sm >= 89

    @property
    def has_redux_sync(self) -> bool:
        """``__reduce_add_sync`` / ``redux.sync`` warp reduction instruction."""
        return self.sm >= 80

    @property
    def has_async_copy(self) -> bool:
        """``cp.async`` global->shared without staging through registers."""
        return self.sm >= 80

    @property
    def has_cluster(self) -> bool:
        return self.sm >= 90

    @property
    def has_independent_thread_scheduling(self) -> bool:
        return self.sm >= 70

    @property
    def fp64_ratio(self) -> float:
        return _lookup(_FP64_RATIO, self.major, self.minor, 1 / 32)

    @property
    def fp64_is_cheap(self) -> bool:
        return self.fp64_ratio >= 1 / 8

    @property
    def cores_per_sm(self) -> int:
        return _lookup(_CORES_PER_SM, self.major, self.minor, 64)

    @property
    def peak_fp32_gflops(self) -> float:
        if not self.clock_khz:
            return 0.0
        return 2.0 * self.cores_per_sm * self.multi_processor_count * self.clock_khz * 1e3 / 1e9

    @property
    def peak_bandwidth_gbps(self) -> float:
        """Theoretical peak DRAM bandwidth.  GDDR is double-data-rate."""
        if not (self.memory_bus_width and self.memory_clock_khz):
            return 0.0
        return 2.0 * self.memory_clock_khz * 1e3 * (self.memory_bus_width / 8) / 1e9

    def summary(self) -> str:
        bw = self.peak_bandwidth_gbps
        gf = self.peak_fp32_gflops
        feats = [
            f"fp16={'y' if self.has_fp16_arith else 'n'}",
            f"fp16tc={'y' if self.has_fp16_tensor_cores else 'n'}",
            f"bf16={'y' if self.has_bf16 else 'n'}",
            f"tf32={'y' if self.has_tf32 else 'n'}",
            f"redux={'y' if self.has_redux_sync else 'n'}",
            f"cp.async={'y' if self.has_async_copy else 'n'}",
            f"fp64=1/{round(1 / self.fp64_ratio)}",
        ]
        return (
            f"[cuda:{self.index}] {self.name} sm_{self.sm} ({self.arch_name}) "
            f"{self.total_memory / 2**30:.1f} GiB, {self.multi_processor_count} SMs, "
            f"{self.shared_memory_per_block // 1024} KiB smem/block, "
            f"peak {bw:.0f} GB/s / {gf / 1000:.1f} TFLOP/s fp32 | " + " ".join(feats)
        )


@dataclasses.dataclass(frozen=True)
class DevicePolicy:
    """Numerical and launch decisions derived from ``DeviceCaps``.

    The same policy object drives both the CUDA kernels and the pure-PyTorch
    fallback so that switching implementations cannot silently change results.
    """

    caps: DeviceCaps
    reduce_dtype: torch.dtype        # accumulator for long reductions
    store_dtype: torch.dtype         # trajectory / basis storage
    compute_dtype: torch.dtype       # matmul compute
    vector_width: int                # elements per vectorised load (1/2/4)
    block_size: int
    max_basis_in_registers: int
    use_redux: bool
    use_async_copy: bool
    allow_tf32: bool

    def describe(self) -> dict[str, Any]:
        return {
            "device": self.caps.name,
            "sm": self.caps.sm,
            "arch": self.caps.arch_name,
            "reduce_dtype": str(self.reduce_dtype).replace("torch.", ""),
            "store_dtype": str(self.store_dtype).replace("torch.", ""),
            "compute_dtype": str(self.compute_dtype).replace("torch.", ""),
            "vector_width": self.vector_width,
            "block_size": self.block_size,
            "use_redux": self.use_redux,
            "use_async_copy": self.use_async_copy,
            "allow_tf32": self.allow_tf32,
            "peak_bandwidth_gbps": round(self.caps.peak_bandwidth_gbps, 1),
            "peak_fp32_gflops": round(self.caps.peak_fp32_gflops, 1),
            "source": self.caps.source,
        }


_CPU_CAPS = DeviceCaps(
    index=-1, name="cpu", major=0, minor=0, total_memory=0, multi_processor_count=0,
    warp_size=1, max_threads_per_block=1, max_threads_per_sm=1,
    shared_memory_per_block=0, shared_memory_per_sm=0, regs_per_block=0,
    l2_cache_size=0, memory_bus_width=0, memory_clock_khz=0, clock_khz=0,
    cooperative_launch=False, source="cpu",
)


def _caps_from_extension(index: int) -> DeviceCaps | None:
    """Ask the compiled extension, which reads the full ``cudaDeviceProp``."""
    try:
        from .kernels import get_extension

        ext = get_extension()
        if ext is None:
            return None
        d = ext.device_info(index)
    except Exception:
        return None
    return DeviceCaps(
        index=index, name=d["name"], major=d["major"], minor=d["minor"],
        total_memory=d["total_memory"], multi_processor_count=d["multi_processor_count"],
        warp_size=d["warp_size"], max_threads_per_block=d["max_threads_per_block"],
        max_threads_per_sm=d["max_threads_per_sm"],
        shared_memory_per_block=d["shared_memory_per_block"],
        shared_memory_per_sm=d["shared_memory_per_sm"], regs_per_block=d["regs_per_block"],
        l2_cache_size=d["l2_cache_size"], memory_bus_width=d["memory_bus_width"],
        memory_clock_khz=d["memory_clock_khz"], clock_khz=d["clock_khz"],
        cooperative_launch=bool(d["cooperative_launch"]), source="cuda_ext",
    )


def _caps_from_torch(index: int) -> DeviceCaps:
    p = torch.cuda.get_device_properties(index)
    getattr_ = lambda k, d=0: int(getattr(p, k, d) or d)  # noqa: E731
    return DeviceCaps(
        index=index, name=p.name, major=p.major, minor=p.minor,
        total_memory=p.total_memory,
        multi_processor_count=p.multi_processor_count,
        warp_size=getattr_("warp_size", 32),
        max_threads_per_block=getattr_("max_threads_per_block", 1024),
        max_threads_per_sm=getattr_("max_threads_per_multi_processor", 1024),
        shared_memory_per_block=getattr_("shared_memory_per_block", 48 * 1024),
        shared_memory_per_sm=getattr_("shared_memory_per_multiprocessor", 64 * 1024),
        regs_per_block=getattr_("regs_per_multiprocessor", 65536),
        l2_cache_size=getattr_("L2_cache_size"),
        memory_bus_width=getattr_("memory_bus_width"),
        memory_clock_khz=getattr_("memory_clock_rate"),
        clock_khz=getattr_("clock_rate"),
        cooperative_launch=False, source="torch",
    )


@functools.lru_cache(maxsize=16)
def get_caps(index: int | None = None) -> DeviceCaps:
    """Capabilities of a CUDA device, or a CPU sentinel when none is available."""
    if not torch.cuda.is_available():
        return _CPU_CAPS
    if index is None:
        index = torch.cuda.current_device()
    caps = _caps_from_extension(index)
    if caps is None:
        caps = _caps_from_torch(index)
    return caps


@functools.lru_cache(maxsize=16)
def get_policy(index: int | None = None) -> DevicePolicy:
    """Derive numerical/launch decisions from what the device reports.

    Rules, in order of importance:

    * **Accumulate wider than you store.**  Long reductions over :math:`10^7`--:math:`10^9`
      elements lose accuracy in fp32.  Use fp64 accumulation only where fp64 is not
      crippled (>= 1/8 rate); otherwise use pairwise/compensated fp32, which the
      kernels implement explicitly.
    * **Never pick bf16 on hardware without native bf16** -- torch reports it as
      "supported" via emulation on Turing, which is slower *and* less accurate than fp32.
    * **Vector width follows alignment guarantees**, and 128-bit (float4) loads are the
      widest the ISA supports.
    """
    caps = get_caps(index)
    if caps.source == "cpu":
        return DevicePolicy(
            caps=caps, reduce_dtype=torch.float64, store_dtype=torch.float32,
            compute_dtype=torch.float32, vector_width=1, block_size=1,
            max_basis_in_registers=8, use_redux=False, use_async_copy=False,
            allow_tf32=False,
        )

    reduce_dtype = torch.float64 if caps.fp64_is_cheap else torch.float32
    # Half-precision trajectory storage halves the dominant memory cost.  fp16 has
    # 10 explicit mantissa bits vs bf16's 7; on hardware without native bf16 fp16 is
    # both faster and more accurate for this use.  Displacements are pre-scaled to
    # unit RMS before casting, so fp16's narrow exponent range is not a hazard.
    if caps.has_bf16:
        store_dtype = torch.bfloat16
    elif caps.has_fp16_arith:
        store_dtype = torch.float16
    else:
        store_dtype = torch.float32

    # A block of 256 threads gives 8 warps: enough to hide latency on every
    # architecture from Kepler on, while keeping register pressure low enough for
    # full occupancy on the 64-core SMs of Volta/Turing.
    block_size = 256 if caps.max_threads_per_block >= 256 else caps.max_threads_per_block

    return DevicePolicy(
        caps=caps,
        reduce_dtype=reduce_dtype,
        store_dtype=store_dtype,
        compute_dtype=torch.float32,
        vector_width=4,
        block_size=block_size,
        max_basis_in_registers=8,
        use_redux=caps.has_redux_sync,
        use_async_copy=caps.has_async_copy,
        allow_tf32=caps.has_tf32,
    )


def all_caps() -> list[DeviceCaps]:
    if not torch.cuda.is_available():
        return [_CPU_CAPS]
    return [get_caps(i) for i in range(torch.cuda.device_count())]


def describe_environment() -> dict[str, Any]:
    """Machine-readable environment record, embedded in every result file."""
    import platform

    env: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cpu_count": os.cpu_count(),
        "devices": [dataclasses.asdict(c) | {"summary": c.summary()} for c in all_caps()],
    }
    if torch.cuda.is_available():
        env["policy"] = get_policy().describe()
    return env


def set_matmul_precision(policy: DevicePolicy | None = None) -> None:
    """Enable TF32 only where it exists; otherwise leave fp32 matmuls exact."""
    policy = policy or get_policy()
    torch.backends.cuda.matmul.allow_tf32 = policy.allow_tf32
    torch.backends.cudnn.allow_tf32 = policy.allow_tf32


def bytes_fmt(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def human_time(seconds: float) -> str:
    if seconds < 1e-6:
        return f"{seconds * 1e9:.1f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    if seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{int(m)}m {s:.0f}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h {int(m)}m"


__all__ = [
    "DeviceCaps", "DevicePolicy", "get_caps", "get_policy", "all_caps",
    "describe_environment", "set_matmul_precision", "bytes_fmt", "human_time",
]
