r"""Probing the restricted loss: value, exact restricted gradient, exact restricted Hessian.

Fix an affine plane :math:`\mathcal S = c + \mathrm{span}\{v_1,v_2\}` and write
:math:`\ell(\alpha) = L(c + \alpha_1 v_1 + \alpha_2 v_2)` for the *restricted* loss on
:math:`\mathbb R^2`.  The observation that drives everything downstream is that at a
point on :math:`\mathcal S` one can obtain, at a small constant multiple of the cost of
the value alone:

.. math::
   \ell(\alpha)\in\mathbb R,\qquad
   \nabla\ell(\alpha) = V\nabla L \in\mathbb R^2,\qquad
   \nabla^2\ell(\alpha) = V\nabla^2 L\,V^\top \in\mathbb R^{2\times2},

where :math:`V\in\mathbb R^{2\times N}` stacks the basis vectors.  The gradient is one
backward pass followed by two inner products.  The Hessian is **two Hessian-vector
products** -- :math:`\nabla^2 L\,v_1` and :math:`\nabla^2 L\,v_2` -- each a
double-backward, followed by four inner products (three after symmetrisation).  Crucially
the count does not depend on :math:`N`: the restricted Hessian of a 2-plane is a
:math:`2\times2` matrix no matter how large the model is.

So one probe yields a complete second-order Taylor model of :math:`\ell` at
:math:`\alpha`, at cost :math:`\kappa_2` times the value-only cost, with
:math:`\kappa_2` a small measured constant.  A value-only scheme would have to recover
that same model by differencing six nearby evaluations, which both costs more and
amplifies their noise by :math:`\Theta(h^{-2})`.

Every probe also returns an *estimate of its own noise*, obtained from the same data:
the sample variance for the value, and a half-split estimator for the derivatives.
Those variances are what make the reconstruction certifiable rather than merely plotted.
"""

from __future__ import annotations

import contextlib
import dataclasses
import time
from typing import Any, Sequence

import numpy as np
import torch

from . import kernels
from .basis import Plane
from .data import EvalSet
from .flat import FlatParams
from .models import Task


@dataclasses.dataclass
class ProbeResult:
    """One measurement of the restricted loss and its derivatives."""

    coords: np.ndarray            # (K,) location in the plane
    order: int                    # 0 value, 1 +gradient, 2 +Hessian
    n_examples: int
    value: float
    value_var: float              # variance of the value estimator
    grad: np.ndarray | None       # (K,)
    grad_var: np.ndarray | None   # (K,) per-component variance of the estimator
    hess: np.ndarray | None       # (K, K) symmetric
    hess_var: np.ndarray | None   # (K, K)
    seconds: float
    cost_units: float             # example-forward-equivalents, including fixed overhead

    def hess_packed(self) -> np.ndarray:
        """Upper-triangular packing ``(xx, xy, yy)`` expected by the reconstruction."""
        if self.hess is None:
            return np.zeros(3, dtype=np.float64)
        H = self.hess
        return np.array([H[0, 0], H[0, 1], H[1, 1]], dtype=np.float64)

    def sharpness(self) -> float:
        """Largest eigenvalue of the restricted Hessian; ``nan`` if not measured."""
        if self.hess is None:
            return float("nan")
        return float(np.linalg.eigvalsh(self.hess)[-1])


@dataclasses.dataclass
class CostModel:
    r"""Measured cost of a probe, in example-forward-equivalents.

    ``cost(order, B) = tau[order] + kappa[order] * B``

    :math:`\kappa` is the marginal cost per example relative to a plain forward pass;
    :math:`\tau` is the fixed per-anchor overhead (writing :math:`\theta` into the model,
    kernel launches, autograd graph setup) expressed in the same unit.  Both are
    measured, never assumed, because :math:`\tau` in particular is what the fused
    kernels change and what shifts the optimal design.
    """

    kappa: dict[int, float]
    tau: dict[int, float]
    seconds_per_example: float
    device: str
    detail: dict[str, Any] = dataclasses.field(default_factory=dict)

    def cost(self, order: int, n_examples: int) -> float:
        return self.tau[order] + self.kappa[order] * n_examples

    def examples_for_budget(self, order: int, budget: float) -> int:
        return max(1, int((budget - self.tau[order]) / self.kappa[order]))

    def to_dict(self) -> dict:
        return {
            "kappa": {str(k): round(v, 4) for k, v in self.kappa.items()},
            "tau_example_equivalents": {str(k): round(v, 2) for k, v in self.tau.items()},
            "seconds_per_example": self.seconds_per_example,
            "device": self.device,
            **self.detail,
        }


def _chunk_examples(chunks: Sequence) -> int:
    """Number of examples in a list of micro-batches, whatever their container type."""
    from .data import _infer_size

    return int(sum(_infer_size(c) for c in chunks))


def _math_sdpa() -> contextlib.AbstractContextManager:
    """Force the math attention backend.

    The fused attention kernels do not implement double backward, which the
    Hessian-vector products require.  Selecting the math path explicitly turns a
    runtime error into a slower but correct computation, and keeps the cost model
    honest about what second-order probing actually costs for a transformer.
    """
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return sdpa_kernel([SDPBackend.MATH])
    except Exception:  # pragma: no cover - older torch
        return contextlib.nullcontext()


class PlaneProbe:
    """Measures the restricted loss on a plane, with variance estimates."""

    def __init__(
        self,
        task: Task,
        flat: FlatParams,
        plane: Plane,
        eval_set: EvalSet,
        micro_batch: int = 250,
        cost_model: CostModel | None = None,
        micro_batch_by_order: dict[int, int] | None = None,
    ):
        self.task = task
        self.flat = flat
        self.plane = plane
        self.eval_set = eval_set
        self.micro_batch = micro_batch
        self.cost_model = cost_model
        # Second-order probing keeps the forward graph alive for a double backward, and
        # for a transformer the math attention backend materialises the full T x T
        # attention weights per layer as well.  Its working set is therefore far larger
        # than a forward pass at the same batch size, and a micro-batch tuned on a
        # forward pass will exhaust device memory here.  Each order gets its own.
        self.micro_batch_by_order = dict(micro_batch_by_order or {})
        self.micro_batch_by_order.setdefault(0, micro_batch)
        self.micro_batch_by_order.setdefault(1, max(1, micro_batch // 2))
        self.micro_batch_by_order.setdefault(2, max(1, micro_batch // 8))

        self._scratch = torch.empty_like(flat.vector)
        self._views = [
            self._scratch[s.offset : s.offset + s.numel].view(s.shape) for s in flat.slices
        ]
        self._params = flat.parameters()
        self._K = plane.dim
        self._basis_views = [
            [plane.basis[k, s.offset : s.offset + s.numel].view(s.shape) for s in flat.slices]
            for k in range(self._K)
        ]

    def micro_for(self, order: int) -> int:
        return self.micro_batch_by_order.get(order, self.micro_batch)

    # -- helpers ---------------------------------------------------------------

    def _project_tuple(self, grads: Sequence[torch.Tensor]) -> torch.Tensor:
        """Project a per-tensor gradient tuple onto the plane basis.

        The tuple is gathered into the flat scratch buffer with a single multi-tensor
        copy, then reduced by the fused projection kernel: two kernel launches instead
        of :math:`2P` inner products.
        """
        torch._foreach_copy_(self._views, list(grads))
        return kernels.project(self._scratch, self.plane.basis)

    def _draw(self, n_examples: int, order: int, generator: torch.Generator) -> list:
        """Micro-batches for one probe, obtained from the evaluation set itself."""
        return list(self.eval_set.sample_chunks(n_examples, self.micro_for(order), generator))

    # -- the three measurement modes ------------------------------------------

    def _value_only(self, chunks: Sequence) -> tuple[float, float]:
        """Mean and sample variance of the per-example loss."""
        s = 0.0
        ss = 0.0
        n = 0
        with torch.no_grad():
            for batch in chunks:
                per = self.task.per_example_loss(batch).double()
                s += float(per.sum())
                ss += float((per * per).sum())
                n += per.numel()
        mean = s / n
        var = max(ss / n - mean * mean, 0.0) * n / max(n - 1, 1)
        return mean, var

    def _value_grad(self, chunks: Sequence) -> tuple[float, float, torch.Tensor]:
        """Mean loss, per-example loss variance, and the exact restricted gradient."""
        self.flat.zero_grad()
        s = 0.0
        ss = 0.0
        n = 0
        for batch in chunks:
            per = self.task.per_example_loss(batch)
            per.sum().backward()
            d = per.detach().double()
            s += float(d.sum())
            ss += float((d * d).sum())
            n += per.numel()
        mean = s / n
        var = max(ss / n - mean * mean, 0.0) * n / max(n - 1, 1)
        # backward accumulated the *sum* of per-example gradients into the flat buffer.
        g = kernels.project(self.flat.grad_vector, self.plane.basis) / n
        return mean, var, g

    def _value_grad_hess(
        self, chunks: Sequence
    ) -> tuple[float, float, torch.Tensor, torch.Tensor]:
        """As above, plus the exact restricted Hessian via two Hessian-vector products."""
        K = self._K
        g_acc = torch.zeros(K, device=self.plane.basis.device, dtype=torch.float32)
        H_acc = torch.zeros(K, K, device=self.plane.basis.device, dtype=torch.float32)
        s = 0.0
        ss = 0.0
        n = 0
        with _math_sdpa():
            for batch in chunks:
                per = self.task.per_example_loss(batch)
                total = per.sum()
                grads = torch.autograd.grad(total, self._params, create_graph=True)
                d = per.detach().double()
                s += float(d.sum())
                ss += float((d * d).sum())
                n += per.numel()

                # <grad, v_k> without materialising a flat gradient: the graph must be
                # kept alive, so the multi-tensor copy path is unavailable here.
                inner = [
                    sum((gi * vi).sum() for gi, vi in zip(grads, self._basis_views[k]))
                    for k in range(K)
                ]
                g_acc += torch.stack([x.detach() for x in inner]).float()
                for k in range(K):
                    hv = torch.autograd.grad(inner[k], self._params, retain_graph=(k < K - 1))
                    H_acc[k] = self._project_tuple(hv)
                del grads, inner
        mean = s / n
        var = max(ss / n - mean * mean, 0.0) * n / max(n - 1, 1)
        H = H_acc / n
        # The two off-diagonal routes to H_12 are equal in exact arithmetic; averaging
        # them symmetrises and their difference is a free numerical health check.
        H = 0.5 * (H + H.t())
        return mean, var, g_acc / n, H

    # -- public interface ------------------------------------------------------

    def probe(
        self,
        coords: Sequence[float] | np.ndarray | torch.Tensor,
        n_examples: int,
        order: int = 2,
        generator: torch.Generator | None = None,
        half_split: bool = True,
    ) -> ProbeResult:
        r"""Measure :math:`\ell` and its derivatives at ``coords`` using ``n_examples`` draws.

        ``half_split`` splits the draw into two independent halves and uses their
        difference to estimate the variance of the derivative estimators.  For half-means
        :math:`\hat g_A,\hat g_B` of :math:`B/2` examples each,
        :math:`\widehat{\mathrm{Var}}(\hat g) = (\hat g_A-\hat g_B)^2/4` is unbiased for
        the variance of the full-sample mean -- a single-degree-of-freedom estimator,
        noisy but exactly right on average, and free.
        """
        device = self.flat.device
        gen = generator or torch.Generator(device="cpu").manual_seed(0)
        coords_t = torch.as_tensor(np.asarray(coords, dtype=np.float32))

        t0 = time.perf_counter()
        self.plane.point(coords_t.to(device), out=self.flat.vector)

        M = len(self.eval_set)
        n_examples = int(min(n_examples, M))
        chunks = self._draw(n_examples, order, gen)
        fpc = max(0.0, 1.0 - n_examples / M)  # finite-population correction

        halves: list[list]
        if half_split and order >= 1 and len(chunks) >= 2:
            mid = len(chunks) // 2
            halves = [chunks[:mid], chunks[mid:]]
        else:
            halves = [chunks]

        vals, vars_, grads, hesses, counts = [], [], [], [], []
        for part in halves:
            if order <= 0:
                v, s2 = self._value_only(part)
                g = h = None
            elif order == 1:
                v, s2, g = self._value_grad(part)
                h = None
            else:
                v, s2, g, h = self._value_grad_hess(part)
            vals.append(v)
            vars_.append(s2)
            grads.append(g)
            hesses.append(h)
            counts.append(_chunk_examples(part))

        n_drawn = int(sum(counts))
        w = np.asarray(counts, dtype=np.float64) / max(sum(counts), 1)
        value = float(np.dot(w, vals))
        # Pooled per-example variance -> variance of the mean, with the FPC applied.
        pooled = float(np.dot(w, vars_))
        value_var = pooled / n_drawn * fpc if n_drawn > 1 else float("nan")

        grad = grad_var = hess = hess_var = None
        if order >= 1:
            gs = torch.stack([g for g in grads if g is not None]).double().cpu().numpy()
            grad = (gs * w[:, None]).sum(0)
            if len(halves) == 2:
                grad_var = ((gs[0] - gs[1]) ** 2 / 4.0) * fpc
            else:
                grad_var = np.full_like(grad, np.nan)
        if order >= 2:
            hs = torch.stack([h for h in hesses if h is not None]).double().cpu().numpy()
            hess = (hs * w[:, None, None]).sum(0)
            if len(halves) == 2:
                hess_var = ((hs[0] - hs[1]) ** 2 / 4.0) * fpc
            else:
                hess_var = np.full_like(hess, np.nan)

        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        cost = (
            self.cost_model.cost(order, n_drawn)
            if self.cost_model is not None
            else float(n_drawn)
        )
        return ProbeResult(
            coords=np.asarray(coords, dtype=np.float64), order=order, n_examples=n_drawn,
            value=value, value_var=value_var, grad=grad, grad_var=grad_var, hess=hess,
            hess_var=hess_var, seconds=seconds, cost_units=cost,
        )

    def exact(self, coords: Sequence[float], order: int = 0, batch_size: int | None = None
              ) -> ProbeResult:
        """Evaluate over the *entire* evaluation set: ground truth, zero sampling error."""
        M = len(self.eval_set)
        saved = dict(self.micro_batch_by_order)
        if batch_size:
            self.micro_batch_by_order = {k: min(batch_size, v)
                                         for k, v in self.micro_batch_by_order.items()}
        try:
            device = self.flat.device
            coords_t = torch.as_tensor(np.asarray(coords, dtype=np.float32))
            t0 = time.perf_counter()
            self.plane.point(coords_t.to(device), out=self.flat.vector)
            chunks = list(self.eval_set.all_chunks(self.micro_for(order)))
            if order <= 0:
                v, s2 = self._value_only(chunks)
                g = h = None
            elif order == 1:
                v, s2, g = self._value_grad(chunks)
                h = None
            else:
                v, s2, g, h = self._value_grad_hess(chunks)
            if device.type == "cuda":
                torch.cuda.synchronize()
            return ProbeResult(
                coords=np.asarray(coords, dtype=np.float64), order=order, n_examples=M,
                value=v, value_var=0.0,
                grad=g.double().cpu().numpy() if g is not None else None,
                grad_var=np.zeros(self._K) if g is not None else None,
                hess=h.double().cpu().numpy() if h is not None else None,
                hess_var=np.zeros((self._K, self._K)) if h is not None else None,
                seconds=time.perf_counter() - t0,
                cost_units=float(M) * (self.cost_model.kappa[order] if self.cost_model else 1.0),
            )
        finally:
            self.micro_batch_by_order = saved


# ---------------------------------------------------------------------------
# Cost calibration
# ---------------------------------------------------------------------------


def calibrate_cost(
    probe: PlaneProbe,
    sizes: Sequence[int] | None = None,
    repeats: int = 3,
    warmup: int = 2,
) -> CostModel:
    r"""Measure :math:`\kappa` and :math:`\tau` by timing probes at several sample sizes.

    Fitting ``seconds = a + b * B`` separates the marginal per-example cost ``b`` from
    the fixed per-anchor overhead ``a``.  Normalising both by the order-0 marginal cost
    puts them in example-forward-equivalents, the unit every budget in this work is
    quoted in.
    """
    device = probe.flat.device
    if sizes is None:
        # Straddle the micro-batch: sample sizes entirely below it all cost one
        # micro-batch, so the regression would attribute their common cost to the
        # intercept and report a fixed overhead that does not exist.
        mb = max(probe.micro_batch, 1)
        sizes = tuple(
            min(int(mb * f), len(probe.eval_set)) for f in (0.5, 1, 2, 4)
        )
        sizes = tuple(sorted(set(s for s in sizes if s >= 1)))
    gen = torch.Generator(device="cpu").manual_seed(12345)
    coords = np.zeros(probe.plane.dim, dtype=np.float32)
    fits: dict[int, tuple[float, float]] = {}
    raw: dict[str, Any] = {}

    for order in (0, 1, 2):
        xs, ys = [], []
        for _ in range(warmup):
            probe.probe(coords, sizes[0], order=order, generator=gen, half_split=False)
        for B in sizes:
            ts = []
            for _ in range(repeats):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                probe.probe(coords, B, order=order, generator=gen, half_split=False)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            xs.append(B)
            ys.append(float(np.median(ts)))
        A = np.stack([np.ones(len(xs)), np.asarray(xs, dtype=np.float64)], 1)
        coef, *_ = np.linalg.lstsq(A, np.asarray(ys), rcond=None)
        fits[order] = (max(float(coef[0]), 0.0), max(float(coef[1]), 1e-12))
        raw[f"order{order}_seconds"] = dict(zip(map(str, xs), [round(y, 6) for y in ys]))

    base = fits[0][1]
    kappa = {o: fits[o][1] / base for o in fits}
    tau = {o: fits[o][0] / base for o in fits}
    return CostModel(
        kappa=kappa, tau=tau, seconds_per_example=base,
        device=str(device),
        detail={"fit": raw, "micro_batch": probe.micro_batch,
                "micro_batch_by_order": dict(probe.micro_batch_by_order)},
    )


def micro_table(value, default: int = 256) -> dict[int, int]:
    """Normalise a recorded micro-batch setting into a per-order table.

    Accepts either a single integer (older runs, tuned on a forward pass) or the
    per-order mapping produced by :func:`tune_micro_batches`.
    """
    if isinstance(value, dict):
        return {int(k): int(v) for k, v in value.items()}
    v = int(value) if value else default
    return {0: v, 1: max(1, v // 2), 2: max(1, v // 8)}


def tune_micro_batches(
    probe: "PlaneProbe",
    orders: Sequence[int] = (0, 1, 2),
    start: int = 32,
    maximum: int | None = None,
    target_fraction: float = 0.72,
) -> dict[int, int]:
    """Autotune the micro-batch separately for each probe order.

    Tuning on a forward pass and reusing the result for second-order probing is the
    obvious mistake and an expensive one: the double-backward working set can be an
    order of magnitude larger, so the run dies partway through a long reference sweep.
    Each order is tuned by actually running it.
    """
    from .parallel import autotune_micro_batch

    device = probe.flat.device
    maximum = maximum or min(1 << 14, len(probe.eval_set))
    gen = torch.Generator(device="cpu").manual_seed(4242)
    coords = np.zeros(probe.plane.dim, dtype=np.float32)
    out: dict[int, int] = {}
    for order in orders:
        def run(b: int, order=order) -> None:
            probe.micro_batch_by_order[order] = b
            probe.probe(coords, b, order=order, generator=gen, half_split=False)

        tuning = autotune_micro_batch(
            run, device, start=start, maximum=maximum,
            target_fraction=target_fraction, repeats=1,
        )
        # The tuner measures one probe in isolation; a long sweep runs thousands
        # back to back, and allocator fragmentation makes the steady-state peak higher
        # than the measured one.  A one-step back-off on the orders that build an
        # autograd graph costs a little throughput and removes the failure mode where a
        # multi-hour reference run dies halfway through.
        backoff = 1 if order == 0 else 2
        out[order] = max(1, tuning.micro_batch // backoff)
        probe.micro_batch_by_order[order] = out[order]
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


__all__ = ["ProbeResult", "CostModel", "PlaneProbe", "calibrate_cost",
           "tune_micro_batches", "micro_table"]
