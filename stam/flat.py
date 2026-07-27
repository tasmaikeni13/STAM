"""Contiguous parameter and gradient views.

Probing a landscape means repeatedly writing a full parameter vector into a model and
reading a full gradient out of it.  Done naively -- a Python loop over parameter
tensors, one ``copy_`` and one ``view`` per tensor -- this costs one kernel launch per
tensor per probe and forces a ``torch.cat`` for every gradient read.  For a model with
:math:`P` parameter tensors that is :math:`\\Theta(P)` launches where :math:`\\Theta(1)`
suffices, and the per-anchor fixed overhead :math:`\\tau` it creates enters the optimal
design directly (Section: budget allocation).

``FlatParams`` removes it by rebinding every parameter to a *view into a single
contiguous buffer* and pre-allocating gradient views into a second buffer.  After
rebinding:

* writing a parameter vector is one ``copy_`` of :math:`N` elements;
* reading the gradient is a zero-copy reference to a contiguous tensor;
* optimiser steps still work, because in-place updates on the views write through
  to the backing buffer.

The one requirement this places on callers is ``zero_grad(set_to_none=False)``: setting
gradients to ``None`` would drop the pre-allocated views and silently restore the slow
path.  ``FlatParams.zero_grad`` does the right thing.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable, Iterator

import torch
import torch.nn as nn


@dataclasses.dataclass(frozen=True)
class ParamSlice:
    name: str
    offset: int
    numel: int
    shape: torch.Size


class FlatParams:
    """A model whose trainable parameters alias one contiguous vector.

    Parameters
    ----------
    model:
        The module to rebind.  Modified in place; the module remains fully usable
        (forward, backward, optimisers) afterwards.
    dtype:
        Storage dtype of the flat buffer.  Defaults to the dtype of the first
        parameter.  All trainable parameters must share it.
    """

    def __init__(self, model: nn.Module, dtype: torch.dtype | None = None):
        params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if not params:
            raise ValueError("model has no trainable parameters")
        dev = params[0][1].device
        dtype = dtype or params[0][1].dtype
        for n, p in params:
            if p.dtype != dtype:
                raise ValueError(
                    f"parameter {n!r} has dtype {p.dtype}, expected {dtype}; "
                    "mixed-dtype models must be cast before flattening"
                )
            if p.device != dev:
                raise ValueError(f"parameter {n!r} is on {p.device}, expected {dev}")

        self.model = model
        self.device = dev
        self.dtype = dtype
        self.slices: list[ParamSlice] = []
        offset = 0
        for n, p in params:
            self.slices.append(ParamSlice(n, offset, p.numel(), p.shape))
            offset += p.numel()
        self.numel = offset

        # Backing storage.  Built from the current values so the rebind is a no-op
        # numerically.
        self._vec = torch.empty(self.numel, dtype=dtype, device=dev)
        self._grad = torch.zeros(self.numel, dtype=dtype, device=dev)
        with torch.no_grad():
            for sl, (_, p) in zip(self.slices, params):
                self._vec[sl.offset : sl.offset + sl.numel].copy_(p.detach().reshape(-1))

        # Rebind: each parameter's storage becomes a window of the flat buffer.
        self._params: list[nn.Parameter] = []
        for sl, (_, p) in zip(self.slices, params):
            window = self._vec[sl.offset : sl.offset + sl.numel].view(sl.shape)
            p.data = window
            p.grad = self._grad[sl.offset : sl.offset + sl.numel].view(sl.shape)
            self._params.append(p)

        # Buffers (batch-norm running statistics, and similar) are *state*, not
        # coordinates: they are not part of the parameter vector and must be restored
        # rather than interpolated when probing.  Keep a reference so probes can
        # snapshot and restore them.
        self._buffers = [(n, b) for n, b in model.named_buffers() if b.dtype.is_floating_point]

    # -- vector access ---------------------------------------------------------

    @property
    def vector(self) -> torch.Tensor:
        """The parameter vector.  Aliases the model's parameters; writes are visible
        to the model immediately and require no synchronisation."""
        return self._vec

    @property
    def grad_vector(self) -> torch.Tensor:
        """The gradient vector.  Populated in place by ``backward()``."""
        return self._grad

    def clone_vector(self) -> torch.Tensor:
        return self._vec.detach().clone()

    @torch.no_grad()
    def set_vector(self, v: torch.Tensor) -> None:
        """Load a parameter vector.  One device-to-device copy of ``numel`` elements."""
        if v.numel() != self.numel:
            raise ValueError(f"expected {self.numel} elements, got {v.numel()}")
        self._vec.copy_(v.reshape(-1))

    @torch.no_grad()
    def add_scaled(self, direction: torch.Tensor, alpha: float) -> None:
        self._vec.add_(direction.reshape(-1), alpha=alpha)

    def zero_grad(self) -> None:
        """Zero gradients *without* dropping the pre-allocated views."""
        self._grad.zero_()

    # -- structured views ------------------------------------------------------

    def parameters(self) -> list[nn.Parameter]:
        return self._params

    def unflatten(self, v: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            sl.name: v[sl.offset : sl.offset + sl.numel].view(sl.shape) for sl in self.slices
        }

    def snapshot_buffers(self) -> list[torch.Tensor]:
        return [b.detach().clone() for _, b in self._buffers]

    @torch.no_grad()
    def restore_buffers(self, snap: Iterable[torch.Tensor]) -> None:
        for (_, b), s in zip(self._buffers, snap):
            b.copy_(s)

    # -- diagnostics -----------------------------------------------------------

    def bytes(self) -> int:
        return 2 * self.numel * self._vec.element_size()

    def __len__(self) -> int:
        return self.numel

    def __iter__(self) -> Iterator[ParamSlice]:
        return iter(self.slices)

    def __repr__(self) -> str:
        return (
            f"FlatParams(numel={self.numel:,}, tensors={len(self.slices)}, "
            f"dtype={self.dtype}, device={self.device})"
        )


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


__all__ = ["FlatParams", "ParamSlice", "count_parameters"]
