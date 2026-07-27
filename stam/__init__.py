"""STAM — Subspace Taylor Anchored Mapping.

Loss-landscape visualisation posed as budgeted stochastic function estimation on a
2-plane: probe few places deeply, take the derivatives that are cheap there, reconstruct
with partition-of-unity Taylor patches, and ship the result with a measured error.

Typical use::

    from stam.pipeline import certified_landscape
    from stam.probe import PlaneProbe, calibrate_cost

    cost = calibrate_cost(probe)
    result = certified_landscape(probe, domain, budget, cost, render_points)
    print(result.certificate.summary())

The submodules stand alone: :mod:`stam.probe` for restricted derivatives,
:mod:`stam.basis` for the plane, :mod:`stam.certify` for the certificate, and
:mod:`stam.kernels` for the fused CUDA operations.
"""

__version__ = "1.0.0"

from .api import CallableTask, LandscapeRecorder, RenderReport  # noqa: E402

__all__ = [
    "LandscapeRecorder", "CallableTask", "RenderReport",
    "basis", "capture", "certify", "data", "design", "device", "fidelity", "flat",
    "kernels", "metrics", "models", "parallel", "pipeline", "probe", "reconstruct",
    "viz",
]
