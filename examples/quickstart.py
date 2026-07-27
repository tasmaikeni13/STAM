"""Attach STAM to a training loop you already have.

This is a complete, self-contained example: a small MLP on synthetic data, trained the
way anyone would train it, with two extra lines. Run it and you get a certified
landscape figure and an animation of the mini-batch surface.

    python examples/quickstart.py

The same two lines work for any ``nn.Module`` and any loss: the only thing STAM needs
from you is a ``loss_fn(model, batch)`` and a fixed set of batches to define the surface.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from stam import LandscapeRecorder


def main() -> None:
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- an ordinary model, an ordinary loss, ordinary data -------------------
    model = nn.Sequential(
        nn.Linear(20, 64), nn.GELU(),
        nn.Linear(64, 64), nn.GELU(),
        nn.Linear(64, 3),
    ).to(device)

    def loss_fn(m: nn.Module, batch) -> torch.Tensor:
        x, y = batch
        # Returning per-example losses is preferred: it gives the certificate its
        # variance estimate for free.  A scalar works too.
        return F.cross_entropy(m(x), y, reduction="none")

    n, bs = 8192, 128
    X = torch.randn(n, 20, device=device)
    W = torch.randn(20, 3, device=device)
    Y = (X @ W + 0.6 * torch.randn(n, 3, device=device)).argmax(1)
    train = [(X[i : i + bs], Y[i : i + bs]) for i in range(0, n, bs)]

    # The evaluation set defines the surface being drawn, so it is fixed, not shuffled.
    eval_batches = train[:16]

    # --- STAM, line 1 ---------------------------------------------------------
    recorder = LandscapeRecorder(model, loss_fn, eval_batches, every=10, name="mlp")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    for epoch in range(30):
        for batch in train:
            recorder.zero_grad()            # keeps STAM's contiguous gradient views
            loss_fn(model, batch).mean().backward()
            opt.step()
            recorder.step()                 # --- STAM, line 2 -------------------
        if epoch % 10 == 0:
            with torch.no_grad():
                ev = torch.cat([loss_fn(model, b) for b in eval_batches]).mean()
            print(f"  epoch {epoch:3d}  loss {ev:.4f}")

    print(f"recorded {recorder.n_snapshots} snapshots of "
          f"{sum(p.numel() for p in model.parameters()):,} parameters")

    # --- STAM, line 3: render, with a stated compute budget -------------------
    report = recorder.render("stam_out", budget_seconds=20.0, resolution=49)
    print(report.summary())
    print(f"  figure    {report.figure}")
    print(f"  animation {report.animation}")
    print(f"  arrays    {report.arrays}")


if __name__ == "__main__":
    main()
