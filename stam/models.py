"""Models used as landscape subjects.

Two are needed, and they are chosen to differ in the property the analysis is most
sensitive to: the smoothness of :math:`L` restricted to a plane.

``cnn``
    A 403k-parameter convolutional network with ReLU activations.  The loss is only
    *piecewise* smooth in the parameters -- the kink set of each example is a
    codimension-1 arrangement, and a generic 2-plane cuts it in curves.  This is the
    hard case for any Taylor-based reconstruction, and the case where the empirical
    certificate has to earn its keep.
``gpt``
    A 5.0M-parameter decoder-only transformer with GELU activations, trained on
    WikiText-2.  GELU is :math:`C^\\infty`, so the restricted loss genuinely is smooth
    and the asymptotic theory applies without caveat.

Neither uses batch normalisation.  BN would make the loss scale-invariant in the
weights of each normalised layer, so distances in the plane would carry no meaning --
a real problem for landscape visualisation, and one deliberately kept out of the
main experiments so that the error analysis is not confounded with it.
"""

from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Convolutional network
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CNNConfig:
    in_channels: int = 3
    num_classes: int = 10
    widths: tuple[int, ...] = (32, 32, 64, 64, 128)
    hidden: int = 128
    image_size: int = 32


class ConvNet(nn.Module):
    """VGG-style network without normalisation layers."""

    def __init__(self, cfg: CNNConfig = CNNConfig()):
        super().__init__()
        self.cfg = cfg
        layers: list[nn.Module] = []
        c_in = cfg.in_channels
        size = cfg.image_size
        for i, c_out in enumerate(cfg.widths):
            layers += [nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU(inplace=True)]
            if i % 2 == 1 or i == len(cfg.widths) - 1:
                layers.append(nn.MaxPool2d(2))
                size //= 2
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c_in * size * size, cfg.hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden, cfg.num_classes),
        )
        self.apply(_init_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def _init_conv(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# Decoder-only transformer
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 7040
    context: int = 256
    d_model: int = 256
    n_layer: int = 4
    n_head: int = 4
    d_ff: int = 1024
    tie_embeddings: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        shape = (B, T, self.n_head, self.d_head)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        # SDPA's fused kernels are used for training; double-backward (needed for the
        # Hessian-vector products) falls back to the math path automatically.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        # GELU rather than ReLU: the restricted loss is then C-infinity, which is the
        # regime the Taylor-order analysis assumes.
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x)), approximate="tanh"))
        return x


class TinyGPT(nn.Module):
    """Pre-LayerNorm decoder-only transformer."""

    def __init__(self, cfg: GPTConfig = GPTConfig()):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.context, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.tok.weight
        self.apply(self._init)
        # Residual-branch scaling keeps activation variance constant with depth.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None]
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


# ---------------------------------------------------------------------------
# Task wrappers: a model plus the loss it defines
# ---------------------------------------------------------------------------


class Task(nn.Module):
    """A model together with per-example loss semantics.

    ``per_example_loss`` is what makes the variance of a probe measurable: the
    certificate needs :math:`\\widehat{\\sigma}^2` at every anchor, and getting it from
    a reduction-free loss costs nothing beyond the forward pass already being done.
    """

    def __init__(self, model: nn.Module, name: str):
        super().__init__()
        self.model = model
        self.name = name

    def forward(self, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return self.per_example_loss(batch).mean()

    def per_example_loss(self, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def accuracy(self, batch: tuple[torch.Tensor, ...]) -> float:
        raise NotImplementedError


class ClassificationTask(Task):
    def per_example_loss(self, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        x, y = batch
        return F.cross_entropy(self.model(x), y, reduction="none")

    @torch.no_grad()
    def accuracy(self, batch: tuple[torch.Tensor, ...]) -> float:
        x, y = batch
        return float((self.model(x).argmax(1) == y).float().mean())


class LanguageModelTask(Task):
    def per_example_loss(self, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Mean cross-entropy per *sequence*.

        Averaging within a sequence before averaging across sequences makes the
        sequence the sampling unit, so a batch of sequences is an i.i.d. sample and the
        variance estimator is valid.  Token-level averaging across a batch would
        correlate the unit of variance with sequence length.
        """
        x, y = batch
        logits = self.model(x)
        tok = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none"
        )
        return tok.view(y.shape).mean(dim=1)

    @torch.no_grad()
    def accuracy(self, batch: tuple[torch.Tensor, ...]) -> float:
        x, y = batch
        return float((self.model(x).argmax(-1) == y).float().mean())


def build_task(name: str, device: torch.device | str = "cuda") -> Task:
    """Construct one of the two landscape subjects."""
    if name == "cnn":
        task: Task = ClassificationTask(ConvNet(), "cnn")
    elif name == "gpt":
        task = LanguageModelTask(TinyGPT(), "gpt")
    else:
        raise ValueError(f"unknown model {name!r}; expected 'cnn' or 'gpt'")
    return task.to(device)


def parameter_report(model: nn.Module) -> dict[str, int | float]:
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    by_kind: dict[str, int] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        kind = name.split(".")[0]
        by_kind[kind] = by_kind.get(kind, 0) + p.numel()
    return {"total": total, "millions": round(total / 1e6, 3), **by_kind}


__all__ = [
    "CNNConfig", "ConvNet", "GPTConfig", "TinyGPT", "Task", "ClassificationTask",
    "LanguageModelTask", "build_task", "parameter_report",
]
