"""Datasets, and the definition of the quantity being visualised.

The central design decision here is what "the loss landscape" *is*.  Almost all prior
work leaves this implicit, which makes the accuracy question unanswerable: if the
target is the population risk, no finite computation ever evaluates it exactly, and
there is no ground truth to compare a reconstruction against.

STAM fixes the target explicitly:

.. math::
   \\mathcal L_{\\mathcal D}(\\theta) = \\frac{1}{|\\mathcal D|}\\sum_{z\\in\\mathcal D}
   \\ell(\\theta; z)

for a **fixed finite evaluation set** :math:`\\mathcal D`, held in device memory and
never reshuffled.  Three things follow:

1. :math:`\\mathcal L_{\\mathcal D}` is computable exactly, so dense reference surfaces
   are genuine ground truth rather than another estimate.
2. A probe that draws :math:`B` elements of :math:`\\mathcal D` uniformly without
   replacement is an unbiased estimator of it, with a variance that is itself
   estimable from the same draw.
3. Cost is measurable in a single natural unit -- one example forward pass -- so
   "budget" means the same thing for every method being compared.

No data augmentation is used anywhere: augmentation would make the training objective
a different (stochastic) function from the surface being drawn.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import pathlib
from typing import Iterator, Sequence

import numpy as np
import torch

DATA_DIR = pathlib.Path(os.environ.get("STAM_DATA", pathlib.Path(__file__).parent.parent / "data"))


@dataclasses.dataclass
class EvalSet:
    """A fixed finite evaluation set resident on the compute device.

    Holding the whole set on the GPU removes the host-to-device transfer from the
    probe cost, so measured throughput reflects the model rather than the input
    pipeline.  Both subjects here are small enough for this to be possible
    (CIFAR-10: 10k x 3 x 32 x 32 fp32 = 123 MB; WikiText-2: 2048 x 257 int64 = 4 MB).
    """

    tensors: tuple[torch.Tensor, ...]
    name: str

    def __len__(self) -> int:
        return self.tensors[0].shape[0]

    @property
    def device(self) -> torch.device:
        return self.tensors[0].device

    def to(self, device: torch.device | str) -> "EvalSet":
        return EvalSet(tuple(t.to(device) for t in self.tensors), self.name)

    def index(self, idx: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(t[idx] for t in self.tensors)

    def batches(self, batch_size: int) -> Iterator[tuple[torch.Tensor, ...]]:
        """Deterministic sequential pass -- used for exact evaluation."""
        n = len(self)
        for lo in range(0, n, batch_size):
            hi = min(lo + batch_size, n)
            yield tuple(t[lo:hi] for t in self.tensors)

    def sample_chunks(
        self, n_examples: int, micro_batch: int, generator: torch.Generator
    ) -> list[tuple[torch.Tensor, ...]]:
        """A without-replacement draw of ``n_examples``, split into micro-batches.

        This is the interface every probe goes through, so an evaluation set only has to
        know how to hand out a random subsample -- which is what lets an arbitrary
        user-supplied dataset be plugged in (see :class:`BatchEvalSet`).
        """
        n = len(self)
        n_examples = min(int(n_examples), n)
        perm = torch.randperm(n, generator=generator)[:n_examples].to(self.device)
        return [
            self.index(perm[lo : lo + micro_batch])
            for lo in range(0, n_examples, max(micro_batch, 1))
        ]

    def all_chunks(self, micro_batch: int) -> list[tuple[torch.Tensor, ...]]:
        """Every element, in order: an exhaustive, exact pass."""
        return list(self.batches(micro_batch))

    def sample(
        self, size: int, generator: torch.Generator, batch_size: int | None = None
    ) -> Iterator[tuple[torch.Tensor, ...]]:
        """Uniform sample without replacement, yielded in chunks.

        Sampling without replacement from a finite set makes the mean of the drawn
        losses an unbiased estimator of the set mean, with variance
        :math:`\\frac{\\sigma^2}{B}\\frac{|\\mathcal D|-B}{|\\mathcal D|-1}` -- the finite
        population correction, which the certificate applies rather than ignores.
        """
        n = len(self)
        size = min(size, n)
        perm = torch.randperm(n, generator=generator, device=generator.device)[:size]
        perm = perm.to(self.device)
        bs = batch_size or size
        for lo in range(0, size, bs):
            yield self.index(perm[lo : lo + bs])

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for t in self.tensors:
            a = t.detach().cpu().numpy()
            h.update(np.ascontiguousarray(a).view(np.uint8))
        return h.hexdigest()[:16]


class BatchEvalSet:
    """An evaluation set defined by a fixed list of pre-collected batches.

    For a caller's own model and data pipeline the natural unit is a batch, not an
    example: the batches may be tuples, dicts, or anything else the caller's loss
    function understands, and there is no general way to index into them.  Sampling is
    therefore at batch granularity -- a draw of :math:`B` examples takes whole batches in
    random order until :math:`B` is reached.

    The consequence for the analysis is mild and worth stating: the realised sample size
    is rounded up to a batch boundary, and the variance estimator sees batch-level rather
    than example-level randomness, which is slightly conservative.  Everything else --
    the cost model, the allocation, the certificate -- is unchanged.
    """

    def __init__(self, batches: Sequence, sizes: Sequence[int] | None = None,
                 name: str = "user-batches", device: torch.device | str | None = None):
        self.batches_list = list(batches)
        if not self.batches_list:
            raise ValueError("evaluation set is empty")
        self.name = name
        self._device = torch.device(device) if device is not None else _infer_device(
            self.batches_list[0]
        )
        self.sizes = list(sizes) if sizes is not None else [
            _infer_size(b) for b in self.batches_list
        ]
        self._n = int(sum(self.sizes))

    def __len__(self) -> int:
        return self._n

    @property
    def device(self) -> torch.device:
        return self._device

    def sample_chunks(self, n_examples: int, micro_batch: int,
                      generator: torch.Generator) -> list:
        order = torch.randperm(len(self.batches_list), generator=generator).tolist()
        out, taken = [], 0
        for i in order:
            if taken >= n_examples:
                break
            out.append(self.batches_list[i])
            taken += self.sizes[i]
        return out or [self.batches_list[order[0]]]

    def all_chunks(self, micro_batch: int) -> list:
        return list(self.batches_list)

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(str([tuple(self.sizes)]).encode())
        for b in self.batches_list[:4]:
            for t in _tensors_of(b):
                h.update(np.ascontiguousarray(t.detach().cpu().numpy()).view(np.uint8))
        return h.hexdigest()[:16]


def _tensors_of(batch) -> list[torch.Tensor]:
    if torch.is_tensor(batch):
        return [batch]
    if isinstance(batch, dict):
        return [v for v in batch.values() if torch.is_tensor(v)]
    if isinstance(batch, (list, tuple)):
        return [v for v in batch if torch.is_tensor(v)]
    return []


def _infer_device(batch) -> torch.device:
    ts = _tensors_of(batch)
    return ts[0].device if ts else torch.device("cpu")


def _infer_size(batch) -> int:
    ts = _tensors_of(batch)
    return int(ts[0].shape[0]) if ts else 1


@dataclasses.dataclass
class DataBundle:
    train: EvalSet          # optimisation stream (full training split)
    eval_train: EvalSet     # the fixed set defining L_train
    eval_val: EvalSet       # the fixed set defining L_val
    meta: dict


# ---------------------------------------------------------------------------
# CIFAR-10
# ---------------------------------------------------------------------------

_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD = (0.2470, 0.2435, 0.2616)


def load_cifar10(
    device: torch.device | str = "cuda",
    eval_train_size: int = 10_000,
    seed: int = 0,
) -> DataBundle:
    import torchvision

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tr = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=True, download=True)
    te = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=False, download=True)

    def pack(ds) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).float().div_(255.0)
        mean = torch.tensor(_CIFAR_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(_CIFAR_STD).view(1, 3, 1, 1)
        x = (x - mean) / std
        y = torch.tensor(ds.targets, dtype=torch.long)
        return x, y

    xtr, ytr = pack(tr)
    xte, yte = pack(te)

    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(xtr.shape[0], generator=g)[:eval_train_size]

    return DataBundle(
        train=EvalSet((xtr, ytr), "cifar10-train").to(device),
        eval_train=EvalSet((xtr[idx], ytr[idx]), "cifar10-eval-train").to(device),
        eval_val=EvalSet((xte, yte), "cifar10-eval-val").to(device),
        meta={
            "dataset": "CIFAR-10",
            "train_size": int(xtr.shape[0]),
            "eval_train_size": int(idx.numel()),
            "eval_val_size": int(xte.shape[0]),
            "augmentation": "none",
            "normalisation": {"mean": _CIFAR_MEAN, "std": _CIFAR_STD},
        },
    )


# ---------------------------------------------------------------------------
# WikiText-2
# ---------------------------------------------------------------------------


def _train_bpe(texts: Sequence[str], vocab_size: int, path: pathlib.Path):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<eos>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(path))
    return tok


def load_wikitext2(
    device: torch.device | str = "cuda",
    context: int = 256,
    vocab_size: int = 7040,
    eval_train_seqs: int = 2048,
    eval_val_seqs: int = 1024,
    seed: int = 0,
) -> DataBundle:
    """Tokenise WikiText-2 with a corpus-trained byte-level BPE and pack into
    fixed-length next-token-prediction sequences."""
    from tokenizers import Tokenizer

    cache = DATA_DIR / f"wikitext2_v{vocab_size}_c{context}.pt"
    tok_path = DATA_DIR / f"wikitext2_bpe_{vocab_size}.json"

    if cache.exists():
        # This cache is written by the branch below and contains numpy arrays, which
        # torch's default restricted unpickler rejects.
        blob = torch.load(cache, map_location="cpu", weights_only=False)
    else:
        from datasets import load_dataset

        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
        splits = {k: ds[k]["text"] for k in ("train", "validation")}

        if tok_path.exists():
            tok = Tokenizer.from_file(str(tok_path))
        else:
            tok = _train_bpe(splits["train"], vocab_size, tok_path)

        def encode(lines: Sequence[str]) -> np.ndarray:
            joined = "\n".join(line for line in lines if line.strip())
            ids: list[int] = []
            step = 200_000
            for lo in range(0, len(joined), step):
                ids.extend(tok.encode(joined[lo : lo + step]).ids)
            return np.asarray(ids, dtype=np.int32)

        blob = {
            "train": encode(splits["train"]),
            "validation": encode(splits["validation"]),
            "vocab_size": tok.get_vocab_size(),
        }
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(blob, cache)

    def pack(ids: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        n = (len(ids) - 1) // context
        arr = torch.from_numpy(ids[: n * context + 1].astype(np.int64))
        x = torch.stack([arr[i * context : (i + 1) * context] for i in range(n)])
        y = torch.stack([arr[i * context + 1 : (i + 1) * context + 1] for i in range(n)])
        return x, y

    xtr, ytr = pack(blob["train"])
    xva, yva = pack(blob["validation"])

    g = torch.Generator().manual_seed(seed)
    itr = torch.randperm(xtr.shape[0], generator=g)[:eval_train_seqs]
    iva = torch.randperm(xva.shape[0], generator=g)[:eval_val_seqs]

    return DataBundle(
        train=EvalSet((xtr, ytr), "wikitext2-train").to(device),
        eval_train=EvalSet((xtr[itr], ytr[itr]), "wikitext2-eval-train").to(device),
        eval_val=EvalSet((xva[iva], yva[iva]), "wikitext2-eval-val").to(device),
        meta={
            "dataset": "WikiText-2 (raw)",
            "tokenizer": f"byte-level BPE, vocab {blob['vocab_size']}",
            "context": context,
            "train_sequences": int(xtr.shape[0]),
            "eval_train_sequences": int(itr.numel()),
            "eval_val_sequences": int(iva.numel()),
            "tokens_train": int(len(blob["train"])),
            "tokens_val": int(len(blob["validation"])),
        },
    )


def load_data(name: str, device: torch.device | str = "cuda", **kw) -> DataBundle:
    if name == "cnn":
        return load_cifar10(device, **kw)
    if name == "gpt":
        return load_wikitext2(device, **kw)
    raise ValueError(f"unknown task {name!r}")


def cost_unit(name: str, bundle: DataBundle) -> dict:
    """Describe the accounting unit for the budget.

    One unit is one *example* forward pass -- one image for the CNN, one 256-token
    sequence for the transformer.  All budgets, and every cost multiplier reported
    later, are in these units.
    """
    if name == "cnn":
        return {"unit": "image", "elements_per_unit": 3 * 32 * 32}
    return {"unit": f"sequence of {bundle.meta['context']} tokens",
            "elements_per_unit": bundle.meta["context"]}


__all__ = ["EvalSet", "BatchEvalSet", "DataBundle", "load_cifar10", "load_wikitext2",
           "load_data", "cost_unit"]
