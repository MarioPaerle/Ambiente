"""PyTorch Dataset utilities.

`PairedOpDataset` is the canonical NACT/Nacturn-2 trainer-side dataset:
each sample is `(A, B, op)` where B = op(A). Pass it any pipeline-or-callable.

`AugmentedDataset` is for on-the-fly augmentation of an existing dataset:
each sample is `pipeline(base[i])`.
"""
from __future__ import annotations

from typing import Callable, Sequence

import torch
from torch.utils.data import Dataset

from .grid import Grid


def _to_tensor(grid: Grid) -> torch.Tensor:
    return grid.torch()


class PairedOpDataset(Dataset):
    """Yields (A, B, op_name) triples.

    - `base`: callable `() -> Grid` (e.g. a generator) OR a sequence of Grids.
    - `ops`: sequence of (name, callable) pairs. The callable is `Grid -> Grid`.
    - `tensorize`: if True, returns torch tensors instead of Grid objects.
    """

    def __init__(self,
                 base: Callable[[], Grid] | Sequence[Grid],
                 ops: Sequence[tuple[str, Callable[[Grid], Grid]]],
                 length: int = 10_000,
                 tensorize: bool = True):
        self.base = base
        self.ops = list(ops)
        if not self.ops:
            raise ValueError("provide at least one (name, op) pair")
        self.length = int(length) if callable(base) else len(base)  # type: ignore[arg-type]
        self.tensorize = tensorize

    def __len__(self) -> int:
        return self.length

    def _sample_grid(self, idx: int) -> Grid:
        if callable(self.base):
            return self.base()
        return self.base[idx % len(self.base)]  # type: ignore[index]

    def __getitem__(self, idx: int):
        a = self._sample_grid(idx)
        op_idx = torch.randint(len(self.ops), (1,)).item()
        name, fn = self.ops[op_idx]
        b = fn(a)
        if self.tensorize:
            return _to_tensor(a), _to_tensor(b), name
        return a, b, name


class AugmentedDataset(Dataset):
    """Wraps any sequence of Grids and applies `pipeline` per __getitem__."""

    def __init__(self, base: Sequence[Grid], pipeline: Callable[[Grid], Grid],
                 tensorize: bool = True):
        self.base = base
        self.pipeline = pipeline
        self.tensorize = tensorize

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        out = self.pipeline(self.base[idx])
        return _to_tensor(out) if self.tensorize else out
