"""Composable pipelines for on-the-fly augmentation.

A pipeline is just a callable `Grid -> Grid`. Combine them freely; they all
implement `__call__`.

    pipe = Compose([
        RandomChoice([rot90, fliplr, transpose]),
        RandomApply(color_permute, p=0.5),
    ])
    augmented = pipe(grid)
"""
from __future__ import annotations

import random
from typing import Callable, Iterable, Sequence

from .grid import Grid
from .registry import OPS


Op = Callable[[Grid], Grid]


def _resolve(op) -> Op:
    """Accept a callable, an op name, or `(name, kwargs)` tuple."""
    if callable(op):
        return op
    if isinstance(op, str):
        return OPS[op]
    if isinstance(op, tuple) and len(op) == 2 and isinstance(op[0], str):
        name, kwargs = op
        fn = OPS[name]
        return lambda g, _f=fn, _k=kwargs: _f(g, **_k)
    raise TypeError(f"cannot resolve op spec {op!r}")


class Compose:
    def __init__(self, ops: Sequence):
        self.ops = [_resolve(o) for o in ops]

    def __call__(self, grid: Grid) -> Grid:
        for op in self.ops:
            grid = op(grid)
        return grid

    def __repr__(self) -> str:
        return f"Compose({len(self.ops)} ops)"


class RandomChoice:
    def __init__(self, ops: Sequence, weights: Sequence[float] | None = None,
                 rng: random.Random | None = None):
        self.ops = [_resolve(o) for o in ops]
        self.weights = list(weights) if weights is not None else None
        self.rng = rng or random

    def __call__(self, grid: Grid) -> Grid:
        if self.weights is None:
            op = self.rng.choice(self.ops)
        else:
            op = self.rng.choices(self.ops, weights=self.weights, k=1)[0]
        return op(grid)


class RandomApply:
    def __init__(self, op, p: float = 0.5, rng: random.Random | None = None):
        self.op = _resolve(op)
        self.p = float(p)
        self.rng = rng or random

    def __call__(self, grid: Grid) -> Grid:
        return self.op(grid) if self.rng.random() < self.p else grid


class RandomOrder:
    """Apply each op once but in shuffled order."""
    def __init__(self, ops: Sequence, rng: random.Random | None = None):
        self.ops = [_resolve(o) for o in ops]
        self.rng = rng or random

    def __call__(self, grid: Grid) -> Grid:
        order = list(self.ops)
        self.rng.shuffle(order)
        for op in order:
            grid = op(grid)
        return grid
