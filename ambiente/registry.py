"""Tiny operation registry.

An "op" is any callable `fn(grid: Grid, **kwargs) -> Grid`. Registered ops
become available in three places:

    >>> from ambiente import ops, Grid
    >>> g = Grid.zeros(3, 3)
    >>> ops.rot90(g)              # functional
    >>> g.rot90()                 # fluent (auto-bound on Grid)
    >>> ops.apply("rot90", g)     # by name (used by pipelines / configs)

Adding a new op is one decorator:

    >>> from ambiente.registry import register
    >>> @register
    ... def shift_right(grid, k=1):
    ...     d = grid.torch().roll(k, dims=1)
    ...     return grid.like(d)
"""
from __future__ import annotations

from typing import Callable, Dict


OPS: Dict[str, Callable] = {}


def register(fn: Callable | None = None, *, name: str | None = None) -> Callable:
    """Register an op. Use as `@register` or `@register(name="alias")`."""
    if fn is None:
        return lambda f: register(f, name=name)
    key = name or fn.__name__
    if key in OPS and OPS[key] is not fn:
        raise ValueError(f"op {key!r} already registered")
    OPS[key] = fn
    return fn


def apply(name: str, grid, **kwargs):
    if name not in OPS:
        raise KeyError(f"unknown op {name!r}; registered: {sorted(OPS)}")
    return OPS[name](grid, **kwargs)


def list_ops() -> list[str]:
    return sorted(OPS)
