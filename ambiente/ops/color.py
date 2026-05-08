"""Color-space ops: recoloring, swaps, random permutations."""
from __future__ import annotations

from typing import Mapping

import torch

from ..palette import N_COLORS
from ..registry import register


def _apply_lut(data: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    return lut[data]


@register
def recolor(grid, mapping: Mapping[int, int]):
    """Apply a {old: new} mapping. Unmapped colors pass through."""
    n = max(N_COLORS, int(grid.data.max().item()) + 1, *(int(k) + 1 for k in mapping))
    lut = torch.arange(n, dtype=torch.long)
    for k, v in mapping.items():
        lut[int(k)] = int(v)
    return grid.like(_apply_lut(grid.data, lut))


@register
def swap_colors(grid, a: int, b: int):
    return recolor(grid, {int(a): int(b), int(b): int(a)})


@register
def color_permute(grid, permutation: list[int] | None = None,
                  keep_background: bool = True,
                  generator: torch.Generator | None = None):
    """Permute colors. If `permutation` is None, sample a random one.

    With `keep_background=True` color 0 is held fixed.
    """
    n = N_COLORS
    if permutation is None:
        if keep_background:
            tail = torch.randperm(n - 1, generator=generator) + 1
            perm = torch.cat([torch.tensor([0], dtype=torch.long), tail])
        else:
            perm = torch.randperm(n, generator=generator)
    else:
        perm = torch.tensor(list(permutation), dtype=torch.long)
        if perm.numel() != n:
            raise ValueError(f"permutation must have length {n}")
    return grid.like(_apply_lut(grid.data, perm))


@register
def set_background(grid, color: int):
    """Replace the current background color with `color`, leaving others alone."""
    bg = int(grid.background)
    new = grid.data.clone()
    new[new == bg] = int(color)
    out = grid.like(new)
    out.background = int(color)
    return out


@register
def mask_color(grid, color: int, replacement: int | None = None):
    """Replace every cell of `color` with `replacement` (default: background)."""
    repl = grid.background if replacement is None else int(replacement)
    new = grid.data.clone()
    new[new == int(color)] = repl
    return grid.like(new)
