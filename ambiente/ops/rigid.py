"""Rigid transforms: rotations, flips, transposes."""
from __future__ import annotations

import torch

from ..registry import register


@register
def rot90(grid, k: int = 1):
    """Rotate counter-clockwise by 90*k degrees."""
    return grid.like(torch.rot90(grid.data, k=int(k) % 4, dims=(0, 1)))


@register
def rot180(grid):
    return rot90(grid, k=2)


@register
def rot270(grid):
    return rot90(grid, k=3)


@register
def fliplr(grid):
    return grid.like(torch.flip(grid.data, dims=(1,)))


@register
def flipud(grid):
    return grid.like(torch.flip(grid.data, dims=(0,)))


@register
def transpose(grid):
    return grid.like(grid.data.t().contiguous())


@register
def anti_transpose(grid):
    """Transpose across the anti-diagonal."""
    return grid.like(torch.flip(grid.data, dims=(0, 1)).t().contiguous())
