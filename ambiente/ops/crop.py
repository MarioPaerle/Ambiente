"""Crop / pad / hole punching."""
from __future__ import annotations

import torch

from ..registry import register


@register
def crop(grid, top: int, left: int, height: int, width: int):
    H, W = grid.shape
    if not (0 <= top and 0 <= left and top + height <= H and left + width <= W):
        raise ValueError(f"crop {top,left,height,width} out of {H,W}")
    return grid.like(grid.data[top:top + height, left:left + width].clone())


@register
def pad(grid, top: int = 0, bottom: int = 0, left: int = 0, right: int = 0,
        fill: int | None = None):
    fill = grid.background if fill is None else int(fill)
    H, W = grid.shape
    out = torch.full((H + top + bottom, W + left + right), fill, dtype=torch.long)
    out[top:top + H, left:left + W] = grid.data
    return grid.like(out)


@register
def punch_hole(grid, top: int, left: int, height: int, width: int,
               fill: int | None = None):
    """Overwrite a rectangular region with `fill` (default: background)."""
    fill = grid.background if fill is None else int(fill)
    H, W = grid.shape
    t = max(0, int(top)); l = max(0, int(left))
    b = min(H, t + int(height)); r = min(W, l + int(width))
    new = grid.data.clone()
    new[t:b, l:r] = fill
    return grid.like(new)


@register
def shift(grid, dy: int = 0, dx: int = 0, fill: int | None = None):
    """Translate the grid by (dy, dx); cells leaving the frame are dropped,
    new cells are filled with `fill` (default: background)."""
    fill = grid.background if fill is None else int(fill)
    H, W = grid.shape
    out = torch.full((H, W), fill, dtype=torch.long)
    sy = max(0, int(dy)); sx = max(0, int(dx))
    ey = min(H, H + int(dy)); ex = min(W, W + int(dx))
    src_y0 = max(0, -int(dy)); src_x0 = max(0, -int(dx))
    h = ey - sy; w = ex - sx
    if h > 0 and w > 0:
        out[sy:sy + h, sx:sx + w] = grid.data[src_y0:src_y0 + h, src_x0:src_x0 + w]
    return grid.like(out)
