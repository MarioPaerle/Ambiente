"""Physics-flavored ops: gravity, reflection, lasers."""
from __future__ import annotations

import torch

from ..registry import register


_DIRS = {
    "down": (1, 0), "up": (-1, 0), "right": (0, 1), "left": (0, -1),
}


@register
def gravity(grid, direction: str = "down", background: int | None = None):
    """Pack non-background cells toward the chosen edge, preserving relative order."""
    bg = grid.background if background is None else int(background)
    if direction not in _DIRS:
        raise ValueError(f"direction must be one of {sorted(_DIRS)}")
    arr = grid.data
    H, W = arr.shape

    def pack_col(col: torch.Tensor, toward_end: bool) -> torch.Tensor:
        nz = col[col != bg]
        pad = torch.full((col.numel() - nz.numel(),), bg, dtype=col.dtype)
        return torch.cat([pad, nz]) if toward_end else torch.cat([nz, pad])

    out = arr.clone()
    if direction in ("down", "up"):
        toward_end = (direction == "down")
        for j in range(W):
            out[:, j] = pack_col(arr[:, j], toward_end)
    else:
        toward_end = (direction == "right")
        for i in range(H):
            out[i, :] = pack_col(arr[i, :], toward_end)
    return grid.like(out)


@register
def reflect(grid, axis: str = "horizontal"):
    """Mirror the grid alongside itself, doubling the corresponding dimension.

    axis="horizontal": output is [grid | flipped(grid)] horizontally.
    axis="vertical":   output is [grid; flipped(grid)] vertically.
    """
    if axis == "horizontal":
        return grid.like(torch.cat([grid.data, torch.flip(grid.data, dims=(1,))], dim=1))
    if axis == "vertical":
        return grid.like(torch.cat([grid.data, torch.flip(grid.data, dims=(0,))], dim=0))
    raise ValueError("axis must be 'horizontal' or 'vertical'")


@register
def laser(grid, row: int, col: int, direction: str = "right",
          color: int = 1, stop_on: int | None = None,
          include_origin: bool = True, overwrite: bool = True):
    """Shoot a beam from (row, col) in `direction`, painting cells with `color`
    until either an edge is hit or a cell equal to `stop_on` is encountered.

    `stop_on=None` means: stop when hitting any non-background cell.
    """
    if direction not in _DIRS:
        raise ValueError(f"direction must be one of {sorted(_DIRS)}")
    dy, dx = _DIRS[direction]
    H, W = grid.shape
    new = grid.data.clone()
    bg = grid.background
    r, c = int(row), int(col)
    if not (0 <= r < H and 0 <= c < W):
        raise ValueError(f"origin ({r},{c}) outside grid {H}x{W}")
    if not include_origin:
        r += dy; c += dx
    while 0 <= r < H and 0 <= c < W:
        cell = int(new[r, c].item())
        is_obstacle = (stop_on is not None and cell == int(stop_on)) \
                      or (stop_on is None and cell != bg
                          and not (r == int(row) and c == int(col)))
        if is_obstacle:
            break
        if overwrite or cell == bg:
            new[r, c] = int(color)
        r += dy; c += dx
    return grid.like(new)
