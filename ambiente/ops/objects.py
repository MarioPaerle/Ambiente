"""Object-level ops.

An "object" is a connected component of same-colored, non-background cells.
We use 4-connectivity by default (8 available via `diagonal=True`).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..registry import register


@dataclass(frozen=True)
class ObjectMask:
    color: int
    coords: torch.Tensor   # (N, 2) long: (row, col) per cell

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        ys, xs = self.coords[:, 0], self.coords[:, 1]
        return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())


def _neighbors(r: int, c: int, diagonal: bool):
    base = ((-1, 0), (1, 0), (0, -1), (0, 1))
    if diagonal:
        base = base + ((-1, -1), (-1, 1), (1, -1), (1, 1))
    return base


def find_objects(grid, diagonal: bool = False,
                 background: int | None = None) -> list[ObjectMask]:
    """Connected components of non-background cells, grouped by color."""
    bg = grid.background if background is None else int(background)
    arr = grid.numpy()
    H, W = arr.shape
    seen = [[False] * W for _ in range(H)]
    out: list[ObjectMask] = []
    nbrs = _neighbors(0, 0, diagonal)
    for i in range(H):
        for j in range(W):
            if seen[i][j] or arr[i, j] == bg:
                continue
            color = int(arr[i, j])
            stack = [(i, j)]
            cells = []
            seen[i][j] = True
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in nbrs:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr][nc] \
                            and arr[nr, nc] == color:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            out.append(ObjectMask(color=color,
                                  coords=torch.tensor(cells, dtype=torch.long)))
    return out


@register
def translate_object(grid, obj: ObjectMask, dy: int = 0, dx: int = 0,
                     fill: int | None = None):
    """Move a single object by (dy, dx). Source cells are filled with `fill`."""
    fill = grid.background if fill is None else int(fill)
    H, W = grid.shape
    new = grid.data.clone()
    rs = obj.coords[:, 0]; cs = obj.coords[:, 1]
    new[rs, cs] = fill
    nr = rs + int(dy); nc = cs + int(dx)
    keep = (nr >= 0) & (nr < H) & (nc >= 0) & (nc < W)
    new[nr[keep], nc[keep]] = obj.color
    return grid.like(new)


@register
def keep_largest_object(grid, diagonal: bool = False,
                        background: int | None = None):
    """Erase every connected component except the largest one."""
    bg = grid.background if background is None else int(background)
    objs = find_objects(grid, diagonal=diagonal, background=bg)
    if not objs:
        return grid.copy()
    best = max(objs, key=lambda o: o.coords.shape[0])
    new = torch.full_like(grid.data, bg)
    new[best.coords[:, 0], best.coords[:, 1]] = best.color
    return grid.like(new)


@register
def n_objects(grid, diagonal: bool = False) -> int:
    return len(find_objects(grid, diagonal=diagonal))
