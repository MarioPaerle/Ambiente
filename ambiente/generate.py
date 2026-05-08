"""Grid generators. Cheap building blocks; compose with ops for richer datasets."""
from __future__ import annotations

import torch

from .grid import Grid
from .palette import N_COLORS


def random_grid(h: int, w: int, n_colors: int = N_COLORS,
                generator: torch.Generator | None = None) -> Grid:
    return Grid.random(h, w, n_colors=n_colors, generator=generator)


def sparse_grid(h: int, w: int, density: float = 0.2,
                n_colors: int = N_COLORS, background: int = 0,
                generator: torch.Generator | None = None) -> Grid:
    """Mostly-background grid with random non-bg cells at the given density."""
    base = torch.full((h, w), background, dtype=torch.long)
    mask = torch.rand((h, w), generator=generator) < float(density)
    colors = torch.randint(1, max(2, n_colors), (h, w),
                           generator=generator, dtype=torch.long)
    base[mask] = colors[mask]
    return Grid(base, background=background)


def rectangle(h: int, w: int, top: int, left: int, height: int, width: int,
              color: int = 1, background: int = 0) -> Grid:
    """Solid rectangle of `color` on a `background` field."""
    g = Grid.full(h, w, background, background=background)
    g.data[top:top + height, left:left + width] = int(color)
    return g


def with_objects(h: int, w: int, objects: list[dict], background: int = 0) -> Grid:
    """Place a list of {top,left,height,width,color} rectangles on a blank field."""
    g = Grid.full(h, w, background, background=background)
    for spec in objects:
        t = spec["top"]; l = spec["left"]
        hh = spec["height"]; ww = spec["width"]
        g.data[t:t + hh, l:l + ww] = int(spec["color"])
    return g
