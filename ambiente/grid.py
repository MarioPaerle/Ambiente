"""Grid: the single object every op consumes and produces.

Wraps a `torch.LongTensor` of shape (H, W). Stays small and explicit:
construction, conversion, equality, fluent op dispatch via the registry.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch

from .palette import DEFAULT_BACKGROUND, DEFAULT_PALETTE, N_COLORS
from .registry import OPS


GridLike = "torch.Tensor | np.ndarray | Sequence[Sequence[int]] | Grid"


def _as_long_tensor(data) -> torch.Tensor:
    if isinstance(data, Grid):
        return data.data.clone()
    if isinstance(data, torch.Tensor):
        t = data.detach().to(torch.long)
    elif isinstance(data, np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(data)).to(torch.long)
    else:
        t = torch.tensor(list(data), dtype=torch.long)
    if t.ndim != 2:
        raise ValueError(f"Grid expects 2D data, got shape {tuple(t.shape)}")
    return t.contiguous()


class Grid:
    __slots__ = ("data", "palette", "background")

    def __init__(self, data: GridLike, palette: Sequence[str] | None = None,
                 background: int = DEFAULT_BACKGROUND):
        self.data = _as_long_tensor(data)
        self.palette = tuple(palette) if palette is not None else DEFAULT_PALETTE
        self.background = int(background)

    # ---- constructors ----

    @classmethod
    def zeros(cls, h: int, w: int, **kw) -> "Grid":
        return cls(torch.zeros(h, w, dtype=torch.long), **kw)

    @classmethod
    def full(cls, h: int, w: int, fill: int, **kw) -> "Grid":
        return cls(torch.full((h, w), int(fill), dtype=torch.long), **kw)

    @classmethod
    def random(cls, h: int, w: int, n_colors: int = N_COLORS,
               generator: torch.Generator | None = None, **kw) -> "Grid":
        t = torch.randint(0, n_colors, (h, w), generator=generator, dtype=torch.long)
        return cls(t, **kw)

    # ---- views / conversions ----

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.data.shape)  # type: ignore[return-value]

    @property
    def H(self) -> int: return int(self.data.shape[0])
    @property
    def W(self) -> int: return int(self.data.shape[1])

    def torch(self) -> torch.Tensor:
        return self.data

    def numpy(self) -> np.ndarray:
        return self.data.cpu().numpy()

    def tolist(self) -> list[list[int]]:
        return self.data.tolist()

    def like(self, data) -> "Grid":
        """Return a new Grid carrying the same palette/background as self."""
        return Grid(data, palette=self.palette, background=self.background)

    def copy(self) -> "Grid":
        return self.like(self.data.clone())

    # ---- equality / repr ----

    def equals(self, other: "Grid") -> bool:
        return isinstance(other, Grid) and bool(torch.equal(self.data, other.data))

    def __eq__(self, other) -> bool:  # type: ignore[override]
        return self.equals(other) if isinstance(other, Grid) else NotImplemented

    def __hash__(self):  # Grid is mutable in principle; opt out of hashing
        raise TypeError("Grid is unhashable")

    def __repr__(self) -> str:
        rows = "\n".join(" ".join(f"{c}" for c in row) for row in self.tolist())
        return f"Grid({self.H}x{self.W})\n{rows}"

    # ---- indexing ----

    def __getitem__(self, key):
        out = self.data[key]
        if isinstance(out, torch.Tensor) and out.ndim == 2:
            return self.like(out)
        return out  # scalar / 1D slice

    # ---- fluent op dispatch ----

    def __getattr__(self, name: str):
        # Called only when normal attribute lookup fails.
        op = OPS.get(name)
        if op is None:
            raise AttributeError(name)
        def bound(*args, **kwargs):
            return op(self, *args, **kwargs)
        bound.__name__ = name
        bound.__doc__ = op.__doc__
        return bound

    def __dir__(self) -> Iterable[str]:
        return sorted(set(super().__dir__()) | set(OPS))
