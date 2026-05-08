"""Matplotlib visualization for grids, pairs, tasks, and diffs.

All `show_*` functions accept `save=<path>` to write a PNG, and `block=False`
for non-blocking display (useful inside notebooks / scripts).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .grid import Grid
from .palette import DEFAULT_PALETTE


# ---- low-level helpers ---------------------------------------------------

def _ensure_mpl():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm
    except ImportError as e:
        raise RuntimeError(
            "ambiente.viz needs matplotlib. Install with `pip install matplotlib`."
        ) from e
    return plt, ListedColormap, BoundaryNorm


def _cmap(palette):
    _, ListedColormap, BoundaryNorm = _ensure_mpl()
    cmap = ListedColormap(list(palette))
    bounds = np.arange(-0.5, len(palette) + 0.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def _luma(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0


def draw_grid(ax, grid: Grid, *, title: str | None = None,
              show_grid: bool = True, annotate: bool = False,
              highlight_mask: np.ndarray | None = None,
              border_color: str | None = None) -> None:
    """Render a Grid into an existing matplotlib Axes."""
    cmap, norm = _cmap(grid.palette)
    arr = grid.numpy()
    H, W = arr.shape

    ax.imshow(arr, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])

    if show_grid:
        ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
        ax.grid(which="minor", color="#222", linewidth=0.6)
        ax.tick_params(which="minor", length=0)

    if annotate:
        for i in range(H):
            for j in range(W):
                v = int(arr[i, j])
                fg = "white" if _luma(grid.palette[v % len(grid.palette)]) < 0.5 \
                    else "black"
                ax.text(j, i, str(v), ha="center", va="center",
                        fontsize=8, color=fg)

    if highlight_mask is not None:
        ys, xs = np.where(highlight_mask)
        for y, x in zip(ys, xs):
            ax.add_patch(_rect(x - 0.5, y - 0.5, 1, 1,
                               edgecolor="#FFD400", linewidth=2.0, fill=False))

    if border_color is not None:
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2.5)

    if title:
        ax.set_title(title, fontsize=10)


def _rect(x, y, w, h, **kw):
    import matplotlib.patches as mpatches
    return mpatches.Rectangle((x, y), w, h, **kw)


def _finalize(fig, save: str | None, block: bool):
    plt, *_ = _ensure_mpl()
    plt.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    backend = plt.get_backend().lower()
    if "agg" not in backend or not save:
        plt.show(block=block)
    return fig


# ---- public API ----------------------------------------------------------

def show(grid: Grid, *, title: str | None = None, show_grid: bool = True,
         annotate: bool = False, save: str | None = None,
         block: bool = True):
    plt, *_ = _ensure_mpl()
    fig, ax = plt.subplots(figsize=(max(2.0, grid.W * 0.35),
                                    max(2.0, grid.H * 0.35)))
    draw_grid(ax, grid, title=title, show_grid=show_grid, annotate=annotate)
    return _finalize(fig, save, block)


def show_many(grids: Sequence[Grid], titles: Sequence[str] | None = None,
              cols: int | None = None, *, show_grid: bool = True,
              annotate: bool = False, save: str | None = None,
              block: bool = True):
    plt, *_ = _ensure_mpl()
    grids = list(grids)
    n = len(grids)
    if titles is not None and len(titles) != n:
        raise ValueError("titles must match grids length")
    cols = cols or min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5),
                             squeeze=False)
    for i, g in enumerate(grids):
        ax = axes[i // cols][i % cols]
        title = titles[i] if titles else None
        draw_grid(ax, g, title=title, show_grid=show_grid, annotate=annotate)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    return _finalize(fig, save, block)


def show_pair(a: Grid, b: Grid, *, titles=("input", "output"), **kw):
    return show_many([a, b], titles=list(titles), cols=2, **kw)


def show_task(train: Sequence[tuple[Grid, Grid]],
              test: Sequence[tuple[Grid, Grid | None]] = (), *,
              annotate: bool = False, save: str | None = None,
              block: bool = True):
    """ARC-style task layout: rows = examples, cols = (input, output).

    Train rows are framed in cyan, test rows in gold. Missing test outputs
    render as a question-mark placeholder.
    """
    plt, *_ = _ensure_mpl()
    rows = list(train) + list(test)
    n = len(rows)
    if n == 0:
        raise ValueError("show_task: no examples to display")
    fig, axes = plt.subplots(n, 2, figsize=(5.5, max(2.5, n * 2.4)),
                             squeeze=False)
    fig.suptitle(f"task — {len(train)} train · {len(test)} test", fontsize=11)
    for i, (a, b) in enumerate(rows):
        is_train = i < len(train)
        border = "#1ABC9C" if is_train else "#FFD400"
        tag = "train" if is_train else "test"
        draw_grid(axes[i][0], a, title=f"{tag} {i}: input",
                  annotate=annotate, border_color=border)
        if b is None:
            axes[i][1].set_facecolor("#222")
            axes[i][1].text(0.5, 0.5, "?", ha="center", va="center",
                            color="#FFD400", fontsize=42, transform=axes[i][1].transAxes)
            axes[i][1].set_xticks([]); axes[i][1].set_yticks([])
            axes[i][1].set_title(f"{tag} {i}: output", fontsize=10)
            for spine in axes[i][1].spines.values():
                spine.set_edgecolor(border); spine.set_linewidth(2.5)
        else:
            draw_grid(axes[i][1], b, title=f"{tag} {i}: output",
                      annotate=annotate, border_color=border)
    return _finalize(fig, save, block)


def show_diff(a: Grid, b: Grid, *, annotate: bool = False,
              save: str | None = None, block: bool = True):
    """Side-by-side, with cells that changed highlighted in gold."""
    if a.shape != b.shape:
        return show_pair(a, b, titles=("a (shape differs)", "b (shape differs)"),
                         annotate=annotate, save=save, block=block)
    plt, *_ = _ensure_mpl()
    mask = (a.numpy() != b.numpy())
    fig, axes = plt.subplots(1, 2, figsize=(6, 3.2), squeeze=False)
    draw_grid(axes[0][0], a, title="a", annotate=annotate, highlight_mask=mask)
    draw_grid(axes[0][1], b, title="b", annotate=annotate, highlight_mask=mask)
    fig.suptitle(f"{int(mask.sum())} / {mask.size} cells differ", fontsize=10)
    return _finalize(fig, save, block)
