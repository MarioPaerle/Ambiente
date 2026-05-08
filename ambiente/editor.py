"""Interactive matplotlib editor for (input, output) grid pairs.

Mouse:
  left-click / drag      paint the focused grid with the selected color
  right-click / drag     erase to background (color 0)
  click on palette       select a color
  click on a grid        focus that grid (next resize affects it)

Keyboard:
  0..9                   select color
  h / H                  shrink / grow rows of focused grid
  w / W                  shrink / grow cols of focused grid
  c                      clear focused grid
  s                      save (file dialog)
  l                      load (file dialog)
  q                      quit

Usage:
    from ambiente.editor import PairEditor
    editor = PairEditor(in_shape=(5, 5))
    a, b = editor.run()                # blocks; returns final (Grid, Grid)

    # or from CLI:
    #   python -m ambiente.editor [path/to/task.json]
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .grid import Grid
from .palette import DEFAULT_BACKGROUND, DEFAULT_PALETTE, N_COLORS
from .viz import draw_grid


def _ensure_mpl():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button
    except ImportError as e:
        raise RuntimeError(
            "ambiente.editor needs matplotlib. Install with `pip install matplotlib`."
        ) from e
    return plt, Button


def _ask_path(action: str, default_name: str = "task.json") -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        if action == "save":
            path = filedialog.asksaveasfilename(
                defaultextension=".json",
                initialfile=default_name,
                filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        else:
            path = filedialog.askopenfilename(
                filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        root.destroy()
        return path or None
    except Exception:
        return default_name if action == "save" else None


def _to_array(g, fallback_shape, background) -> np.ndarray:
    if g is None:
        return np.full(fallback_shape, int(background), dtype=np.int64)
    if isinstance(g, Grid):
        return g.numpy().astype(np.int64, copy=True)
    return np.asarray(g, dtype=np.int64).copy()


class PairEditor:
    """An (input, output) grid pair editor.

    Parameters
    ----------
    in_shape, out_shape : (H, W) — initial sizes; out_shape defaults to in_shape.
    input_grid, output_grid : pre-existing grids to start from (overrides shape).
    palette : per-color hex strings; defaults to ARC palette.
    n_colors : how many palette swatches to show (default 10).
    background : color id treated as background (used by erase + shape grow).
    """

    def __init__(self,
                 in_shape: tuple[int, int] = (5, 5),
                 out_shape: tuple[int, int] | None = None,
                 input_grid: Grid | Iterable | None = None,
                 output_grid: Grid | Iterable | None = None,
                 palette=DEFAULT_PALETTE,
                 n_colors: int = N_COLORS,
                 background: int = DEFAULT_BACKGROUND):
        self.palette = list(palette)
        self.n_colors = int(n_colors)
        self.background = int(background)
        self.in_data = _to_array(input_grid, in_shape, background)
        self.out_data = _to_array(output_grid, out_shape or in_shape, background)
        self.selected_color: int = 1
        self.focus: str = "input"        # "input" or "output"
        self._mouse_down: int | None = None  # 1=left, 3=right, None=up
        self._fig = None
        self._widgets: list = []         # keep button refs alive

    # ---- public API ------------------------------------------------------

    def run(self) -> tuple[Grid, Grid]:
        """Open the editor window, block until closed, return (input, output)."""
        self._build()
        plt, _ = _ensure_mpl()
        plt.show(block=True)
        return self.input_grid, self.output_grid

    @property
    def input_grid(self) -> Grid:
        return Grid(self.in_data, palette=self.palette, background=self.background)

    @property
    def output_grid(self) -> Grid:
        return Grid(self.out_data, palette=self.palette, background=self.background)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "input": self.in_data.tolist(),
            "output": self.out_data.tolist(),
        }, indent=2))

    @classmethod
    def load(cls, path: str | Path, **kw) -> "PairEditor":
        d = json.loads(Path(path).read_text())
        if "train" in d:  # ARC-style task: take the first train pair
            first = d["train"][0]
            d = {"input": first["input"], "output": first["output"]}
        return cls(input_grid=d["input"], output_grid=d["output"], **kw)

    # ---- figure construction --------------------------------------------

    def _build(self):
        plt, Button = _ensure_mpl()
        self._fig = plt.figure(figsize=(11, 7.2))
        self._fig.canvas.manager.set_window_title("Ambiente — Pair Editor")
        gs = self._fig.add_gridspec(
            nrows=4, ncols=12,
            height_ratios=[10, 0.9, 0.8, 0.5],
            hspace=0.45, wspace=0.4,
            left=0.04, right=0.98, top=0.93, bottom=0.06,
        )
        self.ax_in  = self._fig.add_subplot(gs[0, :6])
        self.ax_out = self._fig.add_subplot(gs[0, 6:])
        self.ax_palette = self._fig.add_subplot(gs[1, 1:11])

        # action buttons
        labels = ["Clear In", "Clear Out", "In→Out", "Swap", "Save", "Load"]
        callbacks = [self._clear_in, self._clear_out, self._copy_in_to_out,
                     self._swap, self._save, self._load]
        for i, (lab, cb) in enumerate(zip(labels, callbacks)):
            ax = self._fig.add_subplot(gs[2, i*2:(i+1)*2])
            btn = Button(ax, lab, color="#2c2c2c", hovercolor="#444")
            btn.label.set_color("white")
            btn.on_clicked(lambda _evt, fn=cb: fn())
            self._widgets.append(btn)

        # status row
        self.ax_status = self._fig.add_subplot(gs[3, :])
        self.ax_status.axis("off")
        self._status_txt = self.ax_status.text(
            0.01, 0.5, "", ha="left", va="center",
            fontsize=9, family="monospace")
        self._hint_txt = self.ax_status.text(
            0.99, 0.5,
            "0-9: color · h/H rows · w/W cols · c clear · s save · l load · q quit",
            ha="right", va="center", fontsize=8, family="monospace",
            color="#888")

        # event hooks
        cv = self._fig.canvas
        cv.mpl_connect("button_press_event",   self._on_press)
        cv.mpl_connect("button_release_event", self._on_release)
        cv.mpl_connect("motion_notify_event",  self._on_motion)
        cv.mpl_connect("key_press_event",      self._on_key)

        self._draw_all()

    # ---- drawing ---------------------------------------------------------

    def _draw_all(self):
        if self._fig is None:
            return  # headless mutation (e.g. tests) — no axes yet
        self._draw_grid_axis(self.ax_in,  self.in_data,  "input",  self.focus == "input")
        self._draw_grid_axis(self.ax_out, self.out_data, "output", self.focus == "output")
        self._draw_palette()
        self._update_status()
        self._fig.canvas.draw_idle()

    def _draw_grid_axis(self, ax, data, label, focused):
        ax.clear()
        H, W = data.shape
        g = Grid(data, palette=self.palette, background=self.background)
        title = f"{label}  {H}×{W}" + ("  ← focus" if focused else "")
        draw_grid(ax, g, title=title, show_grid=True,
                  border_color="#1ABC9C" if focused else "#444")

    def _draw_palette(self):
        ax = self.ax_palette
        ax.clear()
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(self.palette[:self.n_colors])
        ax.imshow(np.arange(self.n_colors)[None, :], aspect="auto",
                  cmap=cmap, vmin=-0.5, vmax=self.n_colors - 0.5,
                  interpolation="nearest")
        ax.set_xticks(range(self.n_colors))
        ax.set_xticklabels([str(i) for i in range(self.n_colors)], fontsize=9)
        ax.set_yticks([])
        # selection highlight
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((self.selected_color - 0.5, -0.5), 1, 1,
                               edgecolor="#FFD400", linewidth=3, fill=False))
        ax.set_title("palette (click to pick / digits 0-9)", fontsize=9)

    def _update_status(self):
        H_in, W_in = self.in_data.shape
        H_out, W_out = self.out_data.shape
        self._status_txt.set_text(
            f"color={self.selected_color}  "
            f"input={H_in}×{W_in}  output={H_out}×{W_out}  "
            f"focus={self.focus}")

    # ---- mouse handlers --------------------------------------------------

    def _xy_to_cell(self, ax, event, data):
        if event.xdata is None or event.ydata is None:
            return None
        c = int(round(event.xdata)); r = int(round(event.ydata))
        H, W = data.shape
        if 0 <= r < H and 0 <= c < W:
            return r, c
        return None

    def _which_grid(self, event):
        if event.inaxes is self.ax_in:  return "input"
        if event.inaxes is self.ax_out: return "output"
        return None

    def _data_for(self, which: str) -> np.ndarray:
        return self.in_data if which == "input" else self.out_data

    def _set_data(self, which: str, arr: np.ndarray):
        if which == "input": self.in_data = arr
        else:                self.out_data = arr

    def _paint(self, which: str, event, color: int):
        cell = self._xy_to_cell(
            self.ax_in if which == "input" else self.ax_out,
            event, self._data_for(which))
        if cell is None:
            return False
        r, c = cell
        data = self._data_for(which)
        if data[r, c] == color:
            return False
        data[r, c] = color
        return True

    def _on_press(self, event):
        # palette pick
        if event.inaxes is self.ax_palette and event.xdata is not None:
            i = int(round(event.xdata))
            if 0 <= i < self.n_colors:
                self.selected_color = i
                self._draw_all()
            return
        which = self._which_grid(event)
        if which is None:
            return
        if self.focus != which:
            self.focus = which
        self._mouse_down = event.button
        color = self.selected_color if event.button == 1 else self.background
        if self._paint(which, event, color):
            self._draw_all()
        else:
            # focus may have changed even on no-paint
            self._draw_all()

    def _on_motion(self, event):
        if self._mouse_down is None:
            return
        which = self._which_grid(event)
        if which is None:
            return
        color = self.selected_color if self._mouse_down == 1 else self.background
        if self._paint(which, event, color):
            self._draw_all()

    def _on_release(self, _event):
        self._mouse_down = None

    # ---- keyboard --------------------------------------------------------

    def _on_key(self, event):
        k = event.key or ""
        if k.isdigit():
            i = int(k)
            if 0 <= i < self.n_colors:
                self.selected_color = i
                self._draw_all()
            return
        if k == "c":
            self._clear_focused(); return
        if k == "s":
            self._save(); return
        if k == "l":
            self._load(); return
        if k == "q":
            import matplotlib.pyplot as plt
            plt.close(self._fig); return
        if k in ("h", "H", "w", "W"):
            self._resize_focused(k); return

    # ---- mutators / button callbacks ------------------------------------

    def _clear_in(self):
        self.in_data = np.full_like(self.in_data, self.background)
        self._draw_all()

    def _clear_out(self):
        self.out_data = np.full_like(self.out_data, self.background)
        self._draw_all()

    def _clear_focused(self):
        if self.focus == "input": self._clear_in()
        else:                     self._clear_out()

    def _copy_in_to_out(self):
        self.out_data = self.in_data.copy()
        self._draw_all()

    def _swap(self):
        self.in_data, self.out_data = self.out_data.copy(), self.in_data.copy()
        self._draw_all()

    def _save(self):
        path = _ask_path("save")
        if path:
            self.save(path)

    def _load(self):
        path = _ask_path("open")
        if not path:
            return
        d = json.loads(Path(path).read_text())
        if "train" in d:
            d = {"input": d["train"][0]["input"],
                 "output": d["train"][0]["output"]}
        self.in_data = np.asarray(d["input"], dtype=np.int64)
        self.out_data = np.asarray(d["output"], dtype=np.int64)
        self._draw_all()

    def _resize_focused(self, key: str):
        arr = self._data_for(self.focus)
        H, W = arr.shape
        if   key == "H": new = (H + 1, W)
        elif key == "h": new = (max(1, H - 1), W)
        elif key == "W": new = (H, W + 1)
        elif key == "w": new = (H, max(1, W - 1))
        else: return
        out = np.full(new, self.background, dtype=arr.dtype)
        h_copy = min(H, new[0]); w_copy = min(W, new[1])
        out[:h_copy, :w_copy] = arr[:h_copy, :w_copy]
        self._set_data(self.focus, out)
        self._draw_all()


# ---- CLI -----------------------------------------------------------------

def _main(argv: list[str] | None = None) -> None:
    import sys
    args = argv if argv is not None else sys.argv[1:]
    if args:
        editor = PairEditor.load(args[0])
    else:
        editor = PairEditor(in_shape=(6, 6))
    editor.run()


if __name__ == "__main__":
    _main()
