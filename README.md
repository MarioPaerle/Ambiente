# Ambiente

A small, fluent, PyTorch-native environment for building **ARC-AGI** /
**NACT-style** experiments: colored grids, composable operations, on-the-fly
augmentation, an interactive editor for input/output pairs, and rich
matplotlib visualization.

```
g = Grid.random(8, 8)
g.rot90().color_permute().gravity("down")        # fluent
ops.rot90(g)                                      # functional
ops.apply("rot90", g)                             # by name (config-driven)
```

Three equivalent ways to call any registered op — including ones you add
yourself with a single decorator.

---

## Contents

1. [Quickstart](#1-quickstart)
2. [Install](#2-install)
3. [Core concepts](#3-core-concepts)
4. [Built-in operations](#4-built-in-operations)
5. [Generators](#5-generators)
6. [Pipelines (augmentation)](#6-pipelines-augmentation)
7. [Visualization — cool plots](#7-visualization--cool-plots)
8. [Interactive editor](#8-interactive-editor)
9. [ARC-AGI datasets](#9-arc-agi-datasets)
10. [Efficient PyTorch usage](#10-efficient-pytorch-usage)
11. [Adding your own op](#11-adding-your-own-op)
12. [Project layout](#12-project-layout)
13. [Testing](#13-testing)

---

## 1. Quickstart

```python
import torch
from ambiente import Grid, ops, viz, generate, Compose, RandomChoice

# build / load a grid
g = Grid([[0, 1, 0], [1, 1, 0], [0, 0, 2]])

# transform it (any of three styles)
g2 = g.rot90().swap_colors(1, 2)

# generate random data, then augment with a pipeline
pipe = Compose([
    RandomChoice([ops.rot90, ops.fliplr, ops.transpose]),
    ops.color_permute,
])
augmented = [pipe(generate.sparse_grid(8, 8, density=0.3)) for _ in range(8)]

# visualize
viz.show_pair(g, g2, titles=("input", "rot90 + swap_colors"))
viz.show_many(augmented, cols=4, annotate=True)
```

---

## 2. Install

Ambiente is a plain Python package — no `setup.py` needed. From the repo root:

```bash
pip install torch numpy matplotlib              # required
pip install pytest                              # tests
```

Then either run things from the repo root (so `from ambiente import ...`
just works), or `pip install -e .` if you add a `pyproject.toml` later.

A `conftest.py` at the repo root already puts the project on `sys.path`,
so `pytest` and example scripts import cleanly without configuration.

---

## 3. Core concepts

### `Grid`

A thin, immutable-feeling wrapper around a `torch.LongTensor` of shape
`(H, W)`. Stays explicit:

```python
from ambiente import Grid
import torch

# constructors
Grid([[0, 1], [2, 3]])                         # from list-of-lists
Grid(torch.zeros(4, 4, dtype=torch.long))      # from tensor
Grid.random(6, 6, n_colors=10,
            generator=torch.Generator().manual_seed(0))
Grid.zeros(8, 8); Grid.full(8, 8, fill=3)

# views
g.shape, g.H, g.W
g.torch()         # → torch.LongTensor (no copy)
g.numpy()         # → np.ndarray
g.tolist()
g.copy()

# .like(data) — same palette/background, new data (used inside ops)
g.like(g.torch().roll(1, dims=1))
```

Equality is value-based (`a == b` ⇔ same shape + same cells).
Indexing (`g[1, 2]`) returns ints; slicing returns a sub-`Grid`.

### The ops registry

Every op is a function `Grid → Grid` registered with `@register`. Three
equivalent call styles, all kept in sync:

```python
from ambiente import ops, Grid

g = Grid.zeros(4, 4)
ops.rot90(g)                                   # functional (module attr)
g.rot90()                                      # fluent (auto-bound on Grid)
ops.apply("rot90", g)                          # by name (for configs / loaders)
ops.list_ops()                                 # ['rot90', 'fliplr', ...]
```

The fluent form works because `Grid.__getattr__` falls back to the
registry, so any op you register is *immediately* available as a method —
no Grid subclassing, no boilerplate. Same for the `ops` module:
`ops.<your_new_op>` resolves dynamically.

---

## 4. Built-in operations

| Category | Ops |
|---|---|
| Rigid | `rot90(k)`, `rot180`, `rot270`, `fliplr`, `flipud`, `transpose`, `anti_transpose` |
| Color | `recolor({old:new})`, `swap_colors(a, b)`, `color_permute(permutation=None, keep_background=True)`, `set_background(color)`, `mask_color(color, replacement=None)` |
| Frame | `crop(top, left, h, w)`, `pad(top,bottom,left,right, fill=None)`, `punch_hole(top,left,h,w, fill=None)`, `shift(dy, dx, fill=None)` |
| Object-level | `find_objects(diagonal=False)`, `translate_object(obj, dy, dx)`, `keep_largest_object(diagonal=False)`, `n_objects` |
| Physics | `gravity(direction)` ("down"/"up"/"left"/"right"), `reflect(axis)` ("horizontal"/"vertical"), `laser(row, col, direction, color, stop_on=None)` |

Every op is `Grid → Grid`. Optional kwargs are passed straight through —
e.g. `g.gravity(direction="left")` or `ops.rot90(g, k=2)`.

Object-level ops use 4-connectivity by default; pass `diagonal=True` for 8.
`find_objects(g)` returns a list of `ObjectMask` records (`color`, `coords`,
`bbox`).

---

## 5. Generators

```python
from ambiente import generate

generate.random_grid(8, 8, n_colors=10)
generate.sparse_grid(8, 8, density=0.25)        # mostly-background random
generate.rectangle(10, 10, top=2, left=3, height=4, width=4, color=2)
generate.with_objects(10, 10, [
    {"top": 1, "left": 1, "height": 2, "width": 3, "color": 1},
    {"top": 5, "left": 4, "height": 3, "width": 3, "color": 4},
])
```

All accept an optional `generator=torch.Generator()` for reproducibility.

---

## 6. Pipelines (augmentation)

Pipelines are plain callables `Grid → Grid`. Compose them however you like:

```python
import random
from ambiente import Compose, RandomChoice, RandomApply, RandomOrder, ops

rng = random.Random(42)

pipe = Compose([
    RandomChoice([ops.rot90, ops.rot180, ops.fliplr, ops.transpose], rng=rng),
    RandomApply(ops.color_permute, p=0.7, rng=rng),
    RandomApply(("gravity", {"direction": "down"}), p=0.5, rng=rng),  # by name
    RandomOrder([ops.shift, ops.fliplr], rng=rng),                    # any order
])
```

Each step accepts a callable, a string name, or a `(name, kwargs)` tuple —
so pipelines can be defined in YAML/JSON config files and built at runtime.

Pipelines are *just callables*: drop them into a `torch.utils.data.Dataset`
unchanged.

---

## 7. Visualization — cool plots

All `show_*` helpers accept `save="out.png"` to write a high-DPI PNG and
`block=False` to open without blocking (useful when stacking figures).

### Single grid, with cell labels

```python
viz.show(grid, title="my grid", annotate=True)
```

`annotate=True` overlays the integer color at each cell, contrast-aware
(white text on dark cells, black text on light ones).

### Pair / many grids

```python
viz.show_pair(input_grid, output_grid)
viz.show_many([g1, g2, g3, g4], titles=["a","b","c","d"], cols=2, annotate=True)
```

### ARC-style task layout

```python
viz.show_task(train, test)        # train: list of (Grid, Grid)
                                  # test:  list of (Grid, Grid|None)
```

Train rows are framed cyan, test rows gold. A `None` test output renders
as a "?" placeholder — perfect for visualizing tasks before you have a
prediction.

### Diff highlight

```python
viz.show_diff(a, b)               # gold outlines on changed cells
```

Title shows `N / total cells differ`. If shapes differ it falls back to a
plain pair view.

### Custom plots — use the low-level draw helper

```python
import matplotlib.pyplot as plt
from ambiente.viz import draw_grid

fig, axes = plt.subplots(2, 3, figsize=(9, 6))
for ax, g in zip(axes.flat, my_grids):
    draw_grid(ax, g, annotate=True, border_color="#FFD400")
fig.suptitle("hand-rolled layout")
plt.tight_layout(); plt.show()
```

`draw_grid` is the building block every other helper uses. It accepts
`title`, `show_grid`, `annotate`, `highlight_mask` (a numpy bool array
to outline cells in gold), and `border_color`.

---

## 8. Interactive editor

`PairEditor` is a matplotlib + tkinter editor for `(input, output)` grid
pairs. Click/drag to paint, right-click to erase, digits 0-9 to pick
colors, `h/H w/W` to resize the focused grid.

```python
from ambiente.editor import PairEditor

editor = PairEditor(in_shape=(6, 6))     # blank pair
# or load an existing task
editor = PairEditor.load("datasets/arc-agi-1/data/training/00d62c1b.json")

a, b = editor.run()                       # blocks; returns (Grid, Grid)
editor.save("my_task.json")               # plain JSON
```

Or launch from the command line:

```bash
python -m ambiente.editor                       # blank 6x6 pair
python -m ambiente.editor my_task.json          # open an existing pair
```

UI layout: input grid (left), output grid (right), color palette strip,
six action buttons (Clear In · Clear Out · In→Out · Swap · Save · Load).
The focused grid is bordered cyan and is the one keyboard resize commands
affect. Save/Load use a tkinter file dialog.

JSON format is the same flat shape ARC tasks use, so files written by the
editor load straight into `ambiente.io.load_pair` / `load_task`.

---

## 9. ARC-AGI datasets

Both ARC-AGI-1 and ARC-AGI-2 use the same JSON schema as our loader, so
no conversion is needed.

### Layout

```
datasets/
├── arc-agi-1/data/training/<id>.json     # 400 tasks
├── arc-agi-1/data/evaluation/<id>.json   # 400 tasks
├── arc-agi-2/data/training/<id>.json     # 1000 tasks (includes ARC-1)
└── arc-agi-2/data/evaluation/<id>.json   # 120 tasks
```

Clone with:

```bash
mkdir -p datasets && cd datasets
git clone --depth 1 https://github.com/fchollet/ARC-AGI.git arc-agi-1
git clone --depth 1 https://github.com/arcprize/ARC-AGI-2.git arc-agi-2
```

### Iterating

```python
from ambiente.arc import (load_arc, list_task_ids, iter_tasks,
                          solves, find_solving_ops, CANDIDATES)

# load one task by id
task = load_arc("3c9b0459", version=1, split="training")
# task = {"train": [(Grid, Grid), ...], "test": [(Grid, Grid|None), ...]}

# iterate everything
for task_id, task in iter_tasks(version=1, split="training", limit=50):
    ...

# does a given op solve this task's training pairs exactly?
solves(ops.rot180, task)                          # → True / False

# what's in our 0-arg candidate library?
find_solving_ops(task, CANDIDATES)                # → ['rot180']
```

`CANDIDATES` is a 15-op pre-bound dictionary of arg-free callables (rigid
+ gravity directions + reflect axes + keep-largest), suitable for brute
forcing the easiest tasks. Override the dataset path with the `ARC1_ROOT`
or `ARC2_ROOT` env var.

`examples/07_arc_explorer.py` is a complete visual tour: ARC layout +
op gallery + augmentation gallery + paired consistency demo + auto-found
solver hit. Run it:

```bash
python examples/07_arc_explorer.py
python examples/07_arc_explorer.py 5582e5ca       # specific task (v1)
python examples/07_arc_explorer.py 2 00576224     # ARC-AGI-2 task
```

---

## 10. Efficient PyTorch usage

### The canonical pattern: per-sample augmentation in the DataLoader

`PairedOpDataset` produces `(A, B, op_name)` triples — exactly what NACT /
Nacturn-2 trainers need. The dataset works on `Grid` objects internally
and only converts to tensors at the boundary.

```python
import torch
from torch.utils.data import DataLoader

from ambiente import generate, ops
from ambiente.data import PairedOpDataset

ds = PairedOpDataset(
    base=lambda: generate.sparse_grid(10, 10, density=0.3),
    ops=[
        ("rot90",     lambda g: g.rot90()),
        ("rot180",    lambda g: g.rot180()),
        ("fliplr",    lambda g: g.fliplr()),
        ("transpose", lambda g: g.transpose()),
        ("gravity",   lambda g: g.gravity(direction="down")),
    ],
    length=10_000,
    tensorize=True,           # → torch.LongTensor
)
loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

for A, B, op_names in loader:
    # A, B: (B, H, W) torch.LongTensor  ·  op_names: tuple[str]
    A, B = A.cuda(non_blocking=True), B.cuda(non_blocking=True)
    ...
```

### Augmenting an existing dataset

```python
from ambiente import Compose, RandomChoice, ops
from ambiente.data import AugmentedDataset

base = [Grid(...), Grid(...), ...]                # any sequence of Grids
pipe = Compose([
    RandomChoice([ops.rot90, ops.rot180, ops.fliplr]),
    ops.color_permute,
])
ds = AugmentedDataset(base, pipeline=pipe, tensorize=True)
```

### Notes for speed

- **Stay on CPU in the DataLoader, move to GPU after collation.** Most ops
  are tiny tensor reshapes that don't benefit from GPU launch overhead;
  cross workers + pinned host transfers are the win.
- **Avoid `.numpy()` in hot loops.** All built-ins live in torch already;
  only `find_objects` falls back to numpy (BFS over an array). If you
  need batched object detection, write a torch-native variant and
  `@register` it.
- **Use `num_workers > 0`.** Pipelines + Grid are pure Python and pickle
  fine, so workers parallelize cleanly.
- **Reproducibility.** Pass `generator=torch.Generator().manual_seed(seed)`
  to grid generators, and `rng=random.Random(seed)` to `Random*` pipeline
  components. Seed each worker via `worker_init_fn` if you want strict
  reproducibility under multi-worker loading.
- **Variable-shape batches.** Rigid ops can change `(H, W)`. If you need
  fixed-shape batches, follow with `pad` to a target size or filter to
  one shape; `default_collate` does not stack mismatched tensors.
- **Type contract.** Tensors are always `torch.long` (color ids), 2D
  `(H, W)` per sample. Embeddings/CE loss work directly on these.

### One-shot tensorization

```python
g = Grid.random(16, 16)
t = g.torch()                                     # (16, 16) torch.long, no copy
t = t.unsqueeze(0).cuda()                          # (1, 16, 16) on GPU
```

`Grid.torch()` does not copy; mutate via `Grid.like(new_tensor)` instead
of editing the underlying storage in place.

---

## 11. Adding your own op

One decorator. The op becomes available functionally, fluently, by-name,
and inside config-driven pipelines — instantly.

```python
from ambiente import register, Grid
import torch

@register
def diagonal_stripes(grid, color: int = 4, period: int = 3):
    """Overlay diagonal stripes of `color` every `period` cells."""
    H, W = grid.shape
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    new = grid.data.clone()
    new[(yy + xx) % int(period) == 0] = int(color)
    return grid.like(new)


# all of these now work
ops.diagonal_stripes(g, color=2)
g.diagonal_stripes(color=2)
ops.apply("diagonal_stripes", g, color=2)
Compose([("diagonal_stripes", {"color": 2, "period": 4})])(g)
```

**Op author's contract:**

- First argument is the input `Grid`.
- Return a new `Grid`. Use `grid.like(new_tensor)` to preserve palette
  and background.
- Don't mutate `grid.data` — clone first if you need to write.
- Document any kwargs in the docstring; they show up in the auto-bound
  fluent method's `__doc__`.
- Optional: accept a `generator=torch.Generator()` for stochastic ops.

Register a different name with `@register(name="alias")`.

---

## 12. Project layout

```
ambiente/
  __init__.py        public API
  grid.py            Grid: torch.LongTensor wrapper, fluent dispatch
  palette.py         ARC palette + color-count constants
  registry.py        @register, OPS dict, apply()
  pipeline.py        Compose, RandomChoice, RandomApply, RandomOrder
  generate.py        random_grid, sparse_grid, rectangle, with_objects
  viz.py             show, show_pair, show_many, show_task, show_diff, draw_grid
  editor.py          PairEditor (interactive matplotlib UI)
  data.py            PairedOpDataset, AugmentedDataset (torch Datasets)
  io.py              save_pair / load_pair / save_task / load_task (ARC JSON)
  arc.py             ARC-AGI iteration + 0-arg candidate ops + solver harness
  ops/
    rigid.py         rot90/180/270, fliplr, flipud, transpose, anti_transpose
    color.py         recolor, swap_colors, color_permute, set_background, mask_color
    crop.py          crop, pad, punch_hole, shift
    objects.py       find_objects, translate_object, keep_largest_object, n_objects
    physics.py       gravity, reflect, laser

examples/
  01_quickstart.py             three call styles + auto-discovery
  02_custom_op.py              register a new op
  03_dataloader.py             torch DataLoader with PairedOpDataset
  04_pipeline_and_viz.py       Compose + RandomChoice with viz
  05_editor.py                 launch the PairEditor
  06_viz_gallery.py            annotate, show_diff, show_task
  07_arc_explorer.py           full visual tour over ARC-AGI

tests/
  test_smoke.py
  test_editor_and_viz.py
  test_arc.py
```

---

## 13. Testing

```bash
python -m pytest -q
```

19 tests, all green. The viz tests use the headless `Agg` backend and
assert on rendered PNG file sizes; the editor tests exercise the data
layer (paint / clear / swap / resize / save / load) without opening a
window. The ARC tests skip cleanly if `datasets/` isn't cloned.
