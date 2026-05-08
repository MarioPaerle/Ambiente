"""ARC-AGI dataset helpers (v1 and v2).

The cloned repos already use the same JSON schema as `ambiente.io.load_task`,
so this module is just iteration + a small "does this op solve this task?"
harness. No format conversion is needed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator

from .grid import Grid
from .io import load_task
from .ops import OPS


# ---- locate datasets -----------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent


def arc_root(version: int = 1) -> Path:
    """`<repo>/datasets/arc-agi-<version>/data`. Override by env var if needed."""
    import os
    env = os.environ.get(f"ARC{version}_ROOT")
    if env:
        return Path(env)
    return _REPO_ROOT / f"datasets/arc-agi-{version}/data"


def task_path(name: str, version: int = 1, split: str = "training") -> Path:
    """Resolve a task id (with or without `.json`) to its absolute path."""
    n = name if name.endswith(".json") else f"{name}.json"
    return arc_root(version) / split / n


# ---- iteration -----------------------------------------------------------

def list_task_ids(version: int = 1, split: str = "training") -> list[str]:
    root = arc_root(version) / split
    if not root.exists():
        raise FileNotFoundError(f"{root} not found — did you clone the dataset?")
    return sorted(p.stem for p in root.glob("*.json"))


def iter_tasks(version: int = 1, split: str = "training",
               limit: int | None = None) -> Iterator[tuple[str, dict]]:
    """Yield (task_id, task_dict) for every task in the split.

    `task_dict` is the format returned by `ambiente.io.load_task`:
        {"train": [(Grid, Grid), ...], "test": [(Grid, Grid|None), ...]}
    """
    ids = list_task_ids(version, split)
    if limit is not None:
        ids = ids[:limit]
    for tid in ids:
        yield tid, load_task(task_path(tid, version, split))


def load_arc(name: str, version: int = 1, split: str = "training") -> dict:
    return load_task(task_path(name, version, split))


# ---- op evaluation harness ----------------------------------------------

def solves(op: Callable[[Grid], Grid], task: dict) -> bool:
    """True iff applying `op` to every train input reproduces the train output.

    A safe wrapper: any exception or shape mismatch counts as "doesn't solve".
    """
    pairs = task.get("train", [])
    if not pairs:
        return False
    for a, b in pairs:
        try:
            out = op(a)
        except Exception:
            return False
        if not isinstance(out, Grid) or out.shape != b.shape or not (out == b):
            return False
    return True


def find_solving_ops(task: dict, candidates: dict[str, Callable] | None = None,
                     ) -> list[str]:
    """Return names of every candidate op that perfectly solves the task's
    *training* pairs."""
    cands = candidates if candidates is not None else CANDIDATES
    return [name for name, op in cands.items() if solves(op, task)]


# ---- candidate single-op solvers -----------------------------------------

# A library of fully-bound ops (no extra args needed at call time). Plenty
# of real ARC tasks reduce to one of these — exactly the point of running
# the harness. Add more below to grow coverage.

CANDIDATES: dict[str, Callable[[Grid], Grid]] = {
    "identity":          lambda g: g.like(g.data.clone()),
    "rot90":             OPS["rot90"],
    "rot180":            OPS["rot180"],
    "rot270":            OPS["rot270"],
    "fliplr":            OPS["fliplr"],
    "flipud":            OPS["flipud"],
    "transpose":         OPS["transpose"],
    "anti_transpose":    OPS["anti_transpose"],
    "gravity_down":      lambda g: OPS["gravity"](g, direction="down"),
    "gravity_up":        lambda g: OPS["gravity"](g, direction="up"),
    "gravity_left":      lambda g: OPS["gravity"](g, direction="left"),
    "gravity_right":     lambda g: OPS["gravity"](g, direction="right"),
    "reflect_h":         lambda g: OPS["reflect"](g, axis="horizontal"),
    "reflect_v":         lambda g: OPS["reflect"](g, axis="vertical"),
    "keep_largest":      OPS["keep_largest_object"],
}
