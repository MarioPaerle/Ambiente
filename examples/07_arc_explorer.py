"""Visual tour of ARC-AGI through the Ambiente lens.

This script opens five figures in sequence:

  1. The full ARC task (train pairs + test) rendered ARC-style.
  2. A single train input transformed by every basic op in our library.
  3. The same train input run through a random augmentation pipeline 8 times.
  4. A consistency demo: a single op applied to BOTH the input and the
     expected output preserves the rule — exactly what we want for
     dataset augmentation that doesn't break supervision.
  5. A solver hit: a task that one of our primitives solves outright,
     shown as (train input → predicted → expected) per pair, plus the
     test input → predicted output.

Run from the repo root (datasets/ must be cloned):
    python examples/07_arc_explorer.py
    python examples/07_arc_explorer.py <task_id>   # pick your own task
    python examples/07_arc_explorer.py 2 <task_id> # use ARC-AGI-2
"""
from __future__ import annotations

import random
import sys

import matplotlib.pyplot as plt
import torch

from ambiente import Compose, RandomApply, RandomChoice, ops, viz
from ambiente.arc import (CANDIDATES, find_solving_ops, iter_tasks, load_arc)


# ---- arg parsing ---------------------------------------------------------

def parse_args(argv):
    version = 1
    task_id = "67a3c6ac"   # ARC-AGI-1 task solved by fliplr — good visual demo
    if len(argv) == 1 and argv[0].isdigit() and int(argv[0]) in (1, 2):
        version = int(argv[0])
    elif len(argv) == 1:
        task_id = argv[0]
    elif len(argv) == 2:
        version, task_id = int(argv[0]), argv[1]
    return version, task_id


# ---- figure builders -----------------------------------------------------

def fig_show_task(task: dict, task_id: str):
    viz.show_task(task["train"], task["test"],
                  block=False)
    plt.gcf().suptitle(f"ARC-AGI task {task_id} — full layout", fontsize=11)


def fig_op_gallery(input_grid, op_names: list[str]):
    grids = [input_grid] + [CANDIDATES[n](input_grid) for n in op_names]
    titles = ["original"] + op_names
    viz.show_many(grids, titles=titles, cols=4, block=False)
    plt.gcf().suptitle("the same input under every basic op", fontsize=11)


def fig_augmentation_gallery(input_grid, n: int = 8, seed: int = 0):
    rng = random.Random(seed)
    pipe = Compose([
        RandomChoice([ops.rot90, ops.rot180, ops.rot270,
                      ops.fliplr, ops.flipud, ops.transpose], rng=rng),
        RandomApply(ops.color_permute, p=0.7, rng=rng),
        RandomApply(("shift", {"dy": 1, "dx": 0}), p=0.4, rng=rng),
    ])
    samples = [input_grid] + [pipe(input_grid) for _ in range(n)]
    titles = ["original"] + [f"aug {i+1}" for i in range(n)]
    viz.show_many(samples, titles=titles, cols=3, block=False)
    plt.gcf().suptitle("random augmentation pipeline applied to a real ARC input",
                      fontsize=11)


def fig_paired_consistency(task: dict):
    """Show that augmenting (input, output) jointly preserves the puzzle.

    Apply rot90 to *both* sides of every train pair; the relationship between
    them is unchanged, so this is safe data augmentation for ARC.
    """
    pairs = task["train"][:3]
    grids = []
    titles = []
    for i, (a, b) in enumerate(pairs):
        grids += [a, b, ops.rot90(a), ops.rot90(b)]
        titles += [f"train {i}: in", f"train {i}: out",
                  f"rot90(in)", f"rot90(out)"]
    viz.show_many(grids, titles=titles, cols=4, block=False)
    plt.gcf().suptitle("paired augmentation: rot90 applied to BOTH sides "
                       "preserves the rule", fontsize=11)


def fig_solver_hit(version: int):
    """Find a task solved by a single primitive op and visualize the solution."""
    chosen = None
    for tid, task in iter_tasks(version=version, split="training", limit=200):
        matches = find_solving_ops(task, CANDIDATES)
        # skip 'identity' — boring
        matches = [m for m in matches if m != "identity"]
        if matches:
            chosen = (tid, matches[0], task)
            break
    if chosen is None:
        print("no single-op hit in the first 200 tasks — skipping solver figure")
        return
    tid, op_name, task = chosen
    op = CANDIDATES[op_name]

    grids, titles = [], []
    for i, (a, b) in enumerate(task["train"]):
        grids += [a, op(a), b]
        titles += [f"train {i}: in", f"{op_name}(in)", f"train {i}: out"]
    test_in, test_out = task["test"][0]
    grids += [test_in, op(test_in),
              test_out if test_out is not None else test_in.like(test_in.data * 0)]
    titles += ["TEST in", f"{op_name}(TEST in)",
               "TEST expected" if test_out is not None else "TEST: hidden"]

    viz.show_many(grids, titles=titles, cols=3, block=False)
    plt.gcf().suptitle(f"solved task {tid}: a single '{op_name}' reproduces every "
                       f"output", fontsize=11)
    print(f"  → solver figure: task {tid} solved by '{op_name}'")


# ---- main ----------------------------------------------------------------

def main():
    torch.manual_seed(0)
    version, task_id = parse_args(sys.argv[1:])
    print(f"loading ARC-AGI-{version} task '{task_id}' ...")
    task = load_arc(task_id, version=version, split="training")
    print(f"  {len(task['train'])} train pairs, {len(task['test'])} test pairs")

    a0, _ = task["train"][0]
    print(f"  first train input: {a0.shape}")

    fig_show_task(task, task_id)
    fig_op_gallery(a0, ["rot90", "rot180", "fliplr", "transpose",
                        "gravity_down", "reflect_h", "keep_largest"])
    fig_augmentation_gallery(a0)
    fig_paired_consistency(task)
    fig_solver_hit(version)

    print("\nopen figures — close any to release the script.")
    plt.show()  # blocks until all windows are closed


if __name__ == "__main__":
    main()
