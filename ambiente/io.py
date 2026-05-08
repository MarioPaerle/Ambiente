"""JSON I/O for grids and ARC-style tasks.

The on-disk format mirrors ARC's:

    {
      "train": [{"input": [[..]], "output": [[..]]}, ...],
      "test":  [{"input": [[..]], "output": [[..]]}, ...]
    }

For single-pair editing we also accept the flat form `{"input": ..., "output": ...}`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from .grid import Grid


def grid_to_list(g: Grid) -> list[list[int]]:
    return g.tolist()


def list_to_grid(data) -> Grid:
    return Grid(data)


def save_pair(input_grid: Grid, output_grid: Grid, path: str | Path) -> None:
    Path(path).write_text(json.dumps({
        "input": grid_to_list(input_grid),
        "output": grid_to_list(output_grid),
    }, indent=2))


def load_pair(path: str | Path) -> tuple[Grid, Grid]:
    d = json.loads(Path(path).read_text())
    if "train" in d:  # full task — return first train pair
        first = d["train"][0]
        return list_to_grid(first["input"]), list_to_grid(first["output"])
    return list_to_grid(d["input"]), list_to_grid(d["output"])


def save_task(train: Sequence[tuple[Grid, Grid]],
              test: Sequence[tuple[Grid, Grid | None]] = (),
              path: str | Path = "task.json") -> None:
    payload = {
        "train": [{"input": grid_to_list(a), "output": grid_to_list(b)}
                  for a, b in train],
        "test": [
            {"input": grid_to_list(a),
             **({"output": grid_to_list(b)} if b is not None else {})}
            for a, b in test
        ],
    }
    Path(path).write_text(json.dumps(payload, indent=2))


def load_task(path: str | Path) -> dict:
    """Returns {'train': [(Grid, Grid), ...], 'test': [(Grid, Grid|None), ...]}."""
    d = json.loads(Path(path).read_text())
    if "train" not in d and "input" in d:  # flat pair
        return {"train": [(list_to_grid(d["input"]), list_to_grid(d["output"]))],
                "test": []}
    train = [(list_to_grid(p["input"]), list_to_grid(p["output"]))
             for p in d.get("train", [])]
    test = [(list_to_grid(p["input"]),
             list_to_grid(p["output"]) if "output" in p else None)
            for p in d.get("test", [])]
    return {"train": train, "test": test}
