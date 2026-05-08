"""Smoke tests against the cloned ARC datasets. Skip if not present."""
import pytest

from ambiente.arc import (CANDIDATES, arc_root, find_solving_ops, iter_tasks,
                          list_task_ids, load_arc, solves)
from ambiente.ops import OPS


def _arc_available(version: int) -> bool:
    return arc_root(version).exists()


@pytest.mark.skipif(not _arc_available(1), reason="ARC-AGI-1 not cloned")
def test_arc1_listing_and_load():
    ids = list_task_ids(version=1, split="training")
    assert len(ids) >= 400
    task = load_arc(ids[0], version=1, split="training")
    assert "train" in task and "test" in task
    a, b = task["train"][0]
    assert a.shape and b.shape


@pytest.mark.skipif(not _arc_available(1), reason="ARC-AGI-1 not cloned")
def test_arc1_known_rotation_task_is_solved_by_rot180():
    # 3c9b0459 is a textbook rot180 task from ARC-AGI-1 training.
    task = load_arc("3c9b0459", version=1, split="training")
    assert solves(OPS["rot180"], task)
    matches = find_solving_ops(task, CANDIDATES)
    assert "rot180" in matches


@pytest.mark.skipif(not _arc_available(1), reason="ARC-AGI-1 not cloned")
def test_iter_tasks_respects_limit():
    pairs = list(iter_tasks(version=1, split="training", limit=3))
    assert len(pairs) == 3
    for tid, t in pairs:
        assert isinstance(tid, str) and "train" in t


@pytest.mark.skipif(not _arc_available(2), reason="ARC-AGI-2 not cloned")
def test_arc2_loads():
    ids = list_task_ids(version=2, split="training")
    assert len(ids) >= 100
    task = load_arc(ids[0], version=2, split="training")
    assert "train" in task
