"""Smoke tests covering the public surface. Run with `python -m pytest -q`."""
import random

import torch

from ambiente import (Compose, Grid, RandomApply, RandomChoice, generate, ops,
                      register)
from ambiente.data import PairedOpDataset
from ambiente.ops.objects import find_objects


def test_grid_construction_and_conversions():
    g = Grid([[0, 1], [2, 0]])
    assert g.shape == (2, 2)
    assert g.tolist() == [[0, 1], [2, 0]]
    assert torch.equal(g.torch(), torch.tensor([[0, 1], [2, 0]], dtype=torch.long))
    assert g.numpy().shape == (2, 2)
    assert g == g.copy()


def test_rigid_ops_roundtrip():
    g = Grid.random(5, 7, generator=torch.Generator().manual_seed(0))
    assert g.rot90(k=4) == g
    assert g.fliplr().fliplr() == g
    assert g.flipud().flipud() == g
    assert g.transpose().transpose() == g
    assert g.rot90(k=2) == g.rot180()


def test_color_ops():
    g = Grid([[0, 1, 2], [2, 1, 0]])
    assert g.swap_colors(1, 2) == Grid([[0, 2, 1], [1, 2, 0]])
    assert g.recolor({1: 5}) == Grid([[0, 5, 2], [2, 5, 0]])
    perm = g.color_permute(generator=torch.Generator().manual_seed(0))
    assert perm.shape == g.shape


def test_crop_pad_shift():
    g = Grid([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert g.crop(0, 0, 2, 2) == Grid([[1, 2], [4, 5]])
    padded = g.pad(top=1, fill=0)
    assert padded.shape == (4, 3) and padded.data[0].tolist() == [0, 0, 0]
    assert g.shift(dy=1, dx=0, fill=0) == Grid([[0, 0, 0], [1, 2, 3], [4, 5, 6]])


def test_objects_and_keep_largest():
    g = Grid([[0, 1, 0], [0, 1, 0], [2, 0, 0]])
    objs = find_objects(g)
    assert len(objs) == 2
    assert sorted(o.color for o in objs) == [1, 2]
    largest = g.keep_largest_object()
    assert (largest.torch() == 1).sum() == 2 and (largest.torch() == 2).sum() == 0


def test_physics_gravity_and_laser():
    g = Grid([[1, 0, 2], [0, 0, 0], [0, 3, 0]])
    down = g.gravity(direction="down")
    # column 0: [1,0,0] -> [0,0,1]; column 1: [0,0,3] -> [0,0,3]; column 2: [2,0,0] -> [0,0,2]
    assert down == Grid([[0, 0, 0], [0, 0, 0], [1, 3, 2]])

    field = Grid.zeros(3, 5)
    beam = field.laser(row=1, col=0, direction="right", color=4)
    assert beam.torch()[1].tolist() == [4, 4, 4, 4, 4]


def test_pipeline_and_random_choice():
    rng = random.Random(0)
    pipe = Compose([
        RandomChoice([ops.rot90, ops.fliplr], rng=rng),
        RandomApply(("color_permute", {"keep_background": True}), p=1.0, rng=rng),
    ])
    g = Grid.random(4, 4, generator=torch.Generator().manual_seed(0))
    out = pipe(g)
    assert out.shape == g.shape


def test_register_makes_op_globally_available():
    @register(name="thrice")
    def _(grid):
        return grid.rot90().rot90().rot90()

    g = Grid.random(3, 5, generator=torch.Generator().manual_seed(0))
    assert ops.apply("thrice", g) == g.rot90().rot90().rot90()
    assert g.thrice() == g.rot90(k=3)
    # cleanup so the suite stays idempotent
    from ambiente.registry import OPS
    OPS.pop("thrice", None)


def test_paired_dataset_yields_tensors():
    ds = PairedOpDataset(
        base=lambda: generate.sparse_grid(5, 5, density=0.4),
        ops=[("rot90", lambda g: g.rot90()),
             ("fliplr", lambda g: g.fliplr())],
        length=4, tensorize=True,
    )
    a, b, name = ds[0]
    assert a.shape == (5, 5) and b.shape == (5, 5)
    assert name in ("rot90", "fliplr")
