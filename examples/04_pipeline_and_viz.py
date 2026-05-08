"""Composable pipelines + visualization. Run interactively to see the figures."""
import random

import torch

from ambiente import Compose, Grid, RandomApply, RandomChoice, generate, viz, ops

torch.manual_seed(7)
rng = random.Random(7)

pipe = Compose([
    RandomChoice([ops.rot90, ops.rot180, ops.fliplr, ops.transpose], rng=rng),
    RandomApply(ops.color_permute, p=0.7, rng=rng),
    RandomApply(("gravity", {"direction": "down"}), p=0.5, rng=rng),
])

base = generate.sparse_grid(10, 10, density=0.3)
samples = [pipe(base) for _ in range(7)]

viz.show_many([base, *samples],
              titles=["original"] + [f"aug {i+1}" for i in range(len(samples))],
              cols=4)
