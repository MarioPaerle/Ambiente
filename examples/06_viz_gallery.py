"""Tour of the upgraded plotting helpers: annotation, task layout, diff overlay."""
import torch

from ambiente import Grid, generate, viz

torch.manual_seed(1)

a = generate.sparse_grid(8, 8, density=0.3)
b = a.rot90().color_permute(generator=torch.Generator().manual_seed(0))

viz.show(a, title="single grid + cell labels", annotate=True)
viz.show_pair(a, b, titles=("input", "output"))
viz.show_diff(a, a.shift(dy=1, dx=0))

# A whole ARC-style task: 2 train pairs + 1 test (output unknown)
train = [(generate.sparse_grid(5, 5, density=0.3),
          generate.sparse_grid(5, 5, density=0.3).rot90()) for _ in range(2)]
test = [(generate.sparse_grid(5, 5, density=0.3), None)]
viz.show_task(train, test)
