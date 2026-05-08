"""How to add a new op: one decorator, immediately available everywhere."""
import torch

from ambiente import Grid, ops, register


@register
def diagonal_stripes(grid, color: int = 4, period: int = 3):
    """Overlay diagonal stripes of `color` on top of the grid every `period` cells."""
    H, W = grid.shape
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    new = grid.data.clone()
    new[(yy + xx) % int(period) == 0] = int(color)
    return grid.like(new)


g = Grid.zeros(6, 6)

# functional, fluent, by-name — all work for the new op
print("functional:");           print(ops.diagonal_stripes(g, color=2, period=2))
print("\nfluent:");              print(g.diagonal_stripes(color=3))
print("\napply by name:");       print(ops.apply("diagonal_stripes", g, color=7))

assert "diagonal_stripes" in ops.list_ops()
