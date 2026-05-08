"""Minimal tour: build a grid, run ops, chain them, list everything available."""
import torch

from ambiente import Grid, ops, generate

torch.manual_seed(0)

# 1. Construct grids
g = Grid([
    [0, 0, 1, 0],
    [0, 1, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 2],
])
print(g)

# 2. Functional style
print("\nrot90:")
print(ops.rot90(g))

# 3. Fluent / chained style — every registered op is auto-bound on Grid
chained = g.rot90().fliplr().swap_colors(1, 2)
print("\nrot90 -> fliplr -> swap_colors(1,2):")
print(chained)

# 4. By-name (used by config-driven pipelines)
print("\napply('transpose'):")
print(ops.apply("transpose", g))

# 5. Random generators + an object-level op
rg = generate.sparse_grid(8, 8, density=0.25)
print(f"\nsparse_grid has {rg.n_objects()} objects")
print(rg.keep_largest_object())

# 6. Discoverability
print("\nregistered ops:", ops.list_ops())
