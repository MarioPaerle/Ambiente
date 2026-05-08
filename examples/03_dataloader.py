"""On-the-fly augmentation in a torch DataLoader.

Each batch yields (A, B, op_name) where A is a fresh random grid and
B is the result of one randomly-chosen rigid op. This is the canonical
trainer-side dataset for NACT mode-2 (the (A, B, T) supervised setting).
"""
import torch
from torch.utils.data import DataLoader

from ambiente import generate, ops
from ambiente.data import PairedOpDataset


def base_grid():
    return generate.sparse_grid(10, 10, density=0.3)


op_set = [
    ("rot90",     lambda g: g.rot90()),
    ("rot180",    lambda g: g.rot180()),
    ("fliplr",    lambda g: g.fliplr()),
    ("transpose", lambda g: g.transpose()),
    ("gravity_down", lambda g: g.gravity(direction="down")),
]

ds = PairedOpDataset(base=base_grid, ops=op_set, length=128, tensorize=True)
loader = DataLoader(ds, batch_size=8, shuffle=False)

for A, B, names in loader:
    print(f"A: {A.shape} {A.dtype}  B: {B.shape}  ops: {list(names)}")
    break
