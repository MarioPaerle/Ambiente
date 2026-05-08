"""Ambiente: a fluent grid environment for ARC-AGI / NACT-style experiments."""
from .grid import Grid
from .palette import ARC_PALETTE, DEFAULT_PALETTE, N_COLORS
from .registry import OPS, apply, list_ops, register
from .pipeline import Compose, RandomApply, RandomChoice, RandomOrder
from . import arc
from . import generate
from . import io
from . import ops
from . import viz

__all__ = [
    "Grid",
    "ARC_PALETTE", "DEFAULT_PALETTE", "N_COLORS",
    "OPS", "apply", "list_ops", "register",
    "Compose", "RandomApply", "RandomChoice", "RandomOrder",
    "arc", "generate", "io", "ops", "viz",
]
