"""All built-in ops. Importing this module registers them.

Add a new op by writing a function in any submodule decorated with
`@register` and re-exporting it here. The op then becomes callable via
`ops.<name>(grid, ...)`, `grid.<name>(...)`, and `ops.apply("<name>", grid)`.
"""
from ..registry import OPS, apply, list_ops, register  # re-export

from . import rigid as _rigid
from . import color as _color
from . import crop as _crop
from . import objects as _objects
from . import physics as _physics


def __getattr__(name):
    """Resolve op names registered after import time too."""
    fn = OPS.get(name)
    if fn is None:
        raise AttributeError(f"module 'ambiente.ops' has no attribute {name!r}")
    return fn


def __dir__():
    return sorted(set(globals()) | set(OPS))
