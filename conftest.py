"""Make the repo root importable so `from ambiente import ...` works in tests
and examples without an install or PYTHONPATH gymnastics."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
