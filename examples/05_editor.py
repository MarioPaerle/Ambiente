"""Open the interactive Pair Editor.

Run from the repo root:
    python examples/05_editor.py                 # blank 6x6 pair
    python examples/05_editor.py task.json       # load existing task

You can also launch it directly:
    python -m ambiente.editor [task.json]
"""
import sys
from ambiente.editor import PairEditor

if len(sys.argv) > 1:
    editor = PairEditor.load(sys.argv[1])
else:
    editor = PairEditor(in_shape=(6, 6))

a, b = editor.run()
print("\n--- final input ---");  print(a)
print("\n--- final output ---"); print(b)
