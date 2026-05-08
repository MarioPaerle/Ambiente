"""Headless coverage of viz rendering, IO, and the editor's non-UI logic."""
import json
import os

import matplotlib

matplotlib.use("Agg")  # must precede any pyplot import
import matplotlib.pyplot as plt

from ambiente import Grid, generate, viz
from ambiente.editor import PairEditor
from ambiente.io import load_pair, load_task, save_pair, save_task


def test_viz_helpers_render_to_png(tmp_path):
    g = Grid([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    viz.show(g, save=str(tmp_path / "single.png"), block=False, annotate=True)
    viz.show_pair(g, g.rot90(), save=str(tmp_path / "pair.png"), block=False)
    viz.show_diff(g, g.shift(dy=1, dx=0),
                  save=str(tmp_path / "diff.png"), block=False)
    viz.show_task([(g, g.rot90())], [(g, None)],
                  save=str(tmp_path / "task.png"), block=False)
    plt.close("all")
    for name in ("single.png", "pair.png", "diff.png", "task.png"):
        assert (tmp_path / name).exists()
        assert (tmp_path / name).stat().st_size > 0


def test_io_roundtrip_pair(tmp_path):
    a = Grid([[0, 1], [2, 3]])
    b = Grid([[3, 2], [1, 0]])
    p = tmp_path / "pair.json"
    save_pair(a, b, p)
    a2, b2 = load_pair(p)
    assert a == a2 and b == b2


def test_io_roundtrip_task(tmp_path):
    a = Grid([[0, 1], [1, 0]])
    p = tmp_path / "task.json"
    save_task([(a, a.rot90()), (a, a.fliplr())], [(a, None)], p)
    loaded = load_task(p)
    assert len(loaded["train"]) == 2 and len(loaded["test"]) == 1
    assert loaded["test"][0][1] is None
    assert loaded["train"][0][1] == a.rot90()


def test_editor_state_mutations_without_running_ui():
    ed = PairEditor(in_shape=(4, 4), out_shape=(4, 4))
    ed.selected_color = 3
    # paint by directly hitting the state hook used by the click handler
    ed.in_data[1, 2] = ed.selected_color
    ed.out_data = ed.in_data.copy()
    assert ed.input_grid == ed.output_grid
    # swap and clear preserve invariants
    ed._swap()
    ed._clear_in()
    assert (ed.in_data == 0).all()
    assert ed.output_grid.shape == (4, 4)


def test_editor_resize_preserves_existing_cells():
    ed = PairEditor(in_shape=(3, 3))
    ed.in_data[0, 0] = 5
    ed.in_data[2, 2] = 7
    ed.focus = "input"
    ed._resize_focused("H")  # 3x3 -> 4x3
    assert ed.in_data.shape == (4, 3)
    assert ed.in_data[0, 0] == 5 and ed.in_data[2, 2] == 7
    ed._resize_focused("w")  # 4x3 -> 4x2 (drops last column with 7)
    assert ed.in_data.shape == (4, 2)
    assert ed.in_data[0, 0] == 5


def test_editor_save_and_load_via_class(tmp_path):
    ed = PairEditor(in_shape=(2, 2))
    ed.in_data[0, 0] = 1; ed.in_data[1, 1] = 2
    ed.out_data[0, 1] = 3
    p = tmp_path / "ed.json"
    ed.save(p)
    payload = json.loads(p.read_text())
    assert payload["input"][0][0] == 1 and payload["output"][0][1] == 3

    loaded = PairEditor.load(p)
    assert loaded.input_grid == ed.input_grid
    assert loaded.output_grid == ed.output_grid
