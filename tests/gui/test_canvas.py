"""MplCanvas re-plotting and the small widget factories."""

from __future__ import annotations

import gc
import weakref

import pytest

pytest.importorskip("PyQt6")

from croak.gui.canvas import MplCanvas, check, combo, dspin, spin, with_toolbar


def test_canvas_keeps_its_window_alive_during_a_draw(qapp):
    """A pending canvas draw must outlive its otherwise unreachable parent."""
    canvas = MplCanvas()
    panel = with_toolbar(canvas)
    # Model a discarded stage awaiting cyclic collection while its queued draw
    # still owns a Python reference to the canvas, but not to the parent widget.
    panel.cycle = panel
    owner = weakref.ref(panel)
    owners = [panel]
    del panel
    alive_during_draw = []

    def collect_during_draw(_event):
        # Release it only once the draw has started, regardless of when Python
        # happens to collect cycles from earlier tests.
        owners.clear()
        gc.collect()
        alive_during_draw.append(owner() is not None)

    canvas.mpl_connect("draw_event", collect_during_draw)
    canvas.draw()
    assert alive_during_draw == [True]
    # Drawing should keep the window alive only for the duration of the draw.
    gc.collect()
    assert owner() is None


def test_canvas_draw_without_a_plot_function_is_an_error(qtbot):
    canvas = MplCanvas()
    qtbot.addWidget(canvas)
    with pytest.raises(RuntimeError, match="render"):
        canvas._draw_at(canvas.scale)


def test_canvas_replot_only_when_something_changed(qtbot):
    canvas = MplCanvas()
    qtbot.addWidget(canvas)
    calls = []
    # Nothing rendered yet: a debounced resize has nothing to re-plot.
    canvas._replot()
    canvas.render(lambda fig: calls.append(fig))
    assert len(calls) == 1
    # Same density as drawn: no re-plot.
    canvas._replot()
    assert len(calls) == 1


def test_widget_factories_without_a_change_handler(qapp):
    assert spin(0, 10, 3).value() == 3
    assert dspin(0.0, 1.0, 0.25).value() == pytest.approx(0.25)
    box = combo(["a", "b"], value="z")  # unknown value: keeps the first item
    assert box.currentText() == "a"
    assert check("on", True).isChecked()


def test_widget_factories_connect_the_change_handler(qapp):
    seen = []
    s = spin(0, 10, 3, on_change=seen.append)
    s.setValue(4)
    d = dspin(0.0, 1.0, 0.25, on_change=seen.append)
    d.setValue(0.5)
    c = combo(["a", "b"], value="a", on_change=seen.append)
    c.setCurrentText("b")
    k = check("on", False, on_change=seen.append)
    k.setChecked(True)
    assert seen == [4, 0.5, "b", True]
