"""MplCanvas re-plotting and the small widget factories."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from croak.gui.canvas import MplCanvas, check, combo, dspin, spin


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
