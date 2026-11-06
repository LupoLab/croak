"""The display settings every plot follows (:mod:`croak.gui.display`)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from croak.gui import display, settings
from croak.gui.display import DisplayBar, current, hub, set_display
from croak.gui.view_panel import ViewPanel
from croak.plotting.display import Display
from croak.plotting.screen import CurveView, LineView

LAM = np.linspace(600.0, 1000.0, 41)
SPECTRUM = LineView(
    curves=(CurveView(LAM, np.exp(-(((LAM - 800.0) / 60.0) ** 2)), "measured"),),
    title="Spectrum",
    xlabel="Wavelength (nm)",
    ylabel="Intensity",
)


def test_display_defaults_until_set_and_is_remembered(qtbot):
    assert current() == Display()
    with qtbot.waitSignal(hub().changed):
        set_display(Display("frequency", 45.0))
    assert current() == Display("frequency", 45.0)


def test_display_falls_back_to_the_defaults_for_what_it_cannot_use():
    settings.set_text(display.KEY_AXIS, "sideways")
    assert current() == Display()


def test_display_hub_is_made_once():
    assert hub() is hub()


def test_display_bar_changes_and_resets_the_display(qtbot):
    bar = DisplayBar()
    qtbot.addWidget(bar)
    with qtbot.waitSignal(hub().changed):
        bar.axis.set_value("frequency", emit=True)
    assert current().axis == "frequency"
    bar.db_range.setValue(60)
    assert current() == Display("frequency", 60.0)
    assert bar.display() == current()
    bar.reset.click()
    assert current() == Display()
    assert (bar.axis.value, bar.db_range.value()) == ("native", 30)
    # a display set elsewhere shows on the bar
    set_display(Display("wavelength", 20.0))
    assert bar.display() == Display("wavelength", 20.0)
    # a new bar starts from what is kept
    again = DisplayBar()
    qtbot.addWidget(again)
    assert again.display() == Display("wavelength", 20.0)


def test_display_redraws_every_plot_on_its_new_axis(qtbot):
    panel = ViewPanel("line")
    qtbot.addWidget(panel)
    panel.redisplay()  # nothing shown yet: nothing to do
    assert panel.view is None
    panel.show_view(SPECTRUM)
    assert panel.view is SPECTRUM
    set_display(Display("frequency"))
    assert panel.view is not None and panel.view.xlabel == "Frequency (PHz)"
    axis = panel.plot.item.getAxis("bottom")
    assert axis.labelText == "Frequency (PHz)"
    set_display(Display())
    assert panel.view is SPECTRUM
