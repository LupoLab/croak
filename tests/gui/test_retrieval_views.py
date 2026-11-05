"""The retrieval's panels on screen: the live overview and Analyse's result pages."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QFileDialog

from croak import plotting
from croak.forward import maketrace
from croak.grid import Grid
from croak.gui.retrieval_views import (
    OVERVIEW,
    PANELS,
    ResultPage,
    RetrievalPanels,
)
from croak.gui.state import WizardState
from croak.maths import wlfreq
from croak.processing import TruthPulse
from croak.pulses import gaussian_pulse
from croak.retrieve import retrieve


@pytest.fixture(scope="module")
def data():
    """A short PG retrieval with a measured spectrum and a known truth."""
    g = Grid(96, dt=0.5e-15)
    omega0 = float(wlfreq(800e-9))
    ew = gaussian_pulse(g, 7e-15, phases=[20e-30])
    delays = np.linspace(-50e-15, 50e-15, 60)
    trace = maketrace(g.omega, delays, ew, "pg")
    res = retrieve(
        trace,
        g.omega,
        delays,
        "pg",
        algorithm="copra",
        maxiters=20,
        guess=ew,
        omega0=omega0,
    )
    return plotting.retrieval_plot_data(
        res,
        measured=trace,
        Iomega_meas=np.abs(ew) ** 2,
        lam_min=700e-9,
        lam_max=950e-9,
        truth=TruthPulse.from_spectrum(g, ew, omega0),
    )


def test_retrieval_views_cover_every_figure_panel():
    assert set(PANELS) == set(plotting.RETRIEVAL_PANELS)


def test_retrieval_panels_move_from_progress_to_results(qtbot, data):
    panels = RetrievalPanels()
    qtbot.addWidget(panels)
    assert panels.mode == "empty"
    panels.show_progress([0.5, 0.2, 0.1])
    assert panels.mode == "progress"
    assert panels.convergence.log_y
    panels.set_data(data)
    assert panels.mode == "panels" and panels.data is data
    assert set(panels.boxes) == {k for row in OVERVIEW for k in row}
    temporal = panels.boxes["temporal"]
    assert temporal.title.text() == "Retrieved pulse"
    assert temporal.plot.right_box is not None  # the phase axis
    spectral = panels.boxes["spectral"]
    assert spectral.notes.text().startswith("GDD:")
    assert spectral.notes.property("warn") is False
    panels.clear()
    assert panels.mode == "empty" and panels.data is None


def test_retrieval_panels_draw_every_page(qtbot, data):
    for spec in plotting.RETRIEVAL_PAGES.values():
        panels = RetrievalPanels(spec.mosaic)
        qtbot.addWidget(panels)
        panels.set_data(data)
        for key, box in panels.boxes.items():
            assert box.title.text(), key
    traces = RetrievalPanels(plotting.RETRIEVAL_PAGES["traces"].mosaic)
    qtbot.addWidget(traces)
    traces.resize(900, 700)
    traces.show()
    qtbot.waitExposed(traces)
    traces.set_data(data)
    note = traces.boxes["retrieved_lin"].notes.text()
    assert note.startswith("R ") and "×" in note
    # The trace maps are linked: zooming one zooms the others.
    first = traces.boxes["measured_log"].plot
    first.box.setXRange(-10, 10, padding=0)
    other = traces.boxes["retrieved_lin"].plot
    assert other.box.viewRange()[0] == pytest.approx([-10, 10])


def test_retrieval_panels_say_when_there_is_no_residual(qtbot, data):
    panels = RetrievalPanels((("residual",),))
    qtbot.addWidget(panels)
    panels.set_data(replace(data, processed=replace(data.processed, residual=None)))
    box = panels.boxes["residual"]
    assert box.title.text() == "Residuals"
    assert box.plot.data is None and box.plot.message.isVisible()


def test_retrieval_panels_warn_about_edge_energy(qtbot, data):
    panels = RetrievalPanels((("spectral",),))
    qtbot.addWidget(panels)
    leaky = replace(data, processed=replace(data.processed, edge_energy=0.5))
    panels.set_data(leaky)
    assert panels.boxes["spectral"].notes.property("warn") is True


def test_result_page_follows_the_result_and_pops_out(qtbot, data):
    state = WizardState()
    current = {"data": None}
    page = ResultPage(state, "pulse", lambda: current["data"])
    qtbot.addWidget(page)
    page.show()
    page.on_enter()
    assert page.panels.mode == "empty"
    current["data"] = data
    state.result_changed.emit()  # visible: redrawn at once
    assert page.panels.mode == "panels"
    dialog = page.pop_out()
    follower = dialog.findChild(RetrievalPanels)
    assert follower is not None and follower.mode == "panels"
    current["data"] = replace(data)
    state.result_changed.emit()
    assert follower.data is current["data"]
    dialog.close()
    assert page.popouts == []
    page.hide()
    state.result_changed.emit()  # hidden: drawn when next shown
    empty = ResultPage(state, "overview", lambda: None)
    qtbot.addWidget(empty)
    assert empty.pop_out().findChild(RetrievalPanels).mode == "empty"


def test_result_page_exports_its_figure(qtbot, data, tmp_path, monkeypatch):
    state = WizardState()
    messages = []
    page = ResultPage(state, "overview", lambda: data)
    page.message.connect(messages.append)
    qtbot.addWidget(page)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))
    assert page.export_figure() == ""  # cancelled
    target = tmp_path / "overview.png"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), "")
    )
    assert page.export_figure() == str(target)
    assert target.stat().st_size > 0
    assert messages == ["Saved overview.png"]
    diagnostics = ResultPage(state, "diagnostics", lambda: data)
    qtbot.addWidget(diagnostics)
    target = tmp_path / "diagnostics.svg"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), "")
    )
    assert diagnostics.export_figure() == str(target)
    nothing = ResultPage(state, "traces", lambda: None)
    nothing.message.connect(messages.append)
    qtbot.addWidget(nothing)
    assert nothing.export_figure() == ""
    assert "Run a retrieval first" in messages[-1]


def test_view_panel_draws_bars_bands_spans_points_and_levels(qtbot):
    from test_plotting_screen import _estimate

    from croak.gui.retrieval_views import ViewPanel
    from croak.plotting.screen import uncertainty_views

    panels = [ViewPanel("line") for _ in range(3)]
    for panel in panels:
        qtbot.addWidget(panel)
    hist, band = uncertainty_views(_estimate())
    panels[0].show_view(hist)
    panels[1].show_view(band)
    plot = panels[0].plot
    kinds = {type(item).__name__ for item in plot.extras}
    assert kinds == {"BarGraphItem", "LinearRegionItem"}
    assert plot.box.viewRange()[0] == pytest.approx(
        [hist.bars.edges[0], hist.bars.edges[-1]]
    )
    assert any(type(i).__name__ == "FillBetweenItem" for i in panels[1].plot.extras)
    fwhm, error, _ = uncertainty_views(_estimate("thickness", thickness=True))
    panels[0].show_view(fwhm)  # redrawn: the histogram's extras go
    kinds = {type(item).__name__ for item in plot.extras}
    assert "BarGraphItem" not in kinds and "InfiniteLine" in kinds
    assert plot.items[0].opts["symbol"] == "o"
    panels[2].show_view(error)
