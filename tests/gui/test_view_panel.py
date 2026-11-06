"""A plot's way out of the window: figures and images (:mod:`croak.gui.view_panel`)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QMainWindow,
    QWidget,
)

from croak.gui.view_panel import ViewPanel, export_figure, page_figure
from croak.plotting.screen import CurveView, LineView, MapView

X = np.linspace(0.0, 1.0, 8)
LINE = LineView(
    curves=(CurveView(X, X, "retrieved"),),
    title="Temporal",
    xlabel="Time (fs)",
    ylabel="Intensity",
)
MAP = MapView(
    values=np.outer(X, X),
    x=X,
    y=X,
    levels=(0.0, 1.0),
    title="Measured trace",
    xlabel="Delay (fs)",
    ylabel="Wavelength (nm)",
)


def _save_to(monkeypatch, path):
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (path, ""))


@pytest.fixture
def window(qtbot):
    """A main window, so a panel's notices have a status bar to go to."""
    window = QMainWindow()
    qtbot.addWidget(window)
    window.resize(800, 600)
    return window


def test_view_panel_menu_offers_its_figure_once_it_shows_a_view(qtbot):
    panel = ViewPanel("map", colours="magma", white=False)
    qtbot.addWidget(panel)
    assert panel.figure() is None
    menu = panel.show_menu(QPoint(5, 5))
    actions = {a.text(): a for a in menu.actions() if a.text()}
    assert not actions["Export this plot as a figure…"].isEnabled()
    assert set(actions) == {
        "Export this plot as a figure…",
        "Save as an image…",
        "Copy as an image",
    }
    panel.show_view(MAP)
    figure = panel.figure()
    assert figure is not None
    mesh = figure.axes[0].collections[0]
    assert mesh.cmap(0.0)[:3] != (1.0, 1.0, 1.0)  # the panel's own map, not white
    panel.say("Residuals", "No residual for this result.")
    assert panel.figure() is None


def test_view_panel_right_click_opens_its_menu(qtbot, monkeypatch):
    panel = ViewPanel("line")
    qtbot.addWidget(panel)
    opened = []
    monkeypatch.setattr(panel, "show_menu", opened.append)
    panel.plot.customContextMenuRequested.emit(QPoint(3, 4))
    assert opened == [QPoint(3, 4)]


def test_view_panel_exports_its_figure_and_says_so(window, monkeypatch, tmp_path):
    panel = ViewPanel("line")
    window.setCentralWidget(panel)
    panel.show_view(LINE)
    _save_to(monkeypatch, "")
    assert panel.export_figure() == ""  # cancelled
    target = tmp_path / "temporal.svg"
    _save_to(monkeypatch, str(target))
    assert panel.export_figure() == str(target)
    assert "<svg" in target.read_text()
    assert window.statusBar().currentMessage() == "Saved temporal.svg"


def test_view_panel_saves_and_copies_an_image_of_the_screen(
    qtbot, monkeypatch, tmp_path
):
    panel = ViewPanel("line")
    qtbot.addWidget(panel)
    panel.resize(300, 200)
    panel.show_view(LINE)
    _save_to(monkeypatch, "")
    assert panel.save_image() == ""
    target = tmp_path / "plot.png"
    _save_to(monkeypatch, str(target))
    assert panel.save_image() == str(target)
    assert target.read_bytes()[:4] == b"\x89PNG"
    panel.copy_image()
    assert not QApplication.clipboard().pixmap().isNull()
    panel._say("outside a main window: nothing to tell")  # no status bar, no error


def test_export_figure_says_when_there_is_nothing_to_save(qtbot):
    widget = QWidget()
    qtbot.addWidget(widget)
    said = []
    assert export_figure(widget, None, "page", said.append) == ""
    assert said == ["There is nothing to save as a figure yet."]


def test_page_figure_lays_plots_out_as_on_the_screen(qtbot):
    page = QWidget()
    qtbot.addWidget(page)
    grid = QGridLayout(page)
    panels = {}
    for r, c, kind in ((0, 0, "line"), (0, 1, "map"), (1, 0, "map"), (1, 1, "line")):
        panels[r, c] = ViewPanel(kind)
        grid.addWidget(panels[r, c], r, c)
    page.resize(800, 600)
    page.show()
    qtbot.waitExposed(page)
    assert page_figure(page) is None  # nothing drawn yet
    panels[0, 0].show_view(LINE)
    panels[0, 1].show_view(MAP)
    panels[1, 0].show_view(MAP)  # (1, 1) shows nothing: left out
    figure = page_figure(page)
    assert figure is not None
    titles = [
        ax.get_title(loc="left") for ax in figure.axes if ax.get_title(loc="left")
    ]
    assert titles == ["Temporal", "Measured trace", "Measured trace"]
    assert tuple(figure.get_size_inches()) == pytest.approx((6.8, 5.2))  # 2 × 2
    panels[1, 0].hide()  # a hidden plot is not on the page either
    assert len(page_figure(page).get_axes()) == 3  # a line, a map and its bar
