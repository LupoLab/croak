"""RetrievalPages: pop-out windows and switching between result and convergence."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from croak import plotting
from croak.forward import maketrace
from croak.grid import Grid
from croak.gui.panel_pages import RetrievalPages
from croak.maths import wlfreq
from croak.pulses import gaussian_pulse
from croak.retrieve import retrieve


@pytest.fixture(scope="module")
def plot_data():
    g = Grid(96, dt=0.5e-15)
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
        omega0=wlfreq(800e-9),
        progress=False,
    )
    return plotting.retrieval_plot_data(
        res, measured=trace, lam_min=700e-9, lam_max=950e-9
    )


def test_pop_out_before_any_result_shows_a_blank_page(qtbot):
    pages = RetrievalPages()
    qtbot.addWidget(pages)
    popout = pages.pop_out()
    assert popout.page.title == plotting.RETRIEVAL_PAGES[pages.current_page()].title
    assert not popout.canvas.rendered
    # A second request raises the same window rather than opening another.
    assert pages.pop_out() is popout


def test_closing_a_pop_out_forgets_it(qtbot, plot_data):
    pages = RetrievalPages()
    qtbot.addWidget(pages)
    pages.set_data(plot_data)
    key = pages.page_keys()[1]
    popout = pages.pop_out(key)
    assert popout.canvas.rendered  # drawn with the current result
    with qtbot.waitSignal(popout.finished):
        popout.close()
    assert key not in pages._popouts
    # Forgetting a window that is already gone is harmless.
    pages._forget_popout(key)
    assert pages.pop_out(key) is not popout


def test_progress_after_a_result_replaces_the_page_with_the_curve(qtbot, plot_data):
    pages = RetrievalPages()
    qtbot.addWidget(pages)
    pages.set_data(plot_data)
    key = pages.current_page()
    canvas = pages.canvas(key)
    view = pages._views[canvas]
    assert view.attached
    errors = [0.5, 0.2]
    pages.show_progress(errors)
    # The page's artists went with the cleared figure; the curve took over.
    assert not view.attached
    assert canvas in pages._conv
    # The curve grows in place from the same (live) list.
    artists = pages._conv[canvas]
    assert len(artists.line.get_xdata()) == 2
    errors.append(0.1)
    pages.show_progress(errors)
    assert pages._conv[canvas] is artists
    assert len(artists.line.get_xdata()) == 3
