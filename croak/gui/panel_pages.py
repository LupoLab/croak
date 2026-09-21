"""Tabbed pages of retrieval panels for the Retrieve stage, with pop-out windows.

Why pages rather than one twelve-panel figure: on a 1366×768 display a single
3×4 figure leaves each axes about 124×61 px once titles, tick labels and colorbars
have taken their fixed share, while three 2×2 pages of the same panels give each
axes roughly 370×137 px in the same canvas. The panels are the headless ones in
:mod:`croak.plotting` (:data:`~croak.plotting.RETRIEVAL_PAGES`), so a page shows
exactly what :func:`~croak.plotting.plot_retrieval` shows, and the overview stays
available for export.

Only the visible page is drawn when new data arrive; the others are marked stale
and drawn when their tab is selected. Before the first full live preview of a run
the stage shows the convergence curve instead ("convergence mode"), on whichever
page is visible.

A page is drawn once and later results are pushed onto its artists in place
(:class:`~croak.plotting.RetrievalPageView`), so a live preview costs no layout
solve and nothing jumps between frames; the page is redrawn from scratch only
when the result's shape, its optional curves, or the canvas's text density change.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QTabWidget, QToolButton, QVBoxLayout, QWidget

from .. import plotting
from ..plotting import (
    RETRIEVAL_PAGES,
    ConvergenceArtists,
    PageSpec,
    PlotScale,
    RetrievalPageView,
    RetrievalPlotData,
)
from .canvas import MplCanvas, with_toolbar

__all__ = ["RetrievalPages", "PagePopout", "PAGE_FIGSIZE"]

#: Design size (inches) of one 2×2 page: the overview's default 17.0×9.45 in
#: gives each of its twelve panels this much, so a page panel is drawn at the same
#: size as its overview twin whenever the canvas has the room.
PAGE_FIGSIZE = (8.5, 6.3)


class PagePopout(QDialog):
    """A top-level window showing one page of retrieval panels.

    Opened by :meth:`RetrievalPages.pop_out`; follows every later result the pages
    widget receives and closes with the stage that owns it.
    """

    def __init__(self, page: PageSpec, parent: QWidget | None = None):
        super().__init__(parent, Qt.WindowType.Window)
        self.page = page
        self.setWindowTitle(f"croak — {page.title}")
        # Closing frees the window; the owner drops its reference on ``finished``.
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.canvas = MplCanvas(figsize=PAGE_FIGSIZE)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(with_toolbar(self.canvas))
        self.resize(int(PAGE_FIGSIZE[0] * 100), int(PAGE_FIGSIZE[1] * 100) + 40)


class RetrievalPages(QWidget):
    """The Retrieve stage's plot area: one tab per page of retrieval panels.

    Parameters
    ----------
    pages : mapping of str to PageSpec
        The pages to offer, in tab order; defaults to
        :data:`~croak.plotting.RETRIEVAL_PAGES`.
    parent : QWidget, optional

    Notes
    -----
    Every canvas goes through :meth:`~croak.gui.canvas.MplCanvas.render`, so the
    pages re-plot themselves when a resize changes their text density. Each tab
    carries the matplotlib toolbar, whose Save writes *that page*; the stage's own
    Save… writes the full overview.
    """

    def __init__(
        self,
        pages: Mapping[str, PageSpec] = RETRIEVAL_PAGES,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._pages: dict[str, PageSpec] = dict(pages)
        self._data: RetrievalPlotData | None = None
        self._errors: Sequence[float] | None = None
        self._stale: set[str] = set()
        self._canvases: dict[str, MplCanvas] = {}
        self._popouts: dict[str, PagePopout] = {}
        # Per-canvas presentation state (tabs and pop-outs alike): the artist-
        # reusing view of its page, the convergence-curve artists while a run has
        # no full preview yet, and the text density the canvas was last drawn at.
        self._views: dict[MplCanvas, RetrievalPageView] = {}
        self._conv: dict[MplCanvas, ConvergenceArtists] = {}
        self._scales: dict[MplCanvas, PlotScale] = {}

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)  # a slim tab bar: every pixel is the plot's
        for key, page in self._pages.items():
            canvas = MplCanvas(figsize=PAGE_FIGSIZE)
            self._canvases[key] = canvas
            self.tabs.addTab(with_toolbar(canvas), page.title)
        self.popout_btn = QToolButton()
        self.popout_btn.setText("Pop out")
        self.popout_btn.setToolTip(
            "Open the current page in its own window, which follows later results."
        )
        self.popout_btn.clicked.connect(lambda: self.pop_out())
        self.tabs.setCornerWidget(self.popout_btn, Qt.Corner.TopRightCorner)
        self.tabs.currentChanged.connect(self._on_tab_changed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)

    # -- queries ------------------------------------------------------------
    def page_keys(self) -> tuple[str, ...]:
        """The page keys in tab order."""
        return tuple(self._pages)

    def current_page(self) -> str:
        """Key of the page whose tab is selected."""
        return self.page_keys()[self.tabs.currentIndex()]

    def set_current_page(self, key: str) -> None:
        """Select the tab of page ``key`` (drawing it if it is stale)."""
        self.tabs.setCurrentIndex(self.page_keys().index(key))

    def is_stale(self, key: str) -> bool:
        """Whether page ``key`` still shows older content than the latest data."""
        return key in self._stale

    def canvas(self, key: str) -> MplCanvas:
        """The canvas of page ``key``."""
        return self._canvases[key]

    @property
    def data(self) -> RetrievalPlotData | None:
        """The data the pages currently show, if any."""
        return self._data

    # -- content ------------------------------------------------------------
    def set_data(self, data: RetrievalPlotData) -> None:
        """Show a result: draw the visible page (and pop-outs), mark the rest stale."""
        self._data = data
        self._errors = None  # leaves convergence mode
        self._present()

    def show_progress(self, errors: Sequence[float]) -> None:
        """Show the convergence curve of a run in progress instead of a result.

        ``errors`` is read live (the stage appends to it), so a re-plot after a
        resize shows the latest history.
        """
        self._errors = errors
        self._present()

    def clear(self) -> None:
        """Blank every page and forget the data (a new run is starting)."""
        self._data = None
        self._errors = None
        self._stale = set()
        self._conv.clear()
        for view in self._views.values():
            view.reset()
        for canvas in self._all_canvases():
            canvas.clear()

    def pop_out(self, key: str | None = None) -> PagePopout:
        """Open (or raise) a window showing page ``key`` (default: the current one)."""
        key = self.current_page() if key is None else key
        popout = self._popouts.get(key)
        if popout is None:
            popout = PagePopout(self._pages[key], self)
            popout.finished.connect(lambda _result, k=key: self._forget_popout(k))
            self._popouts[key] = popout
            self._present_canvas(popout.canvas, key)
        popout.show()
        popout.raise_()
        popout.activateWindow()
        return popout

    # -- internals ----------------------------------------------------------
    def _all_canvases(self) -> list[MplCanvas]:
        return [*self._canvases.values(), *(p.canvas for p in self._popouts.values())]

    def _forget_popout(self, key: str) -> None:
        popout = self._popouts.pop(key, None)
        if popout is not None:
            for store in (self._views, self._conv, self._scales):
                store.pop(popout.canvas, None)

    def _present_canvas(self, canvas: MplCanvas, key: str) -> None:
        """Bring ``canvas`` (showing page ``key``) up to date with the current mode."""
        if self._errors is not None:
            self._present_convergence(canvas)
        elif self._data is not None:
            self._present_data(canvas, key, self._data)
        else:
            canvas.clear()

    def _present_convergence(self, canvas: MplCanvas) -> None:
        """Show the live error curve on ``canvas``, growing it in place if it exists."""
        errors = self._errors
        if errors is None:
            return
        artists = self._conv.get(canvas)
        if (
            artists is not None
            and canvas.rendered
            and self._scales.get(canvas) == canvas.scale
        ):
            plotting.update_convergence_curve(artists, errors)
            canvas.draw()  # synchronous: cheap, and no second idle draw follows
            return
        view = self._views.get(canvas)
        if view is not None:
            view.reset()  # the page's artists go with the cleared figure

        def plot(fig: Figure) -> None:
            # Re-run by the canvas on a density change; `errors` is read live.
            self._conv[canvas] = plotting.plot_convergence(fig.add_subplot(111), errors)
            self._scales[canvas] = canvas.scale

        canvas.render(plot)

    def _present_data(
        self, canvas: MplCanvas, key: str, data: RetrievalPlotData
    ) -> None:
        """Show ``data`` on ``canvas``: update in place when possible, else redraw.

        A redraw is needed when the page has never been drawn on this figure,
        when the result changed shape or gained/lost optional curves
        (:meth:`~croak.plotting.RetrievalPageView.can_update`), or when the
        canvas's text density changed: rc sizes bind at artist creation, so an
        in-place update would keep the stale size.
        """
        view = self._views.get(canvas)
        if view is None:
            view = self._views[canvas] = RetrievalPageView(self._pages[key])
        if (
            view.attached
            and view.fig is canvas.figure
            and self._scales.get(canvas) == canvas.scale
            and view.can_update(data)
        ):
            view.update(data)
            # Synchronous, so the caller's timing of a preview includes the paint
            # (the stage feeds it back to the worker's preview cadence) and no
            # second idle draw follows.
            canvas.draw()
            return
        self._conv.pop(canvas, None)

        def plot(fig: Figure) -> None:
            # Re-run by the canvas on a density change: draw the latest result.
            latest = self._data
            if latest is not None:
                view.draw(fig, latest)
                self._scales[canvas] = canvas.scale

        canvas.render(plot)

    def _present(self) -> None:
        """Bring the visible page and every pop-out up to date; mark the rest stale."""
        current = self.current_page()
        self._stale = set(self._pages) - {current}
        self._present_canvas(self._canvases[current], current)
        for key, popout in self._popouts.items():
            self._present_canvas(popout.canvas, key)

    def _on_tab_changed(self, index: int) -> None:
        key = self.page_keys()[index]
        if key in self._stale:
            self._present_canvas(self._canvases[key], key)
            self._stale.discard(key)
