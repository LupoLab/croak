"""Matplotlib canvas embedded in Qt, plus small widget helpers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QResizeEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from superqt import QCollapsible

from ..plotting import FULL_SCALE, PlotScale, plot_scale, scaled_rc
from . import settings

__all__ = [
    "MplCanvas",
    "with_toolbar",
    "collapsible",
    "group",
    "add_wide_row",
    "spin",
    "dspin",
    "combo",
    "check",
]


class MplCanvas(FigureCanvasQTAgg):
    """A matplotlib canvas whose text density follows its on-screen size.

    ``figsize`` is the size in inches the figure's layout was *designed* for: the
    canvas prefers it, but Qt resizes the figure freely to fill its slot. Text is
    sized in points, so in a small window the axes would shrink while the text did
    not (on a 1366x768 laptop the twelve-panel retrieval figure spent 78 % of its
    area on text and padding). :meth:`render` therefore plots inside a
    :func:`matplotlib.rc_context` scaled by :func:`croak.plotting.plot_scale`
    (down to 70 % of the default sizes), and a resize that crosses a density step
    re-runs the remembered plot function after a short debounce.

    Parameters
    ----------
    figsize : tuple of float
        Design size ``(width, height)`` in inches; also the initial figure size.
    """

    #: Debounce (ms) between a resize and the re-plot: a window drag fires many
    #: resize events, and one redraw once it settles is enough.
    REPLOT_DELAY_MS = 150

    def __init__(self, figsize: tuple[float, float] = (5.0, 4.0)):
        self.design_size: tuple[float, float] = figsize
        figure = Figure(figsize=figsize, layout="constrained")
        super().__init__(figure)
        self._plot_fn: Callable[[Figure], object] | None = None
        self._drawn_scale: PlotScale | None = None
        self._replot_timer = QTimer(self)
        self._replot_timer.setSingleShot(True)
        self._replot_timer.setInterval(self.REPLOT_DELAY_MS)
        self._replot_timer.timeout.connect(self._replot)

    @property
    def rendered(self) -> bool:
        """Whether :meth:`render` has been called (so there is a plot to export)."""
        return self._plot_fn is not None

    @property
    def scale(self) -> PlotScale:
        """Text density for the canvas's current on-screen size."""
        width, height = self.figure.get_size_inches()
        return plot_scale(float(width), float(height), self.design_size)

    def render(self, plot_fn: Callable[[Figure], object]) -> None:
        """Clear the figure, run ``plot_fn(figure)`` at the current density, redraw.

        ``plot_fn`` is remembered and re-run whenever a resize changes the density,
        so it must read current state rather than values that go stale, and it
        must tolerate being called again. Matplotlib resolves font sizes when an
        artist is *created*, so everything that creates text has to happen inside
        ``plot_fn``; call :meth:`draw_idle` directly only for in-place data updates
        that create no new artists (see :meth:`croak.plotting.FrogFilterView.update`).

        Parameters
        ----------
        plot_fn : callable
            Receives the cleared :class:`~matplotlib.figure.Figure` and draws into
            it. Its return value is ignored.
        """
        self._plot_fn = plot_fn
        self._draw_at(self.scale)

    def clear(self) -> None:
        """Blank the figure and forget the plot function (nothing to re-plot)."""
        self._plot_fn = None
        self._drawn_scale = None
        self.figure.clear()
        self.draw_idle()

    def save_figure(self, path: str | Path, *, dpi: int) -> None:
        """Save the remembered plot at the design size and full text density.

        The on-screen figure may be small and drawn with shrunken text; the export
        is drawn afresh into a headless figure of :attr:`design_size` inches at
        matplotlib's default sizes, so a PDF saved from a laptop window is the same
        file a large display would produce.

        Parameters
        ----------
        path : str or Path
            Output file; the format follows the extension (``.pdf``, ``.png``, …).
        dpi : int
            Raster resolution for image content (the pcolormesh panels are
            rasterised even inside a PDF).

        Raises
        ------
        RuntimeError
            If nothing has been rendered on this canvas yet.
        """
        if self._plot_fn is None:
            raise RuntimeError("nothing has been rendered on this canvas yet")
        figure = Figure(figsize=self.design_size, layout="constrained")
        with matplotlib.rc_context(scaled_rc(FULL_SCALE)):
            self._plot_fn(figure)
        figure.savefig(path, dpi=dpi)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 — Qt override
        """Let matplotlib resize the figure, then schedule a re-plot if needed."""
        super().resizeEvent(event)
        if self._plot_fn is not None and self.scale != self._drawn_scale:
            self._replot_timer.start()

    def _draw_at(self, scale: PlotScale) -> None:
        """Run the remembered plot function at ``scale`` and repaint."""
        if self._plot_fn is None:  # pragma: no cover — guarded by both callers
            raise RuntimeError("no plot function to draw; call render() first")
        self._drawn_scale = scale
        with matplotlib.rc_context(scaled_rc(scale)):
            self.figure.clear()
            self._plot_fn(self.figure)
        self.draw_idle()

    def _replot(self) -> None:
        """Debounced resize handler: re-plot only if the density really changed."""
        if self._plot_fn is not None and self.scale != self._drawn_scale:
            self._draw_at(self.scale)


def with_toolbar(canvas: MplCanvas) -> QWidget:
    """Wrap ``canvas`` in a panel with a matplotlib pan/zoom/save toolbar on top.

    Returns a container widget to drop into a layout; the caller keeps its own
    reference to ``canvas`` for plotting. The toolbar adds the standard
    Home/Back/Forward, Pan, Zoom-to-rectangle, Subplots, and Save controls, plus a
    live cursor read-out. A full redraw (:meth:`MplCanvas.render`) resets the
    view, so zoom is most useful between control changes; Home restores it.
    """
    panel = QWidget()
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    layout.addWidget(NavigationToolbar2QT(canvas, panel))
    layout.addWidget(canvas)
    return panel


def collapsible(
    title: str,
    content: QWidget,
    *,
    expanded: bool = True,
    settings_key: str | None = None,
) -> QCollapsible:
    """Wrap ``content`` in a click-to-fold strip headed ``title``.

    For auxiliary widgets that sit under a plot (a numeric read-out, say) and
    have a fixed natural height: folding one hands its pixels back to the plot.
    Not for canvases, which should stretch — put those on a splitter instead.

    Parameters
    ----------
    title : str
        Header text; clicking it toggles the strip.
    content : QWidget
        The widget to fold.
    expanded : bool
        Initial state when nothing is remembered.
    settings_key : str, optional
        With a key, the state is restored from :mod:`croak.gui.settings` and
        every toggle (by click or by :meth:`~superqt.QCollapsible.collapse` /
        :meth:`~superqt.QCollapsible.expand`) is remembered under it.

    Returns
    -------
    superqt.QCollapsible
    """
    strip = QCollapsible(title)
    strip.setContent(content)
    strip.setDuration(120)  # visible but brief
    if settings_key is not None:
        expanded = settings.value(settings_key, expanded)
    if expanded:
        strip.expand(animate=False)
    else:
        strip.collapse(animate=False)
    if settings_key is not None:
        key = settings_key
        strip.toggled.connect(lambda state: settings.set_value(key, state))
    return strip


# -- tiny declarative widget builders ---------------------------------------
def group(title: str) -> tuple[QGroupBox, QFormLayout]:
    """A titled group box with a form layout.

    The field-growth policy is set explicitly because ``QFormLayout``'s default is
    style-dependent: the macOS style resolves it to ``FieldsStayAtSizeHint``, which
    sizes each field to its *sizeHint* and so collapses the slider controls to the
    width of their spin boxes. ``ExpandingFieldsGrow`` widens only the fields that
    actually ask for the space (the ``Expanding`` size policy of
    :class:`~croak.gui.widgets.RangeControl`/:class:`~croak.gui.widgets.ValueControl`
    and of ``QLineEdit``), leaving spin boxes, combos and check boxes at their
    natural size.
    """
    box = QGroupBox(title)
    form = QFormLayout(box)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
    return box, form


def add_wide_row(form: QFormLayout, label: str, widget: QWidget) -> None:
    """Add a form row laid out label-above-field, so the field spans the column.

    Used for the slider controls (:class:`~croak.gui.widgets.RangeControl` /
    :class:`~croak.gui.widgets.ValueControl`), which are unusable when squeezed into
    the narrow right-hand field column. ``WrapAllRows`` is a *form*-wide policy, so
    adding one slider wraps the rest of its group box too — that is the intended
    trade and keeps a group visually uniform. The label stays in the form's label
    role (rather than becoming a separate spanning row), which is what
    :func:`~croak.gui.help.apply_tooltips` matches help entries against.
    """
    form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
    form.addRow(label, widget)


def spin(lo: int, hi: int, value: int, on_change=None, step: int = 1) -> QSpinBox:
    w = QSpinBox()
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setValue(int(value))
    if on_change:
        w.valueChanged.connect(on_change)
    return w


def dspin(
    lo: float,
    hi: float,
    value: float,
    on_change=None,
    decimals: int = 3,
    step: float = 1.0,
) -> QDoubleSpinBox:
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(decimals)
    w.setSingleStep(step)
    w.setValue(float(value))
    if on_change:
        w.valueChanged.connect(on_change)
    return w


def combo(items, value=None, on_change=None) -> QComboBox:
    w = QComboBox()
    w.addItems([str(i) for i in items])
    if value is not None and str(value) in [str(i) for i in items]:
        w.setCurrentText(str(value))
    if on_change:
        w.currentTextChanged.connect(on_change)
    return w


def check(label: str, value: bool, on_change=None) -> QCheckBox:
    w = QCheckBox(label)
    w.setChecked(bool(value))
    if on_change:
        w.toggled.connect(on_change)
    return w
