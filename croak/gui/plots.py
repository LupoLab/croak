"""On-screen plots: white pyqtgraph views with a crosshair and a hover readout.

A view draws the arrays it is given and nothing else. What to show is decided
by the data functions in :mod:`croak.plotting`, which the exported matplotlib
figures share, so the screen and a saved figure agree. Mouse zoom and pan are on
(they replace the matplotlib toolbars); a double-click returns to the view the
data was shown with.

The readout under the pointer comes from a ``reader(x, y) -> str`` the owner
sets; :func:`map_reader` and :func:`curve_reader` build the usual ones.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import cast

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QFont, QMouseEvent, QPainter

from .theme import FONT_FAMILY, PALETTE, PLOT_PX

__all__ = [
    "COLOURS",
    "Curve",
    "LinePlot",
    "MapPlot",
    "Plot",
    "curve_reader",
    "map_reader",
    "signed_colormap",
]

pg.setConfigOptions(antialias=True, background="w", foreground=PALETTE.ink)

COLOURS = (
    "#1f6fb4",  # blue
    "#d2670b",  # orange
    "#2a9d6a",  # green
    "#8c5bb5",  # purple
    "#c2445c",  # red
    "#5a6675",  # grey
    "#b08a16",  # ochre
    "#2b8fb3",  # cyan
)
"""The curves' colours, in order of use."""

PEN_WIDTH = 1.7
"""A curve's line width, in logical pixels."""

NEGATIVE_RGBA = (0.70, 0.15, 0.12, 1.0)
"""The colour of the most negative value on a signed map (fading to white at 0)."""

_CROSSHAIR = (120, 130, 140, 190)
_NOTE_FILL = (27, 36, 48, 235)
_LABEL_STYLE = {
    "color": PALETTE.muted,
    "font-size": f"{PLOT_PX}px",
    "font-family": FONT_FAMILY,
}
# Past these fractions of the view the readout flips to the other side of the
# pointer, so it never runs off the plot.
_FLIP_X = 0.62
_FLIP_Y = 0.9


def _font(px: int = PLOT_PX) -> QFont:
    font = QFont(FONT_FAMILY)
    font.setPixelSize(px)
    return font


def _set_label(axis: pg.AxisItem, text: str) -> None:
    """Label ``axis`` in the plots' label style."""
    # setLabel takes CSS properties as **kwargs; its stubs name only a few.
    axis.setLabel(text, **_LABEL_STYLE)  # pyright: ignore[reportArgumentType]


def _style_axis(axis: pg.AxisItem, label: str) -> None:
    """Give ``axis`` the plots' look: grey, inward ticks, no SI rescaling."""
    axis.setPen(pg.mkPen(PALETTE.axis))
    axis.setTextPen(pg.mkPen(PALETTE.muted))
    axis.setTickFont(_font())
    axis.setStyle(tickLength=-4, tickTextOffset=3)
    # The label states its unit; a silent "(×1e-3)" would contradict it.
    axis.enableAutoSIPrefix(False)
    _set_label(axis, label)


class _SmoothImage(pg.ImageItem):
    """An image drawn with bilinear smoothing between its samples."""

    def paint(self, painter: QPainter, *args: object) -> None:
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        super().paint(painter, *args)


class Plot(pg.PlotWidget):
    """A plot box with a crosshair and a hover readout.

    Parameters
    ----------
    xlabel, ylabel : str
        The axis labels, with their units.
    """

    def __init__(self, *, xlabel: str = "", ylabel: str = "") -> None:
        super().__init__(background="w")
        self.setMinimumSize(140, 100)
        self.reader: Callable[[float, float], str] | None = None
        self._home: tuple[tuple[float, float], tuple[float, float]] | None = None
        # A PlotWidget always owns its PlotItem; the stubs allow None.
        self.item = cast(pg.PlotItem, self.getPlotItem())
        self.item.hideButtons()
        self.item.setMenuEnabled(False)
        self.item.layout.setContentsMargins(0, 8, 8, 0)  # pyright: ignore[reportAttributeAccessIssue]
        self.box = self.item.getViewBox()
        _style_axis(self.item.getAxis("bottom"), xlabel)
        _style_axis(self.item.getAxis("left"), ylabel)
        self.item.getAxis("left").setWidth(54)
        self.item.getAxis("bottom").setHeight(38)
        # A closed box: top and right axes drawn, without numbers.
        for side in ("top", "right"):
            self.item.showAxis(side)
            axis = self.item.getAxis(side)
            axis.setPen(pg.mkPen(PALETTE.axis))
            axis.setStyle(showValues=False, tickLength=0)
        pen = pg.mkPen(_CROSSHAIR)
        self.cross_x = pg.InfiniteLine(angle=90, pen=pen)
        self.cross_y = pg.InfiniteLine(angle=0, pen=pen)
        self.note = pg.TextItem("", color="#fff", fill=_NOTE_FILL, anchor=(0, 1))
        self.note.setFont(_font(PLOT_PX + 1))
        for extra in (self.cross_x, self.cross_y, self.note):
            self.addItem(extra, ignoreBounds=True)  # pyright: ignore[reportCallIssue]
            extra.setZValue(50)
            extra.hide()
        self.scene().sigMouseMoved.connect(self.hover)  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]

    # -- labels and view ----------------------------------------------------
    def set_labels(self, xlabel: str | None = None, ylabel: str | None = None) -> None:
        """Change the axis labels (``None`` keeps a label)."""
        if xlabel is not None:
            _set_label(self.item.getAxis("bottom"), xlabel)
        if ylabel is not None:
            _set_label(self.item.getAxis("left"), ylabel)

    def set_view(
        self, xrange: tuple[float, float], yrange: tuple[float, float]
    ) -> None:
        """Show ``xrange`` × ``yrange``, and return there on a double-click."""
        self._home = (xrange, yrange)
        self.box.setRange(xRange=xrange, yRange=yrange, padding=0)

    def reset_view(self) -> None:
        """Return to the view last set with :meth:`set_view` (else fit the data)."""
        if self._home is None:
            self.box.autoRange(padding=0.02)
        else:
            self.box.setRange(xRange=self._home[0], yRange=self._home[1], padding=0)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802 — Qt override
        self.reset_view()
        event.accept()

    # -- hover --------------------------------------------------------------
    def hover(self, position: QPointF) -> None:
        """Show the crosshair and the readout at scene point ``position``."""
        inside = self.box.sceneBoundingRect().contains(position)
        if not inside or self.reader is None:
            self.clear_hover()
            return
        point = self.box.mapSceneToView(position)
        text = self.reader(point.x(), point.y())
        if not text:
            self.clear_hover()
            return
        self.cross_x.setPos(point.x())
        self.cross_y.setPos(point.y())
        self.note.setText(f" {text} ")
        (x0, x1), (y0, y1) = self.box.viewRange()
        right = (point.x() - x0) / ((x1 - x0) or 1.0) > _FLIP_X
        top = (point.y() - y0) / ((y1 - y0) or 1.0) > _FLIP_Y
        self.note.setAnchor((1.0 if right else 0.0, 0.0 if top else 1.0))
        self.note.setPos(point.x(), point.y())
        for extra in (self.cross_x, self.cross_y, self.note):
            extra.show()

    def clear_hover(self) -> None:
        """Hide the crosshair and the readout."""
        for extra in (self.cross_x, self.cross_y, self.note):
            extra.hide()

    def leaveEvent(self, event: object) -> None:  # noqa: N802 — Qt override  # pyright: ignore[reportIncompatibleMethodOverride]
        self.clear_hover()
        super().leaveEvent(event)  # pyright: ignore[reportArgumentType]


def signed_colormap(name: str, low: float, high: float) -> pg.ColorMap:
    """Return colour map ``name`` for levels ``(low, high)``, white at zero.

    A measured trace can dip below zero after background subtraction. One map
    shows both signs: the negative part fades from red to white and the positive
    part runs white → ``name`` (whose own lowest colour is replaced by white,
    so a zero background reads as paper). ``low >= 0`` gives the plain
    white-based map.

    Parameters
    ----------
    name : str
        A pyqtgraph/matplotlib colour map name, e.g. ``"viridis"``.
    low, high : float
        The levels the map will span.

    Returns
    -------
    pyqtgraph.ColorMap
        Stops at the positions of ``low``, ``0`` and ``high``.

    Raises
    ------
    ValueError
        If ``high <= 0``, ``low >= high`` or ``name`` is not a colour map.
    """
    if not high > 0 or not low < high:
        raise ValueError(f"need low < high and high > 0, got ({low}, {high})")
    try:
        base = pg.colormap.get(name)
    except FileNotFoundError:  # pyqtgraph looks the name up as a file
        base = None
    if base is None:
        raise ValueError(f"unknown colour map {name!r}")
    stops = np.linspace(0.0, 1.0, 9)
    colours = np.array(base.map(stops, mode=pg.ColorMap.FLOAT), dtype=float)
    colours[0] = (1.0, 1.0, 1.0, 1.0)
    if low >= 0:
        # pg.ColorMap reads numeric colours on a 0–255 scale.
        return pg.ColorMap(stops, 255 * colours)
    zero = -low / (high - low)
    positions = np.concatenate([[0.0], zero + (1.0 - zero) * stops])
    return pg.ColorMap(positions, 255 * np.vstack([NEGATIVE_RGBA, colours]))


def is_uniform(axis: np.ndarray) -> bool:
    """Whether ``axis`` increases in equal steps (to 0.1 %).

    Examples
    --------
    >>> is_uniform(np.array([0.0, 1.0, 2.0]))
    True
    >>> is_uniform(np.array([0.0, 1.0, 3.0]))
    False
    """
    steps = np.diff(np.asarray(axis, float))
    if steps.size == 0 or not np.all(steps > 0):
        return False
    return bool(np.allclose(steps, steps[0], rtol=1e-3, atol=0))


class MapPlot(Plot):
    """A false-colour image with a colour bar.

    Parameters
    ----------
    xlabel, ylabel : str
        The axis labels.
    colours : str
        The colour map, e.g. ``"viridis"`` or ``"bwr"``.
    unit : str
        The colour bar's label.
    signed : bool
        Draw values below zero in red fading to white (:func:`signed_colormap`).
    """

    def __init__(
        self,
        *,
        xlabel: str = "",
        ylabel: str = "",
        colours: str = "viridis",
        unit: str = "",
        signed: bool = False,
    ) -> None:
        super().__init__(xlabel=xlabel, ylabel=ylabel)
        self.colours = colours
        self.signed = signed
        self.data: np.ndarray | None = None
        self.x_axis = np.zeros(0)
        self.y_axis = np.zeros(0)
        self.image = _SmoothImage(axisOrder="row-major")
        self.addItem(self.image)
        self.bar = pg.ColorBarItem(
            colorMap=pg.colormap.get(colours),
            interactive=False,
            width=10,
            colorMapMenu=False,
        )
        self.bar.setImageItem(self.image, insert_in=self.item)
        _style_axis(self.bar.axis, unit)
        self.message = pg.TextItem("", color=PALETTE.muted, anchor=(0.5, 0.5))
        self.addItem(self.message, ignoreBounds=True)  # pyright: ignore[reportCallIssue]

    def show_map(
        self,
        data: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        levels: tuple[float, float],
    ) -> None:
        """Show ``data`` (rows along ``y``, columns along ``x``) spanning ``levels``.

        Parameters
        ----------
        data : numpy.ndarray, shape (len(y), len(x))
            The image; NaN draws as blank.
        x, y : numpy.ndarray
            The sample positions, increasing and evenly spaced (the data
            functions resample a non-uniform axis before it reaches the screen).
        levels : tuple of float
            The colour bar's span.

        Raises
        ------
        ValueError
            If the shapes disagree or an axis is not evenly spaced.
        """
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        data = np.asarray(data, float)
        if data.shape != (y.size, x.size):
            raise ValueError(f"data {data.shape} does not match y×x {(y.size, x.size)}")
        if not (is_uniform(x) and is_uniform(y)):
            raise ValueError("a map's axes must be increasing and evenly spaced")
        self.data, self.x_axis, self.y_axis = data, x, y
        self.message.hide()
        self.image.setImage(data, autoLevels=False)
        dx, dy = x[1] - x[0], y[1] - y[0]
        # Each sample is a pixel centred on its position.
        self.image.setRect(
            QRectF(x[0] - dx / 2, y[0] - dy / 2, dx * x.size, dy * y.size)
        )
        self.set_levels(levels)
        self.set_view((x[0] - dx / 2, x[-1] + dx / 2), (y[0] - dy / 2, y[-1] + dy / 2))

    def update_map(self, data: np.ndarray) -> None:
        """Replace the image's values in place, on the same axes and levels."""
        data = np.asarray(data, float)
        if self.data is None or data.shape != self.data.shape:
            raise ValueError("update_map needs an image of the shape already shown")
        self.data = data
        self.image.setImage(data, autoLevels=False)

    def set_levels(self, levels: tuple[float, float]) -> None:
        """Change the colour bar's span without touching the data."""
        low, high = float(levels[0]), float(levels[1])
        if self.signed and high > 0 and low < high:
            self.bar.setColorMap(signed_colormap(self.colours, low, high))
        self.bar.setLevels(values=(low, high))

    def value_at(self, x: float, y: float) -> float | None:
        """The sample nearest ``(x, y)``, or ``None`` outside the image."""
        if self.data is None:
            return None
        xs, ys = self.x_axis, self.y_axis
        col = int(round((x - xs[0]) / (xs[1] - xs[0])))
        row = int(round((y - ys[0]) / (ys[1] - ys[0])))
        if not (0 <= col < xs.size and 0 <= row < ys.size):
            return None
        return float(self.data[row, col])

    def say(self, text: str) -> None:
        """Show ``text`` in place of the image."""
        self.image.clear()
        self.data = None
        self.message.setText(text)
        (x0, x1), (y0, y1) = self.box.viewRange()
        self.message.setPos((x0 + x1) / 2, (y0 + y1) / 2)
        self.message.show()


@dataclass(frozen=True)
class Curve:
    """One line of a :class:`LinePlot`.

    Attributes
    ----------
    x, y : numpy.ndarray
        The points.
    name : str
        The legend entry ("" for none).
    colour : str
        Its colour (default: the next of :data:`COLOURS`).
    dashed : bool
        Draw dashed.
    right : bool
        Plot against the right-hand axis (e.g. a phase beside an intensity).
    """

    x: np.ndarray
    y: np.ndarray
    name: str = ""
    colour: str = ""
    dashed: bool = False
    right: bool = False


class LinePlot(Plot):
    """Curves against a shared x axis, with an optional right-hand y axis.

    Parameters
    ----------
    xlabel, ylabel : str
        The axis labels.
    right_label : str or None
        The right-hand axis's label; ``None`` for no right-hand axis.
    """

    def __init__(
        self, *, xlabel: str = "", ylabel: str = "", right_label: str | None = None
    ) -> None:
        super().__init__(xlabel=xlabel, ylabel=ylabel)
        self.curves: list[Curve] = []
        self.items: list[pg.PlotDataItem] = []
        self.log_y = False
        self.legend: pg.LegendItem | None = None
        self.right_box: pg.ViewBox | None = None
        self.marks: list[pg.InfiniteLine] = []
        if right_label is not None:
            right = pg.ViewBox()
            self.right_box = right
            self.item.scene().addItem(right)  # pyright: ignore[reportOptionalMemberAccess]
            axis = self.item.getAxis("right")
            axis.linkToView(right)
            _style_axis(axis, right_label)
            axis.setStyle(showValues=True)
            right.setXLink(self.box)
            match = partial(self._match_right, right)
            self.box.sigResized.connect(match)
            match()

    def _match_right(self, right: pg.ViewBox, _resized: object = None) -> None:
        """Keep the right-hand view box over the main one as it resizes.

        ``_resized`` is the view box ``sigResized`` passes along; unused.
        """
        right.setGeometry(self.box.sceneBoundingRect())
        right.linkedViewChanged(self.box, right.XAxis)

    def _points(self, curve: Curve, x: np.ndarray, y: np.ndarray):
        """The points of ``curve`` to draw: on a log axis, only ``y > 0``."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        if self.log_y and not curve.right:
            keep = y > 0
            return x[keep], y[keep]
        return x, y

    def set_curves(self, curves: Sequence[Curve], *, log_y: bool = False) -> None:
        """Replace the curves; a legend appears when more than one is named.

        On a log axis values ``<= 0`` are dropped rather than drawn as -inf.
        """
        for item, curve in zip(self.items, self.curves, strict=True):
            box = self.right_box if curve.right else None
            (box or self.item).removeItem(item)
        if self.legend is not None:
            self.legend.clear()
        self.items = []
        self.curves = list(curves)
        self.log_y = log_y
        self.item.setLogMode(False, log_y)
        named = [c for c in curves if c.name]
        if len(named) > 1 and self.legend is None:
            self.legend = self.item.addLegend(offset=(-8, 8), labelTextSize="8pt")
        if self.legend is not None:
            self.legend.setVisible(len(named) > 1)
        auto = 0
        for curve in self.curves:
            colour = curve.colour or COLOURS[auto % len(COLOURS)]
            auto += not curve.colour
            style = Qt.PenStyle.DashLine if curve.dashed else Qt.PenStyle.SolidLine
            pen = pg.mkPen(colour, width=PEN_WIDTH, style=style)
            x, y = self._points(curve, curve.x, curve.y)
            item = pg.PlotDataItem(x, y, pen=pen)
            if curve.right and self.right_box is not None:
                self.right_box.addItem(item)
            else:
                self.item.addItem(item)
            if self.legend is not None and curve.name:
                self.legend.addItem(item, curve.name)
            self.items.append(item)

    def update_curve(self, index: int, x: np.ndarray, y: np.ndarray) -> None:
        """Move curve ``index``'s points in place (cheap: no legend or layout)."""
        curve = replace(self.curves[index], x=np.asarray(x), y=np.asarray(y))
        self.curves[index] = curve
        self.items[index].setData(*self._points(curve, curve.x, curve.y))

    def set_right_range(self, low: float, high: float) -> None:
        """Fix the right-hand axis's range."""
        if self.right_box is not None:
            self.right_box.setYRange(low, high, padding=0)

    def set_marks(self, positions: Sequence[float], colour: str = "") -> None:
        """Draw dashed vertical lines at ``positions`` (replacing the last set)."""
        for mark in self.marks:
            self.item.removeItem(mark)
        pen = pg.mkPen(colour or PALETTE.accent, width=1.2, style=Qt.PenStyle.DashLine)
        self.marks = []
        for position in positions:
            line = pg.InfiniteLine(pos=position, angle=90, pen=pen)
            self.addItem(line, ignoreBounds=True)  # pyright: ignore[reportCallIssue]
            self.marks.append(line)

    def fit(self) -> None:
        """Fit the view to the left-axis curves, and make that the home view."""
        drawn = [
            self._points(c, c.x, c.y)
            for c in self.curves
            if not c.right and np.size(c.x)
        ]
        if not drawn:
            return
        x = np.concatenate([d[0] for d in drawn])
        y = np.concatenate([d[1] for d in drawn])
        y = y[np.isfinite(y)]
        if self.log_y:
            y = np.log10(y)
        yr = (float(np.min(y)), float(np.max(y))) if y.size else (0.0, 1.0)
        pad = 0.05 * ((yr[1] - yr[0]) or 1.0)
        self.set_view((float(np.min(x)), float(np.max(x))), (yr[0] - pad, yr[1] + pad))


def map_reader(
    plot: MapPlot, xname: str, yname: str, value: str = "{:.3g}"
) -> Callable[[float, float], str]:
    """Return a reader that says the position and the map's value there."""

    def read(x: float, y: float) -> str:
        found = plot.value_at(x, y)
        if found is None:
            return ""
        return f"{xname} {x:.4g}   {yname} {y:.4g}   {value.format(found)}"

    return read


def curve_reader(
    plot: LinePlot, xname: str, value: str = "{:.3g}"
) -> Callable[[float, float], str]:
    """Return a reader that says x and each left-axis curve's value there."""

    def read(x: float, _y: float) -> str:
        parts = [f"{xname} {x:.4g}"]
        for curve in plot.curves:
            cx = np.asarray(curve.x, float)
            if curve.right or cx.size == 0 or not (cx[0] <= x <= cx[-1]):
                continue
            found = float(np.interp(x, cx, np.asarray(curve.y, float)))
            name = f"{curve.name} " if curve.name else ""
            parts.append(f"{name}{value.format(found)}")
        return "   ".join(parts) if len(parts) > 1 else ""

    return read
