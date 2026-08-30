"""Matplotlib canvas embedded in Qt, plus small widget helpers."""

from __future__ import annotations

from matplotlib.backends.backend_qt import NavigationToolbar2QT
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
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

__all__ = [
    "MplCanvas",
    "with_toolbar",
    "group",
    "add_wide_row",
    "spin",
    "dspin",
    "combo",
    "check",
]


class MplCanvas(FigureCanvasQTAgg):
    """A reusable matplotlib canvas with a constrained-layout figure."""

    def __init__(self, figsize=(5, 4)):
        self.figure = Figure(figsize=figsize, layout="constrained")
        super().__init__(self.figure)

    def show_figure(self, fig: Figure) -> None:
        """Replace the canvas figure's content by copying axes from ``fig``.

        Simplest robust approach: swap the managed figure. We clear and re-draw
        by transferring the new figure's manager — but since axes can't be moved
        between figures, callers instead use :meth:`clear_axes` + plot helpers.
        """
        raise NotImplementedError  # use draw_with(callback) instead

    def draw_with(self, plot_fn) -> None:
        """Clear the figure and let ``plot_fn(figure)`` populate it, then redraw."""
        self.figure.clear()
        plot_fn(self.figure)
        self.draw_idle()

    def single_axes(self):
        """Clear and return a single Axes for simple plots."""
        self.figure.clear()
        return self.figure.add_subplot(111)


def with_toolbar(canvas: MplCanvas) -> QWidget:
    """Wrap ``canvas`` in a panel with a matplotlib pan/zoom/save toolbar on top.

    Returns a container widget to drop into a layout; the caller keeps its own
    reference to ``canvas`` for plotting. The toolbar adds the standard
    Home/Back/Forward, Pan, Zoom-to-rectangle, Subplots, and Save controls, plus a
    live cursor read-out. A full redraw (``single_axes``/``figure.clear``) resets
    the view, so zoom is most useful between control changes; Home restores it.
    """
    panel = QWidget()
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    layout.addWidget(NavigationToolbar2QT(canvas, panel))
    layout.addWidget(canvas)
    return panel


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
