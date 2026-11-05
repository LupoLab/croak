"""Small builders for a stage's form controls.

:func:`group` makes a folding section and the form its rows go in; the others
build a spin box, combo or check box already bound to a callback, and
:func:`collapsible` folds a strip (the retrieval read-out) under the plots.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QSpinBox,
    QWidget,
)
from superqt import QCollapsible

from . import settings
from .sheet import Section

__all__ = [
    "collapsible",
    "group",
    "add_wide_row",
    "spin",
    "dspin",
    "combo",
    "check",
]


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
def group(
    title: str, *, folded: bool = False, settings_key: str = ""
) -> tuple[Section, QFormLayout]:
    """A folding section and the form layout its rows go in.

    The section is added to a stage's ``controls``; its rows are added to the
    form with ``form.addRow(label, widget)``. Help tooltips match rows by
    ``(title, label)`` (:func:`croak.gui.help.apply_tooltips`). ``folded`` and
    ``settings_key`` are as for :class:`~croak.gui.sheet.Section`.
    """
    section = Section(title, folded=folded, settings_key=settings_key)
    return section, section.form


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
