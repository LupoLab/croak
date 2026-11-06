"""The display settings every plot follows: spectral axis and logarithmic range.

The setting is the viewer's, not the session's: it is remembered between
launches (:mod:`croak.gui.settings`) and holds on every page until it is changed
or reset. :func:`current` reads it, :func:`set_display` changes it and tells
every plot (:func:`hub`'s ``changed``), and :class:`DisplayBar` is the control
at the foot of the window. The conversion itself is Qt-free:
:func:`croak.plotting.display.displayed`.
"""

from __future__ import annotations

from typing import cast

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QHBoxLayout, QSpinBox, QWidget

from ..plotting.display import DB_RANGE_LIMITS, Display
from . import settings
from .sheet import Segments
from .theme import button, label

__all__ = ["DisplayBar", "DisplayHub", "current", "hub", "set_display"]

KEY_AXIS = "display/spectral_axis"
KEY_DB_RANGE = "display/db_range"

AXIS_CHOICES = (
    ("native", "As computed"),
    ("wavelength", "Wavelength"),
    ("frequency", "Frequency"),
)
"""The spectral axis control's choices and captions."""

DB_STEP = 5
"""The logarithmic range control's step, dB."""


class DisplayHub(QObject):
    """Tells every plot that the display settings changed."""

    changed = Signal()


_HUB: list[DisplayHub] = []


def hub() -> DisplayHub:
    """Return the one :class:`DisplayHub` (made on first use)."""
    if not _HUB:
        _HUB.append(DisplayHub())
    return _HUB[0]


def current() -> Display:
    """Read the display settings in force.

    The defaults if none were kept, or if what was kept is no longer valid.
    """
    store = settings.user_settings()
    axis = str(store.value(KEY_AXIS, "native", str))
    # QSettings converts what it kept to the type asked for.
    db_range = cast(float, store.value(KEY_DB_RANGE, Display().db_range, float))
    try:
        return Display(axis, db_range)
    except ValueError:
        return Display()


def set_display(display: Display) -> None:
    """Keep ``display`` and redraw every plot with it."""
    store = settings.user_settings()
    store.setValue(KEY_AXIS, display.axis)
    store.setValue(KEY_DB_RANGE, display.db_range)
    store.sync()
    hub().changed.emit()


class DisplayBar(QWidget):
    """The spectral-axis and log-range controls, with a reset."""

    def __init__(self) -> None:
        super().__init__()
        shown = current()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 8, 0)
        row.setSpacing(8)
        row.addWidget(label("Spectral axis", "help"))
        self.axis = Segments(AXIS_CHOICES, shown.axis)
        self.axis.setToolTip(
            "Draw every spectral plot against wavelength or frequency, with the "
            "density Jacobian; 'As computed' keeps each on its own axis "
            "(wavelength for a loaded trace, frequency for a retrieval)."
        )
        self.axis.changed.connect(lambda _value: self._apply())
        row.addWidget(self.axis)
        row.addWidget(label("Log range", "help"))
        self.db_range = QSpinBox()
        low, high = DB_RANGE_LIMITS
        self.db_range.setRange(int(low), int(high))
        self.db_range.setSingleStep(DB_STEP)
        self.db_range.setSuffix(" dB")
        self.db_range.setValue(int(shown.db_range))
        self.db_range.setToolTip(
            "How far below its peak every logarithmic map reaches. Drag a colour "
            "bar to adjust one map on its own."
        )
        self.db_range.valueChanged.connect(lambda _value: self._apply())
        row.addWidget(self.db_range)
        self.reset = button("Reset", "link")
        self.reset.setToolTip("Back to each plot's own axis and a 30 dB range.")
        self.reset.clicked.connect(self.restore_defaults)
        row.addWidget(self.reset)
        hub().changed.connect(self._follow)

    def _follow(self) -> None:
        """Show the display in force, however it was set."""
        shown = current()
        self.db_range.blockSignals(True)
        self.db_range.setValue(int(shown.db_range))
        self.db_range.blockSignals(False)
        self.axis.set_value(shown.axis)

    def display(self) -> Display:
        """Return the display the controls show."""
        return Display(self.axis.value, float(self.db_range.value()))

    def _apply(self) -> None:
        set_display(self.display())

    def restore_defaults(self) -> None:
        """Return to the default display (the controls follow)."""
        set_display(Display())
