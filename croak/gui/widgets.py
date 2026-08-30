"""Reusable composite widgets: a range control and a single-value control.

Both pair a horizontal slider with spin box(es). Programmatic updates block the
child widgets' signals so seeding values never feeds back into the model; the
slider updates continuously while dragging, and the spin boxes commit on
Enter / focus-out.
"""

from __future__ import annotations

import math

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QValidator
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from superqt import QRangeSlider

__all__ = ["RangeControl", "ValueControl", "SciSpinBox"]


class SciSpinBox(QDoubleSpinBox):
    """A spin box that shows scientific notation with decade-aware stepping.

    Values display as e.g. ``5.0e-03``; stepping moves within the current
    decade (…→4e-3→5e-3→…→9e-3→1e-2…) and rolls over at the decade edges, so the
    user gets fine control (5e-3, 7e-4, …) instead of only whole powers of ten.
    """

    def __init__(
        self,
        lo: float,
        hi: float,
        value: float,
        on_change=None,
        *,
        sigfigs: int = 1,
    ):
        super().__init__()
        self._sig = int(sigfigs)
        self.setDecimals(15)  # internal storage only; display via textFromValue
        self.setRange(lo, hi)
        self.setValue(value)
        if on_change:
            self.valueChanged.connect(on_change)

    def textFromValue(self, v: float) -> str:
        if v <= 0:
            return "0"
        return f"{v:.{self._sig}e}"

    def valueFromText(self, text: str) -> float:
        try:
            return float(text)
        except ValueError:
            return self.value()

    def validate(self, text: str, pos: int):
        if text in ("", "-", "+"):
            return QValidator.State.Intermediate, text, pos
        try:
            float(text)
        except ValueError:
            # allow partial scientific entry like "5e", "5e-"
            if text.rstrip("eE+-").replace(".", "", 1).lstrip("+-").isdigit():
                return QValidator.State.Intermediate, text, pos
            return QValidator.State.Invalid, text, pos
        return QValidator.State.Acceptable, text, pos

    def stepBy(self, steps: int) -> None:
        v = self.value()
        if v <= 0:
            self.setValue(self.minimum() if steps > 0 and self.minimum() > 0 else 0.0)
            return
        decade = 10.0 ** math.floor(math.log10(v))
        mant = round(v / decade) + steps
        while mant > 9:
            mant -= 9
            decade *= 10.0
        while mant < 1:
            mant += 9
            decade /= 10.0
        self.setValue(max(self.minimum(), min(self.maximum(), mant * decade)))


def _block(*widgets):
    for w in widgets:
        w.blockSignals(True)


def _unblock(*widgets):
    for w in widgets:
        w.blockSignals(False)


class RangeControl(QWidget):
    """A min/max range control: two spin boxes above a double-handled slider.

    ``changed(lo, hi)`` fires when the user drags the slider or commits a spin
    box. Programmatic :meth:`set_bounds`/:meth:`set_values` are signal-safe.
    """

    changed = pyqtSignal(float, float)

    def __init__(
        self,
        lo: float,
        hi: float,
        low: float,
        high: float,
        *,
        decimals: int = 0,
        step: float = 1.0,
        slider_scale: float = 1.0,
    ):
        super().__init__()
        self._scale = slider_scale
        # Fill the available column width (so the slider is wide and easy to nudge)
        # rather than collapsing to the spins' preferred width.
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.lo_spin = QDoubleSpinBox()
        self.hi_spin = QDoubleSpinBox()
        for sp in (self.lo_spin, self.hi_spin):
            sp.setDecimals(decimals)
            sp.setSingleStep(step)
            sp.setKeyboardTracking(False)
        self.slider = QRangeSlider(Qt.Orientation.Horizontal)
        self.slider.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

        self.set_bounds(lo, hi)
        self.set_values(low, high)

        # user edits only: spin commits, slider drags continuously
        self.lo_spin.editingFinished.connect(self._spins_changed)
        self.hi_spin.editingFinished.connect(self._spins_changed)
        self.lo_spin.valueChanged.connect(self._spins_changed)
        self.hi_spin.valueChanged.connect(self._spins_changed)
        self.slider.valueChanged.connect(self._slider_changed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.lo_spin)
        rl.addWidget(self.hi_spin)
        layout.addWidget(row)
        layout.addWidget(self.slider)

    def set_bounds(self, lo: float, hi: float) -> None:
        _block(self.lo_spin, self.hi_spin, self.slider)
        self.lo_spin.setRange(lo, hi)
        self.hi_spin.setRange(lo, hi)
        self.slider.setRange(int(round(lo * self._scale)), int(round(hi * self._scale)))
        _unblock(self.lo_spin, self.hi_spin, self.slider)

    def set_values(self, low: float, high: float) -> None:
        _block(self.lo_spin, self.hi_spin, self.slider)
        self.lo_spin.setValue(low)
        self.hi_spin.setValue(high)
        self.slider.setValue(
            (int(round(low * self._scale)), int(round(high * self._scale)))
        )
        _unblock(self.lo_spin, self.hi_spin, self.slider)

    def values(self) -> tuple[float, float]:
        return self.lo_spin.value(), self.hi_spin.value()

    def _spins_changed(self, *_):
        lo, hi = self.lo_spin.value(), self.hi_spin.value()
        if lo > hi:
            return
        _block(self.slider)
        self.slider.setValue(
            (int(round(lo * self._scale)), int(round(hi * self._scale)))
        )
        _unblock(self.slider)
        self.changed.emit(lo, hi)

    def _slider_changed(self, value):
        lo, hi = value[0] / self._scale, value[1] / self._scale
        _block(self.lo_spin, self.hi_spin)
        self.lo_spin.setValue(lo)
        self.hi_spin.setValue(hi)
        _unblock(self.lo_spin, self.hi_spin)
        self.changed.emit(lo, hi)


class ValueControl(QWidget):
    """A single-value control: a spin box beside a horizontal slider.

    ``changed(value)`` fires on slider drag or spin commit.
    """

    changed = pyqtSignal(float)

    def __init__(
        self,
        lo: float,
        hi: float,
        value: float,
        *,
        decimals: int = 0,
        step: float = 1.0,
        slider_scale: float = 1.0,
    ):
        super().__init__()
        self._scale = slider_scale
        # Fill the available column width (see RangeControl) so the slider is wide
        # and easy to nudge, rather than collapsing to the spin's preferred width.
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(step)
        self.spin.setKeyboardTracking(False)
        self.slider = QSlider(Qt.Orientation.Horizontal)

        self.set_bounds(lo, hi)
        self.set_value(value)

        self.spin.editingFinished.connect(self._spin_changed)
        self.spin.valueChanged.connect(self._spin_changed)
        self.slider.valueChanged.connect(self._slider_changed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.spin)
        layout.addWidget(self.slider, stretch=1)

    def set_bounds(self, lo: float, hi: float) -> None:
        _block(self.spin, self.slider)
        self.spin.setRange(lo, hi)
        self.slider.setRange(int(round(lo * self._scale)), int(round(hi * self._scale)))
        _unblock(self.spin, self.slider)

    def set_value(self, value: float) -> None:
        _block(self.spin, self.slider)
        self.spin.setValue(value)
        self.slider.setValue(int(round(value * self._scale)))
        _unblock(self.spin, self.slider)

    def value(self) -> float:
        return self.spin.value()

    def bounds(self) -> tuple[float, float]:
        return self.spin.minimum(), self.spin.maximum()

    def _spin_changed(self, *_):
        v = self.spin.value()
        _block(self.slider)
        self.slider.setValue(int(round(v * self._scale)))
        _unblock(self.slider)
        self.changed.emit(v)

    def _slider_changed(self, value):
        v = value / self._scale
        _block(self.spin)
        self.spin.setValue(v)
        _unblock(self.spin)
        self.changed.emit(v)
