"""SciSpinBox text handling and stepping; RangeControl and ValueControl sync."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtGui import QValidator

from croak.gui.widgets import RangeControl, SciSpinBox, ValueControl

State = QValidator.State


def test_sci_spinbox_text_round_trip(qapp):
    box = SciSpinBox(0.0, 1.0, 5e-3)
    assert box.textFromValue(5e-3) == "5.0e-03"
    assert box.textFromValue(0.0) == "0"
    assert box.valueFromText("2e-3") == pytest.approx(2e-3)
    # Unparseable text keeps the current value.
    assert box.valueFromText("abc") == pytest.approx(5e-3)


@pytest.mark.parametrize(
    ("text", "state"),
    [
        ("", State.Intermediate),
        ("-", State.Intermediate),
        ("5e", State.Intermediate),
        ("5e-", State.Intermediate),
        ("1.5e-", State.Intermediate),
        ("x1", State.Invalid),
        ("3e-4", State.Acceptable),
    ],
)
def test_sci_spinbox_validates_partial_scientific_entry(qapp, text, state):
    box = SciSpinBox(0.0, 1.0, 5e-3)
    assert box.validate(text, 0)[0] == state


@pytest.mark.parametrize("start", [0.0, -0.5])
def test_sci_spinbox_steps_from_non_positive_values_to_zero(qapp, start):
    # No decade exists at or below zero: any step lands on zero.
    box = SciSpinBox(-1.0, 1.0, start)
    box.stepBy(1)
    assert box.value() == 0.0


def test_sci_spinbox_steps_within_and_across_decades(qapp):
    seen = []
    box = SciSpinBox(1e-6, 1.0, 5e-3, on_change=seen.append)
    box.stepBy(1)
    assert box.value() == pytest.approx(6e-3)
    box.stepBy(4)  # 6e-3 + 4 -> 10e-3 rolls over to 1e-2
    assert box.value() == pytest.approx(1e-2)
    box.stepBy(-1)  # 1e-2 - 1 -> 0e-2 rolls down to 9e-3
    assert box.value() == pytest.approx(9e-3)
    box.setValue(0.9)
    box.stepBy(5)  # past the maximum: clamped
    assert box.value() == pytest.approx(1.0)
    assert seen[-1] == pytest.approx(1.0)


def test_range_control_slider_drives_spins_and_signal(qapp):
    rc = RangeControl(0.0, 100.0, 10.0, 90.0, slider_scale=10.0)
    seen = []
    rc.changed.connect(lambda lo, hi: seen.append((lo, hi)))
    rc.slider.setValue((200, 800))
    assert rc.values() == pytest.approx((20.0, 80.0))
    assert seen == [pytest.approx((20.0, 80.0))]


def test_range_control_spins_drive_slider_and_ignore_crossed_values(qapp):
    rc = RangeControl(0.0, 100.0, 10.0, 90.0)
    seen = []
    rc.changed.connect(lambda lo, hi: seen.append((lo, hi)))
    rc.lo_spin.setValue(30.0)
    assert rc.slider.value() == (30, 90)
    assert seen == [(30.0, 90.0)]
    # lo above hi is a transient state while typing: nothing propagates.
    rc.lo_spin.setValue(95.0)
    assert seen == [(30.0, 90.0)]
    assert rc.slider.value() == (30, 90)


def test_value_control_spin_and_slider_stay_in_sync(qapp):
    vc = ValueControl(0.0, 10.0, 2.0, decimals=1, slider_scale=10.0)
    seen = []
    vc.changed.connect(seen.append)
    vc.spin.setValue(3.5)
    assert vc.slider.value() == 35
    vc.slider.setValue(70)
    assert vc.value() == pytest.approx(7.0)
    assert seen == [pytest.approx(3.5), pytest.approx(7.0)]
    assert vc.bounds() == (0.0, 10.0)
