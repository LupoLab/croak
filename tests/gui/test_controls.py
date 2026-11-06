"""The small form-control factories."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from croak.gui.controls import check, combo, dspin, spin


def test_widget_factories_without_a_change_handler(qapp):
    assert spin(0, 10, 3).value() == 3
    assert dspin(0.0, 1.0, 0.25).value() == pytest.approx(0.25)
    box = combo(["a", "b"], value="z")  # unknown value: keeps the first item
    assert box.currentText() == "a"
    assert check("on", True).isChecked()


def test_widget_factories_connect_the_change_handler(qapp):
    seen = []
    s = spin(0, 10, 3, on_change=seen.append)
    s.setValue(4)
    d = dspin(0.0, 1.0, 0.25, on_change=seen.append)
    d.setValue(0.5)
    c = combo(["a", "b"], value="a", on_change=seen.append)
    c.setCurrentText("b")
    k = check("on", False, on_change=seen.append)
    k.setChecked(True)
    assert seen == [4, 0.5, "b", True]


def test_controls_collapsible_draws_its_chevrons_from_svg(qtbot):
    from PySide6.QtWidgets import QLabel

    from croak.gui.controls import collapsible

    strip = collapsible("Read-out", QLabel("x"))
    qtbot.addWidget(strip)
    # Real icons, not superqt's text arrows painted into an empty pixmap.
    for icon in (strip.expandedIcon(), strip.collapsedIcon()):
        assert not icon.pixmap(12, 12).isNull()
