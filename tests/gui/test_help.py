"""Help content: HTML rendering and tooltip matching over form layouts."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from croak.gui.help import HelpDialog, apply_tooltips, build_html, flatten


def test_build_html_escapes_and_skips_an_empty_intro():
    html = build_html("A & B", "", [("Sec", [("x<y", "t")])])
    assert "<h2>A &amp; B</h2>" in html
    assert "<p>" not in html
    assert "x&lt;y" in html


def test_build_html_includes_the_intro():
    assert "<p>Intro</p>" in build_html("T", "Intro", [])


def test_help_dialog_builds(qtbot):
    dialog = HelpDialog("Stage", "Intro", [("S", [("a", "b")])])
    qtbot.addWidget(dialog)
    assert dialog.windowTitle() == "Help — Stage"


def test_apply_tooltips_skips_rows_without_a_usable_field_or_label(qtbot):
    root = QWidget()
    qtbot.addWidget(root)
    outer = QVBoxLayout(root)
    box = QGroupBox("Group && more")
    form = QFormLayout(box)
    labelled = QLineEdit()
    form.addRow("Plain", labelled)
    # A layout (not a widget) in the field column: no field widget, skipped.
    form.addRow("Layout row", QHBoxLayout())
    # A spanning widget with no label and not a checkbox: no label, skipped.
    unlabelled = QLineEdit()
    form.addRow(unlabelled)
    # A non-QLabel label widget: falls back to a checkbox's own text.
    toggle = QCheckBox("Toggle && go")
    form.addRow(QPushButton("not a label"), toggle)
    # A row whose label widget is not a QLabel and whose field is no checkbox.
    other = QLineEdit()
    form.addRow(QPushButton("btn"), other)
    outer.addWidget(box)
    loose = QFormLayout()
    loose_field = QLineEdit()
    loose.addRow("Loose", loose_field)
    outer.addLayout(loose)

    help_map = flatten(
        [
            ("Group & more", [("Plain", "p"), ("Toggle & go", "t"), ("Gone", "g")]),
            ("", [("Loose", "l")]),
        ]
    )
    matched = apply_tooltips(root, help_map)
    assert matched == {
        ("Group & more", "Plain"),
        ("Group & more", "Toggle & go"),
        ("", "Loose"),
    }
    assert labelled.toolTip() == "p"
    assert toggle.toolTip() == "t"
    assert loose_field.toolTip() == "l"
    assert unlabelled.toolTip() == other.toolTip() == ""
