"""Sheet building blocks: sections, rows, help, find, reflow and status widgets."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QCheckBox, QLabel, QLineEdit, QSpinBox, QWidget

from croak.gui import settings
from croak.gui.sheet import (
    LABEL_WIDTH,
    NARROW_WIDTH,
    AdviceBox,
    DerivedGrid,
    Form,
    ProgressStrip,
    ReadyLine,
    Row,
    Segments,
    Sheet,
)

HELP = [
    ("Grid", [("Points", "How many samples."), ("Resample t", "Resample in time.")]),
    ("Filtering", [("Threshold", "Clip below this.")]),
    ("Unused", [("Nothing", "No control has this label.")]),
]


def _form(qtbot, **kwargs) -> Form:
    form = Form(HELP, **kwargs)
    qtbot.addWidget(form)
    return form


def test_sheet_rows_take_their_help_from_section_and_label(qtbot):
    form = _form(qtbot)
    grid = form.add_section("Grid")
    points = grid.add_row("Points", QSpinBox())
    resample = grid.add_row("ignored", QCheckBox("Resample t"))
    loose = grid.add_row("Undocumented", QLineEdit())
    assert points.editor.toolTip() == "How many samples."
    assert points.label is not None and points.label.width() == LABEL_WIDTH
    # A check box is its own label and is keyed by its text.
    assert resample.label is None
    assert resample.text == "Resample t"
    assert resample.editor.toolTip() == "Resample in time."
    assert loose.editor.toolTip() == ""
    assert form.matched == {("Grid", "Points"), ("Grid", "Resample t")}


def test_sheet_check_box_with_escaped_ampersand_finds_its_help(qtbot):
    form = Form([("S", [("Skip filtering & regrid", "Raw.")])])
    qtbot.addWidget(form)
    row = form.add_section("S").add_row("", QCheckBox("Skip filtering && regrid"))
    assert row.help_text == "Raw."


def test_sheet_row_label_greys_with_its_row(qtbot):
    row = Row("Points", QSpinBox())
    qtbot.addWidget(row)
    row.setEnabled(False)
    assert row.label is not None and not row.label.isEnabled()
    row.setEnabled(True)
    assert row.label.isEnabled()
    box = Row("", QCheckBox("Flag"))
    qtbot.addWidget(box)
    box.setEnabled(False)  # a check box row has no separate label to grey


def test_sheet_focused_row_shows_its_help_line(qtbot):
    form = _form(qtbot)
    form.show()
    qtbot.waitExposed(form)
    spin = QSpinBox()
    form.add_section("Grid").add_row("Points", spin)
    spin.setFocus()
    qtbot.waitUntil(lambda: form.help.text() == "How many samples.")
    other = QLineEdit()
    form.add_widget(other)  # outside any row: the help line keeps its text
    other.setFocus()
    qtbot.waitUntil(other.hasFocus)
    assert form.help.text() == "How many samples."


def test_sheet_help_ignores_rows_of_another_form(qtbot):
    form, other = _form(qtbot), _form(qtbot)
    spin = QSpinBox()
    other.add_section("Grid").add_row("Points", spin)
    form._focus_moved(None, spin)
    assert form.help.text() == ""
    form._focus_moved(None, None)


def test_sheet_sections_fold_and_remember_it(qtbot):
    form = _form(qtbot)
    section = form.add_section("Grid", folded=True, settings_key="test/grid_open")
    assert section.folded and section.body.isHidden()
    section.set_folded(False)
    assert not section.folded
    assert settings.value("test/grid_open", False) is True
    second = _form(qtbot)  # held, so Python does not delete the widget
    again = second.add_section("Grid", folded=True, settings_key="test/grid_open")
    assert not again.folded  # the remembered state wins over the default
    plain = form.add_section("Filtering")
    plain.set_folded(True)  # no key: nothing remembered
    plain.add_widget(QLabel("extra"))
    assert plain.folded


def test_sheet_find_shows_only_matching_rows(qtbot):
    form = _form(qtbot, find=True)
    form.show()
    grid = form.add_section("Grid")
    points = grid.add_row("Points", QSpinBox())
    resample = grid.add_row("", QCheckBox("Resample t"))
    filtering = form.add_section("Filtering", folded=True)
    threshold = filtering.add_row("Threshold", QSpinBox())
    assert form.find is not None
    form.find.setText("clip")  # matches the threshold's help text
    assert threshold.isVisible() and filtering.body.isVisible()
    assert not grid.isVisible()
    form.find.setText("samples")
    assert points.isVisible() and not resample.isVisible()
    form.find.setText("")
    assert grid.isVisible() and resample.isVisible()
    assert filtering.folded and not filtering.body.isVisible()


def test_sheet_form_without_find_has_no_box(qtbot):
    form = Form()
    qtbot.addWidget(form)
    assert form.find is None
    assert form.help_map == {}


def test_sheet_plots_move_below_the_form_on_a_narrow_window(qtbot):
    form, plots = _form(qtbot), QWidget()
    sheet = Sheet(form, plots)
    qtbot.addWidget(sheet)
    sheet.resize(NARROW_WIDTH + 200, 700)
    sheet.show()
    qtbot.waitExposed(sheet)
    assert not sheet.narrow
    sheet.resize(NARROW_WIDTH - 200, 700)
    qtbot.waitUntil(lambda: plots.geometry().top() >= form.geometry().bottom())
    assert sheet.narrow
    sheet.resize(NARROW_WIDTH - 100, 700)  # still narrow: no change
    qtbot.wait(10)
    assert sheet.narrow


@pytest.mark.parametrize(
    ("state", "warn", "error"),
    [("ok", False, False), ("warn", True, False), ("error", False, True)],
)
def test_sheet_ready_line_colours_its_state(qtbot, state, warn, error):
    line = ReadyLine()
    qtbot.addWidget(line)
    line.say("Ready", state)
    assert line.text() == "Ready" and line.state == state
    assert bool(line.property("warn")) is warn
    assert bool(line.property("error")) is error


def test_sheet_ready_line_rejects_an_unknown_state(qtbot):
    line = ReadyLine()
    qtbot.addWidget(line)
    with pytest.raises(ValueError, match="ok, warn or error"):
        line.say("?", "maybe")


def test_sheet_advice_box_appears_when_it_has_something_to_say(qtbot):
    box = AdviceBox()
    qtbot.addWidget(box)
    assert box.isHidden()
    box.say("Retrieval failed", "Try a smaller grid.")
    assert not box.isHidden()
    assert box.title.text() == "Retrieval failed"


def test_sheet_derived_numbers_show_values_and_states(qtbot):
    grid = DerivedGrid(columns=2)
    qtbot.addWidget(grid)
    grid.add("FWHM")
    grid.add("Edge energy", tooltip="Energy at the grid's edge.")
    grid.add("Grid")
    assert grid.grid.itemAtPosition(1, 0) is not None  # wrapped onto row 2
    grid.set("Edge energy", "2 %", "warn")
    assert grid.values["Edge energy"].property("warn") is True
    grid.clear()
    assert grid.values["FWHM"].text() == "–"
    assert grid.values["Edge energy"].property("warn") is False


def test_sheet_segments_choose_one_value(qtbot):
    seg = Segments(
        [("frequency", "Frequency"), ("wavelength", "Wavelength")], "frequency"
    )
    qtbot.addWidget(seg)
    with qtbot.waitSignal(seg.changed) as blocker:
        seg.buttons["wavelength"].click()
    assert blocker.args == ["wavelength"]
    assert seg.buttons["wavelength"].property("current") is True
    assert seg.buttons["frequency"].property("current") is False
    with qtbot.assertNotEmitted(seg.changed):
        seg.buttons["wavelength"].click()  # no change: no signal
        seg.set_value("frequency")  # programmatic: no signal
    with pytest.raises(ValueError, match="not one of"):
        seg.set_value("time")
    with pytest.raises(ValueError, match="not one of"):
        Segments([("a", "A")], "b")


def test_sheet_progress_strip_clamps_its_bar(qtbot):
    strip = ProgressStrip()
    qtbot.addWidget(strip)
    assert strip.isHidden()
    strip.add_end(QLabel("sparkline"))
    strip.show_progress("Retrieving", 1.7, "iteration 10")
    assert strip.bar.value() == 1000 and not strip.isHidden()
    strip.show_progress("Retrieving", -1.0, "")
    assert strip.bar.value() == 0
    assert strip.state.text() == "Retrieving"
