"""Sheet building blocks: sections, help, find, reflow and status widgets."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QCheckBox, QLabel, QLineEdit, QSpinBox, QWidget

from croak.gui import settings
from croak.gui.help import apply_tooltips
from croak.gui.sheet import (
    NARROW_WIDTH,
    AdviceBox,
    DerivedGrid,
    Form,
    ProgressStrip,
    ReadyLine,
    Section,
    Segments,
    Sheet,
    row_text,
)

HELP = {
    ("Grid", "Points"): "How many samples.",
    ("Grid", "Resample t"): "Resample in time.",
    ("Filtering", "Threshold"): "Clip below this.",
}


def _form(qtbot, **kwargs) -> Form:
    form = Form(**kwargs)
    qtbot.addWidget(form)
    return form


def _grid(form: Form):
    grid = form.add_section("Grid")
    points = QSpinBox()
    resample = QCheckBox("Resample t")
    grid.form.addRow("Points", points)
    grid.form.addRow(resample)
    return grid, points, resample


def test_sheet_section_rows_take_help_by_section_and_label(qtbot):
    form = _form(qtbot)
    _, points, resample = _grid(form)
    loose = QLineEdit()
    form.add_section("Other").form.addRow("Undocumented", loose)
    matched = apply_tooltips(form, HELP)
    assert matched == {("Grid", "Points"), ("Grid", "Resample t")}
    assert points.toolTip() == "How many samples."
    assert resample.toolTip() == "Resample in time."
    assert loose.toolTip() == ""


def test_sheet_section_title_reads_an_escaped_ampersand(qtbot):
    section = Section("Units && interaction")
    qtbot.addWidget(section)
    assert section.section_title == "Units & interaction"
    assert section.header.text() == "Units && interaction"


def test_sheet_row_text_reads_a_label_or_a_check_box(qtbot):
    form = _form(qtbot)
    grid, _, _ = _grid(form)
    grid.form.addRow(QWidget())  # neither: no text
    assert [row_text(grid.form, r) for r in range(3)] == ["Points", "Resample t", ""]


def test_sheet_focused_control_shows_its_help_line(qtbot):
    form = _form(qtbot)
    form.show()
    qtbot.waitExposed(form)
    _, points, _ = _grid(form)
    apply_tooltips(form, HELP)
    points.setFocus()
    qtbot.waitUntil(lambda: form.help.text() == "How many samples.")
    plain = QLineEdit()  # no tooltip anywhere up to the form: help unchanged
    form.add_widget(plain)
    plain.setFocus()
    qtbot.waitUntil(plain.hasFocus)
    assert form.help.text() == "How many samples."


def test_sheet_help_ignores_controls_of_another_form(qtbot):
    form, other = _form(qtbot), _form(qtbot)
    elsewhere = QSpinBox()
    elsewhere.setToolTip("Not mine.")
    other.add_widget(elsewhere)
    form._focus_moved(None, elsewhere)
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
    assert plain.folded
    assert form.sections == [section, plain]


def test_sheet_find_shows_only_matching_rows(qtbot):
    form = _form(qtbot, find=True)
    form.show()
    grid, points, resample = _grid(form)
    filtering = form.add_section("Filtering", folded=True)
    threshold = QSpinBox()
    filtering.form.addRow("Threshold", threshold)
    apply_tooltips(form, HELP)
    assert form.find is not None
    form.find.setText("clip")  # matches the threshold's help text
    assert threshold.isVisible() and filtering.body.isVisible()
    assert not grid.isVisible()
    form.find.setText("samples")
    assert points.isVisible() and not resample.isVisible()
    form.find.setText("grid")  # the section's own name shows all of it
    assert points.isVisible() and resample.isVisible()
    form.find.setText("")
    assert grid.isVisible() and resample.isVisible()
    assert filtering.folded and not filtering.body.isVisible()


def test_sheet_form_without_find_has_no_box(qtbot):
    form = Form()
    qtbot.addWidget(form)
    assert form.find is None
    assert form.sections == []


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
    box.clear()
    assert box.isHidden() and box.title.text() == ""


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


def test_sheet_combo_boxes_shrink_to_the_column(qtbot):
    from PySide6.QtWidgets import QComboBox

    from croak.gui.sheet import FORM_WIDTH

    form, plots = _form(qtbot), QWidget()
    long = QComboBox()
    long.addItems(["Automatic (centroid phase) from a very long description"])
    form.add_section("Mode").form.addRow("Initial guess", long)
    sheet = Sheet(form, plots)
    qtbot.addWidget(sheet)
    sheet.resize(NARROW_WIDTH + 400, 700)
    sheet.show()
    qtbot.waitExposed(sheet)
    assert long.sizeAdjustPolicy() == (
        QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    )
    assert form.width() == FORM_WIDTH  # the long item does not widen the column
    sheet.resize(NARROW_WIDTH - 200, 700)  # stacked: the form spans the sheet
    qtbot.waitUntil(lambda: form.width() > FORM_WIDTH)
