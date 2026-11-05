"""The window's look: palette, style sheet, fonts and styled widgets."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication, QWidget

from croak.gui import theme


def test_theme_stylesheet_is_built_from_the_palette():
    sheet = theme.stylesheet()
    assert theme.PALETTE.accent in sheet
    assert "QPushButton#primary" in sheet
    # Every token reaches the sheet only through the palette: a new palette
    # recolours every rule that used the old value.
    recoloured = theme.stylesheet(theme.Palette(accent="#123456"))
    assert "#123456" in recoloured
    assert theme.PALETTE.accent not in recoloured
    # The combo chevron points at a file that ships with the package.
    assert (theme.ASSETS / "chevron.svg").is_file()


def test_theme_fonts_load_and_become_the_application_font(qapp):
    assert theme.load_fonts()
    assert QApplication.font().family() == theme.FONT_FAMILY
    assert theme.FONT_FAMILY in QFontDatabase.families()
    assert (theme.ASSETS / "fonts" / "LICENSE.txt").is_file()


@pytest.mark.parametrize("found", [[], [0, -1]])
def test_theme_missing_fonts_fall_back_to_the_system_font(
    qapp, monkeypatch, tmp_path, caplog, found
):
    fonts = tmp_path / "fonts"
    fonts.mkdir()
    for index in range(len(found)):
        (fonts / f"f{index}.ttf").write_bytes(b"")
    ids = iter(found)
    monkeypatch.setattr(theme, "ASSETS", tmp_path)
    monkeypatch.setattr(
        theme.QFontDatabase, "addApplicationFont", lambda _path: next(ids)
    )
    before = QApplication.font()
    with caplog.at_level("WARNING", logger="croak.gui.theme"):
        assert not theme.load_fonts()
    assert "system font" in caplog.text
    assert QApplication.font() == before


def test_theme_set_flag_restyles_only_on_change(qtbot, monkeypatch):
    widget = QWidget()
    qtbot.addWidget(widget)
    calls = []
    monkeypatch.setattr(theme, "repolish", calls.append)
    theme.set_flag(widget, "current", True)
    theme.set_flag(widget, "current", True)
    assert widget.property("current") is True
    assert calls == [widget]


def test_theme_repolish_reapplies_the_style(qtbot):
    widget = QWidget()
    qtbot.addWidget(widget)
    widget.setStyleSheet(theme.stylesheet())
    theme.repolish(widget)  # must not raise on a styled widget


def test_theme_builders_set_the_role(qtbot):
    heading = theme.label("Hello", "h1")
    plain = theme.label("text", wrap=True)
    primary = theme.button("Go", "primary")
    plain_button = theme.button("Open")
    box = theme.frame("form")
    for widget in (heading, plain, primary, plain_button, box):
        qtbot.addWidget(widget)
    assert heading.objectName() == "h1"
    assert plain.objectName() == ""
    assert plain.wordWrap()
    assert primary.objectName() == "primary"
    assert plain_button.objectName() == "plain"
    assert box.objectName() == "form"
