"""The window's look: light and calm, with white plots, in IBM Plex Sans.

Every colour is a named token of :class:`Palette`, and the style sheet is built
from those tokens, so one change of a token restyles every widget that uses it.
Widgets take their style from their *role*, which is their ``objectName``
(:func:`label`, :func:`button`, :func:`frame`), and from dynamic properties such
as ``current`` or ``warn`` that :func:`set_flag` changes and re-applies. Nothing
else in the GUI calls ``setStyleSheet``.

The look follows the solprop front end's, so the two LupoLab applications read
alike. The fonts are IBM Plex Sans, carried in ``assets/fonts`` under the SIL
Open Font License (``assets/fonts/LICENSE.txt``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication, QFrame, QLabel, QPushButton, QWidget

__all__ = [
    "FONT_FAMILY",
    "PALETTE",
    "Palette",
    "button",
    "frame",
    "label",
    "load_fonts",
    "repolish",
    "set_flag",
    "stylesheet",
]

_log = logging.getLogger(__name__)

ASSETS = Path(__file__).parent / "assets"
"""Where the fonts and the drawn assets live."""

FONT_FAMILY = "IBM Plex Sans"
"""The window's font family."""

# The type scale, in logical pixels. Plot ticks take the application font's
# PLOT size; widgets take theirs from the style sheet.
BODY_PX = 13
SMALL_PX = 12
PLOT_PX = 11
H1_PX = 28
H2_PX = 15
VALUE_PX = 19


@dataclass(frozen=True)
class Palette:
    """The window's colours, by what they are for."""

    ink: str = "#1b2430"
    """Text."""
    muted: str = "#5a6675"
    """Secondary text: help, units, plot tick labels."""
    heading: str = "#3d4957"
    """Small headings and unselected rail items."""
    disabled: str = "#a4adb8"
    """Disabled text."""
    axis: str = "#b4bcc6"
    """Plot axes."""
    accent: str = "#0b6e99"
    """The primary action, focus and the selection."""
    accent_hover: str = "#095d82"
    """The primary action under the pointer."""
    accent_off: str = "#b9c4cc"
    """A disabled primary action."""
    selected_bg: str = "#e8f1f5"
    """What is selected: its background."""
    selected_ink: str = "#0a5a7d"
    """What is selected: its text; also links."""
    hover_bg: str = "#f1f3f5"
    """A flat control under the pointer."""
    button_hover_bg: str = "#f6f7f8"
    """A plain button under the pointer."""
    hairline: str = "#e3e6ea"
    """Lines between the window's regions."""
    border: str = "#cdd3da"
    """Inputs and plain buttons."""
    tile_border: str = "#dfe3e8"
    """Home's tiles."""
    popup_border: str = "#d7dce2"
    """Popups and dashed placeholders."""
    track: str = "#e8ebee"
    """Slider grooves and progress tracks."""
    text_selection: str = "#bcdbe8"
    """Selected text in an input."""
    ok: str = "#1c7a4a"
    """A ready line that is ready."""
    warn: str = "#8a5200"
    """A ready line with a warning."""
    error: str = "#b3261e"
    """A ready line that is not ready, and errors."""
    advice_border: str = "#e6cfa8"
    """The amber advice box's edge."""
    advice_bg: str = "#fdf8ee"
    """The amber advice box."""
    background: str = "#ffffff"
    """Everything else."""


PALETTE = Palette()
"""The palette the window uses."""


def stylesheet(p: Palette = PALETTE) -> str:
    """Return the window's style sheet in palette ``p``.

    Parameters
    ----------
    p : Palette, optional
        The colours.

    Returns
    -------
    str
        A Qt style sheet, addressing widgets by role (object name).
    """
    chevron = (ASSETS / "chevron.svg").as_posix()
    input_box = (
        f"border: 1px solid {p.border}; border-radius: 5px; padding: 3px 6px;"
        f" background: {p.background}; color: {p.ink};"
        f" selection-background-color: {p.text_selection}; selection-color: {p.ink};"
    )
    focus = f"border: 2px solid {p.accent}; padding: 2px 5px;"
    flat = "border: 0; border-radius: 6px; background: none;"
    selected = f"background: {p.selected_bg}; color: {p.selected_ink};"
    rules = [
        # the frame
        f"* {{ font-size: {BODY_PX}px; }}",
        f"QMainWindow, QWidget#page, QScrollArea, QStackedWidget,"
        f" QScrollArea > QWidget > QWidget {{ background: {p.background}; }}",
        f"QWidget#page QLabel {{ color: {p.ink}; }}",
        f"QFrame#top {{ border-bottom: 1px solid {p.hairline};"
        f" background: {p.background}; }}",
        f"QPushButton#brand {{ border: 0; background: none; padding: 4px;"
        f" font-size: {H2_PX}px; font-weight: 600; color: {p.ink}; }}",
        f"QLabel#context {{ color: {p.muted}; }}",
        f"QPushButton#stage {{ {flat} padding: 6px 14px; color: {p.muted};"
        f" font-weight: 500; }}",
        f"QPushButton#stage:hover {{ background: {p.hover_bg}; }}",
        f'QPushButton#stage[current="true"] {{ {selected} }}',
        f"QPushButton#stage:disabled {{ color: {p.disabled}; background: none; }}",
        f"QPushButton#primary {{ border: 0; border-radius: 6px; padding: 7px 14px;"
        f" min-width: 96px; background: {p.accent}; color: #fff; font-weight: 500; }}",
        f"QPushButton#primary:hover {{ background: {p.accent_hover}; }}",
        f"QPushButton#primary:disabled {{ background: {p.accent_off}; }}",
        f"QToolButton#overflow {{ {flat} padding: 5px 8px; font-size: 16px;"
        f" color: {p.muted}; }}",
        f"QToolButton#overflow:hover {{ background: {p.hover_bg}; }}",
        "QToolButton#overflow::menu-indicator { image: none; }",
        # headings and text
        f"QLabel#h1 {{ font-size: {H1_PX}px; font-weight: 600; }}",
        f"QLabel#h2 {{ font-size: {H2_PX}px; font-weight: 600; }}",
        f"QLabel#h3 {{ font-weight: 600; color: {p.heading}; }}",
        f"QLabel#help, QLabel#note, QLabel#pageNote {{ color: {p.muted};"
        f" font-size: {SMALL_PX}px; }}",
        f"QLabel#control, QLabel#plotTitle {{ color: {p.heading};"
        f" font-size: {SMALL_PX}px; font-weight: 500; }}",
        f"QLabel#rowLabel {{ color: {p.ink}; }}",
        f"QLabel#rowLabel:disabled {{ color: {p.disabled}; }}",
        # Home
        f"QFrame#tile {{ border: 1px solid {p.tile_border}; border-radius: 10px;"
        f" background: {p.background}; min-height: 78px; }}",
        f"QFrame#tile:hover {{ border-color: {p.accent}; background: #fafcfd; }}",
        "QLabel#tileTitle { font-weight: 600; background: none; }",
        f"QLabel#tileText {{ color: {p.muted}; background: none; }}",
        "QFrame#item { border-radius: 6px; background: none; }",
        f"QFrame#item:hover {{ background: {p.hover_bg}; }}",
        "QLabel#itemTitle { font-weight: 500; background: none; }",
        f"QLabel#itemText {{ color: {p.muted}; font-size: {SMALL_PX}px;"
        " background: none; }",
        # buttons
        f"QPushButton#plain, QToolButton#plain {{ border: 1px solid {p.border};"
        f" border-radius: 6px; padding: 5px 12px; background: {p.background};"
        f" color: {p.ink}; }}",
        f"QPushButton#plain:hover, QToolButton#plain:hover"
        f" {{ background: {p.button_hover_bg}; }}",
        f"QPushButton#plain:disabled, QToolButton#plain:disabled"
        f" {{ color: {p.disabled}; }}",
        "QToolButton#plain::menu-indicator { image: none; }",
        f"QPushButton#link {{ border: 0; padding: 0; background: none;"
        f" color: {p.selected_ink}; text-decoration: underline;"
        f" font-size: {SMALL_PX}px; }}",
        f"QPushButton#link:disabled {{ color: {p.disabled}; }}",
        # sheets
        f"QFrame#form {{ border-right: 1px solid {p.hairline};"
        f" background: {p.background}; }}",
        f"QToolButton#sectionHeader {{ border: 0; background: none;"
        f" padding: 6px 0 4px 0; font-weight: 600; color: {p.heading}; }}",
        f"QToolButton#sectionHeader:disabled {{ color: {p.disabled}; }}",
        f"QLineEdit, QComboBox, QAbstractSpinBox {{ {input_box} }}",
        f"QLineEdit:focus, QComboBox:focus, QAbstractSpinBox:focus {{ {focus} }}",
        f"QLineEdit:disabled, QComboBox:disabled, QAbstractSpinBox:disabled"
        f" {{ color: {p.disabled}; border-color: {p.hairline}; }}",
        "QComboBox::drop-down { border: 0; width: 20px; }",
        f'QComboBox::down-arrow {{ image: url("{chevron}");'
        " width: 12px; height: 12px; }",
        f"QComboBox QAbstractItemView {{ border: 1px solid {p.popup_border};"
        f" background: {p.background}; color: {p.ink}; outline: 0;"
        f" selection-background-color: {p.selected_bg};"
        f" selection-color: {p.selected_ink}; }}",
        "QAbstractSpinBox::up-button, QAbstractSpinBox::down-button"
        " { width: 0; border: 0; }",
        f"QCheckBox {{ color: {p.ink}; spacing: 6px; }}",
        f"QCheckBox:disabled {{ color: {p.disabled}; }}",
        f"QLabel#derivedLabel {{ color: {p.muted}; font-size: {SMALL_PX}px; }}",
        f"QLabel#derivedValue {{ font-size: {VALUE_PX}px; font-weight: 500; }}",
        f'QLabel#derivedValue[warn="true"] {{ color: {p.warn}; }}',
        f'QLabel#derivedValue[error="true"] {{ color: {p.error}; }}',
        f"QLabel#ready {{ color: {p.ok}; font-weight: 500; }}",
        f'QLabel#ready[warn="true"] {{ color: {p.warn}; }}',
        f'QLabel#ready[error="true"] {{ color: {p.error}; }}',
        f"QFrame#advice {{ border: 1px solid {p.advice_border}; border-radius: 8px;"
        f" background: {p.advice_bg}; }}",
        f"QFrame#placeholder {{ border: 1px dashed {p.popup_border};"
        f" border-radius: 8px; background: {p.background}; }}",
        # sliders and progress
        f"QSlider::groove:horizontal {{ height: 6px; border-radius: 3px;"
        f" background: {p.track}; }}",
        f"QSlider::sub-page:horizontal {{ border-radius: 3px;"
        f" background: {p.accent}; }}",
        f"QSlider::handle:horizontal {{ width: 16px; height: 16px; margin: -5px 0;"
        f" border-radius: 8px; background: {p.accent}; }}",
        f"QSlider::sub-page:horizontal:disabled, QSlider::handle:horizontal:disabled"
        f" {{ background: {p.accent_off}; }}",
        f"QFrame#strip {{ border-bottom: 1px solid {p.hairline};"
        f" background: {p.background}; }}",
        f"QProgressBar#progress {{ border: 0; border-radius: 3px;"
        f" background: {p.track}; min-height: 6px; max-height: 6px;"
        " color: transparent; }",
        f"QProgressBar#progress::chunk {{ border-radius: 3px;"
        f" background: {p.accent}; }}",
        # Analyse
        f"QFrame#rail {{ border-right: 1px solid {p.hairline};"
        f" background: {p.background}; }}",
        f"QPushButton#railItem {{ {flat} text-align: left; padding: 7px 10px;"
        f" color: {p.heading}; }}",
        f"QPushButton#railItem:hover {{ background: {p.hover_bg}; }}",
        f'QPushButton#railItem[current="true"] {{ {selected} font-weight: 500; }}',
        f"QPushButton#railItem:disabled {{ color: {p.disabled}; background: none; }}",
        f"QFrame#panel {{ border-left: 1px solid {p.hairline};"
        f" background: {p.background}; }}",
        f"QFrame#displayBar {{ border-top: 1px solid {p.hairline};"
        f" background: {p.background}; }}",
        f"QFrame#segments {{ border: 1px solid {p.border}; border-radius: 6px;"
        f" background: {p.background}; }}",
        f"QPushButton#segment {{ border: 0; border-radius: 5px; padding: 4px 8px;"
        f" background: none; color: {p.ink}; font-size: {SMALL_PX}px; }}",
        f'QPushButton#segment[current="true"] {{ {selected} font-weight: 500; }}',
        f"QPushButton#segment:disabled {{ color: {p.disabled}; }}",
        f"QListWidget {{ border: 1px solid {p.border}; border-radius: 5px;"
        f" background: {p.background}; color: {p.ink}; outline: 0; }}",
        f"QListWidget::item:selected {{ {selected} }}",
        # the drawer
        f"QFrame#drawer {{ border-left: 1px solid {p.hairline};"
        f" background: {p.background}; }}",
        f"QPlainTextEdit#drawerText {{ border: 0; background: {p.background};"
        f' color: {p.ink}; font-family: "Menlo", "Consolas", monospace;'
        f" font-size: {SMALL_PX}px; }}",
    ]
    return "\n".join(rules)


def load_fonts() -> bool:
    """Make IBM Plex Sans the application's font, the plots' ticks included.

    A missing or unreadable font file is a cosmetic failure: the window keeps
    the system font and is logged, not stopped.

    Returns
    -------
    bool
        Whether every weight loaded.
    """
    paths = sorted((ASSETS / "fonts").glob("*.ttf"))
    ids = [QFontDatabase.addApplicationFont(str(path)) for path in paths]
    if not ids or -1 in ids:
        _log.warning("IBM Plex Sans did not load; the system font is used instead")
        return False
    font = QFont(FONT_FAMILY)
    font.setPixelSize(PLOT_PX + 1)
    QApplication.setFont(font)
    return True


def repolish(widget: QWidget) -> None:
    """Re-apply the style sheet to ``widget`` after one of its properties changed."""
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def set_flag(widget: QWidget, name: str, on: bool) -> None:
    """Set the boolean style property ``name`` of ``widget`` and restyle it.

    Parameters
    ----------
    widget : QWidget
        The widget.
    name : str
        The property the style sheet tests, e.g. ``"current"``, ``"warn"``.
    on : bool
        Its value.
    """
    if widget.property(name) == on:
        return
    widget.setProperty(name, on)
    repolish(widget)


def label(text: str, role: str = "", *, wrap: bool = False) -> QLabel:
    """Return a label styled as ``role`` (``"h1"``, ``"help"``, …)."""
    widget = QLabel(text)
    if role:
        widget.setObjectName(role)
    widget.setWordWrap(wrap)
    return widget


def button(text: str, role: str = "plain") -> QPushButton:
    """Return a push button styled as ``role`` (``"primary"``, ``"plain"``, …)."""
    widget = QPushButton(text)
    widget.setObjectName(role)
    return widget


def frame(role: str) -> QFrame:
    """Return an empty frame styled as ``role`` (``"form"``, ``"rail"``, …)."""
    widget = QFrame()
    widget.setObjectName(role)
    return widget
