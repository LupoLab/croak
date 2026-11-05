"""The building blocks of a stage: a form of folding sections beside its plots.

A :class:`Sheet` is the layout every stage shares: a :class:`Form` on the left
(sections of rows, the one-line help for the focused row, and a
:class:`ReadyLine`) and the stage's plots on the right, which move below the
form when the window is narrow.

Help text comes from the stage's ``HELP`` sections (:mod:`croak.gui.help`):
a row added to section ``S`` with label ``L`` takes ``HELP[(S, L)]`` as its
tooltip and as the help line shown while it has focus; a row whose editor is a
check box is keyed by the box's text. :attr:`Form.matched` records the keys that
found a row, so a test can check that no help entry is orphaned.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QBoxLayout,
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from . import settings
from .help import HelpSections, flatten
from .theme import label, set_flag

__all__ = [
    "AdviceBox",
    "DerivedGrid",
    "Form",
    "ProgressStrip",
    "ReadyLine",
    "Row",
    "Section",
    "Segments",
    "Sheet",
]

LABEL_WIDTH = 124
"""The width of a row's label column, logical px."""

NARROW_WIDTH = 1000
"""Below this window width (logical px) a sheet's plots move under its form."""

FORM_MIN_WIDTH = 360
"""The narrowest the form column may be, logical px."""


def plain(text: str) -> str:
    """Undo Qt's ``&&`` escape, giving a label as the user reads it.

    Examples
    --------
    >>> plain("Skip filtering && regrid")
    'Skip filtering & regrid'
    """
    return text.replace("&&", "&")


class Row(QWidget):
    """One setting: a label in the label column beside its editor.

    A check box carries its own text and takes the whole width.
    """

    def __init__(self, text: str, editor: QWidget, help_text: str = "") -> None:
        super().__init__()
        self.editor = editor
        self.help_text = help_text
        self.text = plain(editor.text()) if isinstance(editor, QCheckBox) else text
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(8)
        self.label: QLabel | None = None
        if not isinstance(editor, QCheckBox):
            self.label = label(text, "rowLabel")
            self.label.setFixedWidth(LABEL_WIDTH)
            self.label.setWordWrap(True)
            layout.addWidget(self.label)
        layout.addWidget(editor, stretch=1)
        if help_text:
            editor.setToolTip(help_text)
            if self.label is not None:
                self.label.setToolTip(help_text)

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802 — Qt override
        # The label greys with its editor: a disabled row reads as unavailable.
        if event.type() == QEvent.Type.EnabledChange and self.label is not None:
            self.label.setEnabled(self.isEnabled())
        super().changeEvent(event)

    def matches(self, query: str) -> bool:
        """Whether ``query`` (case-insensitive) is in this row's label or help."""
        query = query.casefold()
        return query in self.text.casefold() or query in self.help_text.casefold()


class Section(QWidget):
    """A titled group of rows that folds open and shut in place.

    Parameters
    ----------
    title : str
        Its heading, and the first half of its rows' help keys.
    help_map : mapping
        ``(section, label) -> text``, from the stage's ``HELP``.
    matched : set
        Where the help keys this section uses are recorded.
    folded : bool
        Its initial state; a remembered state (``settings_key``) wins.
    settings_key : str
        Remember the fold state under this key ("" to not remember it).
    """

    toggled = Signal(bool)

    def __init__(
        self,
        title: str,
        help_map: Mapping[tuple[str, str], str],
        matched: set[tuple[str, str]],
        *,
        folded: bool = False,
        settings_key: str = "",
    ) -> None:
        super().__init__()
        self.title = title
        self.rows: list[Row] = []
        self._help = help_map
        self._matched = matched
        self._key = settings_key
        if settings_key:
            folded = not settings.value(settings_key, not folded)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 6)
        outer.setSpacing(0)
        self.header = QToolButton()
        self.header.setObjectName("sectionHeader")
        self.header.setText(title)
        self.header.setCheckable(True)
        self.header.setChecked(not folded)
        self.header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.header.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.header.toggled.connect(self._fold)
        outer.addWidget(self.header)
        self.body = QWidget()
        self.layout_ = QVBoxLayout(self.body)
        self.layout_.setContentsMargins(14, 0, 0, 2)
        self.layout_.setSpacing(2)
        outer.addWidget(self.body)
        self._fold(not folded, remember=False)

    @property
    def folded(self) -> bool:
        """Whether the body is hidden."""
        return not self.header.isChecked()

    def set_folded(self, folded: bool) -> None:
        """Fold or unfold (remembered, when the section has a key)."""
        self.header.setChecked(not folded)

    def _fold(self, open_: bool, remember: bool = True) -> None:
        self.header.setArrowType(
            Qt.ArrowType.DownArrow if open_ else Qt.ArrowType.RightArrow
        )
        self.body.setVisible(open_)
        if remember and self._key:
            settings.set_value(self._key, open_)
        self.toggled.emit(open_)

    def add_row(self, text: str, editor: QWidget) -> Row:
        """Add a setting labelled ``text`` (ignored for a check box) and return it."""
        key_label = plain(editor.text()) if isinstance(editor, QCheckBox) else text
        key = (self.title, key_label)
        help_text = self._help.get(key, "")
        if help_text:
            self._matched.add(key)
        row = Row(text, editor, help_text)
        self.rows.append(row)
        self.layout_.addWidget(row)
        return row

    def add_widget(self, widget: QWidget) -> None:
        """Add a widget across the section's width (a button row, a table, …)."""
        self.layout_.addWidget(widget)


class Form(QFrame):
    """The left column of a sheet: sections in a scroll area, help, ready line.

    Parameters
    ----------
    help_sections : HelpSections
        The stage's ``HELP``.
    find : bool
        Offer "Find a setting", which shows only the rows that match.
    """

    def __init__(
        self, help_sections: HelpSections | None = None, *, find: bool = False
    ) -> None:
        super().__init__()
        self.setObjectName("form")
        self.setMinimumWidth(FORM_MIN_WIDTH)
        self.help_map = flatten(help_sections or [])
        self.matched: set[tuple[str, str]] = set()
        self.sections: list[Section] = []
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.find: QLineEdit | None = None
        if find:
            self.find = QLineEdit()
            self.find.setObjectName("find")
            self.find.setPlaceholderText("Find a setting")
            self.find.setClearButtonEnabled(True)
            self.find.textChanged.connect(self.filter)
            box = QWidget()
            bl = QVBoxLayout(box)
            bl.setContentsMargins(18, 12, 14, 0)
            bl.addWidget(self.find)
            outer.addWidget(box)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        self.content = QVBoxLayout(content)
        self.content.setContentsMargins(18, 10, 14, 10)
        self.content.setSpacing(4)
        self.content.addStretch(1)
        scroll.setWidget(content)
        outer.addWidget(scroll, stretch=1)
        foot = QWidget()
        fl = QVBoxLayout(foot)
        fl.setContentsMargins(18, 8, 14, 12)
        fl.setSpacing(6)
        self.help = label("", "help", wrap=True)
        self.help.setMinimumHeight(32)
        self.help.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.ready = ReadyLine()
        fl.addWidget(self.help)
        fl.addWidget(self.ready)
        outer.addWidget(foot)
        # A widget exists only under a QApplication; the stubs allow None.
        cast(QApplication, QApplication.instance()).focusChanged.connect(
            self._focus_moved
        )

    def add_section(
        self, title: str, *, folded: bool = False, settings_key: str = ""
    ) -> Section:
        """Append a section and return it."""
        section = Section(
            title,
            self.help_map,
            self.matched,
            folded=folded,
            settings_key=settings_key,
        )
        self.sections.append(section)
        self.content.insertWidget(self.content.count() - 1, section)
        return section

    def add_widget(self, widget: QWidget) -> None:
        """Append a widget outside any section (a heading, a file picker, …)."""
        self.content.insertWidget(self.content.count() - 1, widget)

    def _focus_moved(self, _old: QWidget | None, new: QWidget | None) -> None:
        widget: QObject | None = new
        while widget is not None and not isinstance(widget, Row):
            widget = widget.parent()
        if isinstance(widget, Row) and self.isAncestorOf(widget):
            self.help.setText(widget.help_text)

    def filter(self, query: str) -> None:
        """Show only the rows whose label or help contains ``query``."""
        query = query.strip()
        for section in self.sections:
            hits = 0
            for row in section.rows:
                hit = not query or row.matches(query)
                row.setVisible(hit)
                hits += hit
            section.setVisible(not query or hits > 0)
            if query and hits:
                section.body.setVisible(True)
            elif not query:
                section.body.setVisible(not section.folded)


class Sheet(QWidget):
    """A form beside the stage's plots; the plots move below on a narrow window."""

    def __init__(self, form: Form, plots: QWidget) -> None:
        super().__init__()
        self.form = form
        self.plots = plots
        self.split = QBoxLayout(QBoxLayout.Direction.LeftToRight, self)
        self.split.setContentsMargins(0, 0, 0, 0)
        self.split.setSpacing(0)
        self.split.addWidget(form, stretch=2)
        self.split.addWidget(plots, stretch=3)

    @property
    def narrow(self) -> bool:
        """Whether the plots are below the form."""
        return self.split.direction() == QBoxLayout.Direction.TopToBottom

    def resizeEvent(self, event) -> None:  # noqa: N802, ANN001 — Qt override
        narrow = self.width() < NARROW_WIDTH
        if narrow != self.narrow:
            self.split.setDirection(
                QBoxLayout.Direction.TopToBottom
                if narrow
                else QBoxLayout.Direction.LeftToRight
            )
        super().resizeEvent(event)


class ReadyLine(QLabel):
    """Whether the stage can go on: green, amber with a warning, red when not."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("ready")
        self.setWordWrap(True)
        self.state = "ok"

    def say(self, text: str, state: str = "ok") -> None:
        """Show ``text`` as ``state``: ``"ok"``, ``"warn"`` or ``"error"``.

        Raises
        ------
        ValueError
            For any other state.
        """
        if state not in ("ok", "warn", "error"):
            raise ValueError(f"state must be ok, warn or error, not {state!r}")
        self.state = state
        self.setText(text)
        set_flag(self, "warn", state == "warn")
        set_flag(self, "error", state == "error")


class AdviceBox(QFrame):
    """An amber box above the plots: what went wrong and what to do about it."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("advice")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(2)
        self.title = label("", "h3")
        self.text = label("", "note", wrap=True)
        self.text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.title)
        layout.addWidget(self.text)
        self.hide()

    def say(self, title: str, text: str) -> None:
        """Show the box."""
        self.title.setText(title)
        self.text.setText(text)
        self.show()


class DerivedGrid(QWidget):
    """Numbers derived from the settings: a small grey name over a large value."""

    def __init__(self, columns: int = 4) -> None:
        super().__init__()
        self.columns = columns
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(24)
        self.grid.setVerticalSpacing(8)
        self.values: dict[str, QLabel] = {}

    def add(self, name: str, tooltip: str = "") -> QLabel:
        """Add a number called ``name`` (shown as an en dash until set)."""
        index = len(self.values)
        cell = QWidget()
        cl = QVBoxLayout(cell)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)
        cl.addWidget(label(name, "derivedLabel"))
        value = label("–", "derivedValue")
        value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        cl.addWidget(value)
        if tooltip:
            cell.setToolTip(tooltip)
        self.grid.addWidget(cell, index // self.columns, index % self.columns)
        self.values[name] = value
        return value

    def set(self, name: str, text: str, state: str = "ok") -> None:
        """Show ``text`` for ``name``; ``state`` ``"warn"``/``"error"`` colours it."""
        value = self.values[name]
        value.setText(text)
        set_flag(value, "warn", state == "warn")
        set_flag(value, "error", state == "error")

    def clear(self) -> None:
        """Show every number as unknown."""
        for name in self.values:
            self.set(name, "–")


class Segments(QFrame):
    """A row of joined buttons choosing one of a few values.

    Parameters
    ----------
    choices : sequence of (value, text)
        The values and their captions, in order.
    current : str
        The value chosen at first.
    """

    changed = Signal(str)

    def __init__(self, choices: Sequence[tuple[str, str]], current: str) -> None:
        super().__init__()
        self.setObjectName("segments")
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        self.buttons: dict[str, QPushButton] = {}
        for value, text in choices:
            b = QPushButton(text)
            b.setObjectName("segment")
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.clicked.connect(lambda _=False, v=value: self.set_value(v, emit=True))
            layout.addWidget(b)
            self.buttons[value] = b
        if current not in self.buttons:
            raise ValueError(f"{current!r} is not one of {list(self.buttons)}")
        self.value = ""
        self.set_value(current)

    def set_value(self, value: str, *, emit: bool = False) -> None:
        """Choose ``value``; with ``emit``, say so (only when it changed)."""
        if value not in self.buttons:
            raise ValueError(f"{value!r} is not one of {list(self.buttons)}")
        changed = value != self.value
        self.value = value
        for key, b in self.buttons.items():
            set_flag(b, "current", key == value)
        if emit and changed:
            self.changed.emit(value)


class ProgressStrip(QFrame):
    """A thin strip over the plots while work runs: state, bar and figures."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("strip")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(14)
        self.state = label("", "h3")
        self.bar = QProgressBar()
        self.bar.setObjectName("progress")
        self.bar.setTextVisible(False)
        self.bar.setRange(0, 1000)
        self.figures = label("", "note")
        layout.addWidget(self.state)
        layout.addWidget(self.bar, stretch=1)
        layout.addWidget(self.figures)
        self.hide()

    def add_end(self, widget: QWidget) -> None:
        """Add a widget at the strip's right end (a sparkline, a lamp, …)."""
        self.layout().addWidget(widget)  # pyright: ignore[reportOptionalMemberAccess]

    def show_progress(self, state: str, fraction: float, figures: str) -> None:
        """Show ``state``, a bar at ``fraction`` (clamped to [0, 1]) and ``figures``."""
        self.state.setText(state)
        self.bar.setValue(round(1000 * min(max(fraction, 0.0), 1.0)))
        self.figures.setText(figures)
        self.show()
