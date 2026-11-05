"""The building blocks of a stage: a form of folding sections beside its plots.

A :class:`Sheet` is the layout every stage shares: a :class:`Form` on the left
(folding :class:`Section` s of rows, a one-line help for the focused control and
a :class:`ReadyLine`) and the stage's plots on the right, which move below the
form when the window is narrow.

A section's rows live in an ordinary :class:`~PySide6.QtWidgets.QFormLayout`
(:attr:`Section.form`), so a stage adds them with ``form.addRow(label, widget)``.
Help text is attached afterwards by :func:`croak.gui.help.apply_tooltips`, keyed
by ``(section title, row label)``; the one-line help under the form shows the
tooltip of whatever has focus, so the tooltip is the help's single source.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QBoxLayout,
    QCheckBox,
    QFormLayout,
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
from .theme import label, set_flag

__all__ = [
    "AdviceBox",
    "DerivedGrid",
    "Form",
    "ProgressStrip",
    "ReadyLine",
    "Section",
    "Segments",
    "Sheet",
    "plain",
    "row_field",
    "row_text",
]

NARROW_WIDTH = 1000
"""Below this window width (logical px) a sheet's plots move under its form."""

FORM_MIN_WIDTH = 360
"""The narrowest the form column may be, logical px."""


def plain(text: str) -> str:
    """Undo Qt's ``&&`` escape, giving a label as the user reads it.

    Qt reads a lone ``&`` in a check box, button or section title as a keyboard
    mnemonic, so a literal ampersand is written ``&&``; help keys are written
    the way the user reads them.

    Examples
    --------
    >>> plain("Skip filtering && regrid")
    'Skip filtering & regrid'
    """
    return text.replace("&&", "&")


def row_text(form: QFormLayout, row: int) -> str:
    """The label of ``form``'s row ``row``: its label, else a check box's own text.

    A row with neither (a button row, a note) has no label: ``""``.
    """
    item = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
    label_widget = item.widget() if item is not None else None
    if isinstance(label_widget, QLabel):
        return plain(label_widget.text())
    field = row_field(form, row)
    return plain(field.text()) if isinstance(field, QCheckBox) else ""


def row_field(form: QFormLayout, row: int) -> QWidget | None:
    """The widget in ``form``'s row ``row``: its field, or one spanning the row."""
    for role in (QFormLayout.ItemRole.FieldRole, QFormLayout.ItemRole.SpanningRole):
        item = form.itemAt(row, role)
        widget = item.widget() if item is not None else None
        if widget is not None:
            return widget
    return None


class Section(QWidget):
    """A titled group of form rows that folds open and shut in place.

    Parameters
    ----------
    title : str
        Its heading (``&&`` for a literal ampersand); the first half of its
        rows' help keys.
    folded : bool
        Its initial state; a remembered state (``settings_key``) wins.
    settings_key : str
        Remember the fold state under this key ("" to not remember it).
    """

    toggled = Signal(bool)

    def __init__(
        self, title: str, *, folded: bool = False, settings_key: str = ""
    ) -> None:
        super().__init__()
        self.section_title = plain(title)
        self._key = settings_key
        if settings_key:
            folded = not settings.value(settings_key, not folded)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 8)
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
        self.form = QFormLayout(self.body)
        self.form.setContentsMargins(14, 2, 0, 2)
        self.form.setHorizontalSpacing(10)
        self.form.setVerticalSpacing(6)
        self.form.setLabelAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        # Grow only the fields that ask for room (sliders, paths): the macOS
        # style's default would size every field to its hint instead.
        self.form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        outer.addWidget(self.body)
        self._fold(not folded, remember=False)

    @property
    def folded(self) -> bool:
        """Whether the rows are hidden."""
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

    def filter(self, query: str) -> int:
        """Show only the rows whose label or help contains ``query``; count them.

        An empty ``query`` shows every row.
        """
        query = query.casefold()
        hits = 0
        for row in range(self.form.rowCount()):
            field = row_field(self.form, row)
            tip = field.toolTip() if field is not None else ""
            hit = not query or query in f"{row_text(self.form, row)}\n{tip}".casefold()
            self.form.setRowVisible(row, hit)
            hits += hit
        return hits


class Form(QFrame):
    """The left column of a sheet: sections in a scroll area, help, ready line.

    Parameters
    ----------
    find : bool
        Offer "Find a setting", which shows only the rows that match.
    """

    def __init__(self, *, find: bool = False) -> None:
        super().__init__()
        self.setObjectName("form")
        self.setMinimumWidth(FORM_MIN_WIDTH)
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
        self.content.setSpacing(2)
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

    @property
    def sections(self) -> list[Section]:
        """The form's sections, in order."""
        return self.findChildren(Section)

    def add_section(
        self, title: str, *, folded: bool = False, settings_key: str = ""
    ) -> Section:
        """Append a section and return it."""
        section = Section(title, folded=folded, settings_key=settings_key)
        self.content.addWidget(section)
        return section

    def add_widget(self, widget: QWidget) -> None:
        """Append a widget outside any section (a button row, a note, …)."""
        self.content.addWidget(widget)

    def _focus_moved(self, _old: QWidget | None, new: QWidget | None) -> None:
        """Show the help of the newly focused control, if it is on this form."""
        widget = new
        while widget is not None and widget is not self:
            if widget.toolTip():
                if self.isAncestorOf(widget):
                    self.help.setText(widget.toolTip())
                return
            widget = widget.parentWidget()

    def filter(self, query: str) -> None:
        """Show only the rows whose label or help contains ``query``."""
        query = query.strip()
        for section in self.sections:
            hits = section.filter(query)
            section.setVisible(not query or hits > 0)
            section.body.setVisible(bool(query and hits) or not section.folded)


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
