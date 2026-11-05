"""The base of a page in the workflow window.

The window owns the navigation and the one primary action; a page says what it
can do through a few hooks (:class:`PageBase`) and does it in :meth:`~PageBase.act`.
Most pages are a :class:`StagePage`: a :class:`~croak.gui.sheet.Sheet` with a form
of folding sections on the left and, on the right, an advice box, a progress
strip, the derived numbers and the stage's plots. Pages made of other pages
(Data's steps, Analyse's rail) build on :class:`PageBase` directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from .help import HelpDialog, HelpSections
from .sheet import AdviceBox, DerivedGrid, Form, ProgressStrip, Sheet
from .state import WizardState

if TYPE_CHECKING:
    from ..plotting import RetrievalPlotData

__all__ = ["PageBase", "StagePage"]


class PageBase(QWidget):
    """What the window asks of a page.

    Attributes
    ----------
    changed : Signal
        Emitted when what the page can do changes (it can now go on, a run
        started or ended), so the window re-reads the hooks.
    message : Signal(str)
        A passing notice for the window's status bar.
    """

    HELP_TITLE: str = ""
    """The page's name in its help dialog."""
    HELP_INTRO: str = ""
    """One paragraph on what the page is for."""
    HELP: HelpSections = []
    """Per-setting help, ``[(section, [(label, text), ...]), ...]``."""

    changed = Signal()
    message = Signal(str)

    def __init__(self, state: WizardState) -> None:
        super().__init__()
        self.state = state
        self._help_dialog: HelpDialog | None = None

    def on_enter(self) -> None:
        """Refresh as the page comes on screen (default: nothing to do)."""

    def can_advance(self) -> bool:
        """Whether the primary action can be pressed now (default: yes)."""
        return True

    def busy(self) -> bool:
        """Whether the page is running work the primary action would stop."""
        return False

    def page_key(self) -> str:
        """Which sub-page is shown, for pages that have several (default: "")."""
        return ""

    def act(self, kind: str) -> bool:
        """Do the primary action ``kind``; return whether the window moves on.

        ``kind`` is a :class:`croak.gui.workflow.Action` kind. By default
        ``"continue"`` has nothing to do here and the window moves to the next
        stage; a page with steps of its own moves through them first.

        Raises
        ------
        ValueError
            If this page has no such action.
        """
        if kind == "continue":
            return True
        raise ValueError(f"{type(self).__name__} has no {kind!r} action")

    def overview(self) -> RetrievalPlotData | None:
        """The retrieval this page shows, for a saved session's overview PDF."""
        return None

    def show_help(self) -> HelpDialog | None:
        """Open (or raise) the page's help dialog; ``None`` if it has no help."""
        if not self.HELP:
            return None
        if self._help_dialog is None:
            self._help_dialog = HelpDialog(
                self.HELP_TITLE, self.HELP_INTRO, self.HELP, self
            )
        self._help_dialog.show()
        self._help_dialog.raise_()
        return self._help_dialog


class StagePage(PageBase):
    """A page that is a sheet: its form beside its plots.

    Subclasses fill :attr:`form` with sections, add their plots with
    :meth:`set_plots`, and override the hooks they need.
    """

    def __init__(self, state: WizardState, *, find: bool = False) -> None:
        super().__init__(state)
        self.form = Form(find=find)
        self.advice = AdviceBox()
        self.strip = ProgressStrip()
        self.derived = DerivedGrid()
        self.derived.hide()  # shown once a page adds a number
        right = QWidget()
        right.setObjectName("page")
        self.plot_column = QVBoxLayout(right)
        self.plot_column.setContentsMargins(0, 0, 0, 0)
        self.plot_column.setSpacing(0)
        self.plot_column.addWidget(self.strip)
        body = QWidget()
        self.body = QVBoxLayout(body)
        self.body.setContentsMargins(16, 12, 16, 12)
        self.body.setSpacing(10)
        self.body.addWidget(self.advice)
        self.body.addWidget(self.derived)
        self.plot_column.addWidget(body, stretch=1)
        self.sheet = Sheet(self.form, right)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.sheet)

    def add_derived(self, name: str, tooltip: str = "") -> None:
        """Add a derived number above the plots (shown as an en dash until set)."""
        self.derived.add(name, tooltip)
        self.derived.show()

    def set_plots(self, plots: QWidget) -> None:
        """Put the stage's plots under its derived numbers; they take the room."""
        self.body.addWidget(plots, stretch=1)
