"""Shared base for wizard stage pages (left controls + right plots)."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .help import HelpDialog, HelpSections, apply_tooltips, flatten
from .state import WizardState

__all__ = ["Stage"]


class Stage(QWidget):
    """A wizard page: a scrollable control column on the left, plots on the right.

    Subclasses build their controls into ``self.controls`` (a vertical layout)
    and their plot area into ``self.plot_area`` (a widget), then implement
    :meth:`on_enter` to refresh when the page becomes visible.

    Subclasses may also set :attr:`HELP_TITLE`, :attr:`HELP_INTRO` and
    :attr:`HELP` (see :mod:`croak.gui.help`) and call :meth:`apply_help` at the
    end of ``__init__`` to attach option tooltips; :meth:`show_help` then opens
    a summary dialog of the same content (wired to the wizard header's ``?``).
    """

    #: Short stage name shown in the help dialog title (e.g. ``"Retrieve"``).
    HELP_TITLE: str = ""
    #: One-paragraph overview shown at the top of the help dialog.
    HELP_INTRO: str = ""
    #: Per-option help; ``[(section, [(label, text), ...]), ...]``.
    HELP: HelpSections = []

    def __init__(self, state: WizardState):
        super().__init__()
        self.state = state
        self._help_dialog: HelpDialog | None = None

        outer = QHBoxLayout(self)
        # left: scrollable controls
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(315)
        panel = QWidget()
        self.controls = QVBoxLayout(panel)
        scroll.setWidget(panel)
        outer.addWidget(scroll)

        # right: plot area (set by subclass via set_plot_area)
        self._right = QWidget()
        outer.addWidget(self._right, stretch=1)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)

    def set_plot_area(self, widget: QWidget) -> None:
        layout = self._right.layout() or QVBoxLayout(self._right)
        layout.addWidget(widget)

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def apply_help(self) -> set[tuple[str, str]]:
        """Attach :attr:`HELP` text as tooltips on the matching controls.

        Call once at the end of the subclass ``__init__`` (after the controls
        exist). Returns the matched ``(section, label)`` keys.
        """
        return apply_tooltips(self, flatten(self.HELP))

    def show_help(self) -> HelpDialog | None:
        """Open (or re-raise) the modeless summary dialog for this stage."""
        if not self.HELP:
            return None
        if self._help_dialog is None:
            self._help_dialog = HelpDialog(
                self.HELP_TITLE, self.HELP_INTRO, self.HELP, self
            )
        self._help_dialog.show()
        self._help_dialog.raise_()
        self._help_dialog.activateWindow()
        return self._help_dialog

    def on_enter(self) -> None:
        """Called when this stage becomes the visible page (override)."""

    # -- entry-page hooks ---------------------------------------------------
    # The *entry pages* (the synthetic generator and the simulated-scan loader)
    # are reached from the welcome menu rather than by walking the linear stage
    # sequence, so the wizard cannot decide from ``state.stage`` what its footer's
    # "Next" button should say or do. They override these three hooks; the linear
    # stages never use them (their Next is driven by ``WizardState.ready``).

    def next_label(self) -> str:
        """Text for the wizard footer's forward button on this page."""
        return "Next →"

    def can_advance(self) -> bool:
        """Whether the forward button should be enabled on this page."""
        return True

    def advance(self) -> None:
        """Do this page's forward action (generate/load, then move on)."""
