"""Shared base for wizard stage pages (left controls + right plots)."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QShowEvent
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from . import settings
from .help import HelpDialog, HelpSections, apply_tooltips, flatten
from .state import WizardState

__all__ = ["Stage", "CONTROLS_MIN_WIDTH"]

#: Narrowest the control column can be dragged (logical px) before it snaps shut.
#: Wide enough for a label, a spin box and its unit on one form row.
CONTROLS_MIN_WIDTH = 240
#: Debounce (ms) between a divider drag and remembering the new width.
_SAVE_WIDTH_DELAY_MS = 250


class Stage(QWidget):
    """A wizard page: a scrollable control column on the left, plots on the right.

    The two sit on a :class:`~PyQt6.QtWidgets.QSplitter`, so on a small display the
    control column can be narrowed or dragged fully closed to give the plots the
    room; its width is shared by every stage and remembered between launches
    (:mod:`croak.gui.settings`). The plot side never collapses.

    Subclasses build their controls into ``self.controls`` (a vertical layout)
    and hand their plot widget to :meth:`set_plot_area`, then implement
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
        outer.setContentsMargins(0, 0, 0, 0)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        # left: scrollable controls
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(CONTROLS_MIN_WIDTH)
        panel = QWidget()
        self.controls = QVBoxLayout(panel)
        scroll.setWidget(panel)
        # right: plot area (filled by the subclass via set_plot_area)
        self._right = QWidget()
        self._right_layout = QVBoxLayout(self._right)
        self._right_layout.setContentsMargins(0, 0, 0, 0)
        self.splitter.addWidget(scroll)
        self.splitter.addWidget(self._right)
        # Dragging the divider fully left closes the controls; the plots never
        # close. The stretch factors are load-bearing: without them QSplitter hands
        # the spare width to the *left* pane ([1910, 6] at 1920 px) and squeezes
        # the controls to their minimum whenever the window shrinks.
        self.splitter.setCollapsible(0, True)
        self.splitter.setCollapsible(1, False)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([settings.controls_width(), 1])
        # A drag fires splitterMoved continuously; remember the width once it rests.
        self._save_width_timer = QTimer(self)
        self._save_width_timer.setSingleShot(True)
        self._save_width_timer.setInterval(_SAVE_WIDTH_DELAY_MS)
        self._save_width_timer.timeout.connect(self._save_controls_width)
        self.splitter.splitterMoved.connect(self._on_splitter_moved)
        outer.addWidget(self.splitter)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)

    def set_plot_area(self, plot: QWidget, *strips: QWidget) -> None:
        """Fill the right-hand side: ``plot`` on top, then ``strips`` beneath it.

        ``plot`` takes every spare pixel; each strip (a read-out, a fold-away
        panel) keeps its natural height.
        """
        self._right_layout.addWidget(plot, stretch=1)
        for strip in strips:
            self._right_layout.addWidget(strip, stretch=0)

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802 — Qt override
        """Apply the shared control-column width, so a drag on one stage reaches all."""
        super().showEvent(event)
        width = settings.controls_width()
        if self.splitter.sizes()[0] != width:
            self.splitter.setSizes([width, 1])

    def _on_splitter_moved(self, _pos: int, _index: int) -> None:
        self._save_width_timer.start()

    def _save_controls_width(self) -> None:
        settings.set_controls_width(self.splitter.sizes()[0])

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
