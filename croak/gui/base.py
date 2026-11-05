"""The stage pages' wizard-era interface, on the workflow window's layout.

:class:`Stage` is a :class:`~croak.gui.stage.StagePage` (form of folding
sections beside the plots) that keeps the calls the stage modules were written
against: ``controls`` to add sections to, :meth:`set_plot_area`,
:meth:`set_status` and the entry-page hooks the old wizard's footer reads.
"""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

from .help import apply_tooltips, flatten
from .stage import StagePage
from .state import WizardState

__all__ = ["Stage"]


class Stage(StagePage):
    """A stage page: build into :attr:`controls`, then :meth:`set_plot_area`.

    Subclasses add their sections (:func:`croak.gui.canvas.group`) to
    ``self.controls``, hand their plots to :meth:`set_plot_area`, call
    :meth:`apply_help` at the end of ``__init__`` and implement :meth:`on_enter`.
    """

    def __init__(self, state: WizardState, *, find: bool = False):
        super().__init__(state, find=find)
        #: Where the sections go: the form's column.
        self.controls = self.form.content
        #: The form's ready line, which :meth:`set_status` writes.
        self.status_label = self.form.ready

    def set_plot_area(self, plot: QWidget, *strips: QWidget) -> None:
        """Show ``plot`` (taking the room) with ``strips`` beneath it."""
        self.set_plots(plot)
        for strip in strips:
            self.body.addWidget(strip, stretch=0)

    def set_status(self, text: str, state: str = "ok") -> None:
        """Show ``text`` on the ready line, as ``"ok"``, ``"warn"`` or ``"error"``."""
        self.status_label.say(text, state)

    def apply_help(self) -> set[tuple[str, str]]:
        """Attach :attr:`HELP` as tooltips; return the ``(section, label)`` matched.

        Call once at the end of the subclass ``__init__``, after the controls exist.
        """
        return apply_tooltips(self, flatten(self.HELP))

    # -- entry-page hooks ---------------------------------------------------
    # The entry pages (the synthetic generator and the simulated-scan loader)
    # are reached from the welcome menu rather than by walking the linear stage
    # sequence, so the old wizard cannot decide from ``state.stage`` what its
    # footer's "Next" button should say or do. They override these hooks.

    def next_label(self) -> str:
        """Text for the wizard footer's forward button on this page."""
        return "Next →"

    def advance(self) -> None:
        """Do this page's forward action (generate/load, then move on)."""
