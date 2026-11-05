"""The stage pages' wizard-era interface, on the workflow window's layout.

:class:`Stage` is a :class:`~croak.gui.stage.StagePage` (form of folding
sections beside the plots) that keeps the calls the stage modules were written
against: ``controls`` to add sections to, :meth:`set_plot_area`,
:meth:`set_status` and the entry pages' :meth:`advance`.
"""

from __future__ import annotations

from PySide6.QtCore import Signal, SignalInstance
from PySide6.QtWidgets import QPushButton, QWidget

from .help import apply_tooltips, flatten
from .stage import StagePage
from .state import WizardState

__all__ = ["RunControls", "Stage"]


class Stage(StagePage):
    """A stage page: build into :attr:`controls`, then :meth:`set_plot_area`.

    Subclasses add their sections (:func:`croak.gui.controls.group`) to
    ``self.controls``, hand their plots to :meth:`set_plot_area`, call
    :meth:`apply_help` at the end of ``__init__`` and implement :meth:`on_enter`.

    Attributes
    ----------
    advanced : Signal(str)
        An entry page finished its forward action and says where the session
        goes next: ``"check"`` (the marginal check) or ``"retrieve"`` (a raw
        simulated scan, straight to Retrieve).
    """

    advanced = Signal(str)

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

    def advance(self) -> None:
        """Do an entry page's forward action (generate or load), then emit
        :attr:`advanced`; the Data page calls it on Continue."""


class RunControls:
    """Run and Stop buttons that follow a worker, for a page that runs one.

    The page creates :attr:`run_btn` and :attr:`stop_btn`; :meth:`set_running`
    toggles them and tells the window (``changed``) so its primary action
    becomes Stop while the work runs. Mix in before :class:`Stage`.
    """

    run_btn: QPushButton
    stop_btn: QPushButton
    changed: SignalInstance

    def set_running(self, running: bool) -> None:
        """Show the work as running (Stop on) or idle (Run on)."""
        self.run_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.changed.emit()

    def busy(self) -> bool:
        """Whether the work is running."""
        return self.stop_btn.isEnabled()
