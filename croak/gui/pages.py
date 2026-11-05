"""The window's four pages for a session, and the pages made of other pages.

:func:`default_pages` builds a session's Data, Prepare, Retrieve and Analyse
pages from the shared state. Data is two steps on one page: the entry's own
loader (a measured trace, a simulated scan or a synthetic pulse) and then the
marginal check of the loaded trace. Analyse is a rail of pages that read the
retrieved result: its panels (Overview, Traces, Pulse, Diagnostics), then
Dispersion and Uncertainty.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import cast

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
)

from ..session.pipeline import (
    assemble_load_data,
    assemble_simulated_load_data,
    assemble_simulated_tracedata,
)
from .base import Stage
from .help import HelpDialog
from .retrieval_views import ResultPage
from .sheet import Segments
from .stage import PageBase
from .stage_dispersion import StageDispersion
from .stage_load import StageLoad, compute_preproc_defaults, trace_signature
from .stage_marginal_check import StageMarginalCheck
from .stage_preprocess import StagePreprocess
from .stage_retrieve import StageRetrieve
from .stage_simulated import StageSimulated, simulated_signature
from .stage_synthetic import StageSynthetic
from .stage_uncertainty import StageUncertainty
from .state import WizardState
from .theme import frame, set_flag
from .workflow import Stage as Place

__all__ = ["AnalysePage", "DataPage", "RAIL_WIDTH", "default_pages"]

RAIL_WIDTH = 170
"""The width of Analyse's rail, logical px."""

_ENTRY_STEP = {
    "experimental": "Load the trace",
    "simulated": "Load the scan",
    "synthetic": "Generate the trace",
}


class DataPage(PageBase):
    """Data: the entry's loader, then the marginal check of what it loaded.

    Parameters
    ----------
    state : WizardState
        The session; its ``entry`` picks the loader.
    adopt_geometry : callable
        Retrieve's ``apply_simulated_geometry``, which a simulated scan's loader
        calls with the geometry the file records.
    """

    def __init__(
        self, state: WizardState, adopt_geometry: Callable[[object], str]
    ) -> None:
        super().__init__(state)
        entry = state.entry
        self.entry_page: Stage
        if entry == "synthetic":
            self.entry_page = StageSynthetic(state)
        elif entry == "simulated":
            self.entry_page = StageSimulated(state, adopt_geometry)
        else:
            self.entry_page = StageLoad(state)
        self.check = StageMarginalCheck(state)
        self._went = ""
        head = frame("strip")
        hl = QHBoxLayout(head)
        hl.setContentsMargins(16, 8, 16, 8)
        self.steps = Segments(
            [("entry", _ENTRY_STEP[entry]), ("check", "Check the marginal")],
            "entry",
        )
        self.steps.changed.connect(self.show_step)
        hl.addWidget(self.steps)
        hl.addStretch(1)
        self.stack = QStackedWidget()
        self.stack.addWidget(self.entry_page)
        self.stack.addWidget(self.check)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(head)
        layout.addWidget(self.stack, stretch=1)
        for page in (self.entry_page, self.check):
            page.changed.connect(self.changed)
            page.message.connect(self.message)
        self.entry_page.advanced.connect(self._advanced)
        state.load_data_changed.connect(self._sync)
        self._sync()

    @property
    def step(self) -> str:
        """The step on screen: ``"entry"`` or ``"check"``."""
        return self.steps.value

    def current(self) -> Stage:
        """The step's page."""
        return self.check if self.step == "check" else self.entry_page

    def _sync(self) -> None:
        """Offer the check only once a trace is loaded."""
        self.steps.buttons["check"].setEnabled(self.state.load_data is not None)
        self.changed.emit()

    def show_step(self, step: str) -> None:
        """Show ``step`` (``"entry"`` or ``"check"``) and refresh it."""
        self.steps.set_value(step)
        self.stack.setCurrentWidget(self.current())
        self.current().on_enter()
        self.changed.emit()

    def _advanced(self, where: str) -> None:
        self._went = where
        if where == "check":
            self.show_step("check")

    # -- hooks the window reads ---------------------------------------------
    def on_enter(self) -> None:
        self.current().on_enter()

    def page_key(self) -> str:
        return self.step

    def can_advance(self) -> bool:
        if self.step == "check" or self.state.entry == "experimental":
            return self.state.load_data is not None
        return self.entry_page.can_advance()

    def act(self, kind: str) -> bool:
        """Continue: from the loader to the check, from the check onwards.

        A measured trace is loaded by its own button, so its Continue only moves
        to the check; the synthetic and simulated loaders build the trace here.
        A raw simulated scan goes on to Retrieve straight from the loader.
        """
        if kind != "continue":
            return super().act(kind)
        if self.step == "check":
            return True
        if self.state.entry == "experimental":
            self.show_step("check")
            return False
        self._went = ""
        self.entry_page.advance()
        return self._went == "retrieve"

    def show_help(self) -> HelpDialog | None:
        return self.current().show_help()

    # -- session replay -----------------------------------------------------
    def restore(self) -> bool:
        """Reload a reopened session's trace with its saved options.

        Returns whether ``state.load_data`` is set. The saved preprocess settings
        are kept (the preprocess defaults are only recorded, for "Reset
        defaults"); when nothing can be reloaded the loader says why.
        """
        if self.state.entry == "simulated":
            return self._restore_simulated()
        return self._restore_measured()

    def _restore_measured(self) -> bool:
        state = self.state
        if not state.load.frog_path:
            self.entry_page.set_status(
                "Options loaded. Choose a trace file and run the stages.", "warn"
            )
            return False
        try:
            data = assemble_load_data(state.load)
        except Exception as exc:  # any reader failure is the user's to see
            self.entry_page.set_status(
                f"Options loaded, but reloading the trace failed: {exc}", "error"
            )
            return False
        state.set_load_data(data)
        state.trace_sig = trace_signature(state.load)
        state.preproc_defaults = compute_preproc_defaults(data)
        # A measured session's loader is a StageLoad (the entry chose it).
        cast(StageLoad, self.entry_page)._draw(data)
        return True

    def _restore_simulated(self) -> bool:
        state = self.state
        # A simulated session's loader is a StageSimulated (the entry chose it).
        loader = cast(StageSimulated, self.entry_page)
        if not state.simulated.frog_path:
            loader.set_status(
                "Options loaded. Choose a simulated scan file and run the stages.",
                "warn",
            )
            return False
        loader.seed_widgets(state.simulated)
        try:
            data = assemble_simulated_load_data(state.simulated)
        except Exception as exc:  # any reader failure is the user's to see
            loader.set_status(
                f"Options loaded, but reloading the simulated scan failed: {exc}",
                "error",
            )
            return False
        state.set_load_data(data)
        state.trace_sig = simulated_signature(state.simulated)
        state.preproc_defaults = compute_preproc_defaults(data)
        loader._draw(data)
        if state.simulated.raw_direct:
            try:
                state.set_tracedata(assemble_simulated_tracedata(state.simulated))
            except Exception as exc:  # as above
                loader.set_status(
                    f"Options loaded, but the raw simulated trace failed: {exc}",
                    "error",
                )
                return False
        return True


class AnalysePage(PageBase):
    """Analyse: a rail of pages that read the retrieved result.

    Parameters
    ----------
    state : WizardState
        The session.
    pages : sequence of (key, title, page)
        The rail's entries, in order; the first is shown first.
    """

    def __init__(
        self, state: WizardState, pages: Sequence[tuple[str, str, PageBase]]
    ) -> None:
        super().__init__(state)
        if not pages:
            raise ValueError("Analyse needs at least one page")
        rail = frame("rail")
        rail.setFixedWidth(RAIL_WIDTH)
        rl = QVBoxLayout(rail)
        rl.setContentsMargins(8, 10, 8, 10)
        rl.setSpacing(2)
        self.stack = QStackedWidget()
        self.pages: dict[str, PageBase] = {}
        self.rail_items: dict[str, QPushButton] = {}
        for key, title, page in pages:
            item = QPushButton(title)
            item.setObjectName("railItem")
            item.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            item.clicked.connect(lambda _=False, key=key: self.show_page(key))
            rl.addWidget(item)
            self.rail_items[key] = item
            self.pages[key] = page
            self.stack.addWidget(page)
            page.changed.connect(self.changed)
            page.message.connect(self.message)
        rl.addStretch(1)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(rail)
        layout.addWidget(self.stack, stretch=1)
        self.key = ""
        # Up and Down step through the rail while focus is anywhere on the page.
        for key, step in ((Qt.Key.Key_Up, -1), (Qt.Key.Key_Down, 1)):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(lambda step=step: self.step(step))
        self.show_page(pages[0][0])

    def step(self, offset: int) -> None:
        """Show the page ``offset`` places along the rail (clamped at its ends)."""
        keys = list(self.pages)
        index = min(max(keys.index(self.key) + offset, 0), len(keys) - 1)
        self.show_page(keys[index])

    def show_page(self, key: str) -> None:
        """Show the rail's page ``key`` and refresh it."""
        page = self.pages[key]
        self.key = key
        self.stack.setCurrentWidget(page)
        for name, item in self.rail_items.items():
            set_flag(item, "current", name == key)
        page.on_enter()
        self.changed.emit()

    def on_enter(self) -> None:
        self.pages[self.key].on_enter()

    def page_key(self) -> str:
        return self.key

    def busy(self) -> bool:
        return any(page.busy() for page in self.pages.values())

    def act(self, kind: str) -> bool:
        return self.pages[self.key].act(kind)

    def show_help(self) -> HelpDialog | None:
        return self.pages[self.key].show_help()


def default_pages(state: WizardState) -> Mapping[Place, PageBase]:
    """Build a session's Data, Prepare, Retrieve and Analyse pages."""
    retrieve = StageRetrieve(state)
    results: list[tuple[str, str, PageBase]] = [
        (key, title, ResultPage(state, key, retrieve.overview))
        for key, title in (
            ("overview", "Overview"),
            ("traces", "Traces"),
            ("pulse", "Pulse"),
            ("diagnostics", "Diagnostics"),
        )
    ]
    analyse = AnalysePage(
        state,
        [
            *results,
            ("dispersion", "Dispersion", StageDispersion(state)),
            ("uncertainty", "Uncertainty", StageUncertainty(state)),
        ],
    )
    return {
        # Looked up at each load, so Retrieve's current method is the one used.
        Place.DATA: DataPage(
            state, lambda geom: retrieve.apply_simulated_geometry(geom)
        ),
        Place.PREPARE: StagePreprocess(state),
        Place.RETRIEVE: retrieve,
        Place.ANALYSE: analyse,
    }
