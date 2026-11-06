"""A test driver for the workflow window, in the old stage numbering.

The GUI tests were written against six numbered stages (1 Load, 2 Marginal
check, 3 Preprocess, 4 Retrieve, 5 Dispersion, 6 Uncertainty). The window
now has five places (Data, Check, Prepare, Retrieve, Analyse with a rail), so
:class:`Driver` maps the numbers onto them and hands back the stage pages,
which keep their own behaviour. Navigation tests use the window itself.
"""

from __future__ import annotations

from typing import cast

from croak.gui.pages import AnalysePage, DataPage
from croak.gui.window import MainWindow
from croak.gui.workflow import Stage

# stage number -> (place, step or rail page)
PLACES = {
    1: (Stage.DATA, ""),
    2: (Stage.CHECK, ""),
    3: (Stage.PREPARE, ""),
    4: (Stage.RETRIEVE, ""),
    5: (Stage.ANALYSE, "dispersion"),
    6: (Stage.ANALYSE, "uncertainty"),
}


class Driver:
    """A :class:`MainWindow` and its stage pages, numbered as the tests know them."""

    def __init__(self, qtbot, entry: str = "") -> None:
        self.win = MainWindow()
        qtbot.addWidget(self.win)
        # Without an entry: a measured session on its Data page, where the old
        # wizard's stage tests began.
        self.start(entry or "experimental")

    @classmethod
    def of(cls, win: MainWindow) -> Driver:
        """A driver for a window the test built and started itself."""
        driver = cls.__new__(cls)
        driver.win = win
        return driver

    @property
    def state(self):
        return self.win.state

    def start(self, entry: str) -> None:
        """Start a new session from ``entry`` (no confirmation asked)."""
        self.win.state.reset()
        self.win.state.entry = entry
        self.win.session_path = ""
        self.win._build_pages()
        self.win.go(Stage.DATA)

    # -- pages --------------------------------------------------------------
    @property
    def data(self) -> DataPage:
        return cast(DataPage, self.win.pages[Stage.DATA])

    @property
    def analyse(self) -> AnalysePage:
        return cast(AnalysePage, self.win.pages[Stage.ANALYSE])

    @property
    def stages(self) -> list:
        """The six stage pages in the old order (the entry's loader first)."""
        return [
            self.data.entry_page,
            self.win.pages[Stage.CHECK],
            self.win.pages[Stage.PREPARE],
            self.win.pages[Stage.RETRIEVE],
            self.analyse.pages["dispersion"],
            self.analyse.pages["uncertainty"],
        ]

    @property
    def synthetic(self):
        return self.data.entry_page

    simulated = synthetic

    # -- where we are ---------------------------------------------------------
    def goto(self, number: int) -> None:
        """Show stage ``number``, as the old ``state.stage = number`` did.

        The route's rules are bypassed (a test may look at a stage before its
        input exists, to check what it says), exactly like the old setter.
        """
        place, sub = PLACES[number]
        win = self.win
        win.stage = place
        page = win.pages[place]
        win.stack.setCurrentWidget(page)
        if place is Stage.ANALYSE:
            self.analyse.show_page(sub)
        else:
            page.on_enter()
        win.refresh()

    @property
    def stage_number(self) -> int:
        """The old number of the stage on screen (0 on Home)."""
        place = self.win.stage
        for number, (where, sub) in PLACES.items():
            if where is not place:
                continue
            if place is Stage.ANALYSE and sub != self.analyse.key:
                continue
            return number
        return 0

    def current_page(self):
        """The stage page on screen (inside Data or Analyse's rail)."""
        return self.stages[self.stage_number - 1] if self.stage_number else None

    def __getattr__(self, name: str):
        """Anything else is the window's (``resize``, ``grab``, ``statusBar``…)."""
        return getattr(self.win, name)

    def rebuild(self) -> None:
        """Rebuild the pages from the state (after loading options into it)."""
        self.win._build_pages()
        self.win.go(Stage.DATA)
