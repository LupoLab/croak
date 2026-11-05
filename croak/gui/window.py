"""The workflow window: Home, then Data, Prepare, Retrieve and Analyse.

The top bar carries the brand (back to Home), what the session is, the four
stage buttons, the one primary action and an overflow menu (⋯). Which stages
open and what the primary action does are the pure rules of
:mod:`croak.gui.workflow`; this module applies them to widgets. A drawer on the
right shows the session's settings as ``options.toml`` or as a Python script.

The pages are built by a factory from the shared :class:`WizardState`, and rebuilt
for each new session, since their controls bind to the state's parameter objects.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import save
from ..scripting import generate_script
from ..session import files as session_files
from . import settings
from .branding import app_icon
from .home import ENTRY_TILES, Home
from .stage import StagePage
from .state import WizardState
from .theme import button, frame, label, load_fonts, set_flag, stylesheet
from .workflow import (
    STAGES,
    Action,
    Progress,
    Stage,
    blocked_reason,
    next_stage,
    primary_action,
)

__all__ = ["DRAWER_WIDTH", "MainWindow", "PageFactory"]

PageFactory = Callable[[WizardState], Mapping[Stage, StagePage]]
"""Builds a session's four stage pages from the shared state."""

DRAWER_WIDTH = 440
"""The width of the settings drawer, logical px."""
TOP_BAR_HEIGHT = 52
"""The height of the top bar, logical px."""
NOTICE_MS = 20_000
"""How long a passing notice stays in the status bar, ms."""

_ENTRY_TITLES = {entry: title for entry, title, _ in ENTRY_TILES}


class MainWindow(QMainWindow):
    """The croak window.

    Parameters
    ----------
    make_pages : PageFactory
        Builds the Data, Prepare, Retrieve and Analyse pages for a session.
    state : WizardState, optional
        The shared session state (a fresh one by default).
    """

    def __init__(self, make_pages: PageFactory, state: WizardState | None = None):
        super().__init__()
        load_fonts()
        self.setWindowTitle("croak")
        self.setWindowIcon(app_icon())
        settings.restore_window(self)
        self.state = state or WizardState()
        self.make_pages = make_pages
        self.pages: dict[Stage, StagePage] = {}
        self.stage = Stage.HOME
        self.session_path = ""

        central = QWidget()
        central.setObjectName("page")
        self.setCentralWidget(central)
        column = QVBoxLayout(central)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addWidget(self._make_top())
        middle = QHBoxLayout()
        middle.setContentsMargins(0, 0, 0, 0)
        middle.setSpacing(0)
        self.stack = QStackedWidget()
        self.home = Home()
        self.home.start.connect(self.new_session)
        self.home.open_session.connect(self.open_session)
        self.stack.addWidget(self.home)
        middle.addWidget(self.stack, stretch=1)
        middle.addWidget(self._make_drawer())
        column.addLayout(middle, stretch=1)
        self._make_menus()
        status = self.statusBar()
        status.hide()
        status.messageChanged.connect(lambda text: status.setVisible(bool(text)))
        self.setStyleSheet(stylesheet())
        for signal in (
            self.state.load_data_changed,
            self.state.tracedata_changed,
            self.state.result_changed,
            self.state.uncertainty_changed,
        ):
            signal.connect(self.refresh)
        self.show_home()

    # -- building -------------------------------------------------------------
    def _make_top(self) -> QFrame:
        top = frame("top")
        top.setFixedHeight(TOP_BAR_HEIGHT)
        bar = QHBoxLayout(top)
        bar.setContentsMargins(16, 0, 16, 0)
        bar.setSpacing(12)
        self.brand = button("croak", "brand")
        self.brand.setToolTip("Home")
        self.brand.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.brand.clicked.connect(self.go_home)
        self.context = label("", "context")
        bar.addWidget(self.brand)
        bar.addWidget(self.context)
        bar.addStretch(1)
        self.stage_buttons: dict[Stage, QPushButton] = {}
        for stage in STAGES:
            b = button(stage.title, "stage")
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.clicked.connect(lambda _=False, stage=stage: self.go(stage))
            bar.addWidget(b)
            self.stage_buttons[stage] = b
        bar.addStretch(1)
        self.primary = button("", "primary")
        self.primary.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.primary.clicked.connect(self.do_primary)
        bar.addWidget(self.primary)
        self.overflow = QToolButton()
        self.overflow.setObjectName("overflow")
        self.overflow.setText("⋯")
        self.overflow.setToolTip("More")
        self.overflow.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.overflow.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        bar.addWidget(self.overflow)
        return top

    def _make_drawer(self) -> QFrame:
        self.drawer = frame("drawer")
        self.drawer.setFixedWidth(DRAWER_WIDTH)
        layout = QVBoxLayout(self.drawer)
        layout.setContentsMargins(16, 12, 16, 12)
        head = QHBoxLayout()
        self.drawer_title = label("", "h3")
        copy = button("Copy")
        copy.clicked.connect(self.copy_drawer)
        close = button("Close")
        close.clicked.connect(self.drawer.hide)
        head.addWidget(self.drawer_title, stretch=1)
        head.addWidget(copy)
        head.addWidget(close)
        layout.addLayout(head)
        self.drawer_text = QPlainTextEdit()
        self.drawer_text.setObjectName("drawerText")
        self.drawer_text.setReadOnly(True)
        layout.addWidget(self.drawer_text, stretch=1)
        self.drawer.hide()
        return self.drawer

    def _action(
        self,
        text: str,
        slot: Callable[[], object],
        shortcut: QKeySequence | None = None,
    ) -> QAction:
        action = QAction(text, self)
        if shortcut is not None:
            action.setShortcut(shortcut)
        action.triggered.connect(lambda _=False: slot())
        return action

    def _make_menus(self) -> None:
        keys = QKeySequence.StandardKey
        self.actions_: dict[str, QAction] = {
            "home": self._action("Home", self.go_home),
            "open": self._action(
                "Open a session…",
                lambda: self.open_session(""),
                QKeySequence(keys.Open),
            ),
            "save": self._action(
                "Save session…", self.save_session, QKeySequence(keys.Save)
            ),
            "options": self._action(
                "Show the settings (options.toml)", self.show_options
            ),
            "python": self._action("Show as a Python script", self.show_python),
            "help": self._action(
                "Help for this stage", self.show_help, QKeySequence(keys.HelpContents)
            ),
        }
        overflow = QMenu(self)
        for key in ("save", "open"):
            overflow.addAction(self.actions_[key])
        overflow.addSeparator()
        for key in ("options", "python"):
            overflow.addAction(self.actions_[key])
        overflow.addSeparator()
        overflow.addAction(self.actions_["help"])
        self.overflow.setMenu(overflow)
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")
        for key in ("home", "open", "save"):
            file_menu.addAction(self.actions_[key])
        view_menu = bar.addMenu("&View")
        for key in ("options", "python"):
            view_menu.addAction(self.actions_[key])
        bar.addMenu("&Help").addAction(self.actions_["help"])

    def _build_pages(self) -> None:
        """Replace the stage pages with fresh ones bound to the state's parameters."""
        for page in self.pages.values():
            self.stack.removeWidget(page)
            page.deleteLater()
        self.pages = dict(self.make_pages(self.state))
        missing = set(STAGES) - set(self.pages)
        if missing:
            names = ", ".join(sorted(s.value for s in missing))
            raise ValueError(f"the page factory built no page for {names}")
        for page in self.pages.values():
            self.stack.addWidget(page)
            page.changed.connect(self.refresh)
            page.message.connect(self.notify)

    # -- the route ------------------------------------------------------------
    def progress(self) -> Progress:
        """What the session holds, for the route rules."""
        if not self.pages:
            return Progress()
        state = self.state
        data = self.pages[Stage.DATA]
        return Progress(
            entry=state.entry,
            data_ready=data.can_advance(),
            has_load_data=state.load_data is not None,
            has_tracedata=state.tracedata is not None,
            has_result=state.result is not None,
            raw_direct=state.entry == "simulated" and state.simulated.raw_direct,
            retrieving=self.pages[Stage.RETRIEVE].busy(),
            estimating=self.pages[Stage.ANALYSE].busy(),
        )

    def current_action(self) -> Action:
        """The primary action the top bar offers now."""
        page = self.pages.get(self.stage)
        key = page.page_key() if page is not None else ""
        return primary_action(self.stage, self.progress(), key)

    def refresh(self) -> None:
        """Re-apply the route rules to the top bar."""
        progress = self.progress()
        in_session = self.stage is not Stage.HOME or bool(self.pages)
        for stage, b in self.stage_buttons.items():
            reason = blocked_reason(stage, progress) if self.pages else "no session"
            b.setVisible(in_session)
            b.setEnabled(not reason)
            b.setToolTip(reason)
            set_flag(b, "current", stage is self.stage)
        action = self.current_action()
        self.primary.setVisible(bool(action.kind))
        self.primary.setText(action.text)
        self.primary.setEnabled(action.enabled)
        entry = _ENTRY_TITLES.get(progress.entry, "")
        name = Path(self.session_path).parent.name if self.session_path else ""
        self.context.setText("  ·  ".join(p for p in (entry, name) if p))
        self.actions_["save"].setEnabled(progress.has_result)
        for key in ("options", "python"):
            self.actions_[key].setEnabled(bool(progress.entry))

    def go(self, stage: Stage) -> bool:
        """Show ``stage`` if it can be opened; say why not otherwise."""
        if stage is Stage.HOME or not self.pages:
            reason = "" if stage is Stage.HOME else "no session"
        else:
            reason = blocked_reason(stage, self.progress())
        if reason:
            self.notify(reason)
            return False
        self.stage = stage
        if stage is Stage.HOME:
            self.home.show_recent(settings.recent_sessions())
            self.stack.setCurrentWidget(self.home)
        else:
            page = self.pages[stage]
            self.stack.setCurrentWidget(page)
            page.on_enter()
        self.refresh()
        return True

    def show_home(self) -> None:
        """Show Home, keeping any open session."""
        self.go(Stage.HOME)

    def go_home(self) -> None:
        """Show Home (the session stays open until a new one starts)."""
        self.show_home()

    def do_primary(self) -> None:
        """Do what the primary action says."""
        action = self.current_action()
        if not action.enabled:
            return
        if action.kind == "save":
            self.save_session()
            return
        page = self.pages[self.stage]
        page.act(action.kind)
        if action.kind == "continue":
            self.go(next_stage(self.stage, self.progress()))
        self.refresh()

    # -- sessions ---------------------------------------------------------------
    def confirm_discard(self) -> bool:
        """Ask before a new session replaces one holding data; True to go ahead."""
        if not self.state.has_data:
            return True
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Start a new session?")
        box.setText("The current session's trace and retrieval will be discarded.")
        box.setInformativeText("Save session… in the ⋯ menu keeps them.")
        discard = box.addButton("Discard", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        return box.clickedButton() is discard

    def new_session(self, entry: str) -> bool:
        """Start a session from ``entry`` and open its Data stage."""
        if not self.confirm_discard():
            return False
        self.state.reset()
        self.state.entry = entry
        self.session_path = ""
        self._build_pages()
        self.go(Stage.DATA)
        return True

    def open_session(self, path: str) -> bool:
        """Reopen the session saved at ``path`` (``""`` asks for the file)."""
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self, "Open a session", filter="Session settings (options.toml *.toml)"
            )
            if not path:
                return False
        if not self.confirm_discard():
            return False
        try:
            options = save.load_options(path)
        except (OSError, ValueError) as exc:
            self.notify(f"Could not open {path}: {exc}")
            return False
        self.state.reset()
        self.state.from_options(options)
        self.session_path = path
        settings.remember_session(path)
        self._build_pages()
        # TODO(gui-redesign): replay the data, preparation and retrieval (or
        # rehydrate result.h5) once the new stage pages expose them.
        self.go(Stage.DATA)
        return True

    def save_session(self) -> list[str]:
        """Ask for a folder and save the session there; return the files written."""
        if self.state.result is None:
            self.notify("Run a retrieval first: there is nothing to save yet.")
            return []
        folder = QFileDialog.getExistingDirectory(self, "Save the session in")
        if not folder:
            return []
        written = session_files.save_session(
            folder,
            self.state.session_options(),
            result=self.state.result,
            processed=self.state.processed,
            truth=self.state.truth,
            uncertainty=self.state.uncertainty_results,
        )
        self.session_path = str(Path(folder) / session_files.OPTIONS_FILE)
        settings.remember_session(self.session_path)
        self.notify(f"Saved {', '.join(written)} in {folder}")
        self.refresh()
        return written

    # -- the drawer and help ------------------------------------------------------
    def show_drawer(self, title: str, text: str) -> None:
        """Open the drawer on ``text``."""
        self.drawer_title.setText(title)
        self.drawer_text.setPlainText(text)
        self.drawer.show()

    def show_options(self) -> None:
        """Show the session's settings as the ``options.toml`` it would save."""
        self.show_drawer(
            "Settings (options.toml)", save.format_options(self.state.to_options())
        )

    def show_python(self) -> None:
        """Show a Python script that replays the session headless."""
        self.show_drawer(
            "As a Python script", generate_script(self.state.session_options())
        )

    def copy_drawer(self) -> None:
        """Put the drawer's text on the clipboard."""
        QApplication.clipboard().setText(self.drawer_text.toPlainText())
        self.notify("Copied.")

    def show_help(self) -> None:
        """Open the help for the stage on screen."""
        page = self.pages.get(self.stage)
        if page is None or page.show_help() is None:
            self.notify("Choose a stage to see its help.")

    def notify(self, text: str) -> None:
        """Show a passing notice in the status bar."""
        self.statusBar().showMessage(text, NOTICE_MS)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 — Qt override
        """Remember the window's geometry, then close."""
        settings.save_window(self)
        super().closeEvent(event)
