"""The workflow window: Home, then Data, Check, Prepare, Retrieve and Analyse.

The top bar carries the brand (back to Home), what the session is, the five
stage buttons, the primary action (with Retrieve again beside it once a
retrieval is done) and an overflow menu (⋯). Which stages open and what the
actions do are the pure rules of
:mod:`croak.gui.workflow`; this module applies them to widgets. A drawer on the
right shows the session's settings as ``options.toml`` or as a Python script.

The pages are built by a factory from the shared :class:`WizardState`, and rebuilt
for each new session, since their controls bind to the state's parameter objects.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, cast

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QCloseEvent, QIcon, QKeySequence
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
from ..session import retarget
from . import settings
from .about import About
from .branding import app_icon
from .display import DisplayBar
from .home import ENTRY_TILES, Home
from .stage import PageBase
from .state import WizardState
from .theme import (
    ASSETS,
    NOTICE_MS,
    button,
    frame,
    label,
    load_fonts,
    set_flag,
    stylesheet,
)
from .workflow import (
    STAGES,
    Action,
    Progress,
    Stage,
    blocked_reason,
    next_stage,
    primary_action,
    secondary_action,
)

if TYPE_CHECKING:
    from .pages import DataPage
    from .stage_preprocess import StagePreprocess
    from .stage_retrieve import StageRetrieve

__all__ = ["DRAWER_WIDTH", "MainWindow", "PageFactory"]

PageFactory = Callable[[WizardState], Mapping[Stage, PageBase]]
"""Builds a session's stage pages from the shared state."""

DRAWER_WIDTH = 440
"""The width of the settings drawer, logical px."""
TOP_BAR_HEIGHT = 52
"""The height of the top bar, logical px."""

_ENTRY_TITLES = {entry: title for entry, title, _ in ENTRY_TILES}


class MainWindow(QMainWindow):
    """The croak window.

    Parameters
    ----------
    make_pages : PageFactory, optional
        Builds the Data, Check, Prepare, Retrieve and Analyse pages for a session
        (:func:`croak.gui.pages.default_pages` by default).
    state : WizardState, optional
        The shared session state (a fresh one by default).
    """

    def __init__(
        self, make_pages: PageFactory | None = None, state: WizardState | None = None
    ):
        super().__init__()
        load_fonts()
        self.setWindowTitle("croak")
        self.setWindowIcon(app_icon())
        settings.restore_window(self)
        self.state = state or WizardState()
        if make_pages is None:
            from .pages import default_pages

            make_pages = default_pages
        self.make_pages = make_pages
        self.pages: dict[Stage, PageBase] = {}
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
        self.home.about.connect(self.show_about)
        self.stack.addWidget(self.home)
        self.about = About()
        self.stack.addWidget(self.about)
        middle.addWidget(self.stack, stretch=1)
        middle.addWidget(self._make_drawer())
        column.addLayout(middle, stretch=1)
        self._make_menus()
        # The foot of the window: notices on the left and, in a session, the
        # display settings every plot follows on the right.
        status = self.statusBar()
        self.display_bar = DisplayBar()
        status.addPermanentWidget(self.display_bar)
        status.hide()
        status.messageChanged.connect(lambda _text: self._show_status())
        self.setStyleSheet(stylesheet())
        self.state.retarget_requested.connect(self.retarget)
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
        # Home leads the stage row, where navigation is looked for; the brand
        # also goes Home, but nobody guesses that a name is a button.
        self.home_button = button("Home", "stage")
        self.home_button.setIcon(QIcon(str(ASSETS / "home.svg")))
        self.home_button.setToolTip("Home: start or reopen a session")
        self.home_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.home_button.clicked.connect(self.go_home)
        bar.addWidget(self.home_button)
        divider = frame("divider")
        divider.setFixedSize(1, 20)
        bar.addWidget(divider)
        self.stage_buttons: dict[Stage, QPushButton] = {}
        for stage in STAGES:
            b = button(stage.title, "stage")
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.clicked.connect(lambda _=False, stage=stage: self.go(stage))
            bar.addWidget(b)
            self.stage_buttons[stage] = b
        bar.addStretch(1)
        # Beside the primary action, the one step that is not the way on
        # (Retrieve again, once a retrieval is done).
        self.secondary = button("")
        self.secondary.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.secondary.clicked.connect(self.do_secondary)
        bar.addWidget(self.secondary)
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
            "figure": self._action(
                "Export this page as a figure…",
                self.export_figure,
                QKeySequence("Ctrl+E"),
            ),
            "help": self._action(
                "Help for this stage", self.show_help, QKeySequence(keys.HelpContents)
            ),
            "about": self._action("About croak", self.show_about),
        }
        # On macOS, Qt moves it to the application menu, where About belongs.
        self.actions_["about"].setMenuRole(QAction.MenuRole.AboutRole)
        overflow = QMenu(self)
        for key in ("save", "open", "figure"):
            overflow.addAction(self.actions_[key])
        overflow.addSeparator()
        for key in ("options", "python"):
            overflow.addAction(self.actions_[key])
        overflow.addSeparator()
        overflow.addAction(self.actions_["help"])
        overflow.addAction(self.actions_["about"])
        self.overflow.setMenu(overflow)
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")
        for key in ("home", "open", "save", "figure"):
            file_menu.addAction(self.actions_[key])
        view_menu = bar.addMenu("&View")
        for key in ("options", "python"):
            view_menu.addAction(self.actions_[key])
        help_menu = bar.addMenu("&Help")
        for key in ("help", "about"):
            help_menu.addAction(self.actions_[key])

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
        for stage, page in self.pages.items():
            self.stack.addWidget(page)
            page.changed.connect(self.refresh)
            page.message.connect(self.notify)
            page.moved_on.connect(lambda stage=stage: self.move_on(stage))

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
        in_session = bool(self.pages)
        for stage, b in self.stage_buttons.items():
            reason = blocked_reason(stage, progress) if self.pages else "no session"
            b.setVisible(in_session)
            b.setEnabled(not reason)
            b.setToolTip(reason)
            set_flag(b, "current", stage is self.stage)
        set_flag(self.home_button, "current", self.stage is Stage.HOME)
        action = self.current_action()
        self.primary.setVisible(bool(action.kind))
        self.primary.setText(action.text)
        self.primary.setEnabled(action.enabled)
        second = secondary_action(self.stage, progress)
        self.secondary.setVisible(bool(second.kind))
        self.secondary.setText(second.text)
        self.secondary.setEnabled(second.enabled)
        entry = _ENTRY_TITLES.get(progress.entry, "")
        name = Path(self.session_path).parent.name if self.session_path else ""
        self.context.setText("  ·  ".join(p for p in (entry, name) if p))
        self.actions_["save"].setEnabled(progress.has_result)
        for key in ("options", "python"):
            self.actions_[key].setEnabled(bool(progress.entry))
        self.actions_["figure"].setEnabled(self.stage in self.pages)
        self.display_bar.setVisible(in_session)
        self._show_status()

    def _show_status(self) -> None:
        """Show the foot of the window while it has a notice or a session."""
        status = self.statusBar()
        status.setVisible(bool(self.pages) or bool(status.currentMessage()))

    def go(self, stage: Stage) -> bool:
        """Show ``stage`` if it can be opened; say why not otherwise."""
        outside = stage in (Stage.HOME, Stage.ABOUT)
        if outside or not self.pages:
            reason = "" if outside else "no session"
        else:
            reason = blocked_reason(stage, self.progress())
        if reason:
            self.notify(reason)
            return False
        self.stage = stage
        if stage is Stage.HOME:
            self.home.show_recent(settings.recent_sessions())
            self.stack.setCurrentWidget(self.home)
        elif stage is Stage.ABOUT:
            self.stack.setCurrentWidget(self.about)
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

    def export_figure(self) -> str:
        """Save the page on screen as a publication figure; return the path."""
        page = self.pages.get(self.stage)
        return page.export_figure() if page is not None else ""

    def show_about(self) -> None:
        """Show About, keeping any open session."""
        self.go(Stage.ABOUT)

    def do_primary(self) -> None:
        """Do what the primary action says."""
        action = self.current_action()
        if not action.enabled:
            return
        if action.kind == "save":
            self.save_session()
            return
        page = self.pages[self.stage]
        if page.act(action.kind) and action.kind == "continue":
            self.go(next_stage(self.stage, self.progress()))
        self.refresh()

    def do_secondary(self) -> None:
        """Do what the action beside the primary one says."""
        action = secondary_action(self.stage, self.progress())
        if not action.enabled:
            return
        self.pages[self.stage].act(action.kind)
        self.refresh()

    def move_on(self, stage: Stage) -> None:
        """Go to the stage after ``stage``, whose page handed its work on."""
        self.go(next_stage(stage, self.progress()))

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
        self.go(Stage.DATA)
        result = Path(path).with_name(session_files.RESULT_FILE)
        self.replay(str(result) if result.is_file() else "")
        return True

    def replay(self, result_path: str = "") -> None:
        """Bring a reopened session back: its trace, preparation and retrieval.

        The trace is reloaded with the saved options and prepared with the saved
        settings. The retrieval is then taken from ``result_path`` (a sibling
        ``result.h5``) when it was fitted to exactly this trace
        (:func:`croak.session.files.result_mismatch`), and re-run otherwise. A
        synthetic session cannot be replayed: its generator's settings are not
        part of the saved options.
        """
        state = self.state
        data = cast("DataPage", self.pages[Stage.DATA])
        if state.entry == "synthetic":
            data.entry_page.set_status(
                "Options loaded, but the synthetic generator's settings are not "
                "saved with a session — regenerate the trace to replay the rest.",
                "warn",
            )
            return
        if not data.restore():
            return
        prepare = cast("StagePreprocess", self.pages[Stage.PREPARE])
        if state.tracedata is None:
            prepare._seed_all()
            prepare.apply(block=True)
        if state.tracedata is None:
            self.go(Stage.PREPARE)
            prepare.set_status(
                "Options loaded and data restored; preprocess did not complete.",
                "warn",
            )
            return
        retrieve = cast("StageRetrieve", self.pages[Stage.RETRIEVE])
        # Adopt the saved result before Retrieve is shown: entering it draws
        # (and processes) whatever result the session holds.
        note = self._rehydrate(result_path) if result_path else "no saved result"
        self.go(Stage.RETRIEVE)
        if not note:
            retrieve.set_status(
                "Options and saved retrieval loaded — no re-run needed."
            )
            return
        if result_path:
            retrieve.set_status(f"Options loaded; re-running because {note}.")
        retrieve._run()

    def _rehydrate(self, result_path: str) -> str:
        """Adopt the saved result at ``result_path``; return why not, or ``""``."""
        tracedata = self.state.tracedata
        try:
            saved = save.load_result(result_path)
            result = save.result_from_saved(saved, grid=tracedata.grid)  # pyright: ignore[reportOptionalMemberAccess]
        except (KeyError, ValueError, OSError) as exc:
            return f"the saved result could not be read ({exc})"
        reason = session_files.result_mismatch(
            saved,
            tracedata,  # pyright: ignore[reportArgumentType]
            self.state.retrieve,
        )
        if not reason:
            self.state.set_result(result)
        return reason

    def retarget(self, folder: str) -> None:
        """Point the session at the dataset in ``folder`` and replay it.

        The dataset-folder paths are rewritten (the shared calibration files
        stay) by :func:`croak.session.retarget`; only a measured session has
        such a folder, and for the others the reason is shown instead.
        """
        try:
            moved = retarget(self.state.session_options(), folder)
        except ValueError as exc:
            self.notify(str(exc))
            return
        for field in self.state.load.dataset_path_fields():
            setattr(self.state.load, field, getattr(moved.load, field))
        self._build_pages()
        self.go(Stage.DATA)
        self.replay()

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
            overview=self.pages[Stage.RETRIEVE].overview(),
        )
        # What Analyse's pages add: the dispersed pulse, the uncertainty figure.
        written += self.pages[Stage.ANALYSE].save_extras(Path(folder))
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
