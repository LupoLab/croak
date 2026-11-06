"""The workflow window, Home and the stage-page base, driven with stub pages."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QWidget

from croak import save
from croak.gui import settings
from croak.gui.home import Home, describe_session
from croak.gui.stage import StagePage
from croak.gui.state import WizardState
from croak.gui.window import MainWindow
from croak.gui.workflow import STAGES, Stage
from croak.session.options import SessionOptions


class StubPage(StagePage):
    """A stage page whose hooks the tests set."""

    HELP_TITLE = "Stub"
    HELP = [("Section", [("Row", "Help text.")])]

    def __init__(self, state, stage):
        super().__init__(state)
        self.stage = stage
        self.ready = True
        self.running = False
        self.key = ""
        self.acted: list[str] = []
        self.entered = 0

    def on_enter(self):
        self.entered += 1

    def can_advance(self):
        return self.ready

    def busy(self):
        return self.running

    def page_key(self):
        return self.key

    def act(self, kind):
        self.acted.append(kind)
        if self.stage is Stage.DATA and kind == "continue":
            self.state.set_load_data({"trace": "loaded"})
        return kind == "continue"

    def restore(self):
        """Data's session replay: the stub has no trace to reload."""
        self.acted.append("restore")
        return False


def stub_pages(state):
    return {stage: StubPage(state, stage) for stage in STAGES}


@pytest.fixture
def window(qtbot, monkeypatch):
    # A dialog would block the test; each test says what the user answers.
    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)
    win = MainWindow(stub_pages)
    qtbot.addWidget(win)
    return win


def _choose(monkeypatch, answer: str):
    """Make the discard confirmation answer ``answer`` ("Discard" or "Cancel")."""

    def clicked(box):
        for b in box.buttons():
            if answer in b.text():
                return b
        return None

    monkeypatch.setattr(QMessageBox, "clickedButton", clicked)


# -- Home -------------------------------------------------------------------------
def test_window_opens_on_home_with_no_session(window):
    assert window.stage is Stage.HOME
    assert window.stack.currentWidget() is window.home
    assert not window.primary.isVisible()
    assert not any(b.isVisibleTo(window) for b in window.stage_buttons.values())
    assert not window.go(Stage.DATA)  # no session to open
    assert window.statusBar().currentMessage() == "no session"
    assert not window.actions_["save"].isEnabled()


def test_window_home_tile_starts_a_session_at_data(window, qtbot):
    tile = window.home.tiles["synthetic"]
    press = QMouseEvent(
        QMouseEvent.Type.MouseButtonRelease,
        QPointF(5, 5),
        QPointF(5, 5),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    tile.mouseReleaseEvent(press)
    assert window.stage is Stage.DATA
    assert window.state.entry == "synthetic"
    assert window.context.text() == "Synthetic trace"
    assert window.pages[Stage.DATA].entered == 1
    assert window.stage_buttons[Stage.DATA].property("current") is True
    assert not window.stage_buttons[Stage.PREPARE].isEnabled()
    assert "Load a trace" in window.stage_buttons[Stage.PREPARE].toolTip()


def test_window_home_tile_ignores_a_right_click_or_a_release_outside(qtbot):
    home = Home()
    qtbot.addWidget(home)
    started = []
    home.start.connect(started.append)
    tile = home.tiles["experimental"]
    for button, point in (
        (Qt.MouseButton.RightButton, QPointF(5, 5)),
        (Qt.MouseButton.LeftButton, QPointF(-50, -50)),
    ):
        tile.mouseReleaseEvent(
            QMouseEvent(
                QMouseEvent.Type.MouseButtonRelease,
                point,
                point,
                button,
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )
    assert started == []


def test_window_home_lists_recent_sessions_that_still_exist(qtbot, tmp_path):
    saved = tmp_path / "run-7" / "options.toml"
    saved.parent.mkdir()
    saved.write_text("")
    home = Home()
    qtbot.addWidget(home)
    opened = []
    home.open_session.connect(opened.append)
    home.show_recent([str(tmp_path / "gone" / "options.toml"), str(saved)])
    assert len(home.recent_items) == 1
    title, detail = describe_session(saved)
    assert title == "run-7" and str(tmp_path) in detail
    home.recent_items[0].clicked.emit()
    assert opened == [str(saved)]
    home.open_button.click()
    assert opened[-1] == ""
    shown = list(home.recent_items)
    home.show_recent([])  # back to the hint
    assert home.recent_items == []
    assert all(old.isHidden() for old in shown)  # not left drawn until deleted


# -- the route ----------------------------------------------------------------------
def test_window_primary_action_continues_through_the_stages(window):
    window.new_session("experimental")
    data = window.pages[Stage.DATA]
    data.ready = False
    window.refresh()
    assert window.primary.text() == "Continue" and not window.primary.isEnabled()
    window.do_primary()  # disabled: does nothing
    assert data.acted == []
    data.ready = True
    window.state.set_load_data({"trace": "loaded"})
    window.do_primary()
    assert data.acted == ["continue"]
    assert window.stage is Stage.CHECK
    window.do_primary()  # the check passes the trace on
    assert window.stage is Stage.PREPARE
    # Prepare cannot continue until a trace is prepared.
    assert not window.primary.isEnabled()
    window.state.set_tracedata("prepared")
    assert window.primary.isEnabled()
    window.do_primary()
    assert window.stage is Stage.RETRIEVE
    assert window.primary.text() == "Retrieve"


def test_window_follows_a_page_that_moves_on_by_itself(window):
    window.new_session("synthetic")
    window.state.set_load_data({"trace": "generated"})
    window.pages[Stage.DATA].moved_on.emit()  # e.g. the generator's own Generate
    assert window.stage is Stage.CHECK


def test_window_raw_simulated_scan_continues_straight_to_retrieve(window):
    window.new_session("simulated")
    window.state.simulated.raw_direct = True
    data = window.pages[Stage.DATA]

    def raw_continue(kind):
        window.state.set_load_data({"trace": "raw"})
        window.state.set_tracedata("native grid")
        return True

    data.act = raw_continue
    window.do_primary()
    assert window.stage is Stage.RETRIEVE
    assert not window.stage_buttons[Stage.PREPARE].isEnabled()


def test_window_retrieve_and_analyse_actions_reach_their_pages(window):
    window.new_session("synthetic")
    window.state.set_load_data({"trace": 1})
    window.state.set_tracedata("td")
    assert window.go(Stage.RETRIEVE)
    retrieve = window.pages[Stage.RETRIEVE]
    window.do_primary()
    assert retrieve.acted == ["retrieve"]
    retrieve.running = True
    window.refresh()
    assert window.primary.text() == "Stop"
    window.do_primary()
    assert retrieve.acted[-1] == "stop"
    retrieve.running = False
    window.state.set_result("result")
    assert window.primary.text() == "Retrieve again"
    assert window.go(Stage.ANALYSE)
    analyse = window.pages[Stage.ANALYSE]
    assert window.primary.text() == "Save"
    analyse.key = "uncertainty"
    window.refresh()
    assert window.primary.text() == "Estimate"
    window.do_primary()
    assert analyse.acted == ["estimate"]
    # The stage buttons reach a finished stage directly.
    window.stage_buttons[Stage.DATA].click()
    assert window.stage is Stage.DATA


def test_window_brand_returns_home_keeping_the_session(window):
    window.new_session("synthetic")
    window.brand.click()
    assert window.stage is Stage.HOME
    assert window.pages  # still open: the stages stay reachable
    assert window.stage_buttons[Stage.DATA].isVisibleTo(window)
    assert window.go(Stage.DATA)


def test_window_home_button_returns_home_and_marks_it(window):
    assert window.home_button.property("current") is True
    window.new_session("synthetic")
    assert window.home_button.property("current") is False
    window.home_button.click()
    assert window.stage is Stage.HOME and window.pages
    assert window.home_button.property("current") is True


@pytest.mark.parametrize("route", ["menu", "home"])
def test_window_about_opens_with_or_without_a_session(window, route):
    if route == "menu":
        window.actions_["about"].trigger()
    else:
        window.home.about_button.click()
    assert window.stage is Stage.ABOUT
    assert window.stack.currentWidget() is window.about
    assert not window.primary.isVisible()
    assert not any(b.isVisibleTo(window) for b in window.stage_buttons.values())
    window.new_session("synthetic")
    window.show_about()  # a session stays open behind About
    assert window.stage is Stage.ABOUT
    assert window.go(Stage.DATA)


def test_window_exports_the_page_on_screen_as_a_figure(window, monkeypatch):
    assert not window.actions_["figure"].isEnabled()  # Home has no figure
    assert window.export_figure() == ""
    window.new_session("synthetic")
    assert window.actions_["figure"].isEnabled()
    page = window.pages[Stage.DATA]
    monkeypatch.setattr(page, "export_figure", lambda: "data.pdf")
    assert window.export_figure() == "data.pdf"
    assert window.actions_["figure"].shortcut().toString() == "Ctrl+E"


def test_window_new_session_asks_before_discarding_data(window, monkeypatch):
    window.new_session("experimental")
    window.state.set_load_data({"trace": 1})
    _choose(monkeypatch, "Cancel")
    assert not window.new_session("synthetic")
    assert window.state.entry == "experimental"
    _choose(monkeypatch, "Discard")
    assert window.new_session("synthetic")
    assert window.state.load_data is None


def test_window_rejects_a_factory_missing_a_stage(qtbot):
    def partial(state):
        return {Stage.DATA: StubPage(state, Stage.DATA)}

    win = MainWindow(partial)
    qtbot.addWidget(win)
    with pytest.raises(
        ValueError, match="no page for analyse, check, prepare, retrieve"
    ):
        win.new_session("synthetic")


def test_stage_page_base_hooks_have_defaults(qtbot):
    page = StagePage(WizardState())
    qtbot.addWidget(page)
    page.on_enter()
    assert page.can_advance() and not page.busy() and page.page_key() == ""
    page.act("continue")  # nothing to do before moving on
    with pytest.raises(ValueError, match="no 'retrieve' action"):
        page.act("retrieve")
    assert page.show_help() is None  # no help written
    assert page.derived.isHidden()
    page.add_derived("FWHM")
    assert not page.derived.isHidden()
    plots = QWidget()
    page.set_plots(plots)
    assert plots.parent() is not None


# -- sessions, the drawer and help ------------------------------------------
def test_window_opens_a_saved_session_and_remembers_it(window, tmp_path, monkeypatch):
    options = SessionOptions(entry="simulated")
    path = tmp_path / "run" / "options.toml"
    path.parent.mkdir()
    save.save_options(options.to_dict(), str(path))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(path), ""))
    assert window.open_session("")
    assert window.state.entry == "simulated"
    assert window.stage is Stage.DATA
    assert window.pages[Stage.DATA].acted == ["restore"]  # replay was asked
    assert window.context.text() == "Simulated scan  ·  run"
    assert settings.recent_sessions() == [str(path)]
    window.go_home()
    assert len(window.home.recent_items) == 1


def test_window_open_session_cancelled_or_unreadable(window, tmp_path, monkeypatch):
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: ("", ""))
    assert not window.open_session("")
    bad = tmp_path / "options.toml"
    bad.write_text("not = [toml")
    assert not window.open_session(str(bad))
    assert "Could not open" in window.statusBar().currentMessage()
    window.new_session("experimental")
    window.state.set_load_data({"trace": 1})
    _choose(monkeypatch, "Cancel")
    assert not window.open_session(str(bad))


def test_window_saves_a_session_with_a_result(window, tmp_path, monkeypatch):
    window.new_session("synthetic")
    assert window.save_session() == []  # nothing retrieved yet
    assert "nothing to save" in window.statusBar().currentMessage()
    window.state.result = "result"
    window.refresh()
    assert window.actions_["save"].isEnabled()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: "")
    assert window.save_session() == []  # cancelled
    calls = {}

    def fake_save(folder, options, **kwargs):
        calls.update(folder=folder, **kwargs)
        return ["result.h5", "options.toml"]

    monkeypatch.setattr("croak.session.files.save_session", fake_save)
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path)
    )
    assert window.save_session() == ["result.h5", "options.toml"]
    assert calls["folder"] == str(tmp_path) and calls["result"] == "result"
    assert settings.recent_sessions() == [str(tmp_path / "options.toml")]
    # Save is also the primary action on Analyse.
    window.state.set_load_data({"trace": 1})
    window.state.tracedata = "td"
    window.state.result = "result"
    assert window.go(Stage.ANALYSE)
    window.do_primary()
    assert "Saved result.h5, options.toml" in window.statusBar().currentMessage()


def test_window_drawer_shows_the_settings_and_the_script(window, qtbot):
    window.new_session("experimental")
    window.actions_["options"].trigger()
    assert not window.drawer.isHidden()
    assert "[retrieve]" in window.drawer_text.toPlainText()
    window.actions_["python"].trigger()
    assert "import croak" in window.drawer_text.toPlainText()
    window.copy_drawer()
    assert QApplication.clipboard().text() == window.drawer_text.toPlainText()
    window.drawer.hide()


def test_window_help_opens_for_the_stage_on_screen(window):
    window.show_help()  # on Home: nothing to explain
    assert "Choose a stage" in window.statusBar().currentMessage()
    window.new_session("experimental")
    window.actions_["help"].trigger()
    dialog = window.pages[Stage.DATA]._help_dialog
    assert dialog is not None and dialog.isVisible()
    window.show_help()  # raised again, not rebuilt
    assert window.pages[Stage.DATA]._help_dialog is dialog
    dialog.close()


def test_window_status_bar_hides_when_empty(window, qtbot):
    window.show()
    qtbot.waitExposed(window)
    window.notify("hello")
    assert window.statusBar().isVisible()
    window.statusBar().clearMessage()
    assert not window.statusBar().isVisible()


def test_window_remembers_its_geometry_on_close(window, qtbot):
    window.resize(900, 700)
    window.close()
    assert settings.user_settings().value(settings.KEY_WINDOW_GEOMETRY) is not None


# -- recent sessions ----------------------------------------------------------------
def test_settings_recent_sessions_survive_and_reject_garbage():
    assert settings.recent_sessions() == []
    settings.remember_session("/a/options.toml")
    settings.remember_session("/b/options.toml")
    assert settings.recent_sessions() == ["/b/options.toml", "/a/options.toml"]
    settings.set_text(settings.KEY_RECENT_SESSIONS, "{not json")
    assert settings.recent_sessions() == []
    settings.set_text(settings.KEY_RECENT_SESSIONS, '{"a": 1}')
    assert settings.recent_sessions() == []
    settings.set_text(settings.KEY_RECENT_SESSIONS, '["/x", 3]')
    assert settings.recent_sessions() == ["/x"]


def test_home_describes_a_session_by_folder_and_time(tmp_path):
    path = tmp_path / "scan-12" / "options.toml"
    path.parent.mkdir()
    path.write_text("")
    stamp = time.mktime((2026, 3, 4, 5, 6, 0, 0, 0, -1))
    os.utime(path, (stamp, stamp))
    title, detail = describe_session(Path(path))
    assert title == "scan-12"
    assert detail.endswith("04 Mar 2026, 05:06")
