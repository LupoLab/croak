"""The window with its real pages: Data's steps, reopening and replaying sessions.

The end-to-end flows live in ``test_wizard_flow.py``; these target the window's
own branches: Data's two steps, the session replay's early exits, the saved
result's reuse and retargeting.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from gui_session import Driver
from PySide6.QtWidgets import QFileDialog, QMessageBox

from croak import save
from croak.gui import pages, stage_preprocess
from croak.gui.state import WizardState
from croak.gui.window import MainWindow
from croak.gui.workflow import Stage


@pytest.fixture
def win(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    return window


def _point_at_experiment(state, experiment_h5, experiment):
    p = state.load
    p.frog_path = experiment_h5
    p.ifrog_name, p.lam_name, p.scan_name = "trace", "wavelength", "delay"
    p.lam_unit, p.scan_unit, p.scan_type = "nm", "fs", "delay"
    p.interaction = experiment.interaction


# -- Data and Check --------------------------------------------------------------
def test_window_data_waits_for_a_loaded_trace_then_checks_it(
    qtbot, experiment_h5, experiment
):
    d = Driver(qtbot, "experimental")
    win = d.win
    assert not win.primary.isEnabled()  # nothing loaded yet
    assert not win.stage_buttons[Stage.CHECK].isEnabled()
    _point_at_experiment(win.state, experiment_h5, experiment)
    d.stages[0]._load_preview()
    assert win.primary.isEnabled()
    assert win.stage_buttons[Stage.CHECK].isEnabled()  # no Continue needed
    win.do_primary()  # a measured trace: on to the check
    assert win.stage is Stage.CHECK
    assert d.current_page() is win.pages[Stage.CHECK]
    win.do_primary()  # from the check, on to Prepare
    assert win.stage is Stage.PREPARE
    assert win.state.tracedata is not None


def test_window_data_and_check_answer_for_their_own_page(qtbot):
    d = Driver(qtbot, "synthetic")
    d.data.show_help()  # the generator has no help of its own: nothing opens
    with pytest.raises(ValueError, match="no 'retrieve' action"):
        d.data.act("retrieve")
    d.win.do_primary()  # generated: the check opens
    assert d.win.stage is Stage.CHECK
    dialog = d.win.pages[Stage.CHECK].show_help()
    assert dialog is not None
    dialog.close()
    assert d.win.go(Stage.DATA)  # back to the generator, the trace kept
    assert d.state.load_data is not None


def test_window_synthetic_continue_generates_and_checks(qtbot):
    d = Driver(qtbot, "synthetic")
    assert d.win.primary.isEnabled()  # a synthetic pulse is ready at once
    d.win.do_primary()
    assert d.state.load_data is not None
    assert d.win.stage is Stage.CHECK


def test_window_synthetic_continue_stays_when_generation_fails(qtbot, monkeypatch):
    d = Driver(qtbot, "synthetic")

    def fail():
        raise ValueError("bad grid")

    monkeypatch.setattr(d.synthetic, "_pulse", fail)
    d.win.do_primary()
    assert d.win.stage is Stage.DATA and d.state.load_data is None


def test_window_simulated_waits_for_a_previewed_scan(qtbot, simulated_scan_h5):
    d = Driver(qtbot, "simulated")
    assert not d.win.primary.isEnabled()
    d.win.do_primary()  # disabled: nothing happens
    assert d.win.stage is Stage.DATA
    d.simulated.path_edit.setText(simulated_scan_h5)
    d.simulated._populate_file_options(simulated_scan_h5)
    d.simulated._preview()
    assert d.win.primary.isEnabled()
    d.simulated.raw_direct.setChecked(True)
    d.win.do_primary()  # raw: straight to Retrieve, Prepare skipped
    assert d.win.stage is Stage.RETRIEVE
    assert not d.win.stage_buttons[Stage.PREPARE].isEnabled()


def test_window_analyse_rail_switches_pages(qtbot):
    d = Driver(qtbot, "synthetic")
    analyse = d.analyse
    analyse.rail_items["uncertainty"].click()
    assert analyse.page_key() == "uncertainty"
    assert analyse.rail_items["uncertainty"].property("current") is True
    assert not analyse.busy()
    analyse.on_enter()
    dialog = analyse.show_help()
    assert dialog is not None
    dialog.close()
    with pytest.raises(ValueError, match="needs at least one page"):
        pages.AnalysePage(WizardState(), [])


def test_window_analyse_rail_steps_with_the_arrow_keys(qtbot):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeySequence, QShortcut

    d = Driver(qtbot, "synthetic")
    analyse = d.analyse
    shortcuts = {s.key().toString(): s for s in analyse.findChildren(QShortcut)}
    up = shortcuts[QKeySequence(Qt.Key.Key_Up).toString()]
    down = shortcuts[QKeySequence(Qt.Key.Key_Down).toString()]
    assert up.context() == Qt.ShortcutContext.WidgetWithChildrenShortcut
    assert analyse.page_key() == "overview"
    up.activated.emit()  # at the top: stays
    assert analyse.page_key() == "overview"
    down.activated.emit()
    assert analyse.page_key() == "traces"
    analyse.show_page("uncertainty")
    analyse.step(5)  # past the end: the last page
    assert analyse.page_key() == "uncertainty"


def test_window_new_session_asks_before_discarding_a_loaded_trace(
    win, monkeypatch, experiment_h5, experiment
):
    win.new_session("experimental")
    _point_at_experiment(win.state, experiment_h5, experiment)
    win.pages[Stage.DATA].entry_page._load_preview()

    def answer(label):
        def exec_(box):
            next(b for b in box.buttons() if b.text() == label).click()
            return 0

        monkeypatch.setattr(QMessageBox, "exec", exec_)

    answer("Cancel")
    assert not win.new_session("synthetic")
    assert win.state.load_data is not None
    answer("Discard")
    assert win.new_session("synthetic")
    assert win.state.load_data is None and win.state.entry == "synthetic"


# -- reopening a session ---------------------------------------------------------
def test_window_open_session_cancelled_leaves_the_session(win, monkeypatch):
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: ("", ""))
    win.state.retrieve.maxiters = 7
    assert not win.open_session("")
    assert win.state.retrieve.maxiters == 7


def test_window_reopens_a_synthetic_session_without_replaying(win, tmp_path):
    src = WizardState()
    src.entry = "synthetic"
    src.retrieve.maxiters = 77
    opts = tmp_path / "options.toml"
    save.save_options(src.to_options(), str(opts))
    (tmp_path / "result.h5").write_bytes(b"")  # a sibling result is looked for
    assert win.open_session(str(opts))
    assert win.state.retrieve.maxiters == 77
    data = win.pages[Stage.DATA]
    assert "synthetic generator's settings are not saved" in (
        data.entry_page.status_label.text()
    )


def test_window_replay_needs_a_trace_file(qtbot):
    d = Driver(qtbot, "experimental")
    d.win.replay()
    assert d.stage_number == 1
    assert d.stages[0].status_label.text() == (
        "Options loaded. Choose a trace file and run the stages."
    )


def test_window_replay_reports_an_unreadable_trace(qtbot, tmp_path):
    d = Driver(qtbot, "experimental")
    d.state.load.frog_path = str(tmp_path / "moved.h5")
    d.win.replay()
    assert d.stage_number == 1
    assert (
        d.stages[0]
        .status_label.text()
        .startswith("Options loaded, but reloading the trace failed:")
    )
    assert d.state.load_data is None


def test_window_replay_stops_when_preprocess_fails(
    qtbot, monkeypatch, experiment_h5, experiment
):
    def broken(params, data, **kwargs):
        raise RuntimeError("bad window")

    monkeypatch.setattr(stage_preprocess.pipeline, "build_tracedata", broken)
    d = Driver(qtbot, "experimental")
    _point_at_experiment(d.state, experiment_h5, experiment)
    d.win.replay()
    assert d.state.load_data is not None and d.state.tracedata is None
    assert d.stage_number == 3
    assert d.stages[2].status_label.text() == (
        "Options loaded and data restored; preprocess did not complete."
    )


def test_window_replay_needs_a_simulated_scan_file(qtbot):
    d = Driver(qtbot, "simulated")
    d.win.replay()
    assert d.current_page() is d.simulated
    assert d.simulated.status_label.text() == (
        "Options loaded. Choose a simulated scan file and run the stages."
    )


def test_window_replay_reports_an_unreadable_scan(qtbot, tmp_path):
    d = Driver(qtbot, "simulated")
    d.state.simulated.frog_path = str(tmp_path / "moved.h5")
    d.win.replay()
    assert d.simulated.status_label.text().startswith(
        "Options loaded, but reloading the simulated scan failed:"
    )


def test_window_replay_reports_a_failed_raw_trace(
    qtbot, monkeypatch, simulated_scan_h5
):
    def broken(params):
        raise ValueError("no native grid")

    monkeypatch.setattr(pages, "assemble_simulated_tracedata", broken)
    d = Driver(qtbot, "simulated")
    d.state.simulated.frog_path = simulated_scan_h5
    d.state.simulated.raw_direct = True
    d.win.replay()
    assert d.simulated.status_label.text() == (
        "Options loaded, but the raw simulated trace failed: no native grid"
    )
    assert d.state.tracedata is None


def test_window_replay_of_a_raw_simulated_scan_retrieves(
    qtbot, monkeypatch, simulated_scan_h5
):
    d = Driver(qtbot, "simulated")
    runs = []
    monkeypatch.setattr(d.stages[3], "_run", lambda: runs.append(1))
    d.state.simulated.frog_path = simulated_scan_h5
    d.state.simulated.raw_direct = True
    d.win.replay()
    assert d.state.tracedata is not None  # built on the native grid, no Prepare
    assert d.stage_number == 4 and runs == [1]


def test_window_rehydrate_refuses_an_unreadable_result(qtbot, tmp_path):
    d = Driver(qtbot, "synthetic")
    d.synthetic.advance()
    d.goto(3)  # prepares the trace
    reason = d.win._rehydrate(str(tmp_path / "missing.h5"))
    assert reason.startswith("the saved result could not be read")
    assert d.state.result is None


def test_window_retarget_needs_a_measured_session(qtbot, tmp_path):
    d = Driver(qtbot, "synthetic")
    d.state.retarget_requested.emit(str(tmp_path))
    assert d.win.statusBar().currentMessage()  # the reason is shown
    assert d.state.entry == "synthetic"


def test_window_save_writes_the_overview_too(qtbot, monkeypatch, tmp_path):
    d = Driver(qtbot, "synthetic")
    d.synthetic.advance()
    d.goto(3)
    retrieve = d.stages[3]
    assert retrieve.overview() is None  # nothing retrieved yet
    d.state.retrieve.solver = "copra"
    d.state.retrieve.maxiters = 3
    retrieve._seed_widgets()
    with qtbot.waitSignal(d.state.result_changed, timeout=120_000):
        d.goto(4)
        retrieve._run()
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path)
    )
    written = d.win.save_session()
    assert written == ["result.h5", "options.toml", "retrieval.pdf"]
    assert np.isfinite(d.state.result.error)


def test_window_primary_actions_reach_retrieve_and_uncertainty(qtbot, monkeypatch):
    d = Driver(qtbot, "synthetic")
    retrieve, uncertainty = d.stages[3], d.stages[5]
    calls = []
    for page, name in ((retrieve, "retrieve"), (uncertainty, "uncertainty")):
        monkeypatch.setattr(page, "_run", lambda name=name: calls.append(f"{name} run"))
        monkeypatch.setattr(
            page, "_stop", lambda name=name: calls.append(f"{name} stop")
        )
    # Running work never moves the window on; Continue does, by default.
    assert retrieve.act("retrieve") is False
    assert retrieve.act("stop") is False
    assert retrieve.act("continue") is True
    d.analyse.show_page("uncertainty")
    assert d.analyse.act("estimate") is False
    assert d.analyse.act("stop-estimate") is False
    assert d.analyse.act("continue") is True
    assert calls == [
        "retrieve run",
        "retrieve stop",
        "uncertainty run",
        "uncertainty stop",
    ]


def test_window_figures_come_from_the_page_on_screen(qtbot):
    d = Driver(qtbot, "synthetic")
    d.data.on_enter()  # the generator draws its four plots
    assert d.data.figure_name() == "synthetic_trace"
    figure = d.data.figure()
    assert figure is not None
    titles = {ax.get_title(loc="left") for ax in figure.axes} - {""}
    assert titles == {"Temporal", "Spectral", "Spectrogram", "Synthetic FROG trace"}
    d.analyse.show_page("dispersion")
    assert d.analyse.figure_name() == "dispersion"
    assert d.analyse.figure() is None  # nothing retrieved: nothing drawn
    d.analyse.show_page("traces")
    assert d.analyse.figure_name() == "traces"
