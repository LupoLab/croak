"""Tests for the Load stage (:mod:`croak.gui.stage_load`).

The stage is driven directly on a :class:`WizardState`, with file dialogs
monkeypatched so no modal window ever opens.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QFileDialog, QPushButton

from croak.gui import stage_load
from croak.gui.stage_load import StageLoad
from croak.gui.state import WizardState


@pytest.fixture
def stage(qtbot):
    s = StageLoad(WizardState())
    qtbot.addWidget(s)
    return s


def _write_spectrum_h5(path, lam_nm):
    """An independent-spectrum file: a wavelength axis and one intensity."""
    with h5py.File(path, "w") as f:
        f["wavelength"] = lam_nm
        f["trace"] = np.exp(-(((lam_nm - 800.0) / 30.0) ** 2))
    return str(path)


def _write_csv(path, lam, intensity):
    np.savetxt(path, np.column_stack([lam, intensity]), delimiter=",")
    return str(path)


def _point_at(stage, experiment_h5, experiment):
    p = stage.state.load
    p.frog_path = experiment_h5
    p.ifrog_name, p.lam_name, p.scan_name = "trace", "wavelength", "delay"
    p.lam_unit, p.scan_unit, p.scan_type = "nm", "fs", "delay"
    p.interaction = experiment.interaction
    return p


def _file_buttons(stage):
    """The five "…" browse buttons, in layout order.

    trace background, trace calibration, spectrum, spectrum background,
    spectrum calibration.
    """
    return [b for b in stage.findChildren(QPushButton) if b.text() == "…"]


# -- on_enter: repopulating combos after a session restore ------------------
def test_stage_load_on_enter_seeds_combos_from_saved_names(stage, experiment_h5):
    p = stage.state.load
    p.frog_path = experiment_h5
    p.ifrog_name, p.lam_name = "trace", "wavelength"
    p.scan_name = "not-in-file"  # falls back to the first dataset
    stage.on_enter()
    assert stage.path_edit.text() == experiment_h5
    assert stage.ifrog_combo.currentText().startswith("trace")
    assert stage.lam_combo.currentText().startswith("wavelength")
    assert stage.scan_combo.currentIndex() == 0
    # a second entry with the file already listed does not re-read it
    fd = stage._frog_fd
    stage.on_enter()
    assert stage._frog_fd is fd


def test_stage_load_on_enter_ignores_unreadable_saved_files(stage, tmp_path):
    p = stage.state.load
    missing = str(tmp_path / "gone.h5")
    p.frog_path = missing
    p.use_spec, p.spec_path, p.spec_is_csv = True, missing, False
    p.frog_bg_use, p.frog_bg_path = True, missing
    stage.on_enter()
    assert stage._frog_fd is None
    assert stage._spec_fd is None
    assert stage._bg_fd is None
    assert stage.ifrog_combo.count() == 0


def test_stage_load_on_enter_without_files_leaves_combos_empty(stage):
    stage.on_enter()
    assert stage.path_edit.text() == ""
    assert stage.spec_lam_combo.count() == 0
    assert stage.bg_lam_combo.count() == 0


def test_stage_load_on_enter_restores_csv_spectrum_columns(stage, tmp_path):
    p = stage.state.load
    p.use_spec, p.spec_is_csv = True, True
    p.spec_path = _write_csv(tmp_path / "s.csv", [700, 800], [1, 2])
    p.spec_lam_col, p.spec_int_col = 1, 0
    stage.on_enter()
    assert stage.spec_lam_combo.currentIndex() == 1
    assert stage.spec_int_combo.currentIndex() == 0
    assert stage._spec_fd is None


def test_stage_load_on_enter_restores_hdf5_spectrum_and_background(stage, tmp_path):
    p = stage.state.load
    path = _write_spectrum_h5(tmp_path / "s.h5", np.linspace(700, 900, 50))
    p.use_spec, p.spec_is_csv, p.spec_path = True, False, path
    p.spec_lam_name, p.spec_int_name = "wavelength", "missing"
    p.frog_bg_use, p.frog_bg_path = True, path
    p.frog_bg_lam_name, p.frog_bg_int_name = "missing", "trace"
    stage.on_enter()
    assert stage.spec_lam_combo.currentText().startswith("wavelength")
    assert stage.spec_int_combo.currentIndex() == 0
    assert stage.bg_lam_combo.currentIndex() == 0
    assert stage.bg_int_combo.currentText().startswith("trace")
    assert stage._spec_fd is not None and stage._bg_fd is not None
    # entering again keeps the listings already read
    spec_fd, bg_fd = stage._spec_fd, stage._bg_fd
    stage.on_enter()
    assert stage._spec_fd is spec_fd and stage._bg_fd is bg_fd


# -- file pickers -----------------------------------------------------------
def test_stage_load_start_dir_prefers_first_existing_folder(stage, tmp_path):
    existing = str(tmp_path / "data.h5")
    assert stage._start_dir("", "/no/such/dir/x.h5", existing) == str(tmp_path)
    assert stage._start_dir("", "bare-name.h5") == ""


def test_stage_load_browse_trace_populates_or_cancels(
    stage, monkeypatch, experiment_h5
):
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: ("", ""))
    stage._pick_frog()
    assert stage.state.load.frog_path == ""
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", lambda *a, **k: (experiment_h5, "")
    )
    stage._pick_frog()
    assert stage.state.load.frog_path == experiment_h5
    assert stage.path_edit.text() == experiment_h5
    assert stage.state.load.ifrog_name == "trace"
    assert "datasets found" in stage.status_label.text()


def test_stage_load_file_rows_set_paths_and_run_their_hooks(
    stage, monkeypatch, tmp_path
):
    spec = _write_spectrum_h5(tmp_path / "s.h5", np.linspace(0.7, 0.9, 50))
    calib = _write_csv(tmp_path / "c.csv", [0.7, 0.8, 0.9], [1, 1, 1])
    chosen = iter([spec, calib, spec, str(tmp_path / "bg.csv"), calib])
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", lambda *a, **k: (next(chosen), "")
    )
    for button in _file_buttons(stage):
        button.click()
    p = stage.state.load
    assert p.frog_bg_path == spec and p.frog_bg_int_name == "trace"
    assert p.frog_calib_path == calib and p.frog_calib_lam_unit == "µm"
    assert p.spec_path == spec and not p.spec_is_csv
    assert p.spec_lam_unit == "µm"  # guessed from the 0.7-0.9 values
    assert p.spec_bg_path == str(tmp_path / "bg.csv")  # no hook on this row
    assert p.spec_calib_path == calib and p.spec_calib_lam_unit == "µm"
    # a cancelled dialog changes nothing
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: ("", ""))
    _file_buttons(stage)[0].click()
    assert p.frog_bg_path == spec


# -- dataset listing and unit detection --------------------------------------
def test_stage_load_populate_reports_unreadable_file(stage, tmp_path):
    stage.state.load.frog_path = str(tmp_path / "trace.txt")
    stage._populate_frog()
    assert stage.status_label.text().startswith("Could not read file")
    assert stage._frog_fd is None


def test_stage_load_populate_falls_back_without_matching_datasets(stage, tmp_path):
    # only 1-D datasets, none named like a trace: every picker falls back
    path = tmp_path / "odd.npz"
    np.savez(path, alpha=np.linspace(700, 900, 8), beta=np.linspace(-1, 1, 8))
    stage.state.load.frog_path = str(path)
    stage._populate_frog()
    p = stage.state.load
    assert p.ifrog_name == "alpha"
    assert p.lam_name == "alpha" and p.scan_name == "alpha"
    assert p.lam_unit == "nm"
    assert "2 datasets found" in stage.status_label.text()


def test_stage_load_autodetect_tolerates_missing_or_bad_datasets(stage, experiment_h5):
    stage._autodetect_units()  # no file listed yet: nothing to do
    p = stage.state.load
    p.frog_path = experiment_h5
    stage._populate_frog()
    p.lam_unit, p.scan_unit = "m", "s"
    p.lam_name = p.scan_name = "not-a-dataset"
    stage._autodetect_units()  # load errors are swallowed, units kept
    assert (p.lam_unit, p.scan_unit) == ("m", "s")


def test_stage_load_combo_changes_update_names_and_units(stage, tmp_path):
    path = tmp_path / "axes.npz"
    np.savez(
        path,
        trace=np.ones((4, 3)),
        lam_um=np.linspace(0.7, 0.9, 4),
        position_mm=np.linspace(-1, 1, 3),
        tag=np.array([[1.0, 2.0], [3.0, 4.0]]),
    )
    p = stage.state.load
    p.frog_path = str(path)
    stage._populate_frog()
    names = stage._frog_fd.names
    stage._on_ifrog_changed(names.index("tag"))
    assert p.ifrog_name == "tag"
    stage._on_ifrog_changed(99)  # out of range: ignored
    assert p.ifrog_name == "tag"
    stage._on_lam_changed(names.index("lam_um"))
    assert (p.lam_name, p.lam_unit) == ("lam_um", "µm")
    stage._on_lam_changed(-1)
    assert p.lam_name == "lam_um"
    stage._on_scan_changed(names.index("position_mm"))
    assert (p.scan_name, p.scan_type) == ("position_mm", "position")
    assert p.scan_unit == "mm"
    assert stage.scan_type_combo.currentText() == "position"
    stage._on_scan_changed(99)
    assert p.scan_name == "position_mm"


def test_stage_load_combo_handlers_ignore_changes_before_a_file(stage):
    p = stage.state.load
    before = (p.ifrog_name, p.lam_name, p.scan_name)
    stage._on_ifrog_changed(0)
    stage._on_lam_changed(0)
    stage._on_scan_changed(0)
    assert (p.ifrog_name, p.lam_name, p.scan_name) == before


def test_stage_load_lam_and_scan_change_keep_units_on_unloadable_data(
    stage, monkeypatch, experiment_h5
):
    p = stage.state.load
    p.frog_path = experiment_h5
    stage._populate_frog()
    names = stage._frog_fd.names

    def broken(name):
        raise OSError("disk went away")

    monkeypatch.setattr(stage._frog_fd, "load", broken)
    p.lam_unit, p.scan_unit = "m", "ps"
    stage._on_lam_changed(names.index("wavelength"))
    stage._on_scan_changed(names.index("delay"))
    assert (p.lam_name, p.scan_name) == ("wavelength", "delay")
    assert (p.lam_unit, p.scan_unit) == ("m", "ps")


# -- independent spectrum and background --------------------------------------
def test_stage_load_csv_spectrum_selects_columns(stage, tmp_path):
    p = stage.state.load
    p.spec_path = _write_csv(tmp_path / "s.txt", [0.7, 0.8, 0.9], [1, 2, 1])
    stage._on_spec_chosen()
    assert p.spec_is_csv
    assert (p.spec_lam_col, p.spec_int_col) == (0, 1)
    assert p.spec_lam_unit == "µm"
    stage.spec_lam_combo.setCurrentIndex(1)
    stage.spec_int_combo.setCurrentIndex(0)
    assert (p.spec_lam_col, p.spec_int_col) == (1, 0)


def test_stage_load_hdf5_spectrum_maps_combos_to_names(stage, tmp_path):
    p = stage.state.load
    p.spec_path = _write_spectrum_h5(tmp_path / "s.h5", np.linspace(700, 900, 9))
    stage._on_spec_chosen()
    assert (p.spec_lam_name, p.spec_int_name) == ("wavelength", "trace")
    assert p.spec_lam_unit == "nm"
    names = stage._spec_fd.names
    stage.spec_lam_combo.setCurrentIndex(names.index("trace"))
    stage.spec_int_combo.setCurrentIndex(names.index("wavelength"))
    assert (p.spec_lam_name, p.spec_int_name) == ("trace", "wavelength")


def test_stage_load_hdf5_spectrum_keeps_unit_when_axis_is_2d(stage, tmp_path):
    p = stage.state.load
    path = tmp_path / "s2.h5"
    with h5py.File(path, "w") as f:
        f["wavelength"] = np.ones((3, 3))
        f["trace"] = np.ones(3)
    p.spec_path, p.spec_lam_unit = str(path), "m"
    stage._on_spec_chosen()
    assert p.spec_lam_name == "wavelength"
    assert p.spec_lam_unit == "m"  # to_1d refused the 2-D axis


def test_stage_load_spectrum_combo_changes_before_any_file_are_ignored(stage):
    p = stage.state.load
    p.spec_is_csv = False
    before = (p.spec_lam_name, p.spec_int_name)
    stage.spec_lam_combo.addItems(["a", "b"])
    stage.spec_int_combo.addItems(["a", "b"])
    stage.spec_lam_combo.setCurrentIndex(1)
    stage.spec_int_combo.setCurrentIndex(1)
    assert (p.spec_lam_name, p.spec_int_name) == before


def test_stage_load_background_unreadable_reports(stage, tmp_path):
    stage.state.load.frog_bg_path = str(tmp_path / "nope.h5")
    stage._on_bg_chosen()
    assert stage.status_label.text().startswith("BG file unreadable")
    assert stage._bg_fd is None


def test_stage_load_fill_ignores_missing_or_out_of_range_index(stage):
    stage._fill(stage.bg_lam_combo, ["a", "b"], None)
    assert stage.bg_lam_combo.count() == 2
    stage._fill(stage.bg_lam_combo, ["a", "b"], 5)
    assert stage.bg_lam_combo.currentIndex() == 0


def test_stage_load_sync_named_combos_copies_valid_selections(stage, tmp_path):
    p = stage.state.load
    path = _write_spectrum_h5(tmp_path / "s.h5", np.linspace(700, 900, 9))
    p.spec_path = path
    stage._on_spec_chosen()
    p.frog_bg_path = path
    stage._on_bg_chosen()
    names = stage._spec_fd.names
    # move the selections without the change handlers, as a restore would
    stage.spec_lam_combo.blockSignals(True)
    stage.spec_lam_combo.setCurrentIndex(names.index("trace"))
    stage.spec_lam_combo.blockSignals(False)
    stage.bg_int_combo.blockSignals(True)
    stage.bg_int_combo.setCurrentIndex(names.index("wavelength"))
    stage.bg_int_combo.blockSignals(False)
    stage._sync_named_combos()
    assert p.spec_lam_name == "trace"
    assert p.frog_bg_int_name == "wavelength"
    # empty combos (index -1) leave the names alone
    for box in (
        stage.spec_lam_combo,
        stage.spec_int_combo,
        stage.bg_lam_combo,
        stage.bg_int_combo,
    ):
        box.blockSignals(True)
        box.clear()
        box.blockSignals(False)
    stage._sync_named_combos()
    assert p.spec_lam_name == "trace"
    assert p.frog_bg_int_name == "wavelength"


# -- load & preview -----------------------------------------------------------
def test_stage_load_preview_reports_failure(stage, tmp_path):
    stage.state.load.frog_path = str(tmp_path / "missing.h5")
    stage._load_preview()
    assert stage.status_label.text().startswith("Load failed")
    assert stage.state.load_data is None


def test_stage_load_reload_same_trace_keeps_preprocess_settings(
    stage, experiment_h5, experiment
):
    _point_at(stage, experiment_h5, experiment)
    stage._load_preview()
    assert stage.status_label.text() == "Loaded. Proceed to the marginal check."
    stage.state.preproc.tau_min_fs += 1.0
    kept = stage.state.preproc.tau_min_fs
    stage._load_preview()
    assert "kept your preprocess settings" in stage.status_label.text()
    assert stage.state.preproc.tau_min_fs == kept


def test_stage_load_draw_handles_position_scans_and_out_of_band_data(stage):
    # A position-scanned trace entirely outside the 190-1000 nm trust band and
    # all zero: the preview uses the whole trace, has no vmax and no y-limits.
    data = {
        "lam": np.linspace(1200e-9, 1400e-9, 6),
        "scanaxis": np.linspace(-3e-6, 3e-6, 5),
        "input_unit": "position",
        "trace": np.zeros((6, 5)),
    }
    stage._draw(data)
    ax = stage.trace_canvas.figure.axes[0]
    assert ax.get_title() == "Measured trace"
    # τ = 2z/c: ±3 µm of stage travel is ±20 fs of delay (the mesh adds half a
    # 10 fs cell either side), not the ±3e-6 a raw position axis would give
    lo, hi = ax.get_xlim()
    assert lo == pytest.approx(-25.0, abs=0.1) and hi == pytest.approx(25.0, abs=0.1)


def test_stage_load_draw_handles_an_empty_trace(stage):
    data = {
        "lam": np.zeros(0),
        "scanaxis": np.linspace(-1e-14, 1e-14, 3),
        "input_unit": "delay",
        "trace": np.zeros((0, 3)),
    }
    called = []
    monkey = stage.trace_canvas.render
    stage.trace_canvas.render = lambda fn: called.append(fn)
    try:
        stage._draw(data)
    finally:
        stage.trace_canvas.render = monkey
    assert len(called) == 1


def test_stage_load_signature_ignores_reverse_trace(stage):
    p = stage.state.load
    before = stage_load.trace_signature(p)
    p.reverse_trace = not p.reverse_trace
    assert stage_load.trace_signature(p) == before


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        # regression: an intensity match at index 0 used to be read as "no
        # match" (``0 or fallback``), selecting the wavelength axis instead
        (["trace", "wavelength"], (1, 0)),
        (["wavelength", "trace"], (0, 1)),
        (["a", "b"], (0, 1)),
        (["only"], (0, 0)),
    ],
)
def test_stage_load_spectrum_dataset_indices(names, expected):
    assert stage_load._spectrum_dataset_indices(names) == expected
