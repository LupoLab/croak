"""Mirror rows: built-in and custom-file mirrors, registration and specs."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from croak import dispersion, mirrors
from croak.gui import mirror_stack as ms
from croak.gui.mirror_stack import MirrorRow, MirrorStack
from croak.session.params import BeamPathMirror


@pytest.fixture
def mirror_csv(tmp_path):
    """A custom mirror file: wavelength (nm) and a smooth phase (rad)."""
    lam_nm = np.linspace(600.0, 1000.0, 81)
    phase = 0.5 * ((lam_nm - 800.0) / 100.0) ** 2
    path = tmp_path / "mirror.csv"
    np.savetxt(path, np.column_stack([lam_nm, phase]), delimiter=",")
    return path


@pytest.fixture
def row(qtbot):
    widget = MirrorRow()
    qtbot.addWidget(widget)
    yield widget
    widget.cleanup()


def _pick(monkeypatch, path):
    monkeypatch.setattr(
        ms.QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a: (str(path), "")),
    )


def test_custom_type_unregisters_until_a_file_is_loaded(row):
    assert row.key in dispersion.MIRRORS  # the default built-in coating
    row.type_combo.setCurrentText(ms._CUSTOM)
    assert row.key not in dispersion.MIRRORS
    assert row.range_label.text() == "load a file…"
    assert row.contribution() is None
    assert not row.pol_combo.isEnabled()


def test_pick_custom_fills_columns_and_unit(row, monkeypatch, mirror_csv):
    _pick(monkeypatch, mirror_csv)
    row._pick_custom()
    assert row.cm_path.text() == str(mirror_csv)
    assert row.cm_lam_col.count() == 2
    assert row.cm_lam_col.currentIndex() == 0
    assert row.cm_val_col.currentIndex() == 1
    assert row.cm_unit.currentText() == "nm"


def test_pick_custom_cancelled_keeps_the_path(row, monkeypatch):
    row.cm_path.setText("previous.csv")
    _pick(monkeypatch, "")
    row._pick_custom()
    assert row.cm_path.text() == "previous.csv"


def test_load_custom_needs_a_file(row, qtbot):
    with qtbot.waitSignal(row.status) as blocker:
        row._load_custom()
    assert blocker.args == ["Choose a mirror file first."]


@pytest.mark.parametrize("datatype", ["phase (rad)", "GDD (fs²)"])
def test_load_custom_registers_a_phase_only_mirror(
    row, qtbot, monkeypatch, mirror_csv, datatype
):
    row.type_combo.setCurrentText(ms._CUSTOM)
    _pick(monkeypatch, mirror_csv)
    row._pick_custom()
    row.cm_datatype.setCurrentText(datatype)
    with qtbot.waitSignal(row.changed):
        row._load_custom()
    assert row.key in dispersion.MIRRORS
    assert row.key not in dispersion.MIRROR_AMPLITUDES
    # The default Planck taper sits a 2 % margin inside the file's 600–1000 nm.
    assert row.range_label.text() == "valid λ: 608–992 nm"
    assert row.contribution() == (row.key, -1)  # one bounce, removed
    spec = row.to_spec()
    assert spec.name == "" and spec.custom_file == str(mirror_csv)
    assert spec.custom_datatype == ("gdd" if datatype.startswith("GDD") else "phase")


def test_load_custom_failure_is_reported(row, qtbot, tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("not,numbers\n")
    row.cm_path.setText(str(bad))
    with qtbot.waitSignal(row.status) as blocker:
        row._load_custom()
    assert blocker.args[0].startswith("Custom mirror failed:")


def test_fill_combo_ignores_an_out_of_range_index(qapp):
    box = ms.QComboBox()
    MirrorRow._fill_combo(box, ["a", "b"], 5)
    assert [box.itemText(i) for i in range(box.count())] == ["a", "b"]
    assert box.currentIndex() == 0


def test_builtin_load_failure_unregisters_and_reports(row, qtbot, monkeypatch):
    def fail(*_args):
        raise OSError("missing table")

    monkeypatch.setattr(mirrors, "load_coating", fail)
    with qtbot.waitSignal(row.status) as blocker:
        row._reload()
    assert blocker.args == ["Mirror load failed: missing table"]
    assert row.key not in dispersion.MIRRORS


def test_load_spec_custom_without_a_file_just_signals(row, qtbot):
    spec = BeamPathMirror(name="", polarization="s", bounces=2, direction="add")
    with qtbot.waitSignal(row.changed):
        row.load_spec(spec)
    assert row.type_combo.currentText() == ms._CUSTOM
    assert row.custom.isVisibleTo(row)
    # Regression: the default coating the new row had registered must not
    # stand in for the custom mirror that has no file yet.
    assert row.contribution() is None


def test_load_spec_custom_file_round_trips(row, qtbot, mirror_csv):
    spec = BeamPathMirror(
        name="",
        polarization="s",
        bounces=3,
        direction="add",
        custom_file=str(mirror_csv),
        custom_lam_col=0,
        custom_val_col=1,
        custom_unit="nm",
        custom_datatype="gdd",
    )
    with qtbot.waitSignal(row.changed):
        row.load_spec(spec)
    assert row.cm_datatype.currentText() == "GDD (fs²)"
    assert row.contribution() == (row.key, 3)
    assert row.to_spec() == spec


def test_load_spec_custom_file_unreadable_reports(row, qtbot, tmp_path):
    spec = BeamPathMirror(
        name="",
        polarization="s",
        bounces=1,
        direction="remove",
        custom_file=str(tmp_path / "gone.csv"),
    )
    with qtbot.waitSignal(row.status) as blocker:
        row.load_spec(spec)
    assert blocker.args[0].startswith("Custom mirror failed:")


def test_stack_ignores_removing_an_unknown_row(qtbot):
    stack = MirrorStack()
    qtbot.addWidget(stack)
    stranger = MirrorRow()
    qtbot.addWidget(stranger)
    with qtbot.assertNotEmitted(stack.changed):
        stack._remove_row(stranger)
    stranger.cleanup()
