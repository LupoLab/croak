"""Tests for :mod:`croak.io`."""

import numpy as np
import pytest

from croak import io
from croak.constants import C


def test_list_and_load_hdf5(experiment_h5):
    fd = io.list_datasets(experiment_h5)
    assert fd.filetype == "hdf5"
    assert set(fd.names) >= {"wavelength", "delay", "trace", "wavelength_std"}
    trace = fd.load("trace")
    assert trace.ndim == 2


def test_auto_match_picks_shortest(experiment_h5):
    fd = io.list_datasets(experiment_h5)
    # "wavelength" should win over the longer decoy "wavelength_std"
    assert fd.names[fd.auto_index("lambda")] == "wavelength"
    assert fd.names[fd.auto_index("ifrog")] == "trace"
    assert fd.names[fd.auto_index("scan")] == "delay"


def test_npz_roundtrip(tmp_path):
    path = tmp_path / "data.npz"
    np.savez(path, trace=np.ones((4, 5)), lam=np.arange(4.0))
    fd = io.list_datasets(str(path))
    assert fd.filetype == "npz"
    assert np.allclose(fd.load("trace"), 1.0)


def test_unsupported_extension(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("1 2\n")
    with pytest.raises(ValueError):
        io.list_datasets(str(p))


def test_load_csv_auto_delimiter(tmp_path):
    p = tmp_path / "spec.csv"
    p.write_text("# comment\nwavelength,intensity\n700,0.1\n800,1.0\n900,0.2\n")
    data = io.load_csv(str(p))
    assert data.shape == (3, 2)
    assert np.allclose(data[:, 0], [700, 800, 900])
    assert np.allclose(data[:, 1], [0.1, 1.0, 0.2])


def test_load_csv_whitespace(tmp_path):
    p = tmp_path / "spec.txt"
    p.write_text("700 0.1\n800 1.0\n")
    data = io.load_csv(str(p))
    assert data.shape == (2, 2)
    assert np.allclose(data[:, 1], [0.1, 1.0])


def test_unit_conversions():
    assert io.unit_to_si("nm") == pytest.approx(1e-9)
    assert io.unit_to_si("fs") == pytest.approx(1e-15)
    assert io.unit_to_si("mm") == pytest.approx(1e-3)


def test_guess_units():
    assert io.guess_wavelength_unit(np.array([700.0, 800.0])) == "nm"
    assert io.guess_wavelength_unit(np.array([0.7, 0.8])) == "µm"
    assert io.guess_wavelength_unit(np.array([700e-9, 800e-9])) == "m"
    assert io.guess_scanaxis_unit(np.array([-500.0, 500.0]), "delay") == "fs"
    assert io.guess_scanaxis_unit(np.array([-0.1, 0.1]), "delay") == "ps"


def test_guess_scanaxis_type():
    assert io.guess_scanaxis_type("delay_axis") == "delay"
    assert io.guess_scanaxis_type("stage_position") == "position"
    assert io.guess_scanaxis_type("foobar") is None


def test_position_to_delay():
    z = np.array([0.0, 1e-6])  # metres
    tau = io.position_to_delay(z)
    assert tau[1] == pytest.approx(2 * 1e-6 / C)
