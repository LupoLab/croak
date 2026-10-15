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


def test_dataset_info_str_and_file_dataset_views(tmp_path):
    import h5py

    path = tmp_path / "nested.h5"
    with h5py.File(path, "w") as f:
        f["trace"] = np.ones((3, 4))
        f["axes/lam"] = np.arange(3.0)
        f["label"] = "not numeric"  # skipped: only numeric datasets are listed
    fd = io.list_datasets(str(path))
    assert fd.names == ["axes/lam", "trace"]
    assert fd.matrix_names == ["trace"]
    assert fd.vector_names == ["axes/lam"]
    assert fd.option_labels == ["axes/lam  (3,)", "trace  (3, 4)"]
    assert str(fd.datasets[1]) == "trace (3, 4)"


def test_npz_listing_skips_non_numeric_arrays(tmp_path):
    path = tmp_path / "mixed.npz"
    np.savez(path, trace=np.ones((2, 2)), names=np.array(["a", "b"]))
    assert io.list_datasets(str(path)).names == ["trace"]


def test_load_dataset_infers_the_file_type_and_rejects_groups(tmp_path):
    import h5py

    h5 = tmp_path / "d.h5"
    with h5py.File(h5, "w") as f:
        f["g/x"] = np.arange(2.0)
    np.testing.assert_array_equal(io.load_dataset(str(h5), "g/x"), [0.0, 1.0])
    with pytest.raises(TypeError, match="not an HDF5 dataset"):
        io.load_dataset(str(h5), "g")
    npz = tmp_path / "d.npz"
    np.savez(npz, x=np.arange(3))
    np.testing.assert_array_equal(io.load_dataset(str(npz), "x"), [0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="unsupported file type 'mat'"):
        io.load_dataset(str(npz), "x", "mat")


def test_detect_delimiter_skips_headers_and_defaults_to_comma(tmp_path):
    header_only = tmp_path / "h.csv"
    header_only.write_text("# only a comment\n\nwavelength intensity\n")
    assert io._detect_delimiter(str(header_only)) == ","


def test_load_csv_with_explicit_skip_and_whitespace_header(tmp_path):
    p = tmp_path / "s.txt"
    p.write_text("lam I\n700 0.1\n800 1.0\n")
    np.testing.assert_allclose(io.load_csv(str(p)), [[700, 0.1], [800, 1.0]])
    np.testing.assert_allclose(io.load_csv(str(p), skip=2), [[800, 1.0]])


def test_guess_position_units_and_metre_delays():
    assert io.guess_scanaxis_unit(np.array([-50e-6, 50e-6]), "position") == "m"
    assert io.guess_scanaxis_unit(np.array([10.0, 20.0]), "position") == "mm"
    assert io.guess_scanaxis_unit(np.array([1000.0, 2000.0]), "position") == "µm"
    assert io.guess_scanaxis_unit(np.array([-1e-13, 1e-13]), "delay") == "s"


def test_csv_wavelength_unit(tmp_path):
    good = tmp_path / "g.csv"
    good.write_text("0.7,1\n0.8,2\n")
    assert io.csv_wavelength_unit(str(good)) == "µm"
    one_row = tmp_path / "o.csv"
    one_row.write_text("700,1\n")
    assert io.csv_wavelength_unit(str(one_row)) == "nm"
    # an unreadable file falls back to nm rather than raising
    assert io.csv_wavelength_unit(str(tmp_path / "missing.csv")) == "nm"


def test_to_1d_flattens_singleton_axes_only():
    np.testing.assert_array_equal(io.to_1d(np.arange(3.0)), [0, 1, 2])
    np.testing.assert_array_equal(io.to_1d(np.arange(3.0)[:, None]), [0, 1, 2])
    with pytest.raises(ValueError, match="pick a 1-D wavelength axis"):
        io.to_1d(np.ones((2, 3)), "lam")


def test_reduce_to_1d_averages_the_shot_axis():
    shots = np.array([[1.0, 3.0], [2.0, 4.0], [5.0, 7.0]])  # (N_lambda=3, 2 shots)
    np.testing.assert_allclose(io.reduce_to_1d(shots, "I", 3), [2.0, 3.0, 6.0])
    np.testing.assert_allclose(io.reduce_to_1d(shots.T, "I", 3), [2.0, 3.0, 6.0])
    square = np.array([[1.0, 3.0], [2.0, 4.0]])
    np.testing.assert_allclose(io.reduce_to_1d(square, "I", 2), [2.0, 3.0])
    np.testing.assert_array_equal(io.reduce_to_1d(np.arange(3.0), "I", 3), [0, 1, 2])
    np.testing.assert_array_equal(io.reduce_to_1d(np.ones((1, 3)), "I", 3), [1, 1, 1])
    with pytest.raises(ValueError, match="cannot reduce to length 4"):
        io.reduce_to_1d(shots, "I", 4)
    with pytest.raises(ValueError, match="cannot reduce"):
        io.reduce_to_1d(np.ones((2, 2, 2)), "I", 2)


def test_listing_skips_named_datatypes(tmp_path):
    import h5py

    path = tmp_path / "typed.h5"
    with h5py.File(path, "w") as f:
        f["x"] = np.arange(2.0)
        f["t"] = np.dtype("f8")  # a committed datatype: neither dataset nor group
    assert io.list_datasets(str(path)).names == ["x"]


def test_load_csv_with_an_explicit_delimiter(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("lam;I\n700;0.1\n800;1.0\n")
    np.testing.assert_allclose(
        io.load_csv(str(p), delimiter=";"), [[700, 0.1], [800, 1.0]]
    )


def test_load_csv_without_numeric_rows_is_empty(tmp_path):
    p = tmp_path / "h.csv"
    p.write_text("wavelength,intensity\nnm,counts\n")
    with pytest.warns(UserWarning, match="Empty input file"):
        assert io.load_csv(str(p)).size == 0
