"""Tests for :func:`croak.io.read_simulated_scan` (scansave HDF5 reading)."""

import h5py
import numpy as np
import pytest
from conftest import SIMULATED_ZSAVE, make_simulated_scan, write_simulated_h5

from croak import io


def test_read_orientation_and_sort(simulated_scan_h5, simulated_truth):
    """Trace is oriented ``(Nomega, Ndelay)`` and ω is sorted ascending."""
    scan = io.read_simulated_scan(simulated_scan_h5)
    n_omega = simulated_truth.omega_abs.size
    n_tau = simulated_truth.tau.size
    assert scan.trace.shape == (n_omega, n_tau)
    assert np.all(np.diff(scan.omega) > 0)
    # the file stored everything in FFT order; reading sorts it back to the truth
    np.testing.assert_allclose(scan.omega, simulated_truth.omega_abs)
    np.testing.assert_allclose(scan.trace, simulated_truth.trace_omega)
    np.testing.assert_allclose(scan.tau, simulated_truth.tau)
    assert scan.omega0 == pytest.approx(simulated_truth.omega0)
    assert scan.tau_fwhm == pytest.approx(simulated_truth.tau_fwhm)
    assert scan.Iomega_beamlet is not None
    assert scan.window_key == "Iω_win"


def test_z_index_selects_exit_slice(simulated_scan_h5, simulated_truth):
    """Default ``z_index=-1`` picks the genuine (last) slice, not the decoy."""
    last = io.read_simulated_scan(simulated_scan_h5, z_index=-1)
    first = io.read_simulated_scan(simulated_scan_h5, z_index=0)
    np.testing.assert_allclose(last.trace, simulated_truth.trace_omega)
    # the first slice is a decoy, so it must differ
    assert not np.allclose(first.trace, simulated_truth.trace_omega)


def test_window_key_selects_dataset(simulated_scan_h5):
    """The two stored windows differ (Iω_win_reimaged = 0.7 × Iω_win here)."""
    full = io.read_simulated_scan(simulated_scan_h5, window_key="Iω_win")
    reimaged = io.read_simulated_scan(simulated_scan_h5, window_key="Iω_win_reimaged")
    np.testing.assert_allclose(reimaged.trace, 0.7 * full.trace)


def test_missing_window_key_lists_available(simulated_scan_h5):
    with pytest.raises(KeyError, match="Iω_win"):
        io.read_simulated_scan(simulated_scan_h5, window_key="not_a_window")


def test_missing_grid_raises(tmp_path):
    path = tmp_path / "bad.h5"
    with h5py.File(path, "w") as f:
        f["Iω_win"] = np.zeros((4, 2, 8))
    with pytest.raises(KeyError, match="grid"):
        io.read_simulated_scan(str(path))


def test_missing_delay_axis_raises(tmp_path, simulated_truth):
    path = tmp_path / "nodelay.h5"
    with h5py.File(path, "w") as f:
        gg = f.create_group("grid")
        gg["ω"] = simulated_truth.omega_abs
        gg["ω0"] = simulated_truth.omega0
        gg["Iω"] = simulated_truth.Iomega
        gg["t"] = simulated_truth.t
        gg["It"] = simulated_truth.It
        gg["τfwhm"] = simulated_truth.tau_fwhm
        f["Iω_win"] = np.zeros(
            (simulated_truth.tau.size, 2, simulated_truth.omega_abs.size)
        )
    with pytest.raises(KeyError, match="τ"):
        io.read_simulated_scan(str(path))


def test_two_dimensional_window_supported(tmp_path):
    """A 2-D window (no propagation axis) is accepted and oriented."""
    truth = make_simulated_scan(n=64, ndelay=32)
    n_omega = truth.omega_abs.size
    n_tau = truth.tau.size
    path = tmp_path / "scan2d.h5"
    with h5py.File(path, "w") as f:
        gg = f.create_group("grid")
        gg["ω"] = np.fft.ifftshift(truth.omega_abs)
        gg["ω0"] = truth.omega0
        gg["Iω"] = np.fft.ifftshift(truth.Iomega)
        gg["t"] = truth.t
        gg["It"] = truth.It
        gg["τfwhm"] = truth.tau_fwhm
        sv = f.create_group("scanvariables")
        sv["τ"] = truth.tau
        # 2-D, in (Ntau, Nomega) order (h5py reverse of Julia (Nomega, Ntau))
        f["Iω_win"] = np.fft.ifftshift(truth.trace_omega, axes=0).T
    scan = io.read_simulated_scan(str(path))
    assert scan.trace.shape == (n_omega, n_tau)
    assert scan.Iomega_beamlet is None
    np.testing.assert_allclose(scan.trace, truth.trace_omega)


def test_oversampled_time_grid_preferred(tmp_path, simulated_truth):
    """When ``To``/``Ito`` are present they are used for the reference axes."""
    path = write_simulated_h5(tmp_path / "scan.h5", simulated_truth)
    to = np.linspace(-1e-13, 1e-13, 1024)
    ito = np.ones_like(to)
    with h5py.File(path, "a") as f:
        f["grid"]["To"] = to
        f["grid"]["Ito"] = ito
    scan = io.read_simulated_scan(path)
    assert scan.t.size == 1024
    np.testing.assert_allclose(scan.t, to)


# ---------------------------------------------------------------------------
# Time-domain truth source (beamlet vs source)
# ---------------------------------------------------------------------------
def test_truth_source_selects_beamlet_by_default(tmp_path, simulated_truth):
    """With ``It_beamlet`` present, the default ``truth_source`` returns it."""
    path = write_simulated_h5(tmp_path / "scan.h5", simulated_truth)
    it_beamlet = 0.5 * simulated_truth.It + 0.1  # a distinct beamlet envelope
    with h5py.File(path, "a") as f:
        f["grid"]["It_beamlet"] = it_beamlet
    default = io.read_simulated_scan(path)
    beamlet = io.read_simulated_scan(path, truth_source="beamlet")
    source = io.read_simulated_scan(path, truth_source="source")
    np.testing.assert_allclose(default.It, it_beamlet)
    np.testing.assert_allclose(beamlet.It, it_beamlet)
    np.testing.assert_allclose(source.It, simulated_truth.It)
    assert not np.allclose(beamlet.It, source.It)


def test_truth_source_falls_back_to_source(tmp_path, simulated_truth):
    """Requesting the beamlet truth on a file without it falls back to ``It``."""
    path = write_simulated_h5(tmp_path / "scan.h5", simulated_truth)  # no It_beamlet
    scan = io.read_simulated_scan(path, truth_source="beamlet")
    np.testing.assert_allclose(scan.It, simulated_truth.It)


def test_truth_source_prefers_oversampled_beamlet(tmp_path, simulated_truth):
    """``Ito_beamlet``/``To`` take precedence over native ``It_beamlet``."""
    path = write_simulated_h5(tmp_path / "scan.h5", simulated_truth)
    to = np.linspace(-1e-13, 1e-13, 1024)
    ito_beamlet = np.hanning(1024)
    with h5py.File(path, "a") as f:
        f["grid"]["It_beamlet"] = simulated_truth.It  # native, must be ignored
        f["grid"]["To"] = to
        f["grid"]["Ito_beamlet"] = ito_beamlet
    scan = io.read_simulated_scan(path, truth_source="beamlet")
    assert scan.t.size == 1024
    np.testing.assert_allclose(scan.t, to)
    np.testing.assert_allclose(scan.It, ito_beamlet)


def test_read_simulated_truth_keys(tmp_path, simulated_truth):
    """The truth-key reader lists what the file stores, beamlet first."""
    path = write_simulated_h5(tmp_path / "scan.h5", simulated_truth)
    assert io.read_simulated_truth_keys(path) == ("source",)
    with h5py.File(path, "a") as f:
        f["grid"]["It_beamlet"] = simulated_truth.It
    assert io.read_simulated_truth_keys(path) == ("beamlet", "source")


def test_read_simulated_truth_keys_missing_grid_raises(tmp_path):
    path = tmp_path / "nogrid.h5"
    with h5py.File(path, "w") as f:
        f["Iω_win"] = np.zeros((4, 2, 8))
    with pytest.raises(KeyError, match="grid"):
        io.read_simulated_truth_keys(str(path))


# ---------------------------------------------------------------------------
# Multi-thickness selection (/grid/zsave)
# ---------------------------------------------------------------------------
def test_read_z_positions(simulated_scan_multi_h5, simulated_scan_h5):
    """``read_simulated_z_positions`` returns ``zsave`` (or ``None`` for legacy)."""
    np.testing.assert_allclose(
        io.read_simulated_z_positions(simulated_scan_multi_h5), SIMULATED_ZSAVE
    )
    assert io.read_simulated_z_positions(simulated_scan_h5) is None


def test_read_z_positions_missing_grid_raises(tmp_path):
    path = tmp_path / "nogrid.h5"
    with h5py.File(path, "w") as f:
        f["Iω_win"] = np.zeros((4, 2, 8))
    with pytest.raises(KeyError, match="grid"):
        io.read_simulated_z_positions(str(path))


def test_z_thickness_selects_nearest_slice(simulated_scan_multi_h5):
    """``z_thickness`` loads the slice with the nearest saved position."""
    # 9 µm is closest to the 10 µm slice (index 2 of SIMULATED_ZSAVE).
    near = io.read_simulated_scan(simulated_scan_multi_h5, z_thickness=9e-6)
    by_index = io.read_simulated_scan(simulated_scan_multi_h5, z_index=2)
    assert near.z_index == 2
    np.testing.assert_allclose(near.trace, by_index.trace)
    # and it is genuinely a different slice from the default exit
    exit_slice = io.read_simulated_scan(simulated_scan_multi_h5)
    assert not np.allclose(near.trace, exit_slice.trace)


def test_z_thickness_overrides_z_index(simulated_scan_multi_h5):
    """``z_thickness`` takes precedence over an explicit ``z_index``."""
    exit_thk = SIMULATED_ZSAVE[-1]
    scan = io.read_simulated_scan(
        simulated_scan_multi_h5, z_index=0, z_thickness=exit_thk
    )
    exit_slice = io.read_simulated_scan(simulated_scan_multi_h5, z_index=-1)
    assert scan.z_index == SIMULATED_ZSAVE.size - 1
    np.testing.assert_allclose(scan.trace, exit_slice.trace)


def test_z_thickness_without_zsave_raises(simulated_scan_h5):
    """Selecting by thickness needs ``/grid/zsave`` (absent in the legacy format)."""
    with pytest.raises(KeyError, match="zsave"):
        io.read_simulated_scan(simulated_scan_h5, z_thickness=5e-6)


def test_scan_reports_z_positions_and_index(simulated_scan_multi_h5, simulated_scan_h5):
    """The scan surfaces ``z_positions`` and the resolved non-negative ``z_index``."""
    multi = io.read_simulated_scan(simulated_scan_multi_h5)  # default exit
    np.testing.assert_allclose(multi.z_positions, SIMULATED_ZSAVE)
    assert multi.z_index == SIMULATED_ZSAVE.size - 1  # -1 resolved to the last slice
    legacy = io.read_simulated_scan(simulated_scan_h5)
    assert legacy.z_positions is None
    assert legacy.z_index == 1  # nz=2, -1 -> 1


# ---------------------------------------------------------------------------
# Trace-window selection (Iω_win / Iω_win_reimaged / Iω_full)
# ---------------------------------------------------------------------------
def test_read_window_keys(simulated_scan_multi_h5, simulated_scan_h5):
    """``read_simulated_window_keys`` lists the windows present (``Iω_full`` is new)."""
    assert io.read_simulated_window_keys(simulated_scan_multi_h5) == [
        "Iω_win",
        "Iω_win_reimaged",
        "Iω_full",
    ]
    # the legacy fixture has no full-collection window
    assert io.read_simulated_window_keys(simulated_scan_h5) == [
        "Iω_win",
        "Iω_win_reimaged",
    ]


def test_iomega_full_loads_as_trace(simulated_scan_multi_h5):
    """``Iω_full`` is selectable as the trace and differs from ``Iω_win``."""
    full = io.read_simulated_scan(simulated_scan_multi_h5, window_key="Iω_full")
    win = io.read_simulated_scan(simulated_scan_multi_h5, window_key="Iω_win")
    assert full.window_key == "Iω_full"
    # the fixture stores Iω_full = Iω_win / 0.6
    np.testing.assert_allclose(full.trace, win.trace / 0.6)


def test_read_wavelength_range_matches_positive_band(simulated_scan_h5):
    """``read_simulated_wavelength_range`` is the positive-ω λ span (nm)."""
    lam_min, lam_max = io.read_simulated_wavelength_range(simulated_scan_h5)
    scan = io.read_simulated_scan(simulated_scan_h5)
    pos = scan.omega[scan.omega > 0.0]
    expect_min = 2.0 * np.pi * 299792458.0 / pos.max() / 1e-9
    expect_max = 2.0 * np.pi * 299792458.0 / pos.min() / 1e-9
    assert lam_min == pytest.approx(expect_min)
    assert lam_max == pytest.approx(expect_max)
    assert 0.0 < lam_min < lam_max


def test_read_simulated_scan_delay_convention_marker(tmp_path, simulated_truth):
    """The /grid/delay_convention marker is surfaced; absence means legacy."""
    legacy = write_simulated_h5(tmp_path / "legacy.h5", simulated_truth)
    assert io.read_simulated_scan(legacy).delay_convention == "legacy"

    gate = write_simulated_h5(
        tmp_path / "gate.h5", simulated_truth, delay_convention="gate"
    )
    assert io.read_simulated_scan(gate).delay_convention == "gate"


def test_pipeline_resolves_reverse_trace_from_marker(tmp_path, simulated_truth):
    """reverse_trace=None auto-detects; explicit values override the marker."""
    from croak.session.params import SimulatedLoadParams
    from croak.session.pipeline import _resolve_reverse_trace

    legacy = io.read_simulated_scan(
        write_simulated_h5(tmp_path / "l.h5", simulated_truth)
    )
    gate = io.read_simulated_scan(
        write_simulated_h5(tmp_path / "g.h5", simulated_truth, delay_convention="gate")
    )
    auto = SimulatedLoadParams(frog_path="x")
    assert auto.reverse_trace is None
    assert _resolve_reverse_trace(auto, legacy) is True
    assert _resolve_reverse_trace(auto, gate) is False
    forced = SimulatedLoadParams(frog_path="x", reverse_trace=True)
    assert _resolve_reverse_trace(forced, gate) is True
    off = SimulatedLoadParams(frog_path="x", reverse_trace=False)
    assert _resolve_reverse_trace(off, legacy) is False


def test_read_simulated_scan_complex_beamlet(tmp_path, simulated_truth):
    """Complex spectra load, sorted onto the ascending-ω axis; None if absent."""
    plain = io.read_simulated_scan(
        write_simulated_h5(tmp_path / "plain.h5", simulated_truth)
    )
    assert plain.Eomega_beamlet is None and plain.Eomega is None

    scan = io.read_simulated_scan(
        write_simulated_h5(tmp_path / "cplx.h5", simulated_truth, store_complex=True)
    )
    assert scan.Eomega_beamlet is not None and scan.Eomega is not None
    # amplitude consistent with the stored intensity, on the sorted axis
    np.testing.assert_allclose(
        np.abs(scan.Eomega_beamlet) ** 2, scan.Iomega_beamlet, rtol=1e-10
    )
    # The reader only sorts onto the ascending-ω axis; it deliberately does not
    # touch the file's Fourier convention. Undoing the stored half-window shift
    # ((-1)**n) and the opposite Fourier sign (conjugation) --- what
    # croak.session.pipeline._simulated_truth_spectrum does --- recovers the
    # generating field exactly.
    alternate = (-1.0) ** np.arange(scan.Eomega.size)
    np.testing.assert_allclose(
        np.conj(scan.Eomega * alternate), simulated_truth.ew, rtol=1e-10
    )


def test_read_simulated_scan_mask_geometry(tmp_path, simulated_truth):
    """Recorded mask geometry loads; a file without it reports None, not a guess."""
    plain = io.read_simulated_scan(
        write_simulated_h5(tmp_path / "nogeom.h5", simulated_truth)
    )
    assert plain.mask_diameter is None
    assert plain.mask_spacing is None
    assert plain.geometry is None

    scan = io.read_simulated_scan(
        write_simulated_h5(
            tmp_path / "geom.h5", simulated_truth, geometry=(1e-3, 0.5e-3, "tg")
        )
    )
    assert scan.mask_diameter == pytest.approx(1e-3)
    assert scan.mask_spacing == pytest.approx(0.5e-3)
    assert scan.geometry == "tg"


def test_simulated_geometry_from_scan(tmp_path, simulated_truth):
    """``simulated_geometry`` reports the slice thickness and maps the layout."""
    from croak.session.pipeline import simulated_geometry

    path = write_simulated_h5(
        tmp_path / "sd.h5",
        simulated_truth,
        zsave=SIMULATED_ZSAVE,
        geometry=(0.95e-3, 3.4e-3, "sd"),
    )
    geom = simulated_geometry(io.read_simulated_scan(path, z_thickness=10e-6))
    assert geom.thickness_um == pytest.approx(10.0)
    assert geom.mask_diameter_mm == pytest.approx(0.95)
    assert geom.mask_spacing_mm == pytest.approx(3.4)
    # a two-beam SD simulation must not get the boxcars kernel: it has no
    # gate-shape channel at all, so the p width would be pure invention
    assert geom.layout == "sd2"
    assert geom.has_mask

    tg = simulated_geometry(
        io.read_simulated_scan(
            write_simulated_h5(
                tmp_path / "tg.h5", simulated_truth, geometry=(1e-3, 1e-3, "tg")
            )
        )
    )
    assert tg.layout == "boxcars"

    # a legacy file: thickness still known, mask honestly unknown
    old = simulated_geometry(
        io.read_simulated_scan(
            write_simulated_h5(
                tmp_path / "old.h5", simulated_truth, zsave=SIMULATED_ZSAVE
            ),
            z_thickness=20e-6,
        )
    )
    assert old.thickness_um == pytest.approx(20.0)
    assert not old.has_mask
    assert old.mask_diameter_mm is None


def test_window_keys_include_numbered_apertures(tmp_path, simulated_truth):
    """Multi-aperture scans list every stored window, canonical trio first.

    One propagation reduced through several collection holes stores
    ``Iω_win_2`` … with ``_reimaged`` partners; the selector must offer them
    (they load through ``read_simulated_scan(window_key=...)`` like any other).
    """
    path = write_simulated_h5(tmp_path / "multi.h5", simulated_truth)
    with h5py.File(path, "a") as f:
        base = f["Iω_win"][()]
        for k in ("Iω_win_3", "Iω_win_2_reimaged", "Iω_win_2"):
            f[k] = 0.5 * base
    keys = io.read_simulated_window_keys(path)
    assert keys[:2] == ["Iω_win", "Iω_win_reimaged"]
    tail = keys[2:]
    assert tail == ["Iω_win_2", "Iω_win_2_reimaged", "Iω_win_3"]
    # and the numbered window actually loads
    scan = io.read_simulated_scan(path, window_key="Iω_win_2")
    assert scan.trace.shape == io.read_simulated_scan(path).trace.shape


def test_window_def_prefix_mapping():
    """Selector -> record prefix, for multi-aperture scans."""
    assert io.window_def_prefix(None) == "window_def_"
    assert io.window_def_prefix("Iω_win") == "window_def_"
    assert io.window_def_prefix("Iω_win_reimaged") == "window_def_"
    assert io.window_def_prefix("Iω_win_5") == "window_def_5_"
    assert io.window_def_prefix("Iω_win_5_reimaged") == "window_def_5_"
    assert io.window_def_prefix(1) == "window_def_"
    assert io.window_def_prefix(3) == "window_def_3_"
    with pytest.raises(ValueError, match="cannot map"):
        io.window_def_prefix("Iω_full")


def test_mask_window_selects_numbered_record(tmp_path, simulated_truth):
    """A multi-aperture file's numbered window rebuilds its own hole."""
    mask_window = dict(
        type="PhysicalMaskWindow",
        holex=-1.0e-3,
        holey=-1.0e-3,
        holediam=0.5e-3,
        zmask=0.1,
        apod="tanh",
        apod_param=96.9e-6,
        delta_k=7803.6,
        reference_wavelength=260e-9,
    )
    path = write_simulated_h5(
        tmp_path / "multiwin.h5", simulated_truth, mask_window=mask_window
    )
    with h5py.File(path, "a") as f:
        g = f["grid"]
        for key in ("type", "holex", "holey", "zmask", "apod", "apod_param"):
            g[f"window_def_2_{key}"] = g[f"window_def_{key}"][()]
        g["window_def_2_holediam"] = 2.0e-3  # the wide hole
    base = io.read_simulated_mask_window(path)
    wide = io.read_simulated_mask_window(path, "Iω_win_2")
    assert base is not None and wide is not None
    assert base.hole_diameter == pytest.approx(0.5e-3)
    assert wide.hole_diameter == pytest.approx(2.0e-3)
    assert wide.hole_x == base.hole_x
    # absent numbered record -> None, not a half-built aperture
    assert io.read_simulated_mask_window(path, "Iω_win_3") is None
