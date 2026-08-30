"""Tests for :mod:`croak.save`."""

import numpy as np
import pytest

from croak import save
from croak.forward import maketrace
from croak.grid import Grid
from croak.maths import wlfreq
from croak.pulses import gaussian_pulse
from croak.retrieve import retrieve
from croak.uncertainty import UncertaintyResult


def _mk_uresult(method, *, pe=1.5e-15, sd=0.1e-15, n=200, profiles=False, seed=0):
    """A lightweight UncertaintyResult for save tests (no retrieval)."""
    rng = np.random.default_rng(seed)
    samples = pe + sd * rng.standard_normal(n)
    prof = t = None
    if profiles:
        t = np.linspace(-15e-15, 15e-15, 40)
        base = np.exp(-((t / pe) ** 2))
        prof = np.clip(base + 0.02 * rng.standard_normal((n, t.size)), 0.0, None)
    return UncertaintyResult(
        statistic="fwhm",
        point_estimate=pe,
        samples=samples,
        interval_68=(pe - sd, pe + sd),
        interval_95=(pe - 2 * sd, pe + 2 * sd),
        interval_method="percentile",
        method=method,
        n_resamples=n,
        n_converged=n,
        frog_errors=np.empty(0),
        base_error=0.01,
        profiles=prof,
        t_profile=t,
    )


@pytest.fixture
def result_and_trace():
    g = Grid(64, dt=0.5e-15)
    omega0 = float(wlfreq(800e-9))
    ew = gaussian_pulse(g, 7e-15)
    delays = np.linspace(-40e-15, 40e-15, 50)
    trace = maketrace(g.omega, delays, ew, "pg")
    res = retrieve(
        trace,
        g.omega,
        delays,
        "pg",
        algorithm="copra",
        maxiters=40,
        guess=ew,
        omega0=omega0,
    )
    return res, trace


def test_save_and_load_result(tmp_path, result_and_trace):
    res, trace = result_and_trace
    path = save.save_result(res, str(tmp_path / "r.h5"), measured=trace)
    data = save.load_result(path)
    assert data["interaction"] == "pg"
    assert data["algorithm"] == "copra"
    assert np.iscomplexobj(data["Ew_retr"])
    assert data["Ew_retr"].shape == res.spectrum.shape
    assert data["error_final"] == pytest.approx(res.error)
    assert "Ilam_retr" in data and "gdd_fs2" in data  # processed fields present
    assert data["trace_meas"].shape == trace.shape


def test_save_result_with_truth_and_rehydrate(tmp_path, result_and_trace):
    """A saved truth round-trips, and result_from_saved skips the solver re-run."""
    from croak.processing import TruthPulse

    res, trace = result_and_trace
    truth = TruthPulse.from_spectrum(res.grid, res.spectrum, res.omega0)
    path = save.save_result(res, str(tmp_path / "r.h5"), measured=trace, truth=truth)
    data = save.load_result(path)
    # truth arrays are stored
    assert data["It_truth"].shape == truth.It.shape
    assert data["fwhm_truth_fs"] == pytest.approx(truth.fwhm / 1e-15)
    assert "lam_truth" in data and "Ilam_truth" in data
    assert "phit_truth" in data and "phiw_truth" in data
    assert "omega_truth" in data and "Ew_truth" in data
    assert "tau0" in data  # extras for a faithful reload

    # rehydration reproduces the retrieval without re-running it
    rebuilt = save.result_from_saved(data, grid=res.grid)
    np.testing.assert_allclose(rebuilt.spectrum, res.spectrum)
    np.testing.assert_allclose(rebuilt.delays, res.delays)
    assert rebuilt.interaction == res.interaction
    assert rebuilt.algorithm == res.algorithm
    assert rebuilt.error == pytest.approx(res.error)
    np.testing.assert_allclose(rebuilt.grid.omega, res.grid.omega)


def test_result_from_saved_grid_mismatch_raises(tmp_path, result_and_trace):
    res, trace = result_and_trace
    data = save.load_result(
        save.save_result(res, str(tmp_path / "r.h5"), measured=trace)
    )
    with pytest.raises(ValueError, match="grid size"):
        save.result_from_saved(data, grid=Grid(8, dt=0.5e-15))


def test_save_result_includes_peak_power(tmp_path, result_and_trace):
    from croak.processing import process_result

    res, trace = result_and_trace
    pr = process_result(res, measured=trace, energy=100e-6)
    path = save.save_result(
        res, str(tmp_path / "r.h5"), processed=pr, measured=trace, force=True
    )
    data = save.load_result(path)
    assert data["pulse_energy_J"] == pytest.approx(100e-6)
    assert data["peak_power_W"] == pytest.approx(pr.peak_power)
    assert data["peak_power_tl_W"] == pytest.approx(pr.peak_power_tl)


def test_save_result_omits_peak_power_without_energy(tmp_path, result_and_trace):
    res, trace = result_and_trace
    path = save.save_result(res, str(tmp_path / "r.h5"), measured=trace)
    data = save.load_result(path)
    assert "peak_power_W" not in data
    assert "pulse_energy_J" not in data


def test_save_result_no_overwrite(tmp_path, result_and_trace):
    res, _ = result_and_trace
    p = str(tmp_path / "r.h5")
    save.save_result(res, p)
    with pytest.raises(FileExistsError):
        save.save_result(res, p)
    save.save_result(res, p, force=True)  # ok


def test_save_result_groups_share_one_file(tmp_path, result_and_trace):
    """Several results live in one file, one group each, and load back nested."""
    res, trace = result_and_trace
    p = str(tmp_path / "sweep.h5")
    save.save_result(res, p, measured=trace, group="full/z00")
    save.save_result(res, p, measured=trace, group="naive/z00")
    data = save.load_result(p)
    # both groups present, each a complete flat result (attrs included)
    for grp in ("full", "naive"):
        entry = data[grp]["z00"]
        assert entry["interaction"] == "pg"
        assert entry["error_final"] == pytest.approx(res.error)
        assert entry["trace_meas"].shape == trace.shape
    # a grouped write must not truncate the file: writing the second kept the first
    assert set(data) == {"full", "naive"}


def test_save_result_group_and_root_metadata_coexist(tmp_path, result_and_trace):
    """Root-level metadata written by the caller survives a grouped write."""
    import h5py

    res, trace = result_and_trace
    p = str(tmp_path / "sweep.h5")
    with h5py.File(p, "w") as f:
        f["thickness_um"] = np.array([0.0, 20.0])
        f.attrs["frog_path"] = "scan.h5"
    save.save_result(res, p, measured=trace, group="full/z00")
    data = save.load_result(p)
    np.testing.assert_allclose(data["thickness_um"], [0.0, 20.0])
    assert data["frog_path"] == "scan.h5"
    assert "Ew_retr" in data["full"]["z00"]


def test_save_result_group_no_overwrite(tmp_path, result_and_trace):
    """An existing group needs force; force replaces only that group."""
    res, trace = result_and_trace
    p = str(tmp_path / "sweep.h5")
    save.save_result(res, p, measured=trace, group="full/z00")
    save.save_result(res, p, measured=trace, group="full/z01")
    with pytest.raises(FileExistsError, match="full/z00"):
        save.save_result(res, p, measured=trace, group="full/z00")
    save.save_result(res, p, measured=trace, group="full/z00", force=True)
    assert set(save.load_result(p)["full"]) == {"z00", "z01"}


def test_save_uncertainty_appends_group_with_bands(tmp_path, result_and_trace):
    res, trace = result_and_trace
    p = save.save_result(res, str(tmp_path / "r.h5"), measured=trace)
    a = _mk_uresult("parametric", profiles=True, seed=1)
    b = _mk_uresult("thickness", sd=0.2e-15, profiles=True, seed=2)
    save.save_uncertainty({"parametric": a, "thickness": b}, p, force=True)

    data = save.load_result(p)
    assert data["algorithm"] == "copra"  # original retrieval preserved
    u = data["uncertainty"]
    assert set(u) == {"parametric", "thickness"}
    assert u["thickness"]["point_estimate_fs"] == pytest.approx(
        b.point_estimate / 1e-15
    )
    assert u["parametric"]["interval68_fs"].shape == (2,)
    # the temporal confidence band is stored as median + lower/upper envelopes
    for key in ("band_median", "band_lo68", "band_hi68", "band_lo95", "band_hi95"):
        assert key in u["parametric"]
    assert u["parametric"]["t_profile_fs"].shape == u["parametric"]["band_median"].shape


def test_save_uncertainty_standalone_force_and_empty(tmp_path):
    a = _mk_uresult("parametric", seed=3)
    p = str(tmp_path / "u.h5")
    save.save_uncertainty(a, p)  # single result wrapped under its method
    with pytest.raises(FileExistsError):
        save.save_uncertainty(a, p)  # group exists, no force
    save.save_uncertainty({"parametric": a}, p, force=True)  # ok
    assert "parametric" in save.load_result(p)["uncertainty"]
    with pytest.raises(ValueError):
        save.save_uncertainty({}, str(tmp_path / "e.h5"))


def test_options_roundtrip(tmp_path):
    options = {
        "load": {"frog_path": "/x/y.h5", "interaction": "SHG", "transpose": True},
        "preproc": {"lam_min_nm": 700.0, "lam_max_nm": 900.0, "filter_dc": False},
        "retrieve": {
            "solver": "lbfgs",
            "maxiters": 100,
            "weights": np.array([1.0, 0.0]),
        },
    }
    path = save.save_options(options, str(tmp_path / "options.toml"))
    loaded = save.load_options(path)
    assert loaded["load"]["interaction"] == "SHG"
    assert loaded["preproc"]["lam_min_nm"] == pytest.approx(700.0)
    assert loaded["retrieve"]["weights"] == [1.0, 0.0]
    # version header present for forward-compatible reloads
    assert "croak_version" in loaded
    assert loaded["options_version"] == save.OPTIONS_VERSION


def test_split_smear_scales_round_trip(tmp_path, result_and_trace):
    """Both split smearing multipliers survive a save/load/rehydrate cycle."""
    from dataclasses import replace

    res, trace = result_and_trace
    res = replace(res, smear_scale=1.31, smear_scale_delta=0.87)
    path = save.save_result(res, str(tmp_path / "r.h5"), measured=trace)
    data = save.load_result(path)
    assert data["smear_scale"] == pytest.approx(1.31)
    assert data["smear_scale_delta"] == pytest.approx(0.87)
    rebuilt = save.result_from_saved(data, grid=res.grid)
    assert rebuilt.smear_scale == pytest.approx(1.31)
    assert rebuilt.smear_scale_delta == pytest.approx(0.87)
