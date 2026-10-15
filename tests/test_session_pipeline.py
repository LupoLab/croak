"""Tests for the loader and stage helpers of :mod:`croak.session.pipeline`.

The end-to-end replay is exercised in ``test_session.py``; here each option of
:func:`~croak.session.pipeline.assemble_load_data` (orientation, delay sign,
FROG background, calibration curves, independent spectrum) and the smaller
stage helpers are checked against the arithmetic they promise.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from croak import io
from croak.forward import maketrace, quadrature_nodes_weights
from croak.grid import Grid
from croak.maths import wlfreq
from croak.pulses import gaussian_pulse
from croak.result import RetrievalResult
from croak.retrieve import retrieve
from croak.session import pipeline as session_pipeline
from croak.session.dispersion import transfer_function
from croak.session.params import (
    DispersionParams,
    LoadParams,
    RetrieveParams,
    SimulatedLoadParams,
    UncertaintyParams,
)
from croak.session.pipeline import (
    apply_post_filter,
    assemble_load_data,
    assemble_simulated_load_data,
    assemble_simulated_tracedata,
    extra_param_centres,
    focal_mixture_from_params,
    generation_envelope,
    resolve_guess,
    run_retrieval,
    run_uncertainty,
    simulated_geometry,
    smearing_kernel,
)

LAM_NM = np.linspace(700.0, 900.0, 21)
DELAY_FS = np.linspace(-40.0, 40.0, 9)


def _trace(n_lam: int = LAM_NM.size, n_delay: int = DELAY_FS.size) -> np.ndarray:
    """A strictly positive (Nlambda, Ndelay) trace with no symmetry to hide bugs."""
    rng = np.random.default_rng(1)
    return 0.5 + rng.random((n_lam, n_delay))


def _write_frog(path, trace, lam_nm=LAM_NM, delay_fs=DELAY_FS) -> str:
    with h5py.File(path, "w") as f:
        f["wavelength"] = lam_nm
        f["delay"] = delay_fs
        f["trace"] = trace
    return str(path)


def _params(frog_path: str, **kw) -> LoadParams:
    return LoadParams(
        frog_path=frog_path,
        ifrog_name="trace",
        lam_name="wavelength",
        scan_name="delay",
        lam_unit="nm",
        scan_unit="fs",
        **kw,
    )


def _write_curve(path, lam, values) -> str:
    np.savetxt(path, np.column_stack([lam, values]), delimiter=",")
    return str(path)


def test_assemble_load_data_normalises_and_converts_units(tmp_path):
    trace = _trace()
    d = assemble_load_data(_params(_write_frog(tmp_path / "f.h5", trace)))
    np.testing.assert_allclose(d["lam"], LAM_NM * 1e-9)
    # the scan axis is centred and sorted (already both here); centring leaves
    # ~1e-30 s of rounding at zero delay
    np.testing.assert_allclose(d["scanaxis"], DELAY_FS * 1e-15, atol=1e-25)
    np.testing.assert_allclose(d["trace"], trace / trace.max())
    assert d["input_unit"] == "delay"
    assert d["lam_spec"] is None and d["Ilam_spec"] is None


def test_assemble_load_data_orients_a_delay_major_trace(tmp_path):
    """A file stored (Ndelay, Nlambda) is transposed to (Nlambda, Ndelay)."""
    trace = _trace()
    d = assemble_load_data(_params(_write_frog(tmp_path / "f.h5", trace.T)))
    np.testing.assert_allclose(d["trace"], trace / trace.max())


def test_assemble_load_data_transpose_flag_for_square_traces(tmp_path):
    """Shape cannot tell a square trace's orientation; ``transpose`` does."""
    lam = np.linspace(700.0, 900.0, 9)
    trace = _trace(9, 9)
    path = _write_frog(tmp_path / "f.h5", trace, lam_nm=lam)
    plain = assemble_load_data(_params(path))
    flipped = assemble_load_data(_params(path, transpose=True))
    np.testing.assert_allclose(flipped["trace"], plain["trace"].T)


def test_assemble_load_data_reverse_trace_mirrors_the_delay_axis(tmp_path):
    trace = _trace()
    path = _write_frog(tmp_path / "f.h5", trace)
    d = assemble_load_data(_params(path, reverse_trace=True))
    np.testing.assert_allclose(d["scanaxis"], DELAY_FS * 1e-15, atol=1e-25)
    np.testing.assert_allclose(d["trace"], trace[:, ::-1] / trace.max())


def test_assemble_load_data_position_scan_sets_the_unit(tmp_path):
    d = assemble_load_data(
        _params(_write_frog(tmp_path / "f.h5", _trace()), scan_type="position")
    )
    assert d["input_unit"] == "position"


def test_assemble_load_data_all_zero_trace_is_left_unscaled(tmp_path):
    trace = np.zeros((LAM_NM.size, DELAY_FS.size))
    d = assemble_load_data(_params(_write_frog(tmp_path / "f.h5", trace)))
    np.testing.assert_array_equal(d["trace"], trace)


def test_assemble_load_data_subtracts_the_frog_background(tmp_path):
    """The background's own λ axis is sliced to the trace's λ range."""
    trace = _trace()
    lam_bg = np.linspace(650.0, 950.0, 31)  # same 10 nm step, wider span
    bg = 0.01 * lam_bg / 800.0
    with h5py.File(tmp_path / "bg.h5", "w") as f:
        f["lam"] = lam_bg
        f["bg"] = bg
    p = _params(
        _write_frog(tmp_path / "f.h5", trace),
        frog_bg_use=True,
        frog_bg_path=str(tmp_path / "bg.h5"),
        frog_bg_lam_name="lam",
        frog_bg_int_name="bg",
    )
    d = assemble_load_data(p)
    inside = (lam_bg >= 700.0) & (lam_bg <= 900.0)
    expected = trace - bg[inside][:, None]
    np.testing.assert_allclose(d["trace"], expected / expected.max())


def test_assemble_load_data_rejects_a_mismatched_frog_background(tmp_path):
    with h5py.File(tmp_path / "bg.h5", "w") as f:
        f["lam"] = np.linspace(700.0, 900.0, 7)  # coarser grid: overlap is 7 ≠ 21
        f["bg"] = np.zeros(7)
    p = _params(
        _write_frog(tmp_path / "f.h5", _trace()),
        frog_bg_use=True,
        frog_bg_path=str(tmp_path / "bg.h5"),
        frog_bg_lam_name="lam",
        frog_bg_int_name="bg",
    )
    with pytest.raises(ValueError, match="background λ overlap has 7 points"):
        assemble_load_data(p)


@pytest.mark.parametrize(("unit", "factor"), [("nm", 1.0), ("µm", 1e-3)])
def test_assemble_load_data_frog_calibration_honours_its_unit(tmp_path, unit, factor):
    """Regression: a µm calibration file is read at the right wavelengths.

    The curve's wavelengths were once scaled by unit(calib)/unit(trace) instead of
    being converted to nm, so only nm files applied the intended correction.
    """
    trace = _trace()
    curve_nm = np.linspace(600.0, 1000.0, 41)
    scale = 1.0 + (curve_nm - 600.0) / 400.0  # 1 at 600 nm → 2 at 1000 nm
    p = _params(
        _write_frog(tmp_path / "f.h5", trace),
        frog_calib_use=True,
        frog_calib_path=_write_curve(tmp_path / "cal.csv", curve_nm * factor, scale),
        frog_calib_lam_unit=unit,
    )
    d = assemble_load_data(p)
    expected_scale = 1.0 + (LAM_NM - 600.0) / 400.0
    expected = trace * expected_scale[:, None]
    # the spline reproduces a linear curve to rounding
    np.testing.assert_allclose(d["trace"], expected / expected.max(), rtol=1e-10)


def test_assemble_load_data_third_order_scaling(tmp_path):
    trace = _trace()
    p = _params(
        _write_frog(tmp_path / "f.h5", trace), third_order=True, third_order_exp=2.0
    )
    d = assemble_load_data(p)
    expected = trace * ((LAM_NM * 1e-3) ** 2)[:, None]
    np.testing.assert_allclose(d["trace"], expected / expected.max(), rtol=1e-12)


def test_assemble_load_data_reads_a_csv_spectrum_with_background_and_calibration(
    tmp_path,
):
    lam = np.linspace(750.0, 850.0, 11)
    spec = np.exp(-(((lam - 800.0) / 20.0) ** 2))
    bg = np.full_like(lam, 0.05)
    p = _params(
        _write_frog(tmp_path / "f.h5", _trace()),
        use_spec=True,
        spec_path=_write_curve(tmp_path / "spec.csv", lam, spec),
        spec_bg_use=True,
        spec_bg_path=_write_curve(tmp_path / "bg.csv", lam, bg),
        spec_calib_use=True,
        spec_calib_path=_write_curve(
            tmp_path / "cal.csv", np.linspace(0.6, 1.0, 41), np.linspace(1, 2, 41)
        ),
        spec_calib_lam_unit="µm",
    )
    d = assemble_load_data(p)
    expected = (spec - bg) * (1.0 + (lam - 600.0) / 400.0)
    np.testing.assert_allclose(d["lam_spec"], lam * 1e-9)
    # the spline reproduces a linear calibration curve to rounding
    np.testing.assert_allclose(d["Ilam_spec"], expected / expected.max(), rtol=1e-10)


def test_assemble_load_data_ignores_a_spectrum_background_of_another_length(tmp_path):
    lam = np.linspace(750.0, 850.0, 11)
    spec = 1.0 + lam / 1000.0
    p = _params(
        _write_frog(tmp_path / "f.h5", _trace()),
        use_spec=True,
        spec_path=_write_curve(tmp_path / "spec.csv", lam, spec),
        spec_bg_use=True,
        spec_bg_path=_write_curve(tmp_path / "bg.csv", lam[:5], np.ones(5)),
    )
    d = assemble_load_data(p)
    np.testing.assert_allclose(d["Ilam_spec"], spec / spec.max())


def test_assemble_load_data_reads_an_hdf5_spectrum(tmp_path):
    lam = np.linspace(750.0, 850.0, 11)
    shots = np.stack([lam / 800.0, lam / 800.0 + 0.2], axis=1)  # (Nlambda, Nshots)
    with h5py.File(tmp_path / "spec.h5", "w") as f:
        f["lam"] = lam
        f["I"] = shots
    p = _params(
        _write_frog(tmp_path / "f.h5", _trace()),
        use_spec=True,
        spec_path=str(tmp_path / "spec.h5"),
        spec_lam_name="lam",
        spec_int_name="I",
    )
    d = assemble_load_data(p)
    mean = shots.mean(axis=1)
    np.testing.assert_allclose(d["Ilam_spec"], mean / mean.max())


def test_assemble_load_data_rejects_spectrum_length_mismatch(tmp_path):
    with h5py.File(tmp_path / "spec.h5", "w") as f:
        f["lam"] = np.linspace(750.0, 850.0, 11)
        f["I"] = np.ones(7)
    p = _params(
        _write_frog(tmp_path / "f.h5", _trace()),
        use_spec=True,
        spec_path=str(tmp_path / "spec.h5"),
        spec_lam_name="lam",
        spec_int_name="I",
    )
    with pytest.raises(ValueError, match="lengths differ"):
        assemble_load_data(p)


def test_assemble_load_data_zero_spectrum_is_left_unscaled(tmp_path):
    lam = np.linspace(750.0, 850.0, 11)
    p = _params(
        _write_frog(tmp_path / "f.h5", _trace()),
        use_spec=True,
        spec_path=_write_curve(tmp_path / "spec.csv", lam, np.zeros(11)),
    )
    d = assemble_load_data(p)
    np.testing.assert_array_equal(d["Ilam_spec"], np.zeros(11))


# ---------------------------------------------------------------------------
# Stage helpers: guess, warm-start centres, focal guards, uncertainty dispatch
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("flags", "override", "expected"),
    [
        ({}, None, "auto"),
        ({"tl_init": True}, None, "tl"),
        ({"reuse_result": True}, None, "auto"),  # nothing compatible to reuse
        ({"reuse_result": True}, "spectrum", "spectrum"),
        ({"random_init": True}, "spectrum", "spectrum"),  # explicit override wins
    ],
)
def test_resolve_guess_modes(flags, override, expected):
    assert resolve_guess(RetrieveParams(**flags), override) == expected


def _seed(**kw) -> RetrievalResult:
    g = Grid(8, dt=1e-15)
    defaults = dict(
        spectrum=np.ones(8, complex),
        grid=g,
        delays=np.zeros(1),
        interaction="pg",
        algorithm="lbfgs-ad",
        error=0.0,
        thickness=12e-6,
        tau0=3e-15,
        smear_scale=1.2,
        smear_scale_delta=0.8,
    )
    defaults.update(kw)
    return RetrievalResult(**defaults)


def test_extra_param_centres_cold_start_uses_nominal_values():
    p = RetrieveParams(dispersive=True, thickness_um=10.0)
    assert extra_param_centres(p, None) == (pytest.approx(10e-6), 0.0, 1.0, 1.0)
    assert extra_param_centres(RetrieveParams(), None) == (0.0, 0.0, 1.0, 1.0)


def test_extra_param_centres_warm_start_only_refits_ticked_extras():
    fit_all = RetrieveParams(
        dispersive=True,
        thickness_um=10.0,
        smearing=True,
        fit_thickness=True,
        fit_tau0=True,
        fit_smearing=True,
        fit_smearing_split=True,
    )
    assert extra_param_centres(fit_all, _seed()) == (12e-6, 3e-15, 1.2, 0.8)
    # unticked: the explicit nominal values, not the last fit's drift
    held = RetrieveParams(dispersive=True, thickness_um=10.0, smearing=True)
    assert extra_param_centres(held, _seed()) == (pytest.approx(10e-6), 0.0, 1.0, 1.0)


def test_extra_param_centres_ignores_a_seed_without_fitted_values():
    p = RetrieveParams(
        dispersive=True,
        thickness_um=10.0,
        smearing=True,
        fit_thickness=True,
        fit_smearing=True,
        fit_smearing_split=True,
    )
    seed = _seed(thickness=None, smear_scale=None, smear_scale_delta=None)
    assert extra_param_centres(p, seed) == (pytest.approx(10e-6), 0.0, 1.0, 1.0)


def test_focal_mixture_rejects_smearing_and_non_pg():
    td = SimpleNamespace(interaction="pg", omega0_pulse=2.35e15)
    assert focal_mixture_from_params(RetrieveParams(), td) is None
    with pytest.raises(ValueError, match="mutually exclusive"):
        focal_mixture_from_params(RetrieveParams(focal=True, smearing=True), td)
    td_shg = SimpleNamespace(interaction="shg", omega0_pulse=2.35e15)
    with pytest.raises(ValueError, match="interaction is 'shg'"):
        focal_mixture_from_params(RetrieveParams(focal=True), td_shg)


@pytest.fixture(scope="module")
def tiny_retrieval():
    """A small measured-style trace and a short converged-ish retrieval of it."""
    g = Grid(64, dt=0.8e-15)
    ew = gaussian_pulse(g, 6e-15, phases=(10e-30,))
    delays = np.linspace(-30e-15, 30e-15, 31)
    trace = maketrace(g.omega, delays, ew, "pg")
    result = retrieve(
        trace,
        g.omega,
        delays,
        "pg",
        algorithm="copra",
        guess=ew,
        maxiters=5,
        omega0=wlfreq(800e-9),
        progress=False,
    )
    td = SimpleNamespace(trace=trace, Iomega=None, interaction="pg")
    return result, td


@pytest.fixture
def spies(monkeypatch):
    """Record what run_uncertainty hands each estimator (tested elsewhere)."""
    calls: dict[str, dict] = {}

    def fake(name):
        def estimator(result, measured, **kw):
            calls[name] = dict(kw, result=result, measured=measured)
            return name

        return estimator

    monkeypatch.setattr(session_pipeline, "covariance_uncertainty", fake("cov"))
    monkeypatch.setattr(session_pipeline, "estimate_fwhm_uncertainty", fake("boot"))
    return calls


@pytest.mark.parametrize("noise_source", ["residual", "manual"])
def test_run_uncertainty_covariance_dispatch(tiny_retrieval, spies, noise_source):
    result, td = tiny_retrieval
    up = UncertaintyParams(
        method="covariance", noise_source=noise_source, noise_sigma=0.02
    )
    assert run_uncertainty(up, RetrieveParams(), result, td) == "cov"
    kw = spies["cov"]
    assert kw["measured"] is td.trace
    assert kw["propagation"] is None and kw["propagation_label"] is None
    assert kw["omega0"] == result.omega0
    if noise_source == "manual":
        assert kw["noise"].sigma == 0.02
    else:
        # the residual of a near-exact fit to a noiseless trace is tiny
        assert kw["noise"].sigma < 0.02


def test_run_uncertainty_parametric_propagates_to_the_dispersion_point(
    tiny_retrieval, spies
):
    result, td = tiny_retrieval
    up = UncertaintyParams(
        method="parametric",
        noise_source="manual",
        noise_sigma=0.03,
        propagate_to_dispersion_point=True,
    )
    dispersion = DispersionParams(gdd_fs2=20.0)
    run_uncertainty(up, RetrieveParams(), result, td, dispersion=dispersion)
    kw = spies["boot"]
    assert kw["method"] == "parametric" and kw["noise"].sigma == 0.03
    np.testing.assert_array_equal(
        kw["propagation"], transfer_function(dispersion, result.grid, result.omega0)
    )
    assert kw["propagation_label"] == "dispersion-stage point"


def test_run_uncertainty_skips_propagation_without_dispersion(tiny_retrieval, spies):
    result, td = tiny_retrieval
    up = UncertaintyParams(method="delay", propagate_to_dispersion_point=True)
    run_uncertainty(up, RetrieveParams(), result, td, dispersion=DispersionParams())
    kw = spies["boot"]
    assert kw["propagation"] is None and kw["noise"] is None


def test_run_uncertainty_parametric_residual_noise(tiny_retrieval, spies):
    result, td = tiny_retrieval
    run_uncertainty(
        UncertaintyParams(method="parametric"), RetrieveParams(), result, td
    )
    assert spies["boot"]["noise"].sigma < 0.02


def test_run_uncertainty_thickness_drops_the_thickness_fit(tiny_retrieval, spies):
    """The thickness systematic and a thickness fit are alternatives, never both."""
    result, td = tiny_retrieval
    rp = RetrieveParams(
        dispersive=True, thickness_um=20.0, fit_thickness=True, polish_two_phase=True
    )
    up = UncertaintyParams(method="thickness", thickness_sigma_um=2.0)
    run_uncertainty(up, rp, result, td)
    kw = spies["boot"]
    assert kw["central_thickness"] == pytest.approx(20e-6)
    assert kw["thickness_sigma"] == pytest.approx(2e-6)
    assert kw["material"] == rp.material
    for dropped in ("thickness", "fit_thickness", "polish"):
        assert dropped not in kw


# ---------------------------------------------------------------------------
# Simulated scans: edge cases on a real scan read from disk, then modified
# ---------------------------------------------------------------------------
@pytest.fixture
def stored_scan(simulated_scan_h5, monkeypatch):
    """Return ``(scan, use)``: a real scan and a setter making the loader read it."""
    scan = io.read_simulated_scan(simulated_scan_h5)

    def use(modified):
        monkeypatch.setattr(io, "read_simulated_scan", lambda *a, **k: modified)

    return scan, use


def _sim_params(**kw) -> SimulatedLoadParams:
    base = dict(
        frog_path="unused.h5",
        third_order=False,
        reverse_trace=False,
        use_spectrum=True,
        lam_min_nm=140.0,
        lam_max_nm=900.0,
    )
    base.update(kw)
    return SimulatedLoadParams(**base)


def test_uniform_delay_core_is_a_no_op_on_a_uniform_scan(stored_scan):
    scan, use = stored_scan
    use(scan)
    plain = assemble_simulated_load_data(_sim_params())
    core = assemble_simulated_load_data(_sim_params(uniform_delay_core=True))
    np.testing.assert_array_equal(core["scanaxis"], plain["scanaxis"])
    np.testing.assert_array_equal(core["trace"], plain["trace"])


def test_simulated_zero_trace_and_spectrum_are_left_unscaled(stored_scan):
    """A blank scan stays blank instead of dividing by a zero peak."""
    scan, use = stored_scan
    use(
        replace(
            scan,
            trace=np.zeros_like(scan.trace),
            Iomega=np.zeros_like(scan.Iomega),
            Iomega_beamlet=None,
        )
    )
    d = assemble_simulated_load_data(_sim_params())
    assert not np.any(d["trace"]) and not np.any(d["Ilam_spec"])


def test_simulated_tracedata_zero_reference_spectrum_is_left_unscaled(stored_scan):
    scan, use = stored_scan
    use(replace(scan, Iomega=np.zeros_like(scan.Iomega), Iomega_beamlet=None))
    td = assemble_simulated_tracedata(_sim_params())
    assert not np.any(td.Iomega)


def test_simulated_tracedata_rejects_a_non_fft_frequency_axis(stored_scan):
    scan, use = stored_scan
    warped = scan.omega + 1e-3 * (scan.omega - scan.omega0) ** 2 / scan.omega0
    use(replace(scan, omega=warped))
    with pytest.raises(ValueError, match="not a uniform FFT grid"):
        assemble_simulated_tracedata(_sim_params())


def test_simulated_geometry_reads_the_recorded_instrument(stored_scan):
    scan, _ = stored_scan
    geo = simulated_geometry(
        replace(
            scan,
            z_positions=np.array([0.0, 20e-6]),
            z_index=1,
            geometry="SD",
            mask_diameter=1e-3,
            mask_spacing=4e-3,
            f_foc=0.5,
        )
    )
    assert geo.thickness_um == pytest.approx(20.0)
    assert geo.layout == "sd2"
    assert geo.has_mask and geo.mask_spacing_mm == pytest.approx(4.0)
    assert geo.f_foc_mm == pytest.approx(500.0)
    assert simulated_geometry(replace(scan, geometry="boxcars")).layout == "boxcars"


def test_simulated_geometry_tolerates_an_out_of_range_slice(stored_scan):
    scan, _ = stored_scan
    geo = simulated_geometry(replace(scan, z_positions=np.array([0.0]), z_index=5))
    assert geo.thickness_um is None
    assert not geo.has_mask


def test_reimaged_beamlet_is_preferred_when_stored(stored_scan):
    scan, _ = stored_scan
    reimaged = 0.5 * scan.Iomega
    e_reimaged = np.sqrt(reimaged).astype(complex)
    modified = replace(
        scan, Iomega_beamlet_reimaged=reimaged, Eomega_beamlet_reimaged=e_reimaged
    )
    assert (
        session_pipeline._reference_spectrum(modified, "beamlet_reimaged") is reimaged
    )
    assert session_pipeline._truth_field(modified, "beamlet_reimaged") is e_reimaged


def test_truth_temporal_phase_needs_light(stored_scan):
    """No field, or a temporal reference with under two bright samples: no phase."""
    scan, _ = stored_scan
    blank = np.zeros(scan.omega.size, complex)
    assert session_pipeline._truth_temporal_phase(scan, blank) is None
    spike = np.zeros_like(scan.It)
    spike[0] = 1.0
    field = np.ones(scan.omega.size, complex)
    assert (
        session_pipeline._truth_temporal_phase(replace(scan, It=spike), field) is None
    )


def test_simulated_truth_with_a_blank_stored_field_has_no_phase(stored_scan):
    scan, _ = stored_scan
    blank = np.zeros(scan.omega.size, complex)
    truth = session_pipeline._simulated_truth(
        replace(scan, Eomega=blank, Eomega_beamlet=blank), "source"
    )
    assert truth.phi_w is None and truth.phi_t is None


def test_simulated_tracedata_zero_trace_is_left_unscaled(stored_scan):
    scan, use = stored_scan
    use(replace(scan, trace=np.zeros_like(scan.trace)))
    td = assemble_simulated_tracedata(_sim_params(use_spectrum=False))
    assert not np.any(td.trace)


def test_assemble_load_data_zero_trace_with_a_scale_curve(tmp_path):
    """The post-calibration renormalisation also skips a blank trace."""
    trace = np.zeros((LAM_NM.size, DELAY_FS.size))
    p = _params(_write_frog(tmp_path / "f.h5", trace), third_order=True)
    np.testing.assert_array_equal(assemble_load_data(p)["trace"], trace)


# ---------------------------------------------------------------------------
# Forward-model options built from the parameters
# ---------------------------------------------------------------------------
TD_PG = SimpleNamespace(interaction="pg", omega0_pulse=wlfreq(800e-9))


def test_smearing_kernel_layouts_and_fixed_scale():
    assert smearing_kernel(RetrieveParams(), TD_PG) is None
    boxcars = smearing_kernel(RetrieveParams(smearing=True), TD_PG)
    sd2 = smearing_kernel(RetrieveParams(smearing=True, smear_layout="sd2"), TD_PG)
    # two-beam self-diffraction has no p channel; the boxcars closed form does
    assert sd2.sigma_p == 0.0 and boxcars.sigma_p > 0.0
    doubled = smearing_kernel(RetrieveParams(smearing=True, smear_scale=2.0), TD_PG)
    assert doubled.sigma_delta == pytest.approx(2.0 * boxcars.sigma_delta)


def test_generation_envelope_off_unless_dispersive():
    assert generation_envelope(RetrieveParams(envelope=True)) is None
    assert generation_envelope(RetrieveParams(dispersive=True)) is None


def test_generation_envelope_rejects_non_positive_lengths():
    p = RetrieveParams(envelope=True, dispersive=True, envelope_zr_um=0.0)
    with pytest.raises(ValueError, match="envelope_zr_um > 0"):
        generation_envelope(p)


def test_generation_envelope_matches_the_gaussian_crossing_form():
    p = RetrieveParams(
        envelope=True,
        dispersive=True,
        thickness_um=50.0,
        npoints=7,
        envelope_zr_um=30.0,
        envelope_lw_um=40.0,
        envelope_zf_um=25.0,
        envelope_gouy=1.0,
    )
    g = generation_envelope(p)
    nodes, _ = quadrature_nodes_weights(50e-6, 7, "gausslegendre")
    u = (nodes - 25e-6) / 30e-6
    expected = (
        (1 + u**2) ** -0.5
        * np.exp(-(((nodes - 25e-6) / 40e-6) ** 2) / (1 + u**2))
        * np.exp(1j * np.arctan(u))
    )
    # the same closed form, evaluated independently
    np.testing.assert_allclose(g, expected, rtol=1e-12)
    # the midpoint node sits at the focus: unit amplitude, zero Gouy phase
    assert g[3] == pytest.approx(1.0)


def test_run_retrieval_uses_an_explicit_focal_mixture(monkeypatch):
    """A supplied mixture is forwarded as-is and not rebuilt from the params."""
    sentinel = object()
    seen = {}

    def no_rebuild(*a, **k):
        raise AssertionError("focal mixture rebuilt despite an explicit one")

    def fake_retrieve(td, **kw):
        seen.update(kw)
        return "result"

    monkeypatch.setattr(session_pipeline, "focal_mixture_from_params", no_rebuild)
    monkeypatch.setattr(session_pipeline, "retrieve_from_tracedata", fake_retrieve)
    td = SimpleNamespace(Iomega=None, delays=np.zeros(3))
    p = RetrieveParams(solver="copra")
    assert run_retrieval(p, td, focal=sentinel) == "result"
    assert seen["focal"] is sentinel
    assert seen["reg_spectrum"] is None  # no measured spectrum: inert, silent


def test_run_retrieval_passes_the_spectral_weight_with_a_spectrum(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        session_pipeline,
        "retrieve_from_tracedata",
        lambda td, **kw: seen.update(kw),
    )
    td = SimpleNamespace(Iomega=np.ones(4), delays=np.zeros(3))
    run_retrieval(RetrieveParams(solver="copra", reg_spectrum=0.3), td)
    assert seen["reg_spectrum"] == 0.3


@pytest.mark.parametrize(
    ("lam", "tau", "expected"),
    [
        (False, False, None),
        (True, False, ((1e-7, 2e-7), None)),
        (False, True, (None, (-5e-14, 5e-14))),
    ],
)
def test_apply_post_filter_windows_only_when_toggled(monkeypatch, lam, tau, expected):
    calls = []
    monkeypatch.setattr(
        session_pipeline,
        "post_filter",
        lambda result, *, lam_lims, tau_lims: calls.append((lam_lims, tau_lims)),
    )
    td = SimpleNamespace(lamm_lims=(1e-7, 2e-7), tau_lims=(-5e-14, 5e-14))
    p = RetrieveParams(post_filter_lam=lam, post_filter_tau=tau)
    out = apply_post_filter(p, "result", td)
    if expected is None:
        assert out == "result" and calls == []
    else:
        assert calls == [expected]
