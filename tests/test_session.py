"""Tests for the headless session engine (:mod:`croak.session`).

Exercises the Qt-free replay path end-to-end on a synthetic experiment: building
a :class:`~croak.session.options.SessionOptions`, round-tripping it through TOML,
retargeting it at a new dataset folder, and replaying it with
:func:`~croak.session.run_session` — checking the engine reproduces a direct
``croak`` pipeline run.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest
from conftest import write_experiment_h5

import croak
from croak.session import (
    SessionOptions,
    assemble_load_data,
    build_tracedata,
    run_retrieval,
    run_session,
)
from croak.session.dispersion import compose_dispersion, has_dispersion
from croak.session.params import (
    BeamPathMirror,
    DispersionParams,
    LoadParams,
    PreprocParams,
    RetrieveParams,
    SimulatedLoadParams,
)
from croak.session.pipeline import resolve_guess


def test_initial_guess_flags_are_mutually_exclusive():
    """Headless sessions reject conflicting legacy initial-condition flags."""
    params = RetrieveParams(random_init=True, truth_init=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_guess(params)


def _options_for(experiment_h5: str, experiment) -> SessionOptions:
    """A minimal but complete session for the synthetic experiment fixture."""
    load = LoadParams(
        frog_path=experiment_h5,
        ifrog_name="trace",
        lam_name="wavelength",
        scan_name="delay",
        lam_unit="nm",
        scan_unit="fs",
        scan_type="delay",
        interaction=experiment.interaction,
    )
    # Seed the windows/grid as a real saved options.toml would. The delay
    # windows sit a little *outside* the scan extents so the τm Planck taper does
    # not attenuate this synthetic's signal (which runs right to the scan edges).
    tmin = 2.0 * float(experiment.delay_fs.min())
    tmax = 2.0 * float(experiment.delay_fs.max())
    preproc = PreprocParams(
        lam_min_nm=experiment.lam_min / 1e-9,
        lam_max_nm=experiment.lam_max / 1e-9,
        lamm_min_nm=float(experiment.lam_nm.min()),
        lamm_max_nm=float(experiment.lam_nm.max()),
        tau_min_fs=tmin,
        tau_max_fs=tmax,
        taum_min_fs=tmin,
        taum_max_fs=tmax,
        trange_fs=2.0 * (tmax - tmin),
    )
    retrieve = RetrieveParams(solver="copra", maxiters=200)
    return SessionOptions(load=load, preproc=preproc, retrieve=retrieve)


def test_run_session_end_to_end(experiment_h5, experiment):
    """A full replay populates every intermediate and converges."""
    options = _options_for(experiment_h5, experiment)
    out = run_session(options, rng=np.random.default_rng(0))

    assert out.data is not None
    assert out.tracedata is not None
    assert out.result is not None
    assert out.processed is not None
    # A sane fit on the noiseless synthetic (rigorous engine == pipeline
    # equivalence is checked separately in test_run_session_matches_direct_pipeline).
    assert out.result.error < 0.15
    assert out.processed.fwhm_retr > 0


def test_run_session_matches_direct_pipeline(experiment_h5, experiment):
    """The engine reproduces a hand-written load → clean → retrieve run (same seed)."""
    options = _options_for(experiment_h5, experiment)

    engine = run_session(
        options, stages=("load", "preproc", "retrieve"), rng=np.random.default_rng(7)
    )

    # The same three mappings, called directly, with an identically seeded rng.
    data = assemble_load_data(options.load)
    td = build_tracedata(options.preproc, data)
    direct = run_retrieval(options.retrieve, td, rng=np.random.default_rng(7))

    assert engine.result is not None
    np.testing.assert_allclose(engine.result.error, direct.error, rtol=0, atol=1e-12)
    np.testing.assert_allclose(engine.result.spectrum, direct.spectrum)


def test_session_options_toml_roundtrip(tmp_path, experiment_h5, experiment):
    """Writing and re-reading options.toml preserves every persisted field."""
    options = _options_for(experiment_h5, experiment)
    options.retrieve.reg_spectrum = 0.25
    options.uncertainty.n_resamples = 33
    options.load.energy_j = 1.5e-4  # 150 µJ
    # the de-fringe / arPLS-baseline preprocessing options
    options.preproc.defringe = True
    options.preproc.defringe_fraction = 0.37
    options.preproc.defringe_on_alias = "warn"
    options.preproc.baseline = True
    options.preproc.baseline_smoothness = 4.2e3
    options.preproc.baseline_ratio = 3e-3
    options.preproc.baseline_clip = True
    options.preproc.baseline_tau_exclude_fs = 9.5

    path = str(tmp_path / "options.toml")
    options.to_toml(path)
    loaded = SessionOptions.from_toml(path)

    assert loaded.load.frog_path == experiment_h5
    assert loaded.load.interaction == experiment.interaction
    assert loaded.load.energy_j == pytest.approx(1.5e-4)
    assert loaded.preproc.lam_min_nm == pytest.approx(experiment.lam_min / 1e-9)
    assert loaded.retrieve.solver == "copra"
    assert loaded.retrieve.reg_spectrum == pytest.approx(0.25)
    assert loaded.uncertainty.n_resamples == 33
    # new preprocessing options survive the TOML round-trip
    assert loaded.preproc.defringe is True
    assert loaded.preproc.defringe_fraction == pytest.approx(0.37)
    assert loaded.preproc.defringe_on_alias == "warn"
    assert loaded.preproc.baseline is True
    assert loaded.preproc.baseline_smoothness == pytest.approx(4.2e3)
    assert loaded.preproc.baseline_ratio == pytest.approx(3e-3)
    assert loaded.preproc.baseline_clip is True
    assert loaded.preproc.baseline_tau_exclude_fs == pytest.approx(9.5)


def test_simulated_options_toml_roundtrip(tmp_path, simulated_scan_h5):
    """A simulated session records its loader (entry + [simulated]) in the TOML."""
    options = SessionOptions(
        entry="simulated",
        simulated=SimulatedLoadParams(
            frog_path=simulated_scan_h5,
            window_key="Iω_win_reimaged",
            z_thickness_um=20.0,
            lam_min_nm=131.0,
            lam_max_nm=793.0,
            third_order_exp=2.7,
            reverse_trace=False,
            interaction="sd",
            spectrum_source="source",
            truth_source="source",
        ),
    )

    path = str(tmp_path / "options.toml")
    options.to_toml(path)
    loaded = SessionOptions.from_toml(path)

    assert loaded.entry == "simulated"
    assert loaded.simulated == options.simulated
    # the experimental table is still written (at its defaults), so which loader
    # produced the trace is told by `entry`, never guessed from an empty path
    assert loaded.load == LoadParams()


def test_simulated_options_omit_unset_thickness(tmp_path, simulated_scan_h5):
    """``z_thickness_um=None`` round-trips: TOML has no null, so the key is dropped."""
    options = SessionOptions(
        entry="simulated",
        simulated=SimulatedLoadParams(frog_path=simulated_scan_h5),
    )
    assert options.simulated.z_thickness_um is None

    path = str(tmp_path / "options.toml")
    options.to_toml(path)
    assert "z_thickness_um" not in (tmp_path / "options.toml").read_text()
    assert SessionOptions.from_toml(path).simulated.z_thickness_um is None


def test_options_without_entry_read_as_experimental(tmp_path):
    """Files written before ``entry`` existed still load (as experimental)."""
    path = tmp_path / "options.toml"
    path.write_text('options_version = 1\n\n[load]\nfrog_path = "/data/acq.h5"\n')
    loaded = SessionOptions.from_toml(str(path))
    assert loaded.entry == "experimental"
    assert loaded.load.frog_path == "/data/acq.h5"
    assert loaded.simulated == SimulatedLoadParams()


def test_run_session_replays_a_simulated_session(simulated_scan_h5, simulated_truth):
    """The headless engine loads through the simulated loader when told to."""
    options = SessionOptions(
        entry="simulated",
        simulated=SimulatedLoadParams(
            frog_path=simulated_scan_h5,
            third_order=False,
            reverse_trace=False,
            lam_min_nm=simulated_truth.lam_min / 1e-9,
            lam_max_nm=simulated_truth.lam_max / 1e-9,
        ),
        preproc=PreprocParams(
            lam_min_nm=simulated_truth.lam_min / 1e-9,
            lam_max_nm=simulated_truth.lam_max / 1e-9,
            lamm_min_nm=simulated_truth.lam_min / 1e-9,
            lamm_max_nm=simulated_truth.lam_max / 1e-9,
        ),
        retrieve=RetrieveParams(solver="copra", maxiters=100),
    )

    out = run_session(options, rng=np.random.default_rng(0))

    assert out.data is not None and out.tracedata is not None
    assert out.result is not None
    assert out.result.error < 0.1
    # the loader's ground truth rides along, as in the GUI
    assert out.data["truth"] is not None


def test_run_session_simulated_raw_direct_skips_the_regrid(simulated_scan_h5):
    """``raw_direct`` replays on the simulation's native grid (no build_tracedata)."""
    simulated = SimulatedLoadParams(
        frog_path=simulated_scan_h5, use_spectrum=True, raw_direct=True
    )
    options = SessionOptions(
        entry="simulated",
        simulated=simulated,
        retrieve=RetrieveParams(solver="copra", maxiters=100),
    )

    out = run_session(
        options, stages=("load", "preproc", "retrieve"), rng=np.random.default_rng(0)
    )

    assert out.tracedata is not None
    native = croak.session.assemble_simulated_tracedata(simulated)
    np.testing.assert_allclose(out.tracedata.omega, native.omega)
    assert out.result is not None and out.result.error < 0.1


def test_run_session_rejects_synthetic_sessions():
    """A synthetic session has no saved generator settings, so replay refuses."""
    with pytest.raises(ValueError, match="cannot replay a synthetic session"):
        run_session(SessionOptions(entry="synthetic"), stages=("load",))


def test_run_session_energy_gives_peak_power(experiment_h5, experiment):
    """A load-stage pulse energy flows through to absolute peak power."""
    options = _options_for(experiment_h5, experiment)
    options.load.energy_j = 100e-6
    out = run_session(options)
    assert out.processed is not None
    assert out.processed.peak_power is not None
    assert out.processed.peak_power_tl is not None
    assert out.processed.energy == pytest.approx(100e-6)


def test_retarget_swaps_dataset_dir_keeps_calibration(tmp_path):
    """Retarget moves dataset files to a new folder but leaves calibration fixed."""
    options = SessionOptions(
        load=LoadParams(
            frog_path="/data/run01/acq.h5",
            spec_path="/data/run01/spectrum.h5",
            frog_bg_path="/data/run01/bg.h5",
            frog_calib_path="/calib/response.dat",
            frog_calib_use=True,
            spec_calib_path="/calib/spec_response.dat",
            spec_calib_use=True,
        )
    )

    moved = croak.retarget(options, "/data/run02")

    assert moved.load.frog_path == "/data/run02/acq.h5"
    assert moved.load.spec_path == "/data/run02/spectrum.h5"
    assert moved.load.frog_bg_path == "/data/run02/bg.h5"
    # calibration paths are shared and must not move
    assert moved.load.frog_calib_path == "/calib/response.dat"
    assert moved.load.spec_calib_path == "/calib/spec_response.dat"
    # empty (unused) dataset paths stay empty, not joined onto the new base
    assert moved.load.spec_bg_path == ""
    # the original is untouched (retarget returns a copy)
    assert options.load.frog_path == "/data/run01/acq.h5"


def test_retarget_rejects_non_experimental_sessions():
    """Only experimental sessions have a per-dataset folder to re-point."""
    options = SessionOptions(
        entry="simulated", simulated=SimulatedLoadParams(frog_path="/sim/scan.h5")
    )
    with pytest.raises(ValueError, match="cannot retarget a simulated session"):
        croak.retarget(options, "/data/run02")


def test_retarget_then_replay(tmp_path, experiment_h5, experiment):
    """A retargeted session replays against a copied dataset folder."""
    # Lay out the dataset under one folder, then copy it to a sibling "run02".
    src_dir = tmp_path / "run01"
    src_dir.mkdir()
    write_experiment_h5(src_dir / "acq.h5", experiment)
    dst_dir = tmp_path / "run02"
    shutil.copytree(src_dir, dst_dir)

    options = _options_for(str(src_dir / "acq.h5"), experiment)
    moved = croak.retarget(options, str(dst_dir))
    assert moved.load.frog_path == str(dst_dir / "acq.h5")

    # The retargeted folder is a byte-for-byte copy, so replaying it with the same
    # seed must reproduce the original run exactly — the strongest retarget check.
    original = run_session(
        options, stages=("load", "preproc", "retrieve"), rng=np.random.default_rng(0)
    )
    out = run_session(
        moved, stages=("load", "preproc", "retrieve"), rng=np.random.default_rng(0)
    )
    assert out.result is not None and original.result is not None
    np.testing.assert_allclose(out.result.spectrum, original.result.spectrum)
    assert out.result.error == pytest.approx(original.result.error)


# ---------------------------------------------------------------------------
# Dispersion persistence + composition
# ---------------------------------------------------------------------------
def _dispersion_with_everything() -> DispersionParams:
    """A dispersion config exercising Taylor, material, gas and a mirror row."""
    return DispersionParams(
        gdd_fs2=42.0,
        tod_fs3=12.0,
        fod_fs4=3.0,
        gas_name=croak.gases.GASES[0],
        gas_pressure_bar=2.5,
        gas_path_cm=15.0,
        material_thickness_mm={"SiO2": 0.4},
        mirrors=[
            BeamPathMirror(
                name=next(iter(croak.mirrors.BUILTIN_MIRRORS)),
                bounces=3,
                direction="remove",
            )
        ],
    )


def test_dispersion_options_roundtrip(tmp_path):
    """A full [dispersion] section (gas + materials + mirrors) survives TOML."""
    options = SessionOptions(dispersion=_dispersion_with_everything())
    path = str(tmp_path / "options.toml")
    options.to_toml(path)

    loaded = SessionOptions.from_toml(path)
    assert loaded.dispersion == options.dispersion
    # the mirror reconstructs as a BeamPathMirror, not a raw dict
    assert isinstance(loaded.dispersion.mirrors[0], BeamPathMirror)


def test_has_dispersion_detects_empty_vs_set():
    assert not has_dispersion(DispersionParams())
    assert has_dispersion(DispersionParams(gdd_fs2=10.0))
    assert has_dispersion(_dispersion_with_everything())


def _toy_result(interaction: str = "shg"):
    """A quick low-iteration retrieval result to compose dispersion onto."""
    g = croak.Grid(128, dt=0.3e-15)
    ew = croak.gaussian_pulse(g, 6e-15)
    omega0 = croak.maths.wlfreq(800e-9)
    delays = np.linspace(-30e-15, 30e-15, 60)
    trace = croak.maketrace(g.omega, delays, ew, interaction)
    return croak.retrieve(
        trace,
        g.omega,
        delays,
        interaction,
        algorithm="copra",
        maxiters=2,
        omega0=omega0,
    )


def test_compose_dispersion_matches_apply_dispersion():
    """compose_dispersion equals a direct apply_dispersion (Taylor + material)."""
    result = _toy_result()
    params = DispersionParams(
        gdd_fs2=50.0, tod_fs3=20.0, material_thickness_mm={"SiO2": 0.3}
    )
    composed = compose_dispersion(params, result)
    direct = croak.apply_dispersion(
        result.grid.omega,
        result.omega0,
        result.spectrum,
        gdd=50e-30,
        tod=20e-45,
        material_thicknesses={"SiO2": 0.3e-3},
    )
    np.testing.assert_allclose(composed.spectrum, direct)


def test_compose_dispersion_no_op_returns_input():
    result = _toy_result()
    assert compose_dispersion(DispersionParams(), result) is result


def test_compose_dispersion_mirror_leaves_registry_clean():
    """A mirror row changes the spectrum and leaves no leaked registry keys."""
    result = _toy_result()
    params = DispersionParams(
        mirrors=[
            BeamPathMirror(
                name=next(iter(croak.mirrors.BUILTIN_MIRRORS)),
                bounces=2,
                direction="remove",
            )
        ]
    )
    before = set(croak.dispersion.MIRRORS)
    composed = compose_dispersion(params, result)
    # the mirror actually altered the spectrum ...
    assert not np.allclose(composed.spectrum, result.spectrum)
    # ... and the transient registration was cleaned up
    assert set(croak.dispersion.MIRRORS) == before


def test_run_session_applies_dispersion(experiment_h5, experiment):
    """The dispersion stage produces a compensated spectrum distinct from the raw."""
    options = _options_for(experiment_h5, experiment)
    options.dispersion.gdd_fs2 = 100.0
    out = run_session(options, rng=np.random.default_rng(0))
    assert out.result is not None and out.dispersed is not None
    assert not np.allclose(out.dispersed.spectrum, out.result.spectrum)


def test_retrieve_params_extra_fit_defaults():
    """New extra-fit fields default to off."""
    p = RetrieveParams()
    assert p.fit_thickness is False
    assert p.fit_tau0 is False
    assert p.polish_two_phase is False


def test_solver_kwargs_forward_extra_fit_flags():
    """run_retrieval / bootstrap / covariance kwargs carry the extra-fit flags."""
    from croak.session.pipeline import bootstrap_solver_kwargs, covariance_kwargs

    p = RetrieveParams(
        solver="lm",
        dispersive=True,
        fit_thickness=True,
        fit_tau0=True,
        polish_two_phase=True,
    )
    cov = covariance_kwargs(p)
    assert cov["fit_thickness"] is True
    assert cov["fit_tau0"] is True

    class _TD:
        Iomega = None

    boot = bootstrap_solver_kwargs(p, _TD())
    assert boot["fit_thickness"] is True
    assert boot["fit_tau0"] is True
    assert boot["polish"] is True


def _seed_result(**changes):
    """A stand-in previous result carrying fitted extras."""
    from croak.result import RetrievalResult

    base = dict(thickness=36.4e-6, tau0=1.5e-15, smear_scale=1.457)
    base.update(changes)
    return RetrievalResult(
        spectrum=np.ones(4, dtype=complex),
        grid=croak.Grid(4, dt=1e-15),
        delays=np.zeros(3),
        interaction="pg",
        algorithm="lbfgs-ad",
        error=1e-3,
        **base,
    )


def test_extra_param_centres_cold_start_uses_the_nominal_settings():
    """Without a seed the centres are the parameters' own nominal values."""
    from croak.session.pipeline import extra_param_centres

    p = RetrieveParams(
        dispersive=True, thickness_um=30.0, fit_thickness=True, fit_tau0=True
    )
    thickness, tau0, smear, _smear_delta = extra_param_centres(p, None)
    assert thickness == pytest.approx(30e-6)
    # exact comparisons: a femtosecond-scale τ₀ would slip under approx's default
    # absolute tolerance, so an unwanted seed must not be able to hide here
    assert tau0 == 0.0
    assert smear == 1.0


def test_extra_param_centres_warm_starts_the_refitted_extras():
    """Reusing a result seeds every extra that is being fitted again."""
    from croak.session.pipeline import extra_param_centres

    p = RetrieveParams(
        dispersive=True,
        thickness_um=30.0,
        fit_thickness=True,
        fit_tau0=True,
        smearing=True,
        fit_smearing=True,
    )
    thickness, tau0, smear, _smear_delta = extra_param_centres(p, _seed_result())
    assert thickness == pytest.approx(36.4e-6)
    # compare τ₀ in fs: approx's 1e-12 absolute default swamps SI-scale delays
    assert tau0 / 1e-15 == pytest.approx(1.5)
    assert smear == pytest.approx(1.457)


def test_extra_param_centres_keeps_nominal_for_unfitted_extras():
    """An extra that is no longer fitted is held at the value the user set."""
    from croak.session.pipeline import extra_param_centres

    p = RetrieveParams(dispersive=True, thickness_um=30.0)  # every fit_* off
    thickness, tau0, smear, _smear_delta = extra_param_centres(p, _seed_result())
    assert thickness == pytest.approx(30e-6)
    assert tau0 == 0.0
    assert smear == 1.0


def test_extra_param_centres_ignores_extras_the_seed_never_fitted():
    """A first (pulse-only) run seeds nothing: its extras are absent, not zero."""
    from croak.session.pipeline import extra_param_centres

    p = RetrieveParams(
        dispersive=True,
        thickness_um=30.0,
        fit_thickness=True,
        smearing=True,
        fit_smearing=True,
    )
    cold = _seed_result(thickness=None, tau0=0.0, smear_scale=None)
    thickness, tau0, smear, _smear_delta = extra_param_centres(p, cold)
    assert thickness == pytest.approx(30e-6)
    assert tau0 == 0.0
    assert smear == 1.0


def test_run_retrieval_forwards_the_warm_start_centres(monkeypatch):
    """``extras_seed`` reaches :func:`croak.pipeline.retrieve_from_tracedata`."""
    from croak.session import pipeline as session_pipeline

    captured: dict = {}

    def _fake(td, **kwargs):
        captured.update(kwargs)
        return "result"

    monkeypatch.setattr(session_pipeline, "retrieve_from_tracedata", _fake)
    p = RetrieveParams(
        solver="lbfgs-ad",
        dispersive=True,
        thickness_um=30.0,
        fit_thickness=True,
        fit_tau0=True,
    )

    class _TD:
        Iomega = None
        interaction = "pg"
        omega0_pulse = 2.4e15

    session_pipeline.run_retrieval(p, _TD(), extras_seed=_seed_result())

    assert captured["thickness"] == pytest.approx(36.4e-6)
    assert captured["tau0"] / 1e-15 == pytest.approx(1.5)
    # smearing is off here, so its multiplier stays at the unscaled kernel
    assert captured["smear_scale"] == pytest.approx(1.0)


def test_run_session_with_smearing_end_to_end(experiment_h5, experiment):
    """A full replay carries the stage-4 smearing geometry into the forward model."""
    options = _options_for(experiment_h5, experiment)
    options.retrieve.solver = "lbfgs-ad"
    options.retrieve.maxiters = 60
    options.retrieve.smearing = True

    out = run_session(options, rng=np.random.default_rng(0))
    assert out.result is not None
    assert out.result.error < 0.15
    # Held fixed unless asked for, so nothing is reported.
    assert out.result.smear_scale is None

    options.retrieve.fit_smearing = True
    fitted = run_session(options, rng=np.random.default_rng(0))
    assert fitted.result is not None
    assert fitted.result.smear_scale is not None
    assert fitted.result.smear_scale >= 0.0


def test_run_session_smearing_rejects_projection_solvers(experiment_h5, experiment):
    """COPRA cannot model the incoherent node sum, so the replay must fail loudly."""
    options = _options_for(experiment_h5, experiment)
    options.retrieve.smearing = True  # solver is copra
    with pytest.raises(ValueError, match="cannot model geometrical smearing"):
        run_session(options, rng=np.random.default_rng(0))


def test_retrieve_params_smearing_defaults():
    """The smearing block defaults to off, at the reference DUV mask geometry."""
    p = RetrieveParams()
    assert p.smearing is False
    assert p.fit_smearing is False
    assert p.smear_hole_diameter_mm == pytest.approx(1.0)
    assert p.smear_hole_spacing_mm == pytest.approx(0.5)
    assert p.smear_nodes == 5


def test_smearing_kernel_from_params_matches_geometry():
    """The stage-4 geometry builds the square-BOXCARS kernel at the trace carrier."""
    from croak.maths import wlfreq
    from croak.session.pipeline import smearing_kernel
    from croak.smearing import square_boxcars_kernel

    class _TD:
        interaction = "pg"
        omega0_pulse = wlfreq(260e-9)

    p = RetrieveParams(smearing=True, smear_nodes=7)
    kernel = smearing_kernel(p, _TD())
    assert kernel == square_boxcars_kernel(
        "pg", hole_diameter=1e-3, hole_spacing=0.5e-3, wavelength=260e-9, npoints=7
    )
    # Off by default, and off means off.
    assert smearing_kernel(RetrieveParams(), _TD()) is None


def test_smearing_kwargs_gated_on_the_enable():
    """``fit_smearing`` is only forwarded when a kernel is actually being modelled."""
    from croak.maths import wlfreq
    from croak.session.pipeline import bootstrap_solver_kwargs, covariance_kwargs

    class _TD:
        Iomega = None
        interaction = "pg"
        omega0_pulse = wlfreq(260e-9)

    on = RetrieveParams(solver="lbfgs-ad", smearing=True, fit_smearing=True)
    off = RetrieveParams(solver="lbfgs-ad", smearing=False, fit_smearing=True)
    assert bootstrap_solver_kwargs(on, _TD())["smearing"] is not None
    assert bootstrap_solver_kwargs(on, _TD())["fit_smearing"] is True
    assert bootstrap_solver_kwargs(off, _TD())["smearing"] is None
    assert bootstrap_solver_kwargs(off, _TD())["fit_smearing"] is False
    assert covariance_kwargs(on, _TD())["fit_smearing"] is True
    # Without a TraceData there is no carrier to build the kernel from.
    assert covariance_kwargs(on)["smearing"] is None
    assert covariance_kwargs(on)["fit_smearing"] is False


def test_fit_thickness_requires_dispersive_in_kwargs():
    """Thickness fitting is gated on dispersive propagation being enabled."""
    from croak.session.pipeline import covariance_kwargs

    p = RetrieveParams(fit_thickness=True, dispersive=False)
    assert covariance_kwargs(p)["fit_thickness"] is False
