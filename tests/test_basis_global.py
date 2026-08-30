"""Reduced B-spline phase basis + the global (CMA-ES/evosax) retriever."""

import numpy as np
import pytest

from croak._jax_pulse import (
    build_parameterisation,
    ls_parameterisation,
    make_spline_phase_basis,
)
from croak.forward import maketrace
from croak.grid import Grid
from croak.lbfgs_ad import LBFGSAD
from croak.pulses import gaussian_pulse


@pytest.fixture
def chirped():
    g = Grid(96, dt=0.4e-15)
    ew = gaussian_pulse(g, 3e-15, phases=[5e-30])  # GDD chirp
    delays = np.linspace(-35e-15, 35e-15, 75)
    return g, ew, delays


# ---------------------------------------------------------------------------
# Basis
# ---------------------------------------------------------------------------
def test_spline_basis_reconstructs_seed_and_preserves_amplitude(chirped):
    import jax.numpy as jnp

    g, ew, _ = chirped
    amp, ph = np.abs(ew), np.angle(ew)
    spectrum_fn, c0, phase_pen, support = make_spline_phase_basis(
        amp, ph, g.omega, n_nodes=25
    )
    s0, s1 = support
    assert s1 - s0 < g.n  # nodes only span the non-zero spectrum
    ew_rec = np.asarray(spectrum_fn(jnp.asarray(c0)))
    # amplitude is held fixed
    assert np.allclose(np.abs(ew_rec), amp)
    # the fitted spline reproduces the seed phase factor over the support
    factor_err = np.max(
        np.abs(np.exp(1j * np.angle(ew_rec))[s0:s1] - np.exp(1j * ph)[s0:s1])
    )
    assert factor_err < 1e-8
    assert float(phase_pen(jnp.asarray(c0))) >= 0.0


def test_spline_basis_is_differentiable(chirped):
    import jax
    import jax.numpy as jnp

    g, ew, _ = chirped
    spectrum_fn, c0, _pen, _sup = make_spline_phase_basis(
        np.abs(ew), np.angle(ew), g.omega, n_nodes=20
    )
    grad = jax.grad(
        lambda c: jnp.sum(jnp.abs(spectrum_fn(c)) ** 2 * jnp.arange(c.size).sum())
    )(jnp.asarray(c0))
    assert grad.shape == (20,) and np.all(np.isfinite(np.asarray(grad)))


def test_spline_basis_rejects_too_few_nodes(chirped):
    g, ew, _ = chirped
    with pytest.raises(ValueError, match="n_nodes"):
        make_spline_phase_basis(np.abs(ew), np.angle(ew), g.omega, n_nodes=3)


def test_build_parameterisation_dims(chirped):
    g, ew, _ = chirped
    pointwise = build_parameterisation(ew, g.omega, phase_only=True)
    bspline = build_parameterisation(
        ew, g.omega, phase_only=True, phase_basis="bspline", n_nodes=22
    )
    assert pointwise.u0.size == g.n  # one phase value per frequency
    assert bspline.u0.size == 22  # reduced to the node count
    _fn, _u0, nidcs, phase_only = ls_parameterisation(
        ew, g.omega, phase_only=False, phase_basis="bspline", n_nodes=22
    )
    assert nidcs == 0 and phase_only is True  # spline implies phase-only


@pytest.mark.parametrize("n_nodes", [15, 25])
def test_spline_phase_retrieval_lbfgs_ad(chirped, n_nodes):
    g, ew, delays = chirped
    trace = maketrace(g.omega, delays, ew, "pg")
    guess = gaussian_pulse(g, 3e-15)  # right amplitude, flat (wrong) phase
    res = LBFGSAD(
        phase_only=True, phase_basis="bspline", n_nodes=n_nodes, maxiters=300
    ).run(trace, g.omega, delays, "pg", guess=guess)
    assert res.error < 5e-3
    assert res.spectrum.shape == (g.n,)


# ---------------------------------------------------------------------------
# Global optimiser (evosax)
# ---------------------------------------------------------------------------
evosax = pytest.importorskip("evosax")
from croak.cmaes import CMAES  # noqa: E402 - must follow the evosax importorskip


@pytest.mark.parametrize("strategy", ["cma", "sep-cma", "de"])
def test_cmaes_strategies_run(chirped, strategy):
    g, ew, delays = chirped
    trace = maketrace(g.omega, delays, ew, "pg")
    guess = gaussian_pulse(g, 3e-15)
    res = CMAES(
        strategy=strategy,
        phase_only=True,
        phase_basis="bspline",
        n_nodes=15,
        maxiters=60,
        seed=0,
    ).run(trace, g.omega, delays, "pg", guess=guess)
    assert res.spectrum.shape == (g.n,)
    assert np.isfinite(res.error)


def test_cmaes_beats_local_from_bad_guess(chirped):
    """CMA-ES (global) finds a basin a local solver misses from the same guess."""
    g, ew, delays = chirped
    trace = maketrace(g.omega, delays, ew, "pg")
    guess = gaussian_pulse(g, 3e-15)  # correct amplitude, flat phase (no chirp)
    cma = CMAES(
        phase_only=True,
        phase_basis="bspline",
        n_nodes=18,
        maxiters=150,
        std_init=1.0,
        seed=0,
    ).run(trace, g.omega, delays, "pg", guess=guess)
    assert cma.error < 5e-2
    # reuse the global result as the local guess -> refine
    local = LBFGSAD(
        phase_only=True, phase_basis="bspline", n_nodes=18, maxiters=300
    ).run(trace, g.omega, delays, "pg", guess=cma.spectrum)
    assert local.error <= cma.error + 1e-9


def test_cmaes_via_pipeline(chirped):
    from croak import preprocess
    from croak.pipeline import retrieve_from_tracedata

    g, ew, delays = chirped
    trace = maketrace(g.omega, delays, ew, "pg")
    # wrap into a TraceData-like path through the registry dispatch
    from croak.maths import wlfreq

    lam = wlfreq(g.omega + wlfreq(800e-9))
    td = preprocess.TraceData(
        interaction="pg",
        grid=g,
        delays=delays,
        trace=trace,
        omega0_pulse=wlfreq(800e-9),
        omega0_trace=wlfreq(800e-9),
        Iomega=None,
        lam_min=700e-9,
        lam_max=900e-9,
        lamm_lims=None,
        tau_lims=None,
        lam_frog=lam,
        tau_meas=delays,
        Ifrog_meas=trace.T,
        tau_filt=delays,
        Ifrog_filt=trace.T,
        lam_spec=None,
        Ilam_spec_meas=None,
        Ilam_spec_filt=None,
        scalecurve=np.ones(g.n),
    )
    res = retrieve_from_tracedata(
        td,
        algorithm="cma-es",
        full=False,
        phase_basis="bspline",
        n_nodes=15,
        maxiters=40,
        seed=1,
    )
    assert res.algorithm == "cma-es"
    assert res.spectrum.shape == (g.n,)


# ---------------------------------------------------------------------------
# Extra fitted parameters (thickness / tau0) on the global search
# ---------------------------------------------------------------------------
def test_fit_params_in_cmaes_algorithm_params():
    from croak.retrieve import algorithm_params

    params = algorithm_params("cma-es")
    assert {"fit_thickness", "fit_tau0"} <= params
    assert "polish" not in params  # the global search has no two-phase polish


def test_warm_start_centres_in_algorithm_params():
    """Every fitting solver accepts the τ₀ / smearing warm-start centres."""
    from croak.retrieve import algorithm_params

    for name in ("lbfgs-ad", "lm", "lm-optx", "cma-es"):
        assert {"tau0", "smear_scale"} <= algorithm_params(name), name


@pytest.mark.parametrize("fit_tau0", [False, True])
def test_augment_holds_unfitted_extras_at_their_centres(fit_tau0):
    """A non-fitted extra is held at its centre — the same rule for all three.

    Regression test: ``tau0`` used to fall back to a hard ``0`` while thickness
    and the smearing multiplier fell back to their centres, so a known delay-zero
    offset was silently dropped from the model whenever it was not being fitted
    (e.g. when re-linearising for the covariance about a fitted ``tau0``).
    """
    from croak._jax_pulse import ExtraParamSpec, augment

    spec = ExtraParamSpec(fit_tau0=fit_tau0, tau0_0=2e-15, scale_tau=1e-15)
    aug = augment(np.zeros(4), spec)
    _thickness, tau0, smear = aug.physical(aug.u0)
    # the optimised offsets start at zero, so u0 evaluates exactly at the centre
    assert tau0 / 1e-15 == pytest.approx(2.0)
    assert smear == 1.0


def test_cmaes_fits_tau0():
    """From a good pulse guess the global search recovers a known delay-zero."""
    g = Grid(96, dt=0.4e-15)
    ew = gaussian_pulse(g, 3e-15)  # transform-limited
    delays = np.linspace(-35e-15, 35e-15, 75)
    tau0_true = 1.0e-15
    trace = maketrace(g.omega, delays - tau0_true, ew, "shg")
    common = dict(phase_basis="bspline", n_nodes=12, maxiters=150, std_init=0.6, seed=0)
    no_fit = CMAES(**common).run(trace, g.omega, delays, "shg", guess=ew)
    fit = CMAES(fit_tau0=True, **common).run(trace, g.omega, delays, "shg", guess=ew)
    assert fit.tau0 == pytest.approx(tau0_true, abs=0.4e-15)
    assert fit.error < no_fit.error


def test_cmaes_fits_thickness_records_value():
    """fit_thickness runs end-to-end for cma-es and records a sane thickness."""
    from croak.maths import wlfreq

    g = Grid(96, dt=2e-15)
    omega0 = wlfreq(800e-9)
    ew = gaussian_pulse(g, 8e-15)
    delays = np.linspace(-80e-15, 80e-15, 60)
    L0 = 300e-6
    trace = maketrace(
        g.omega,
        delays,
        ew,
        "pg",
        material="SiO2",
        thickness=L0,
        npoints=12,
        omega0=omega0,
    )
    res = CMAES(
        material="SiO2",
        thickness=L0,
        npoints=12,
        omega0=omega0,
        phase_basis="bspline",
        n_nodes=10,
        maxiters=40,
        std_init=0.3,
        fit_thickness=True,
        seed=0,
    ).run(trace, g.omega, delays, "pg", guess=ew, omega0=omega0)
    assert res.thickness is not None
    assert np.isfinite(res.thickness) and res.thickness > 0
    # the search centres on the prior, so it stays in its neighbourhood
    assert res.thickness == pytest.approx(L0, rel=0.5)


# ---------------------------------------------------------------------------
# Temporal (pedestal) regularisation
# ---------------------------------------------------------------------------
@pytest.fixture
def satellite():
    """A main pulse plus a satellite well inside the grid but outside ±120 fs."""
    g = Grid(256, dt=4e-15)  # ~1024 fs span, satellite at +250 fs is representable
    t = g.t
    field = np.exp(-((t / 30e-15) ** 2)) + 0.35 * np.exp(
        -(((t - 250e-15) / 30e-15) ** 2)
    )
    ew = g.fft(field.astype(complex))
    delays = g.t[::2]
    trace = maketrace(g.omega, delays, ew, "shg")
    win = (-120e-15, 120e-15)  # excludes the satellite
    return g, ew, delays, trace, win


def _out_of_window_fraction(grid, spectrum, win):
    from croak._jax_pulse import time_window_mask

    mask = time_window_mask(grid.t, win[0], win[1], grid.dt)
    energy = np.abs(grid.ifft(spectrum)) ** 2
    return float((mask * energy).sum() / energy.sum())


def test_reg_time_in_algorithm_params():
    from croak.retrieve import algorithm_params

    for name in ("cma-es", "lbfgs-ad"):
        params = algorithm_params(name)
        assert {"reg_time", "time_window"} <= params
    # LM / COPRA do not take it (penalty is scalar-objective only, for now)
    for name in ("lm", "copra"):
        assert "reg_time" not in algorithm_params(name)


def test_time_window_mask_complement():
    from croak._jax_pulse import time_window_mask

    g = Grid(128, dt=4e-15)
    mask = time_window_mask(g.t, -100e-15, 100e-15, g.dt)
    assert mask.shape == (g.n,)
    assert np.all((mask >= 0) & (mask <= 1))
    assert mask[np.argmin(np.abs(g.t))] == pytest.approx(0.0)  # ~0 inside window
    assert mask[0] == pytest.approx(1.0) and mask[-1] == pytest.approx(1.0)  # 1 outside


def test_reg_time_suppresses_pedestal_lbfgs_ad(satellite):
    """The temporal penalty trades a little FROG error to remove out-of-window
    energy, self-consistently (no post-filter)."""
    g, ew, delays, trace, win = satellite
    truth_frac = _out_of_window_fraction(g, ew, win)
    assert truth_frac > 0.05  # the satellite really is out of the window

    rng = np.random.default_rng(1)
    amp = np.abs(ew)
    guess = amp * np.exp(1j * rng.uniform(-0.3, 0.3, size=amp.shape))

    base = LBFGSAD(maxiters=400).run(trace, g.omega, delays, "shg", guess=guess)
    reg = LBFGSAD(maxiters=400, reg_time=0.2, time_window=win).run(
        trace, g.omega, delays, "shg", guess=guess
    )
    # Without the penalty the retrieval reproduces the pedestal; with it the
    # out-of-window energy is strongly suppressed.
    assert _out_of_window_fraction(g, base.spectrum, win) > 0.05
    assert _out_of_window_fraction(g, reg.spectrum, win) < 0.02
    assert reg.error >= base.error  # the expected fidelity/pedestal tradeoff


def test_reg_time_reports_bare_frog_error_cmaes(satellite):
    """cma-es still selects/reports on the bare FROG error with reg_time on."""
    g, ew, delays, trace, win = satellite
    res = CMAES(
        phase_only=True,
        phase_basis="bspline",
        n_nodes=20,
        maxiters=60,
        reg_time=0.2,
        time_window=win,
        seed=5,
    ).run(trace, g.omega, delays, "shg", guess=ew)
    assert res.errors[-1] == pytest.approx(res.error, abs=1e-9)
