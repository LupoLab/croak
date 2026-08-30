"""Convergence tests for the Optimistix-backed solvers (``lm-optx``, ``lbfgs-optx``).

JAX-native twins of :class:`croak.lm.LM` and :class:`croak.lbfgs_ad.LBFGSAD`: they
optimise the identical objective/residual over the identical parameterisation,
differing only in the optimiser (Optimistix instead of SciPy ``least_squares`` /
NLopt). They must converge on the same problems as their twins, so the thresholds
here mirror ``test_lm.py`` / ``test_lbfgs_ad.py``.
"""

import numpy as np
import pytest

from croak.forward import maketrace
from croak.grid import Grid
from croak.maths import wlfreq
from croak.optimistix_lbfgs import OptxLBFGS
from croak.optimistix_lm import OptxLM
from croak.pulses import gaussian_pulse
from croak.smearing import square_boxcars_kernel

SOLVERS = [OptxLM, OptxLBFGS]


@pytest.fixture
def setup():
    g = Grid(128, dt=0.3e-15)
    ew = gaussian_pulse(g, 2.5e-15)
    delays = np.linspace(-15e-15, 15e-15, 87)
    return g, ew, delays


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("interaction", ["pg", "sd", "shg"])
def test_optx_near_initial(setup, solver, interaction):
    g, ew, delays = setup
    trace = maketrace(g.omega, delays, ew, interaction)
    near = gaussian_pulse(g, 1.5e-15)
    res = solver(maxiters=300).run(trace, g.omega, delays, interaction, guess=near)
    assert res.error < 5e-3


@pytest.mark.parametrize("solver", SOLVERS)
def test_optx_smeared_trace_needs_the_kernel(setup, solver):
    """Both Optimistix solvers accept a smearing kernel and converge on it.

    The threshold is looser than :func:`test_optx_near_initial` because ``lbfgs-optx``
    plateaus early on this problem, the same documented weakness as
    :func:`test_optx_lbfgs_random_initial_weaker_than_nlopt`. What matters here is that
    modelling the kernel beats ignoring it by a wide margin.
    """
    g, ew, delays = setup
    kernel = square_boxcars_kernel(
        "pg", hole_diameter=1e-3, hole_spacing=2.0e-3, wavelength=800e-9
    )
    trace = maketrace(g.omega, delays, ew, "pg", smearing=kernel)
    near = gaussian_pulse(g, 1.8e-15)
    modelled = solver(maxiters=300, smearing=kernel).run(
        trace, g.omega, delays, "pg", guess=near
    )
    ignored = solver(maxiters=300).run(trace, g.omega, delays, "pg", guess=near)
    assert modelled.error < 1e-2
    assert modelled.error < 0.5 * ignored.error


@pytest.mark.parametrize("solver", SOLVERS)
def test_optx_reltol_plateau_stops_earlier(setup, solver):
    """``reltol`` drives a relative-improvement plateau stop.

    Optimistix's own Cauchy termination is ``atol``-dominated near convergence and
    effectively ignores ``rtol``, so :func:`croak._optimistix.run_optx` adds a
    relative-improvement stop keyed on ``reltol`` (as COPRA does). A looser
    ``reltol`` must therefore stop the solver sooner — and before ``maxiters``.
    """
    g, _, delays = setup
    ew = gaussian_pulse(g, 2.5e-15, phases=[3e-30])
    trace = maketrace(g.omega, delays, ew, "pg")

    def iters(reltol):
        guess = gaussian_pulse(g, 1.8e-15, phases=[1e-30])
        res = solver(maxiters=500, reltol=reltol).run(
            trace, g.omega, delays, "pg", guess=guess
        )
        return len(res.errors)

    loose, tight = iters(3e-1), iters(1e-3)
    assert loose < 500  # the plateau stop fires before maxiters
    assert loose < tight  # and a looser tolerance stops sooner


@pytest.mark.parametrize("interaction", ["pg", "sd", "shg"])
def test_optx_lm_random_initial(setup, interaction):
    """OptxLM converges to the global minimum even from a random guess."""
    g, ew, delays = setup
    trace = maketrace(g.omega, delays, ew, interaction)
    rng = np.random.default_rng(1)
    res = OptxLM(maxiters=500).run(
        trace, g.omega, delays, interaction, guess=None, rng=rng
    )
    assert res.error < 5e-2


def test_optx_lbfgs_random_initial_weaker_than_nlopt(setup):
    """Document a real limitation: from a *random* start the Optimistix L-BFGS
    plateaus at a local minimum (~0.05) on this PG problem, whereas the
    NLopt-backed ``lbfgs-ad`` twin reaches the global minimum (~1e-5). This is
    systematic across seeds (the backtracking line search stalls), so OptxLBFGS
    is best reserved for refinement near a good guess. It still reduces the error
    substantially from the random start, which this test guards.
    """
    from croak.lbfgs_ad import LBFGSAD

    g, ew, delays = setup
    trace = maketrace(g.omega, delays, ew, "pg")
    rng = lambda: np.random.default_rng(1)  # noqa: E731 - a fresh identical rng per solver
    r_optx = OptxLBFGS(maxiters=500).run(
        trace, g.omega, delays, "pg", guess=None, rng=rng()
    )
    r_nlopt = LBFGSAD(maxiters=500).run(
        trace, g.omega, delays, "pg", guess=None, rng=rng()
    )
    assert r_optx.error < 0.1  # substantial reduction, but not the global min
    assert r_nlopt.error < r_optx.error  # NLopt clearly wins from a random start


@pytest.mark.parametrize("solver", SOLVERS)
def test_optx_chirped_phase_only(setup, solver):
    g, _, delays = setup
    ew = gaussian_pulse(g, 2.5e-15, phases=[3e-30])
    trace = maketrace(g.omega, delays, ew, "pg")
    guess = gaussian_pulse(g, 2.5e-15, phases=[1e-30])
    res = solver(maxiters=300, phase_only=True).run(
        trace, g.omega, delays, "pg", guess=guess
    )
    assert res.error < 1e-2


@pytest.mark.parametrize("solver", SOLVERS)
def test_optx_dispersive(solver):
    g = Grid(128, dt=0.5e-15)
    omega0 = wlfreq(800e-9)
    ew = gaussian_pulse(g, 3.0e-15)
    delays = np.linspace(-20e-15, 20e-15, 80)
    trace = maketrace(
        g.omega,
        delays,
        ew,
        "pg",
        material="SiO2",
        thickness=10e-6,
        npoints=20,
        omega0=omega0,
    )
    near = gaussian_pulse(g, 2.0e-15)
    res = solver(
        maxiters=400, material="SiO2", thickness=10e-6, npoints=20, omega0=omega0
    ).run(trace, g.omega, delays, "pg", guess=near)
    assert res.error < 5e-2


@pytest.mark.parametrize("solver", SOLVERS)
def test_optx_spectral_reg_improves_spectrum(setup, solver):
    # Noisy trace so the amplitude is under-constrained (a clean PG trace already
    # determines |E(ω)|); the spectral-match penalty then pulls it toward truth.
    g, _, delays = setup
    ew = gaussian_pulse(g, 2.5e-15) + 0.6 * gaussian_pulse(g, 1.2e-15)
    ew = ew / np.max(np.abs(ew))
    I_true = np.abs(ew) ** 2
    clean = maketrace(g.omega, delays, ew, "pg")
    rng = np.random.default_rng(0)
    trace = np.clip(clean + 0.03 * rng.standard_normal(clean.shape), 0.0, None)
    near = gaussian_pulse(g, 4.0e-15)

    def spec_l2(res):
        a = np.abs(res.spectrum)
        a /= a.max()
        s = np.sqrt(I_true)
        s /= s.max()
        return float(np.linalg.norm(a - s))

    r0 = solver(maxiters=400).run(trace, g.omega, delays, "pg", guess=near)
    r1 = solver(maxiters=400, reg_spectrum=0.5, spectrum_target=I_true).run(
        trace, g.omega, delays, "pg", guess=near
    )
    assert r1.error < 5e-2  # near the noise floor
    assert spec_l2(r1) < spec_l2(r0)


@pytest.mark.parametrize("solver", SOLVERS)
def test_optx_callback_and_errlog(setup, solver):
    """The manual-stepping loop logs per-iteration error and fires the callback."""
    g, ew, delays = setup
    trace = maketrace(g.omega, delays, ew, "pg")
    near = gaussian_pulse(g, 1.5e-15)
    seen = []
    res = solver(maxiters=50).run(
        trace,
        g.omega,
        delays,
        "pg",
        guess=near,
        callback=lambda it, R, best, snapshot=None: seen.append((it, R, best)),
    )
    assert len(seen) == len(res.errors)
    assert seen and seen[0][0] == 1
    # best_R is the running minimum and is monotone non-increasing
    bests = [b for _, _, b in seen]
    assert all(b1 >= b2 for b1, b2 in zip(bests, bests[1:], strict=False))


# -- extra fitted parameters (thickness / tau0), OptxLM only ------------------
# These mirror the LM tests in tests/test_lm.py: OptxLM reuses the same
# augmentation machinery and two-phase polish, only the optimiser differs.


def test_optx_lm_fits_tau0(setup):
    """A known delay-zero offset is recovered, lowering the trace error."""
    g, ew, delays = setup
    tau0_true = 1.5e-15
    trace = maketrace(g.omega, delays - tau0_true, ew, "shg")
    no_fit = OptxLM(maxiters=300).run(trace, g.omega, delays, "shg", guess=ew)
    fit = OptxLM(maxiters=300, fit_tau0=True).run(
        trace, g.omega, delays, "shg", guess=ew
    )
    assert fit.tau0 == pytest.approx(tau0_true, abs=0.2e-15)
    assert fit.error < no_fit.error
    assert fit.error < 1e-3


def test_optx_lm_fits_thickness():
    """A known slab thickness is recovered in the strong-dispersion regime."""
    g = Grid(128, dt=2e-15)
    omega0 = wlfreq(800e-9)
    ew = gaussian_pulse(g, 8e-15)
    delays = np.linspace(-120e-15, 120e-15, 90)
    L_true = 400e-6
    trace = maketrace(
        g.omega,
        delays,
        ew,
        "pg",
        material="SiO2",
        thickness=L_true,
        npoints=20,
        omega0=omega0,
    )
    no_fit = OptxLM(
        maxiters=300, material="SiO2", thickness=250e-6, npoints=20, omega0=omega0
    ).run(trace, g.omega, delays, "pg", guess=ew, omega0=omega0)
    fit = OptxLM(
        maxiters=300,
        material="SiO2",
        thickness=250e-6,
        npoints=20,
        omega0=omega0,
        fit_thickness=True,
    ).run(trace, g.omega, delays, "pg", guess=ew, omega0=omega0)
    assert fit.thickness == pytest.approx(L_true, rel=0.05)
    assert fit.error < no_fit.error


def test_optx_lm_two_phase_polish_thickness():
    """The two-phase polish recovers the thickness even from a cold pulse start."""
    g = Grid(128, dt=2e-15)
    omega0 = wlfreq(800e-9)
    ew = gaussian_pulse(g, 8e-15)
    delays = np.linspace(-120e-15, 120e-15, 90)
    L_true = 400e-6
    trace = maketrace(
        g.omega,
        delays,
        ew,
        "pg",
        material="SiO2",
        thickness=L_true,
        npoints=20,
        omega0=omega0,
    )
    res = OptxLM(
        maxiters=300,
        material="SiO2",
        thickness=250e-6,
        npoints=20,
        omega0=omega0,
        fit_thickness=True,
        polish=True,
    ).run(trace, g.omega, delays, "pg", guess=ew, omega0=omega0)
    assert res.thickness == pytest.approx(L_true, rel=0.05)


def test_optx_lm_thickness_fit_shg_guard(setup):
    """Fitting thickness for SHG is rejected (a slab is a pure spectral phase)."""
    g, ew, delays = setup
    trace = maketrace(g.omega, delays, ew, "shg")
    with pytest.raises(ValueError, match="dispersive"):
        OptxLM(
            material="SiO2", thickness=10e-6, omega0=wlfreq(400e-9), fit_thickness=True
        ).run(trace, g.omega, delays, "shg", guess=ew)


# -- the damped Gauss--Newton linear solver, OptxLM only ----------------------
# Each LM iteration solves [J; sqrt(lambda) I] d = [r; 0]. croak defaults to the
# normal equations (J^T J + lambda I, Cholesky) rather than Optimistix's own QR of
# the stacked operator: the Gram matrix is one BLAS `gemm` at near-peak throughput,
# whereas LAPACK's blocked Householder QR of a tall-skinny matrix is panel-bound.
# The two routes must agree — they solve the same linear system.

LINEAR_SOLVERS = ["normal", "qr"]


def test_optx_lm_linear_solver_is_discoverable():
    """The option is advertised on ``lm-optx`` and only there.

    ``lbfgs-optx`` runs a quasi-Newton loop with no linear solve, so exposing the
    knob there would be meaningless; the GUI and pipeline gate their controls on
    :func:`croak.retrieve.algorithm_params`.
    """
    from croak.retrieve import algorithm_params

    assert "linear_solver" in algorithm_params("lm-optx")
    assert "linear_solver" not in algorithm_params("lbfgs-optx")


def test_optx_lm_rejects_unknown_linear_solver():
    """An unrecognised name fails loudly at construction, not mid-solve."""
    with pytest.raises(ValueError, match="unknown linear_solver"):
        OptxLM(linear_solver="cholesky")


@pytest.mark.parametrize("interaction", ["pg", "sd", "shg"])
def test_optx_lm_linear_solvers_agree(setup, interaction):
    """Both routes solve the same system, so they must land on the same pulse.

    The residual bottoms out near ``1e-16``, and a quadratic minimum means the
    *parameters* are then determined only to its square root — hence agreement is
    asserted at ``1e-5`` on the normalised spectral amplitude and temporal
    intensity (measured: ``~5e-8``), not at machine precision.
    """
    g, ew, delays = setup
    trace = maketrace(g.omega, delays, ew, interaction)
    near = gaussian_pulse(g, 1.5e-15)

    def run(linear_solver):
        return OptxLM(maxiters=300, linear_solver=linear_solver).run(
            trace, g.omega, delays, interaction, guess=near
        )

    normal, qr = run("normal"), run("qr")
    assert normal.error < 5e-3
    assert qr.error < 5e-3

    def normalised(x):
        x = np.asarray(x, dtype=float)
        return x / x.max()

    np.testing.assert_allclose(
        normalised(np.abs(normal.spectrum)),
        normalised(np.abs(qr.spectrum)),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        normalised(normal.intensity_t), normalised(qr.intensity_t), atol=1e-5
    )


@pytest.mark.parametrize("linear_solver", LINEAR_SOLVERS)
def test_optx_lm_stays_finite_with_tolerances_off(setup, linear_solver):
    """A long run with every stop disabled stays finite.

    The Levenberg--Marquardt parameter is ``lambda = 1/step_size`` and
    :class:`optimistix.ClassicalTrustRegion` grows ``step_size`` without an upper
    bound, so ``lambda`` decays towards zero as the solve converges. The FROG
    Jacobian is strongly rank-deficient, so an undamped Gauss--Newton system is
    singular — and the normal equations reach that point sooner because they square
    the condition number. This guards the resulting NaN risk: 500 unrestrained
    iterations must still yield a finite error history and a finite spectrum.
    """
    g, ew, delays = setup
    trace = maketrace(g.omega, delays, ew, "pg")
    near = gaussian_pulse(g, 1.5e-15)
    res = OptxLM(maxiters=500, reltol=0.0, abstol=0.0, linear_solver=linear_solver).run(
        trace, g.omega, delays, "pg", guess=near
    )
    assert np.all(np.isfinite(res.errors))
    assert np.all(np.isfinite(res.spectrum))
    assert res.error < 5e-3


# -- previously untested OptxLM options --------------------------------------


@pytest.mark.parametrize("linear_solver", LINEAR_SOLVERS)
def test_optx_lm_r_omega(setup, linear_solver):
    """The per-frequency adaptive scaling path converges under both solvers."""
    g, ew, delays = setup
    trace = maketrace(g.omega, delays, ew, "pg")
    near = gaussian_pulse(g, 1.5e-15)
    res = OptxLM(maxiters=300, R_omega=True, linear_solver=linear_solver).run(
        trace, g.omega, delays, "pg", guess=near
    )
    assert res.error < 5e-3


@pytest.mark.parametrize("linear_solver", LINEAR_SOLVERS)
def test_optx_lm_fits_smearing_scale(linear_solver):
    """A known smearing-width multiplier is recovered (joint, both channels).

    The OptxLM counterpart of ``test_fit_smearing_split_recovers_channel_scales``
    in ``tests/test_smearing.py``, which only exercised ``lbfgs-ad``. The pulse is
    chirped because a transform-limited gate makes the ``p`` channel a pure
    attenuation that ``mu`` absorbs, leaving the scale unidentifiable.
    """
    import jax.numpy as jnp

    from croak.forward_jax import make_param_trace_fn

    g = Grid(96, dt=0.6e-15)
    delays = np.linspace(-15e-15, 15e-15, 31)
    kernel = square_boxcars_kernel(
        "pg", hole_diameter=1e-3, hole_spacing=1.5e-3, wavelength=260e-9, npoints=5
    )
    ew = gaussian_pulse(g, 2.5e-15, phases=[6e-30])
    trace_fn = make_param_trace_fn(g.omega, delays, "pg", smearing=kernel)
    t_meas = np.asarray(trace_fn(jnp.asarray(ew), 0.0, 0.0, 0.9))

    res = OptxLM(
        maxiters=300,
        smearing=kernel,
        fit_smearing=True,
        linear_solver=linear_solver,
    ).run(t_meas, g.omega, delays, "pg", guess=ew)
    assert res.smear_scale == pytest.approx(0.9, abs=0.02)
    assert res.smear_scale_delta is None  # joint fit reports no split channel


@pytest.mark.parametrize("linear_solver", LINEAR_SOLVERS)
def test_optx_lm_bspline_phase_basis(linear_solver):
    """The B-spline phase basis retrieves a chirped pulse under both solvers.

    Mirrors ``test_spline_phase_retrieval_lbfgs_ad`` in ``tests/test_basis_global.py``,
    which only covered ``lbfgs-ad`` (``lm`` + B-spline is exercised in
    ``tests/test_covariance.py``, but ``lm-optx`` was untested on this basis).
    It is also the small-``n`` regime for the linear solve — the Jacobian's column
    count drops from ``2N`` to ``n_nodes`` — which is the opposite end from
    :func:`test_optx_lm_linear_solvers_agree` and the reason the B-spline basis is
    the main lever on ``lm-optx`` iteration cost.
    """
    g = Grid(96, dt=0.4e-15)
    ew = gaussian_pulse(g, 3e-15, phases=[5e-30])  # GDD chirp
    delays = np.linspace(-35e-15, 35e-15, 75)
    trace = maketrace(g.omega, delays, ew, "pg")
    guess = gaussian_pulse(g, 3e-15)  # right amplitude, flat (wrong) phase
    res = OptxLM(
        phase_only=True,
        phase_basis="bspline",
        n_nodes=20,
        maxiters=300,
        linear_solver=linear_solver,
    ).run(trace, g.omega, delays, "pg", guess=guess)
    assert res.error < 5e-3
    assert res.spectrum.shape == (g.n,)
