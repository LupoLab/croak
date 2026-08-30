"""Convergence + numpy-equivalence tests for the JAX COPRA (``copra-jax``)."""

import numpy as np
import pytest

from croak.copra import COPRA
from croak.copra_jax import COPRAJax
from croak.forward import maketrace
from croak.grid import Grid
from croak.maths import wlfreq
from croak.pulses import gaussian_pulse


@pytest.fixture
def setup():
    g = Grid(128, dt=0.3e-15)
    ew = gaussian_pulse(g, 2.5e-15)
    delays = np.linspace(-15e-15, 15e-15, 87)
    return g, ew, delays


@pytest.mark.parametrize("interaction", ["pg", "shg"])
def test_copra_jax_perfect_initial(setup, interaction):
    g, ew, delays = setup
    trace = maketrace(g.omega, delays, ew, interaction)
    res = COPRAJax(maxiters=30).run(trace, g.omega, delays, interaction, guess=ew)
    assert res.error < 1e-6


@pytest.mark.parametrize("interaction", ["pg", "shg"])
def test_copra_jax_near_initial(setup, interaction):
    g, ew, delays = setup
    trace = maketrace(g.omega, delays, ew, interaction)
    near = gaussian_pulse(g, 1.0e-15)
    res = COPRAJax(maxiters=100).run(trace, g.omega, delays, interaction, guess=near)
    assert res.error < 5e-3


@pytest.mark.parametrize("interaction", ["pg", "shg"])
def test_copra_jax_random_initial(setup, interaction):
    g, ew, delays = setup
    trace = maketrace(g.omega, delays, ew, interaction)
    rng = np.random.default_rng(0)
    res = COPRAJax(maxiters=300).run(
        trace, g.omega, delays, interaction, guess=None, rng=rng
    )
    assert res.error < 5e-2


def test_copra_jax_chirped(setup):
    g, _, delays = setup
    ew = gaussian_pulse(g, 2.5e-15, phases=[3e-30])
    trace = maketrace(g.omega, delays, ew, "pg")
    near = gaussian_pulse(g, 2.0e-15)
    res = COPRAJax(maxiters=150).run(trace, g.omega, delays, "pg", guess=near)
    assert res.error < 5e-3


def test_copra_jax_phase_only_fixes_amplitude(setup):
    g, _, delays = setup
    chirped = gaussian_pulse(g, 2.5e-15, phases=[3e-30])
    trace = maketrace(g.omega, delays, chirped, "pg")
    guess = gaussian_pulse(g, 2.5e-15)  # correct amplitude, no chirp
    res = COPRAJax(maxiters=150, phase_only=True).run(
        trace, g.omega, delays, "pg", guess=guess
    )
    a_guess = np.abs(guess) / np.max(np.abs(guess))
    a_retr = np.abs(res.spectrum) / np.max(np.abs(res.spectrum))
    assert np.allclose(a_retr, a_guess, atol=1e-9)
    assert res.error < 5e-3


def test_copra_jax_romega_zero_measured_rows_no_nan(setup):
    # Regression (mirrors test_copra_romega_zero_measured_rows_no_nan): in Rω mode
    # the per-frequency scale mu is zero on frequency rows whose measured trace is
    # identically zero, making the old measured = t_meas / mu = 0/0 = NaN that the
    # adjoint smeared everywhere. The JAX division was silent (no warning) but still
    # propagated NaN, so copra-jax failed the same way.
    g, ew, delays = setup
    trace = maketrace(g.omega, delays, ew, "pg")
    trace[:5, :] = 0.0
    trace[-5:, :] = 0.0
    near = gaussian_pulse(g, 1.5e-15)
    res = COPRAJax(maxiters=20, R_omega=True).run(
        trace, g.omega, delays, "pg", guess=near
    )
    assert np.isfinite(res.error)
    assert np.all(np.isfinite(res.errors))
    assert np.all(np.isfinite(res.spectrum))


def test_copra_jax_dispersive():
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
    res = COPRAJax(
        maxiters=150, material="SiO2", thickness=10e-6, npoints=20, omega0=omega0
    ).run(trace, g.omega, delays, "pg", guess=near)
    assert res.error < 5e-2


@pytest.mark.parametrize("interaction", ["pg", "shg"])
def test_copra_jax_matches_numpy(setup, interaction):
    """With the same seeded permutation, copra-jax tracks numpy COPRA closely."""
    g, ew, delays = setup
    trace = maketrace(g.omega, delays, ew, interaction)
    near = gaussian_pulse(g, 1.0e-15)
    rn = COPRA(maxiters=80).run(
        trace, g.omega, delays, interaction, guess=near, rng=np.random.default_rng(0)
    )
    rj = COPRAJax(maxiters=80).run(
        trace, g.omega, delays, interaction, guess=near, rng=np.random.default_rng(0)
    )
    # The permutation RNG is shared (host-side numpy), so the iterations should
    # agree to floating-point round-off (FFT ordering differences ~1e-9).
    assert len(rj.errors) == len(rn.errors)
    assert np.allclose(rj.errors, rn.errors, rtol=1e-5, atol=1e-9)
    assert rj.error == pytest.approx(rn.error, rel=1e-5, abs=1e-9)


def test_copra_jax_abstol_early_stop(setup):
    g, ew, delays = setup
    trace = maketrace(g.omega, delays, ew, "pg")
    near = gaussian_pulse(g, 2.0e-15)
    res = COPRAJax(maxiters=300, abstol=1e-3).run(
        trace, g.omega, delays, "pg", guess=near
    )
    assert res.error <= 1e-3
    assert len(res.errors) < 300


def test_copra_jax_reltol_plateau_stop(setup):
    from croak.copra_jax import _COPRA_PATIENCE

    g, ew, delays = setup
    trace = maketrace(g.omega, delays, ew, "pg")
    near = gaussian_pulse(g, 2.0e-15)
    res = COPRAJax(maxiters=300, reltol=1.0).run(
        trace, g.omega, delays, "pg", guess=near
    )
    assert len(res.errors) == _COPRA_PATIENCE + 1


def test_copra_jax_callback_supplies_a_live_snapshot(setup):
    """COPRA hands the GUI an in-progress result, so previews run during it.

    Without a ``snapshot=``, the Retrieve stage can only draw the convergence
    curve for COPRA; with one it can redraw the whole 12-panel view mid-sweep,
    like the gradient solvers. The snapshot must be a usable result — the right
    spectrum, the trace error the sweep just reported, and the error log so far.
    """
    g, ew, delays = setup
    trace = maketrace(g.omega, delays, ew, "pg")
    seen = []

    def callback(iteration, R, best_R, snapshot=None):
        seen.append((iteration, R, None if snapshot is None else snapshot()))

    COPRAJax(maxiters=4).run(
        trace,
        g.omega,
        delays,
        "pg",
        guess=gaussian_pulse(g, 1.0e-15),
        callback=callback,
    )

    assert [it for it, _, _ in seen] == [1, 2, 3, 4]
    for iteration, R, snap in seen:
        assert snap is not None
        assert snap.spectrum.shape == (g.n,)
        assert snap.trace is not None
        # the snapshot's own error is the one the sweep reported for that iterate
        assert snap.error == pytest.approx(R, rel=1e-9)
        assert len(snap.errors) == iteration


def test_copra_numpy_callback_supplies_a_live_snapshot(setup):
    """The NumPy twin snapshots too, so the two do not diverge in behaviour."""
    g, ew, delays = setup
    trace = maketrace(g.omega, delays, ew, "pg")
    snaps = []

    def callback(iteration, R, best_R, snapshot=None):
        snaps.append(None if snapshot is None else snapshot())

    COPRA(maxiters=3).run(
        trace,
        g.omega,
        delays,
        "pg",
        guess=gaussian_pulse(g, 1.0e-15),
        callback=callback,
    )
    assert len(snaps) == 3
    assert all(s is not None and s.spectrum.shape == (g.n,) for s in snaps)
