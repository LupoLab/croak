"""Convergence tests for COPRA (thin and dispersive)."""

import numpy as np
import pytest

from croak.copra import COPRA
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
def test_copra_perfect_initial(setup, interaction):
    g, ew, delays = setup
    trace = maketrace(g.omega, delays, ew, interaction)
    res = COPRA(maxiters=30).run(trace, g.omega, delays, interaction, guess=ew)
    assert res.error < 1e-6


@pytest.mark.parametrize("interaction", ["pg", "shg"])
def test_copra_near_initial(setup, interaction):
    g, ew, delays = setup
    trace = maketrace(g.omega, delays, ew, interaction)
    near = gaussian_pulse(g, 1.0e-15)  # wrong-width Gaussian
    res = COPRA(maxiters=100).run(trace, g.omega, delays, interaction, guess=near)
    assert res.error < 5e-3


@pytest.mark.parametrize("interaction", ["pg", "shg"])
def test_copra_random_initial(setup, interaction):
    g, ew, delays = setup
    trace = maketrace(g.omega, delays, ew, interaction)
    rng = np.random.default_rng(0)
    res = COPRA(maxiters=300).run(
        trace, g.omega, delays, interaction, guess=None, rng=rng
    )
    assert res.error < 5e-2


def test_copra_chirped(setup):
    g, _, delays = setup
    ew = gaussian_pulse(g, 2.5e-15, phases=[3e-30])
    trace = maketrace(g.omega, delays, ew, "pg")
    near = gaussian_pulse(g, 2.0e-15)
    res = COPRA(maxiters=150).run(trace, g.omega, delays, "pg", guess=near)
    assert res.error < 5e-3


def test_copra_phase_only_fixes_amplitude(setup):
    g, ew, delays = setup
    chirped = gaussian_pulse(g, 2.5e-15, phases=[3e-30])
    trace = maketrace(g.omega, delays, chirped, "pg")
    guess = gaussian_pulse(g, 2.5e-15)  # correct amplitude, no chirp
    res = COPRA(maxiters=150, phase_only=True).run(
        trace, g.omega, delays, "pg", guess=guess
    )
    # amplitude must stay exactly the guess amplitude; only the phase is retrieved
    a_guess = np.abs(guess) / np.max(np.abs(guess))
    a_retr = np.abs(res.spectrum) / np.max(np.abs(res.spectrum))
    assert np.allclose(a_retr, a_guess, atol=1e-9)
    assert res.error < 5e-3


def test_shg_dispersive_raises():
    g = Grid(64, dt=0.5e-15)
    delays = np.linspace(-20e-15, 20e-15, 40)
    ew = gaussian_pulse(g, 5e-15)
    with pytest.raises(ValueError, match="dispersive"):
        maketrace(
            g.omega,
            delays,
            ew,
            "shg",
            material="SiO2",
            thickness=10e-6,
            npoints=10,
            omega0=wlfreq(400e-9),
        )


def test_copra_abstol_early_stop(setup):
    # abstol is an absolute target FROG error: COPRA stops once best_R <= abstol,
    # well before maxiters on an easy synthetic problem.
    g, ew, delays = setup
    trace = maketrace(g.omega, delays, ew, "pg")
    near = gaussian_pulse(g, 2.0e-15)
    res = COPRA(maxiters=300, abstol=1e-3).run(trace, g.omega, delays, "pg", guess=near)
    assert res.error <= 1e-3
    assert len(res.errors) < 300  # stopped early


def test_copra_reltol_plateau_stop(setup):
    # A relative tolerance of 1.0 means "stop as soon as the best error has not
    # more-than-halved-to-zero over the look-back window" — always true — so it
    # terminates deterministically one window after the start.
    from croak.copra import _COPRA_PATIENCE

    g, ew, delays = setup
    trace = maketrace(g.omega, delays, ew, "pg")
    near = gaussian_pulse(g, 2.0e-15)
    res = COPRA(maxiters=300, reltol=1.0).run(trace, g.omega, delays, "pg", guess=near)
    assert len(res.errors) == _COPRA_PATIENCE + 1


def test_copra_romega_zero_measured_rows_no_nan(setup):
    # Regression: in Rω mode the optimal per-frequency scale mu is exactly zero on
    # any frequency row whose *measured* trace is identically zero across all
    # delays (out-of-band rows) while the model still has signal there. The old
    # code computed measured = t_meas / mu = 0/0 = NaN on those rows, which the
    # FFT-based adjoint then smeared across every frequency, so every reported
    # error came back NaN. With uniform weights (the GUI/session default) those
    # rows are not down-weighted, so the guard must keep the retrieval finite.
    g, ew, delays = setup
    trace = maketrace(g.omega, delays, ew, "pg")
    # Force several frequency rows to be identically zero in the measurement,
    # exactly the out-of-band / cropped situation that triggered the bug.
    trace[:5, :] = 0.0
    trace[-5:, :] = 0.0
    near = gaussian_pulse(g, 1.5e-15)
    res = COPRA(maxiters=20, R_omega=True).run(trace, g.omega, delays, "pg", guess=near)
    assert np.isfinite(res.error)
    assert np.all(np.isfinite(res.errors))
    assert np.all(np.isfinite(res.spectrum))


def test_copra_dispersive():
    # Dispersive propagation is just COPRA with material/thickness set (the
    # former "D-COPRA"); it must still retrieve a slab-propagated trace.
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
    res = COPRA(
        maxiters=150, material="SiO2", thickness=10e-6, npoints=20, omega0=omega0
    ).run(trace, g.omega, delays, "pg", guess=near)
    assert res.error < 5e-2
