"""Convergence tests for COPRA (thin and dispersive)."""

import numpy as np
import pytest

from croak.copra import COPRA, run_copra
from croak.forward import ForwardModel, maketrace
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


@pytest.fixture
def small():
    g = Grid(32, dt=1e-15)
    ew = gaussian_pulse(g, 6e-15, phases=(10e-30,))
    delays = np.linspace(-15e-15, 15e-15, 11)
    return g, ew, delays, maketrace(g.omega, delays, ew, "pg")


def test_copra_zero_weight_rows_do_not_pull_the_fit(small):
    """Zero-weight rows target the model itself, so corrupting them changes nothing."""
    g, ew, delays, trace = small
    weights = np.ones(g.n)
    weights[:4] = 0.0
    corrupted = trace.copy()
    corrupted[:4] = 0.5  # garbage where the weight is zero
    kw = dict(guess=0.9 * ew, weights=weights, rng=np.random.default_rng(0))
    clean = COPRA(maxiters=10).run(trace, g.omega, delays, "pg", **kw)
    kw["rng"] = np.random.default_rng(0)
    dirty = COPRA(maxiters=10).run(corrupted, g.omega, delays, "pg", **kw)
    # the trace is max-normalised, so a 0.5 plateau in zero-weight rows rescales
    # nothing (the peak is 1) and the iterates coincide to rounding
    np.testing.assert_allclose(dirty.spectrum, clean.spectrum, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"abstol": 1.0}, "COPRA: reached abstol (1.00e+00) at iter 1"),
        ({"reltol": 1e9}, "COPRA: best error plateaued (reltol 1.00e+09)"),
    ],
)
def test_copra_verbose_reports_each_stop(small, capsys, kw, message):
    g, ew, delays, trace = small
    COPRA(maxiters=50, verbose=True, **kw).run(
        trace, g.omega, delays, "pg", guess=0.9 * ew, rng=np.random.default_rng(0)
    )
    out = capsys.readouterr().out
    assert "COPRA: initial R = " in out and "  iter    1: R = " in out
    assert message in out


def test_run_copra_callback_without_snapshots(small):
    """The bare iteration hands a three-argument callback no snapshot."""
    g, ew, delays, trace = small
    calls = []
    model = ForwardModel(g.omega, delays, "pg")
    _, errors, _, _, _ = run_copra(
        model,
        g,
        trace / trace.max(),
        0.9 * ew,
        np.ones(g.n),
        maxiters=3,
        rng=np.random.default_rng(0),
        callback=lambda *args: calls.append(args),
    )
    assert [c[0] for c in calls] == [1, 2, 3]
    assert [c[1] for c in calls] == errors
