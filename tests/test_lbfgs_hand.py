"""Tests for the jitted hand-adjoint retriever :class:`croak.lbfgs_hand.LBFGSHand`.

Its gradient is checked against autodiff in ``test_ad_vs_analytic.py``; here the
options (per-frequency scale, penalties, phase-only mode, early stopping) and
the NLopt plumbing are exercised on a small problem.
"""

from __future__ import annotations

import numpy as np
import pytest
from nlopt_harness import script_nlopt

from croak.forward import maketrace
from croak.grid import Grid
from croak.lbfgs import LBFGS
from croak.lbfgs_hand import LBFGSHand
from croak.pulses import gaussian_pulse


@pytest.fixture
def small():
    g = Grid(32, dt=1e-15)
    ew = gaussian_pulse(g, 6e-15, phases=(10e-30,))
    delays = np.linspace(-15e-15, 15e-15, 11)
    return g, ew, delays, maketrace(g.omega, delays, ew, "pg")


def test_lbfgs_hand_per_frequency_scale(small):
    """R_omega absorbs a per-row response exactly, leaving the truth optimal."""
    g, ew, delays, trace = small
    scaled = trace * np.linspace(1.0, 2.0, g.n)[:, None]
    res = LBFGSHand(maxiters=30, R_omega=True).run(
        scaled, g.omega, delays, "pg", guess=ew
    )
    assert np.shape(res.mu) == (g.n,)
    assert res.error < 1e-6


@pytest.mark.parametrize("phase_only", [False, True])
def test_lbfgs_hand_matches_numpy_lbfgs_with_a_phase_penalty(small, phase_only):
    """Same objective and gradients as LBFGS, so the same first evaluations."""
    g, ew, delays, trace = small
    kw = dict(maxiters=8, reg_amp=0.0, reg_phase=1e-4, phase_only=phase_only)
    guess = gaussian_pulse(g, 6.5e-15, phases=(5e-30,))
    hand = LBFGSHand(**kw).run(trace, g.omega, delays, "pg", guess=guess)
    ref = LBFGS(**kw).run(trace, g.omega, delays, "pg", guess=guess)
    # identical arithmetic up to JAX/NumPy summation order (float64)
    np.testing.assert_allclose(hand.errors[:3], ref.errors[:3], rtol=1e-6)


def test_lbfgs_hand_verbose_target_error_and_callback(small, capsys):
    g, _, delays, trace = small
    seen = []
    res = LBFGSHand(maxiters=200, targeterr=1e-2, verbose=True).run(
        trace,
        g.omega,
        delays,
        "pg",
        guess=gaussian_pulse(g, 7e-15),
        callback=lambda i, err, best, snapshot=None: seen.append(snapshot().error),
    )
    assert res.error <= 1e-2 and len(res.errors) < 200
    assert "  eval    1: R = " in capsys.readouterr().out
    np.testing.assert_allclose(seen, res.errors, rtol=1e-6)


def test_lbfgs_hand_value_only_evaluation(small, monkeypatch):
    g, ew, delays, trace = small
    script_nlopt(monkeypatch, value_only_probe=True)
    res = LBFGSHand(maxiters=10).run(trace, g.omega, delays, "pg", guess=ew)
    assert res.error < 1e-6


def test_lbfgs_hand_converged_start_returns_the_optimum(small):
    """Regression: NLopt's line search fails at an exact optimum.

    Started at the generating pulse, every step is below roundoff and NLopt
    raises ``runtime_error``; LBFGSHand used to let it escape and abort the
    retrieval, unlike LBFGS and LBFGSAD, which salvage the best iterate (here
    silently, since R is at roundoff).
    """
    g, ew, delays, trace = small
    res = LBFGSHand(maxiters=30).run(trace, g.omega, delays, "pg", guess=ew)
    assert res.error < 1e-10


def test_lbfgs_hand_salvages_a_line_search_failure(small, monkeypatch):
    g, _, delays, trace = small
    script_nlopt(monkeypatch, fail_after=2)
    far = gaussian_pulse(g, 12e-15)
    with pytest.warns(RuntimeWarning, match=r"NLopt stopped early \(RoundoffLimited\)"):
        res = LBFGSHand(maxiters=10).run(trace, g.omega, delays, "pg", guess=far)
    assert len(res.errors) == 2 and res.error == pytest.approx(min(res.errors))


def test_lbfgs_hand_spectral_penalty_pulls_towards_the_target(small):
    """With a measured spectrum the spectral term enters value and gradient."""
    g, ew, delays, trace = small
    target = np.abs(ew) ** 2
    far = gaussian_pulse(g, 9e-15)
    kw = dict(maxiters=40, reg_spectrum=1.0, spectrum_target=target)
    hand = LBFGSHand(**kw).run(trace, g.omega, delays, "pg", guess=far)
    ref = LBFGS(**kw).run(trace, g.omega, delays, "pg", guess=far)
    # identical objective: the first evaluation agrees to float64 rounding
    assert hand.errors[0] == pytest.approx(ref.errors[0], rel=1e-9)
    assert hand.error < hand.errors[0]  # and the fit improves on the start
