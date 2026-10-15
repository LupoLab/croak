"""Convergence tests for the L-BFGS solver."""

import numpy as np
import pytest
from nlopt_harness import script_nlopt

from croak.forward import maketrace
from croak.grid import Grid
from croak.lbfgs import LBFGS
from croak.maths import wlfreq
from croak.pulses import ArrayPulse, gaussian_pulse


@pytest.fixture
def setup():
    g = Grid(128, dt=0.3e-15)
    ew = gaussian_pulse(g, 2.5e-15)
    delays = np.linspace(-15e-15, 15e-15, 87)
    return g, ew, delays


@pytest.mark.parametrize("interaction", ["pg", "shg"])
def test_lbfgs_near_initial(setup, interaction):
    g, ew, delays = setup
    trace = maketrace(g.omega, delays, ew, interaction)
    near = gaussian_pulse(g, 1.5e-15)
    res = LBFGS(maxiters=300).run(trace, g.omega, delays, interaction, guess=near)
    assert res.error < 5e-3


@pytest.mark.parametrize("interaction", ["pg", "shg"])
def test_lbfgs_random_initial(setup, interaction):
    g, ew, delays = setup
    trace = maketrace(g.omega, delays, ew, interaction)
    rng = np.random.default_rng(1)
    res = LBFGS(maxiters=500).run(
        trace, g.omega, delays, interaction, guess=None, rng=rng
    )
    assert res.error < 5e-2


def test_lbfgs_chirped_phase_only(setup):
    g, _, delays = setup
    ew = gaussian_pulse(g, 2.5e-15, phases=[3e-30])
    trace = maketrace(g.omega, delays, ew, "pg")
    # correct amplitude, perturbed phase
    guess = gaussian_pulse(g, 2.5e-15, phases=[1e-30])
    res = LBFGS(maxiters=300, phase_only=True).run(
        trace, g.omega, delays, "pg", guess=guess
    )
    assert res.error < 1e-2


def _spectrum_l2(res, I_true):
    """Peak-normalised L2 distance between retrieved |E(ω)| and √I_true."""
    a = np.abs(res.spectrum)
    a /= a.max()
    s = np.sqrt(I_true)
    s /= s.max()
    return float(np.linalg.norm(a - s))


def test_lbfgs_spectral_reg_improves_spectrum(setup):
    # Structured (double-peak) spectrum, deliberately wrong-amplitude guess on a
    # NOISY trace: a clean PG trace already determines |E(ω)|, so noise is what
    # makes the amplitude under-constrained and gives the spectral-match penalty
    # something to pull on — with it, |E(ω)| lands close to the true spectrum.
    g, _, delays = setup
    ew = gaussian_pulse(g, 2.5e-15) + 0.6 * gaussian_pulse(g, 1.2e-15)
    ew = ew / np.max(np.abs(ew))
    I_true = np.abs(ew) ** 2
    clean = maketrace(g.omega, delays, ew, "pg")
    rng = np.random.default_rng(0)
    trace = np.clip(clean + 0.03 * rng.standard_normal(clean.shape), 0.0, None)
    near = gaussian_pulse(g, 4.0e-15)  # wrong amplitude
    r0 = LBFGS(maxiters=400).run(trace, g.omega, delays, "pg", guess=near)
    r1 = LBFGS(maxiters=400, reg_spectrum=0.5, spectrum_target=I_true).run(
        trace, g.omega, delays, "pg", guess=near
    )
    assert r1.error < 5e-2  # near the noise floor
    assert _spectrum_l2(r1, I_true) < _spectrum_l2(r0, I_true)


def test_lbfgs_spectral_reg_none_target_is_noop(setup):
    g, ew, delays = setup
    trace = maketrace(g.omega, delays, ew, "pg")
    near = gaussian_pulse(g, 1.5e-15)
    r0 = LBFGS(maxiters=200).run(trace, g.omega, delays, "pg", guess=near)
    r1 = LBFGS(maxiters=200, reg_spectrum=0.3, spectrum_target=None).run(
        trace, g.omega, delays, "pg", guess=near
    )
    assert np.allclose(r0.spectrum, r1.spectrum)


def test_lbfgs_dispersive():
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
    res = LBFGS(
        maxiters=400, material="SiO2", thickness=10e-6, npoints=20, omega0=omega0
    ).run(trace, g.omega, delays, "pg", guess=near)
    assert res.error < 5e-2


# ---------------------------------------------------------------------------
# Options and NLopt contracts on a small problem
# ---------------------------------------------------------------------------
@pytest.fixture
def small():
    g = Grid(32, dt=1e-15)
    ew = gaussian_pulse(g, 6e-15, phases=(10e-30,))
    delays = np.linspace(-15e-15, 15e-15, 11)
    trace = maketrace(g.omega, delays, ew, "pg")
    return g, ew, delays, trace


def test_lbfgs_per_frequency_mu_and_phase_regularisation(small):
    """R_omega fits a per-row scale; a phase penalty still converges near truth."""
    g, ew, delays, trace = small
    scaled = trace * np.linspace(1.0, 2.0, g.n)[:, None]  # a per-row response
    res = LBFGS(maxiters=60, R_omega=True, reg_phase=1e-6).run(
        scaled, g.omega, delays, "pg", guess=ew
    )
    assert np.shape(res.mu) == (g.n,)
    # per-row scaling is absorbed exactly by mu(ω), leaving the truth optimal
    assert res.error < 1e-6


def test_lbfgs_exact_start_has_a_zero_error_and_gradient(small):
    g, ew, delays, trace = small
    res = LBFGS(maxiters=5).run(trace, g.omega, delays, "pg", guess=ew)
    assert res.errors[0] == pytest.approx(0.0, abs=1e-15)


def test_lbfgs_verbose_and_target_error(small, capsys):
    """targeterr stops as soon as R drops below it; verbose logs each evaluation."""
    g, ew, delays, trace = small
    near = gaussian_pulse(g, 7e-15)
    res = LBFGS(maxiters=200, targeterr=1e-2, verbose=True).run(
        trace, g.omega, delays, "pg", guess=near
    )
    assert res.error <= 1e-2 and len(res.errors) < 200
    assert "  eval    1: R = " in capsys.readouterr().out


def test_lbfgs_value_only_evaluation(small, monkeypatch):
    """NLopt may ask for the objective without a gradient (an empty buffer)."""
    g, ew, delays, trace = small
    script_nlopt(monkeypatch, value_only_probe=True)
    res = LBFGS(maxiters=10).run(trace, g.omega, delays, "pg", guess=ew)
    assert res.error < 1e-6


def test_lbfgs_salvages_a_line_search_failure(small, monkeypatch):
    g, _, delays, trace = small
    script_nlopt(monkeypatch, fail_after=3)
    far = gaussian_pulse(g, 12e-15)
    with pytest.warns(RuntimeWarning, match=r"NLopt stopped early \(RoundoffLimited\)"):
        res = LBFGS(maxiters=10).run(trace, g.omega, delays, "pg", guess=far)
    assert len(res.errors) == 3 and res.error == pytest.approx(min(res.errors))


def test_lbfgs_non_finite_gradient_is_trapped(small, monkeypatch):
    """A NaN gradient is replaced by a penalty so the line search backtracks."""
    g, ew, delays, trace = small
    monkeypatch.setattr(
        ArrayPulse,
        "ew_vjp",
        lambda self, ew_bar: np.full_like(self.get_params(), np.nan),
    )
    script_nlopt(monkeypatch, fail_after=2)
    with pytest.warns(RuntimeWarning) as record:
        LBFGS(maxiters=5).run(trace, g.omega, delays, "pg", guess=0.9 * ew)
    messages = [str(w.message) for w in record]
    assert any("non-finite" in m for m in messages)


def test_lbfgs_non_finite_error_is_trapped(small):
    g, ew, delays, trace = small
    bad = trace.copy()
    bad[0, 0] = np.nan
    with pytest.warns(RuntimeWarning, match="non-finite"):
        LBFGS(maxiters=3).run(bad, g.omega, delays, "pg", guess=ew)


def test_lbfgs_without_amplitude_penalty_reports_progress(small):
    """reg_amp=0 drops the amplitude term; the callback sees every evaluation."""
    g, ew, delays, trace = small
    seen = []

    def callback(iteration, err, best, snapshot=None):
        seen.append((iteration, err, best, snapshot().error))

    res = LBFGS(maxiters=15, reg_amp=0.0).run(
        trace, g.omega, delays, "pg", guess=0.95 * ew, callback=callback
    )
    assert [s[0] for s in seen] == list(range(1, len(res.errors) + 1))
    # the snapshot is the in-progress result at that evaluation
    assert seen[0][3] == pytest.approx(seen[0][1], rel=1e-9)
    assert seen[-1][2] == pytest.approx(min(res.errors))
