"""Option and reporting tests for the evolutionary retriever (:mod:`croak.cmaes`).

The search behaviour itself is tested in ``test_basis_global.py``; these use a
tiny phase-only B-spline problem and a handful of generations.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("evosax")

from croak.cmaes import CMAES, _import_strategy  # noqa: E402 - after importorskip
from croak.forward import maketrace  # noqa: E402
from croak.grid import Grid  # noqa: E402
from croak.pulses import gaussian_pulse  # noqa: E402


@pytest.fixture
def small():
    g = Grid(32, dt=1e-15)
    ew = gaussian_pulse(g, 6e-15, phases=(10e-30,))
    delays = np.linspace(-15e-15, 15e-15, 11)
    return g, ew, delays, maketrace(g.omega, delays, ew, "pg")


def _fast(**kw) -> CMAES:
    base = dict(
        phase_only=True, phase_basis="bspline", n_nodes=6, maxiters=4, popsize=8
    )
    base.update(kw)
    return CMAES(**base)


def test_import_strategy_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown strategy 'pso'"):
        _import_strategy("pso")


def test_cmaes_verbose_and_callback_report_every_generation(small, capsys):
    g, ew, delays, trace = small
    seen = []
    res = _fast(seed=1, verbose=True).run(
        trace,
        g.omega,
        delays,
        "pg",
        guess=ew,
        callback=lambda gen, r, best, snapshot=None: seen.append(
            (gen, r, best, snapshot().error)
        ),
    )
    assert [s[0] for s in seen] == [1, 2, 3, 4]
    assert "  gen    1: best R = " in capsys.readouterr().out
    # the running best never increases, and the snapshot is the all-time best
    bests = [s[2] for s in seen]
    assert bests == sorted(bests, reverse=True)
    assert seen[-1][3] == pytest.approx(bests[-1], rel=1e-6)
    assert res.error == pytest.approx(bests[-1], rel=1e-6)


def test_cmaes_phase_penalty_only(small):
    """reg_amp=0, reg_spectrum=0 and a phase penalty: the bare R is still reported."""
    g, ew, delays, trace = small
    res = _fast(seed=2, reg_amp=0.0, reg_spectrum=0.0, reg_phase=1e-3).run(
        trace, g.omega, delays, "pg", guess=ew
    )
    assert np.isfinite(res.error) and res.error == pytest.approx(min(res.errors))


def test_cmaes_without_a_seed_draws_its_key_from_the_rng(small):
    """No ``seed``: the JAX key comes from ``rng``, so equal rngs give equal runs."""
    g, ew, delays, trace = small

    def run(seed):
        return _fast().run(
            trace, g.omega, delays, "pg", guess=ew, rng=np.random.default_rng(seed)
        )

    a, b = run(3), run(3)
    np.testing.assert_array_equal(a.spectrum, b.spectrum)
