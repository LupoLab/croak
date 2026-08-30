"""Two-stage bookkeeping of the COPRA-warm-started L-BFGS (``warm-lbfgs``)."""

import numpy as np
import pytest

from croak.forward import maketrace
from croak.grid import Grid
from croak.pulses import gaussian_pulse
from croak.warm_lbfgs import WarmLBFGS

WARMUP = 4
MAXITERS = 6


@pytest.fixture
def setup():
    g = Grid(64, dt=2e-15)
    ew = gaussian_pulse(g, 20e-15)
    delays = np.linspace(-60e-15, 60e-15, 32)
    return g, ew, delays


def _run(setup, callback=None):
    g, ew, delays = setup
    trace = maketrace(g.omega, delays, ew, "pg")
    return WarmLBFGS(maxiters=MAXITERS, warmup_iters=WARMUP).run(
        trace,
        g.omega,
        delays,
        "pg",
        guess=gaussian_pulse(g, 30e-15),
        callback=callback,
    )


def test_errors_include_the_warmup_iterations(setup):
    """The convergence history covers both stages, not just the second.

    The L-BFGS stage logs only its own errors, so on its own the curve begins
    partway down with no visible account of the work that got it there.
    """
    res = _run(setup)
    assert len(res.errors) > WARMUP  # the L-BFGS stage contributed too
    assert res.stage_boundaries == [WARMUP]
    assert all(np.isfinite(e) and e > 0 for e in res.errors)
    # The warm-up's errors come first, so the L-BFGS stage picks up from where
    # the sweep left off rather than restarting from the cold guess.
    assert res.errors[WARMUP] <= res.errors[0]


def test_progress_runs_continuously_across_the_handover(setup):
    """Iteration numbers continue across the stages instead of restarting."""
    seen = []

    def callback(iteration, R, best_R, snapshot=None):
        seen.append(iteration)

    _run(setup, callback)
    assert seen[:WARMUP] == list(range(1, WARMUP + 1))
    assert seen == sorted(seen)
    assert seen[WARMUP] == WARMUP + 1  # no restart at 1


def test_live_preview_is_available_during_the_warmup(setup):
    """Every iteration of *both* stages offers a snapshot for the live plot.

    The warm-up is 300 iterations by default, so without this the GUI's full-plot
    preview would sit blank for the majority of a ``warm-lbfgs`` run.
    """
    snaps = []

    def callback(iteration, R, best_R, snapshot=None):
        snaps.append((iteration, None if snapshot is None else snapshot()))

    _run(setup, callback)
    assert all(snap is not None for _, snap in snaps)
    # A snapshot taken mid-warm-up already carries the warm-up history...
    warm_snap = snaps[WARMUP - 1][1]
    assert len(warm_snap.errors) == WARMUP
    # ...and one taken after the handover carries both stages, marked at the join,
    # so the live curve matches the final one rather than jumping when it lands.
    after = snaps[WARMUP][1]
    assert after.stage_boundaries == [WARMUP]
    assert len(after.errors) == WARMUP + 1


def test_warm_start_still_returns_the_lbfgs_answer(setup):
    """Stitching the history must not disturb the retrieval itself."""
    res = _run(setup)
    assert res.error < 1e-2
    assert res.spectrum.shape == (setup[0].n,)
    assert res.algorithm == "warm-lbfgs"
