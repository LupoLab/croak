"""Tests for the live full-plot preview machinery.

Covers the shared :func:`croak.solver.assemble_result` helper, the per-iteration
*snapshot* builders solvers hand to their callback, and the throttling / Stop
logic of :class:`croak.gui.worker.RetrievalWorker`.
"""

import numpy as np
import pytest

from croak.forward import maketrace
from croak.grid import Grid
from croak.lbfgs import LBFGS
from croak.pulses import gaussian_pulse
from croak.result import RetrievalResult
from croak.solver import assemble_result


@pytest.fixture
def problem():
    """A small SHG problem: grid, true field, delays and normalised trace."""
    g = Grid(64, dt=0.4e-15)
    ew = gaussian_pulse(g, 2.5e-15)
    delays = np.linspace(-12e-15, 12e-15, 41)
    trace = maketrace(g.omega, delays, ew, "shg")
    return g, ew, delays, trace / trace.max()


# -- assemble_result ---------------------------------------------------------


def test_assemble_result_optimal_scaling(problem):
    """A perfectly-scaled candidate gives mu absorbing the scale and R = 0."""
    g, ew, delays, t_meas = problem
    weights = np.ones(g.n)
    # A trace that is exactly half the measured one: the optimal mu must be 2,
    # the scaled trace must match T_meas and the FROG error must vanish.
    res = assemble_result(
        spectrum=ew,
        t_sim=0.5 * t_meas,
        grid=g,
        t_meas=t_meas,
        delays=delays,
        weights=weights,
        interaction="shg",
        algorithm="unit",
        errors=[0.3, 0.1],
    )
    assert res.mu == pytest.approx(2.0)
    np.testing.assert_allclose(res.trace, t_meas)
    assert res.error == pytest.approx(0.0, abs=1e-10)
    assert res.errors == [0.3, 0.1]
    assert res.algorithm == "unit"
    assert res.interaction == "shg"


def test_assemble_result_passes_through_extras(problem):
    """Fitted extras and the carrier are recorded verbatim on the result."""
    g, ew, delays, t_meas = problem
    res = assemble_result(
        spectrum=ew,
        t_sim=t_meas.copy(),
        grid=g,
        t_meas=t_meas,
        delays=delays,
        weights=np.ones(g.n),
        interaction="pg",
        algorithm="unit",
        errors=[],
        thickness=1.5e-6,
        tau0=2e-15,
        omega0=3e15,
    )
    assert res.thickness == 1.5e-6
    assert res.tau0 == 2e-15
    assert res.omega0 == 3e15
    # errors is copied, not aliased.
    res.errors.append(9.9)
    assert res.errors == [9.9]


def test_assemble_result_copies_errors(problem):
    """The errors list is copied so later solver appends do not mutate it."""
    g, ew, delays, t_meas = problem
    log = [0.5]
    res = assemble_result(
        spectrum=ew,
        t_sim=t_meas.copy(),
        grid=g,
        t_meas=t_meas,
        delays=delays,
        weights=np.ones(g.n),
        interaction="shg",
        algorithm="unit",
        errors=log,
    )
    log.append(0.2)  # the solver keeps logging after the snapshot is built
    assert res.errors == [0.5]


# -- solver snapshot builders ------------------------------------------------


def _run_with_snapshots(solver, trace, omega, delays, interaction, guess):
    """Run ``solver`` capturing a live snapshot at each iteration (as the GUI does).

    Returns ``(result, captured)`` where ``captured`` is a list of
    ``(iteration, R, snapshot_or_None)`` taken by *calling* the builder inside the
    callback, exactly as the worker does.
    """
    captured: list[tuple[int, float, RetrievalResult | None]] = []

    def callback(iteration, R, best_R, snapshot=None):
        captured.append((iteration, R, snapshot() if snapshot is not None else None))

    result = solver.run(
        trace, omega, delays, interaction, guess=guess, callback=callback
    )
    return result, captured


def test_lbfgs_snapshot_is_valid_in_progress_result(problem):
    """LBFGS hands the callback a builder of a complete, current full result."""
    g, ew, delays, t_meas = problem
    near = gaussian_pulse(g, 1.5e-15)
    res, captured = _run_with_snapshots(
        LBFGS(maxiters=20), t_meas, g.omega, delays, "shg", near
    )
    assert captured  # the solver iterated
    # Every iteration offered a snapshot builder.
    assert all(s is not None for _, _, s in captured)

    it_mid, R_mid, snap = captured[len(captured) // 2]
    assert isinstance(snap, RetrievalResult)
    assert snap.trace is not None and snap.trace.shape == t_meas.shape
    assert snap.spectrum.shape == (g.n,)
    # The snapshot's error equals the live FROG error reported at that iteration,
    # and it carries the convergence history up to that point.
    assert snap.error == pytest.approx(R_mid, rel=1e-6)
    assert len(snap.errors) == it_mid
    assert res.error < 5e-3  # sanity: the run still converged


def test_lbfgs_snapshot_does_not_corrupt_gradient(problem):
    """Building snapshots every iteration must not disturb the analytic gradient.

    LBFGS reuses shared buffers between the objective and gradient; the snapshot
    must use fresh buffers. If it leaked, convergence from a near guess would be
    spoiled — so this run must still reach the same low error as without preview.
    """
    g, ew, delays, t_meas = problem
    near = gaussian_pulse(g, 1.5e-15)
    plain = LBFGS(maxiters=60).run(t_meas, g.omega, delays, "shg", guess=near)
    res, _ = _run_with_snapshots(
        LBFGS(maxiters=60), t_meas, g.omega, delays, "shg", near
    )
    assert res.error == pytest.approx(plain.error, rel=1e-6)


def test_lbfgs_ad_snapshot_is_valid_in_progress_result(problem):
    """The JAX AD solver (a snapshot-priority solver) builds valid snapshots too."""
    pytest.importorskip("jax")
    from croak.lbfgs_ad import LBFGSAD

    g, ew, delays, t_meas = problem
    near = gaussian_pulse(g, 1.5e-15)
    _, captured = _run_with_snapshots(
        LBFGSAD(maxiters=12), t_meas, g.omega, delays, "shg", near
    )
    snaps = [(it, R, s) for it, R, s in captured if s is not None]
    assert snaps
    it_mid, R_mid, snap = snaps[len(snaps) // 2]
    assert isinstance(snap, RetrievalResult)
    assert snap.trace is not None and snap.trace.shape == t_meas.shape
    assert snap.error == pytest.approx(R_mid, rel=1e-4)
    assert len(snap.errors) == it_mid


# -- RetrievalWorker throttling and Stop -------------------------------------


def test_worker_throttles_previews(qtbot, monkeypatch):
    """Previews fire on the first iteration then at most once per interval."""
    from croak.gui import worker as worker_mod

    w = worker_mod.RetrievalWorker(None, None, preview=True)
    seen: list[object] = []
    w.preview.connect(seen.append)

    clock = {"t": 1000.0}
    monkeypatch.setattr(worker_mod.time, "perf_counter", lambda: clock["t"])
    sentinel = object()

    def snap():
        return sentinel

    w._callback(1, 0.5, 0.5, snapshot=snap)  # first → emits
    w._callback(2, 0.4, 0.4, snapshot=snap)  # same instant → throttled
    clock["t"] = 1000.5
    w._callback(3, 0.3, 0.3, snapshot=snap)  # +0.5 s (< interval) → throttled
    clock["t"] = 1001.6
    w._callback(4, 0.2, 0.2, snapshot=snap)  # +1.6 s (≥ interval) → emits
    assert seen == [sentinel, sentinel]


def test_worker_preview_disabled_emits_nothing(qtbot):
    """With the toggle off, no full-plot previews are built or emitted."""
    from croak.gui.worker import RetrievalWorker

    w = RetrievalWorker(None, None, preview=False)
    seen: list[object] = []
    w.preview.connect(seen.append)
    w._callback(1, 0.5, 0.5, snapshot=lambda: object())
    assert seen == []


def test_worker_stop_captures_snapshot(qtbot):
    """Stop raises after capturing the exact stopping-point result for display."""
    from croak.gui.worker import RetrievalWorker, StopRetrieval

    w = RetrievalWorker(None, None, preview=False)  # even with previews off
    w.request_stop()
    sentinel = object()
    with pytest.raises(StopRetrieval):
        w._callback(7, 0.1, 0.1, snapshot=lambda: sentinel)
    assert w._stop_result is sentinel


def test_worker_stop_without_snapshot(qtbot):
    """A convergence-only solver (no snapshot) stops with no captured result."""
    from croak.gui.worker import RetrievalWorker, StopRetrieval

    w = RetrievalWorker(None, None, preview=True)
    w.request_stop()
    with pytest.raises(StopRetrieval):
        w._callback(7, 0.1, 0.1, snapshot=None)
    assert w._stop_result is None


def test_worker_build_swallows_failures():
    """A failing snapshot build yields None rather than aborting the retrieval."""
    from croak.gui.worker import RetrievalWorker

    def bad():
        raise ValueError("degenerate field")

    assert RetrievalWorker._build(bad) is None
    sentinel = object()
    assert RetrievalWorker._build(lambda: sentinel) is sentinel
