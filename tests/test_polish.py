"""Check the two-phase retrieval contract on a small delay-offset problem."""

import numpy as np
import pytest

from croak.forward import maketrace
from croak.grid import Grid
from croak.lm import LM
from croak.optimistix_lm import OptxLM
from croak.pulses import gaussian_pulse


@pytest.mark.parametrize("solver", [LM, OptxLM])
def test_two_phase_polish_holds_then_recovers_delay_offset(solver):
    """Improve the pulse with a held offset, then jointly recover the offset."""
    # Two-phase sequencing needs no depth integration or large convergence
    # problem. Each solver's separate thickness-fit test covers slab recovery.
    g = Grid(32, dt=0.5e-15)
    ew = gaussian_pulse(g, 3e-15)
    guess = gaussian_pulse(g, 2.5e-15)
    delays = np.linspace(-6e-15, 6e-15, 17)
    tau0_true = 0.5e-15
    trace = maketrace(g.omega, delays - tau0_true, ew, "shg")
    snapshots = []

    def record(_iteration, _error, _best, *, snapshot):
        snapshots.append(snapshot())

    result = solver(maxiters=30, fit_tau0=True, polish=True).run(
        trace, g.omega, delays, "shg", guess=guess, callback=record
    )

    held = [item for item in snapshots if item.tau0 == 0.0]
    # A single joint fit starts at zero too; requiring improvement while the
    # offset is still held distinguishes the first phase from that first step.
    assert len(held) > 1
    assert held[-1].error < held[0].error
    # Recover the offset to better than a tenth of one measured delay bin.
    assert result.tau0 == pytest.approx(tau0_true, abs=0.1 * (delays[1] - delays[0]))
    # Freeing the offset must explain the residual left by the fixed-offset fit,
    # rather than merely return a finite result after running the second phase.
    assert result.error < 0.1 * held[-1].error
    assert result.thickness is None
