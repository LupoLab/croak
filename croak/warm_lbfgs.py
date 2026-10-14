"""Retrieve with L-BFGS-AD after a COPRA local-sweep warm start.

COPRA has two stages that behave very differently. Its *local* sweep visits
delays in random order and replaces each one's signal modulus with the
measured value, which injects far more measured information per iteration
than a gradient step; its *global* stage is a gradient descent, and on the
cases we measured engaging it made the result worse. The local sweep alone
therefore produces a good, cheap estimate, but one that cannot represent
geometrical smearing: a smeared trace is an incoherent mixture with no single
signal whose modulus to replace.

This solver uses that estimate as the starting point for the regularised
extended model under L-BFGS. The final objective and options are those of
:class:`croak.lbfgs_ad.LBFGSAD`; only the initial guess changes. The warm-up
uses the core dispersive model (material, thickness, quadrature, and
``phase_only`` when set) and, when supplied, the reduced smearing kernel to
score its iterates. It does not include the chromatic focal or collection
model, fitted extras, regularisation or marginal-peak centring of the final
stage, and its local projection is the unsmeared single-field update.

Measured effect
---------------
On simulated TG-FROG traces (the Gaussian-beam control and the apertured mask,
each at transform limit and ±2 fs², plus a mask ±0.625 fs² pair; eight cases
at a 9.5 µm substrate), compared with a cold ``lbfgs-ad`` start:

* Accuracy: mean complex error 0.333 → 0.208 (38 % lower), median
  0.408 → 0.150 (63 % lower); mean ``|dFWHM|`` 17.7 % → 8.5 %, worst
  69.3 % → 20.8 %. The duration improved in seven of the eight cases and the
  complex error degraded by more than 20 % in none. It rescues bad basins
  rather than refining good solutions.
* Variance over starting points is the largest effect: the spread in
  retrieved duration collapsed five-fold to twelve-fold on every chirped arm
  (e.g. ±36.9 → ±3.1 points), while the transform-limited arms, which were
  never start-sensitive, were unchanged.
* Known regression: on the Gaussian transform-limited arm at depths away from
  9.5 µm the warm complex error reached 0.63 against a cold maximum of 0.18,
  i.e. where the cold start was already reliable.

Noise overfitting
-----------------
The COPRA warm-up has no spectral penalty. On a noisy trace whose grid spans
far more bandwidth than the signal, any solver without a spectral target can fit
noise into signal-free spectral bins near the grid edges; the warm-up does so
before the penalised stage starts. On the noisy trace of
``examples/example_thickness_uncertainty.py``, both ``warm-lbfgs`` and
``lbfgs-ad`` without a target leave ~3 % of the spectral energy at the edges and
return durations from about 1.9 to 3.2 fs for a 3.0 fs pulse. With the measured
spectrum as target, ``lbfgs-ad`` recovers 2.97 fs while ``warm-lbfgs`` still ends
at 3.16 fs. Check :func:`croak.edge_energy_fraction` on noisy data and supply
the measured spectrum.

The result combines both convergence histories. COPRA sweeps and L-BFGS
function evaluations are different units of work; ``stage_boundaries`` marks
the handover.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np

from .copra_jax import COPRAJax
from .lbfgs_ad import LBFGSAD
from .result import RetrievalResult

#: `stall_patience` large enough that the local sweep never hands over to the
#: global stage within any plausible iteration budget.
_LOCAL_ONLY = 10**9


def _with_history(result: RetrievalResult, warm: RetrievalResult) -> RetrievalResult:
    """Return ``result`` carrying both stages' error history and their join.

    The L-BFGS stage only ever logs its own errors, so on its own the
    convergence curve starts partway down with no visible account of the work
    that got it there. Prepending the warm-up's log — and recording where the two
    meet — makes the plot show the whole run.
    """
    return replace(
        result,
        errors=list(warm.errors) + list(result.errors),
        stage_boundaries=[len(warm.errors)],
    )


def _relay_after_warmup(
    callback: Callable[..., None], warm: RetrievalResult
) -> Callable[..., None]:
    """Wrap ``callback`` so the second stage continues the first stage's count.

    Without the offset the progress read-out restarts at iteration 1 halfway
    through the run; without the snapshot wrapper a live preview taken after the
    handover would show only the L-BFGS errors and then jump when the final
    result arrives with both.
    """
    offset = len(warm.errors)

    def relay(iteration, R, best_R, snapshot=None) -> None:
        callback(
            iteration + offset,
            R,
            min(best_R, warm.error),
            snapshot=None
            if snapshot is None
            else lambda: _with_history(snapshot(), warm),
        )

    return relay


class WarmLBFGS(LBFGSAD):
    """Retrieve with a COPRA local-sweep initialisation followed by L-BFGS-AD.

    Accepts every :class:`~croak.lbfgs_ad.LBFGSAD` argument for the final
    stage. The warm-up inherits the dispersive model and the smearing kernel,
    so it judges its iterates against the same physics: its projection step
    cannot *use* the kernel, but the error by which it selects the returned
    iterate is the smeared one. Per-frequency factors are off during the
    warm-up (COPRA's projection has no use for them) and free to move
    afterwards. See :mod:`croak.warm_lbfgs` for the remaining differences.

    Parameters
    ----------
    warmup_iters : int, optional
        Maximum COPRA warm-up sweeps (default 300); the sweep normally
        converges well inside this. Separate from the final L-BFGS
        ``maxiters`` budget.
    """

    #: Name recorded in results and used by uncertainty replicates when no
    #: alternative solver is requested. It must not inherit ``LBFGSAD``'s
    #: label: a warm-started result would then claim to be a plain
    #: ``lbfgs-ad`` (contradicting its ``stage_boundaries``), and a bootstrap
    #: would re-run its replicates with the cold solver.
    name = "warm-lbfgs"

    def __init__(self, *args, warmup_iters: int = 300, **kwargs):
        super().__init__(*args, **kwargs)
        self.warmup_iters = int(warmup_iters)

    def _solve(
        self, grid, trace, delays, interaction, ew0, weights, rng, callback=None
    ) -> RetrievalResult:
        # The warm-up gets the caller's callback unchanged, so progress and the
        # live preview run from the first COPRA iteration rather than starting
        # only once the handover happens.
        warm = COPRAJax(
            maxiters=self.warmup_iters,
            material=self.material,
            thickness=self.thickness,
            npoints=self.npoints,
            omega0=self.omega0,
            quadrature=self.quadrature,
            phase_only=self.phase_only,
            smearing=self.smearing,
            R_omega=False,
            stall_patience=_LOCAL_ONLY,
        )._solve(grid, trace, delays, interaction, ew0, weights, rng, callback=callback)
        result = super()._solve(
            grid,
            trace,
            delays,
            interaction,
            np.asarray(warm.spectrum),
            weights,
            rng,
            callback=None if callback is None else _relay_after_warmup(callback, warm),
        )
        return _with_history(result, warm)
