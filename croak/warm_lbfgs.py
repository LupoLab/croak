"""L-BFGS-AD seeded from COPRA's local projection sweep.

COPRA has two stages that behave very differently. Its *local* sweep visits
delays in random order and replaces each one's signal modulus with the measured
value, which injects far more measured information per iteration than a
gradient step; its *global* stage is a gradient descent, and on the cases
measured here engaging it makes the result worse. Running the local sweep alone
therefore produces a good, cheap estimate — but one that cannot represent
geometrical smearing, because a smeared trace is an incoherent mixture with no
single signal whose modulus to replace.

This solver uses that estimate as the starting point for the full extended
model under L-BFGS. The forward model is unchanged; only the initial guess is.

What it buys, measured across the Gaussian-beam control and the apertured mask
at transform limit and +-2 fs^2 (and the mask +-0.625 pair).

Accuracy, all eight cases at the 9.5 um experimental substrate: mean complex
error 0.333 -> 0.208 (38% lower), median 0.408 -> 0.150 (63% lower), mean
``|dFWHM|`` 17.7% -> 8.5% (52% lower), worst ``|dFWHM|`` 69.3% -> 20.8%. The complex
error improved by more than 20% in four of the eight cases and degraded by more
than 20% in none; the duration improved in seven of eight. The pattern is that
it does little where the answer was already good and a great deal where it was
not: it rescues bad basins rather than refining good solutions, raising the
floor without lowering the ceiling.

Across depth, over a 1-40 um ladder, the complex-error ranges improve on most
arms -- clearly on both chirped Gaussian arms and on mask -2 fs^2 (0.33-0.55 ->
0.15-0.46), with the mask +2 arm unchanged. The retrieved-duration ranges over
depth are mixed and slightly wider on the mask arms; that alone is not the whole
picture and should not be read as one.

Variance over starting points is the largest effect. The spread in retrieved
duration collapses on every chirped arm -- five-fold on Gaussian +2 and mask -2,
twelve-fold on Gaussian -2 (+-36.9 -> +-3.1 points) and mask +2 (+-6.0 -> +-0.5)
-- while the transform-limited arms, which were never start-sensitive, are
unchanged. A retrieval whose answer moves by +-37% with the draw of the initial
guess is not a measurement whatever its mean error, and on real data one never
learns which draw one got.

One regression is known: the Gaussian transform-limited arm at depths away from
9.5 um, where the warm complex error reaches 0.63 against a cold maximum of
0.18. That is the case where the cold start was already reliable.
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
    """L-BFGS-AD with a COPRA local-sweep warm start.

    Takes every argument :class:`~croak.lbfgs_ad.LBFGSAD` does. The warm-up
    inherits the dispersive model (material, thickness, quadrature) and the
    smearing kernel, so it judges its iterates against the same physics; its
    projection step cannot *use* the kernel (the global stage that could never
    engages here), but the error it selects the returned iterate by is the
    smeared one. Per-frequency factors are off during the warm-up: COPRA's
    projection has no use for them, and they are free to move afterwards.

    Parameters
    ----------
    warmup_iters : int, optional
        Iteration budget for the COPRA sweep (default 300). The sweep normally
        converges well inside this.
    """

    #: Report the two-stage solver by its own name. Inheriting ``LBFGSAD``'s
    #: label made a warm-started result claim to be a plain ``lbfgs-ad`` --- which
    #: contradicts the ``stage_boundaries`` it carries, and sent an uncertainty
    #: bootstrap (which reads ``result.algorithm`` when none is given) off to
    #: re-run the replicates with the *cold* solver.
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
