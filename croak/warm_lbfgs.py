"""Retrieve with L-BFGS-AD after a COPRA local-sweep warm start.

The initial COPRA sweep supplies a pulse estimate to the regularised AD solver.
The final objective and available options are those of
:class:`croak.lbfgs_ad.LBFGSAD`. Warm starting can improve convergence from some
initial guesses, but can also lead to a worse local solution. Compare several
starts and inspect fields as well as trace errors.

The warm-up uses the core dispersive model and, when supplied, the reduced
smearing kernel to score candidates. It does not include the chromatic focal
or collection model, fitted extras, regularisation or marginal-peak centring
of the final stage. Its local projection is the unsmeared single-field update.
This is an initialisation strategy, not a second complete fit of the final model.

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

    Accept all :class:`~croak.lbfgs_ad.LBFGSAD` arguments for the final stage.
    The warm-up fixes per-frequency scaling off and uses the subset of model
    options described in the module introduction.

    Parameters
    ----------
    warmup_iters : int, optional
        Maximum COPRA warm-up sweeps (default 300). This is separate from the
        final L-BFGS ``maxiters`` budget.
    """

    #: Name recorded in results and used by uncertainty replicates when no
    #: alternative solver is requested.
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
