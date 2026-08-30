"""Shared manual-stepping driver for the Optimistix-backed retrievers.

Both :class:`~croak.optimistix_lm.OptxLM` and
:class:`~croak.optimistix_lbfgs.OptxLBFGS` run their Optimistix solver through its
low-level ``init`` / ``step`` / ``terminate`` / ``postprocess`` interface (rather
than the all-in-one :func:`optimistix.least_squares` / :func:`optimistix.minimise`)
so that the FROG error can be logged and the progress ``callback`` fired once per
iteration — the all-in-one entry points hide their jitted ``while_loop``.

The solver objective ``fn(y, args)`` returns ``(out, R)`` where ``out`` is the
quantity Optimistix optimises (the least-squares residual vector for LM, the
scalar objective for L-BFGS) and the **aux** ``R`` is the bare FROG error, which
is what we log and forward to the callback.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

#: Plateau window (iterations) for the ``reltol`` relative-improvement stop, as in
#: :data:`croak.copra._COPRA_PATIENCE`: ``reltol`` triggers only once the *best*
#: FROG error has improved by less than ``reltol`` over this many steps.
_OPTX_PATIENCE = 10


def run_optx(
    solver,
    fn: Callable,
    y0: NDArray[np.float64],
    *,
    maxiters: int,
    targeterr: float = 0.0,
    reltol: float = 0.0,
    callback=None,
    make_snapshot: Callable[[NDArray[np.float64]], object] | None = None,
    err_log: list[float] | None = None,
    verbose: bool = False,
) -> tuple[NDArray[np.float64], list[float]]:
    """Step ``solver`` on ``fn`` from ``y0`` and return ``(y, err_log)``.

    Parameters
    ----------
    solver
        An Optimistix iterative solver (e.g. ``LevenbergMarquardt``, ``LBFGS``).
    fn : callable
        ``fn(y, args) -> (out, R)``: ``out`` is the optimiser objective/residual,
        ``R`` the bare FROG error returned as aux (logged each iteration).
    y0 : numpy.ndarray
        Initial real parameter vector.
    maxiters : int
        Maximum number of solver steps.
    targeterr : float, optional
        Stop early once ``R <= targeterr`` (disabled when ``0``).
    reltol : float, optional
        Relative-improvement plateau tolerance (disabled when ``0``): stop once
        the *best* FROG error has improved by less than ``reltol`` (relative) over
        the last :data:`_OPTX_PATIENCE` steps. Optimistix's own ``terminate`` uses
        a ``atol + rtol·|f|`` Cauchy test that is **atol-dominated** near
        convergence and so effectively ignores ``rtol``; this honours ``reltol``
        in the same "relative improvement" sense as :func:`croak.copra.run_copra`
        and SciPy's ``ftol``, for consistency across solvers.
    callback : callable, optional
        Called as ``callback(iteration, R, best_R, snapshot=...)`` after each
        step. ``snapshot`` is a zero-argument builder of the current
        :class:`~croak.result.RetrievalResult` (for the GUI live preview), or
        ``None`` when ``make_snapshot`` is not supplied.
    make_snapshot : callable, optional
        ``make_snapshot(y) -> RetrievalResult`` mapping the current parameter
        vector to an in-progress result; wired to the callback's ``snapshot``.
    err_log : list of float, optional
        Pre-existing list to append each step's error to (rather than a fresh
        one). Lets a multi-pass caller accumulate across passes and lets a
        snapshot builder closing over the list see the live history.
    verbose : bool, optional
        Print per-iteration progress.

    Returns
    -------
    y : numpy.ndarray
        The final (post-processed) parameter vector.
    err_log : list of float
        The FROG error ``R`` after each step, in order.
    """
    args = None
    options: dict = {}
    tags = frozenset()
    y = jnp.asarray(y0)

    f_struct, aux_struct = eqx.filter_eval_shape(fn, y, args)
    step = eqx.filter_jit(solver.step)
    terminate = eqx.filter_jit(solver.terminate)

    state = solver.init(fn, y, args, options, f_struct, aux_struct, tags)
    # Seed aux so postprocess has a value even if we converge before stepping.
    _, aux = fn(y, args)
    done, result = terminate(fn, y, args, options, state, tags)

    err_log = [] if err_log is None else err_log
    best_R = float("inf")
    best_log: list[float] = []  # best R after each step (for the reltol plateau stop)
    for _ in range(maxiters):
        if bool(done):
            break
        y, state, aux = step(fn, y, args, options, state, tags)
        R = float(aux)
        err_log.append(R)
        best_R = min(best_R, R)
        best_log.append(best_R)
        if verbose:
            print(f"  iter {len(err_log):4d}: R = {R:.6e}")
        if callback is not None:
            # Bind the current parameter vector into a zero-arg snapshot builder
            # (a host-side numpy copy, since `y` keeps stepping on device).
            snapshot = (
                functools.partial(make_snapshot, np.asarray(y))
                if make_snapshot is not None
                else None
            )
            callback(len(err_log), R, best_R, snapshot=snapshot)
        if targeterr > 0 and R <= targeterr:
            break
        # Relative-improvement plateau stop (mirrors COPRA): honour reltol since
        # Optimistix's terminate is atol-dominated and ignores rtol in practice.
        if reltol > 0.0 and len(best_log) > _OPTX_PATIENCE:
            past = best_log[-1 - _OPTX_PATIENCE]
            if past - best_R <= reltol * max(past, np.finfo(float).tiny):
                if verbose:
                    print(
                        f"  best error plateaued (reltol {reltol:.2e}) "
                        f"at iter {len(err_log)}"
                    )
                break
        done, result = terminate(fn, y, args, options, state, tags)

    y, _, _ = solver.postprocess(fn, y, aux, args, options, state, tags, result)
    return np.asarray(y), err_log
