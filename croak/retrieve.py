"""The :func:`retrieve` convenience wrapper and the algorithm registry."""

from __future__ import annotations

import inspect
import time

import numpy as np
from numpy.typing import ArrayLike

from .cmaes import CMAES
from .copra import COPRA
from .copra_jax import COPRAJax
from .lbfgs import LBFGS
from .lbfgs_ad import LBFGSAD
from .lbfgs_hand import LBFGSHand
from .lm import LM
from .optimistix_lbfgs import OptxLBFGS
from .optimistix_lm import OptxLM
from .progress import make_reporter
from .pulses import Pulse
from .result import RetrievalResult
from .solver import Retriever
from .warm_lbfgs import WarmLBFGS

__all__ = ["retrieve", "ALGORITHMS", "algorithm_params"]

#: Registry mapping algorithm names to solver classes. Dispersive propagation is
#: a parameter of the solvers (``material``/``thickness``), not a separate
#: algorithm. ``"lbfgs"`` uses hand-derived analytic gradients (pure NumPy);
#: ``"lbfgs-hand"`` uses the *same* hand-derived gradients but jitted in JAX
#: (``vmap`` over delays, no AD); ``"lbfgs-ad"`` and ``"lm"`` use JAX automatic
#: differentiation (gradient / Jacobian). The ``"-optx"`` variants are JAX-native
#: twins backed by Optimistix (the whole optimisation loop runs on-device).
#: ``"copra-jax"`` is the JAX twin of ``"copra"`` (vmap/jit global step,
#: ``lax.scan`` local sweep) — the same algorithm, vectorised over delays.
ALGORITHMS: dict[str, type[Retriever]] = {
    "copra": COPRA,
    "copra-jax": COPRAJax,
    "lbfgs": LBFGS,
    "lbfgs-hand": LBFGSHand,
    "lbfgs-ad": LBFGSAD,
    "warm-lbfgs": WarmLBFGS,
    "lbfgs-optx": OptxLBFGS,
    "lm": LM,
    "lm-optx": OptxLM,
    "cma-es": CMAES,
}


def algorithm_params(name: str) -> set[str]:
    """Return the keyword names a registered algorithm's constructor accepts.

    The solver's ``__init__`` signature *is* the single source of truth for what
    each algorithm supports: ``"copra"``/``"copra-jax"`` lack ``reg_amp`` etc.,
    so they are absent from the returned set. Callers (the pipeline dispatch and
    the GUI's option greying) use this to forward only accepted options and to
    disable controls an algorithm cannot use.
    """
    if name not in ALGORITHMS:
        raise ValueError(f"unknown algorithm {name!r}")
    cls = ALGORITHMS[name]
    params: set[str] = set()
    # A solver that wraps another (WarmLBFGS) forwards through **kwargs, so its
    # own signature does not name the inherited options. Walk the MRO until a
    # class is reached that does not forward, so wrappers advertise everything
    # they can actually accept -- the pipeline filters on this set, and a
    # wrapper that under-reports would silently receive none of its options.
    for klass in cls.__mro__:
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        sig = inspect.signature(init).parameters
        params |= set(sig)
        if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.values()):
            break
    params -= {"self", "args", "kwargs"}
    return params


def retrieve(
    trace: ArrayLike,
    omega: ArrayLike,
    delays: ArrayLike,
    interaction: str,
    *,
    algorithm: str = "warm-lbfgs",
    guess: ArrayLike | Pulse | None = None,
    weights: ArrayLike | None = None,
    rng: np.random.Generator | None = None,
    omega0: float | None = None,
    callback=None,
    progress: bool | str = "auto",
    **kwargs,
) -> RetrievalResult:
    """Retrieve a pulse from a measured trace with the chosen algorithm.

    A thin convenience wrapper around the solver classes
    (:class:`~croak.copra.COPRA`, :class:`~croak.lbfgs.LBFGS`): it constructs the
    solver from ``kwargs`` and calls its :meth:`~croak.solver.Retriever.run`.

    Parameters
    ----------
    trace : array_like
        Measured trace intensity ``(Nomega, Ndelay)``.
    omega : array_like
        Centred angular-frequency axis (rad/s).
    delays : array_like
        Delay axis (s).
    interaction : str
        ``"shg"``, ``"sd"`` or ``"pg"``.
    algorithm : str, optional
        ``"warm-lbfgs"`` (the default: ``lbfgs-ad`` warm-started from COPRA's
        local sweep), ``"copra"``, ``"copra-jax"``, ``"lbfgs"``,
        ``"lbfgs-hand"``, ``"lbfgs-ad"``, ``"lbfgs-optx"``, ``"lm"``,
        ``"lm-optx"`` or ``"cma-es"`` — the keys of :data:`ALGORITHMS`.
    guess : None, array_like or Pulse, optional
        Initial guess.
    weights : array_like, optional
        Per-frequency weights (length ``Nomega``).
    rng : numpy.random.Generator, optional
        Random generator for a random initial guess.
    omega0 : float, optional
        Carrier angular frequency (rad/s). For a **dispersive** retrieval it sets the
        carrier at which the slab dispersion is built (forwarded to the solver
        constructor) as well as labelling the result's wavelength axes; for a thin
        retrieval it only labels the axes.
    callback : callable, optional
        ``callback(iteration, R, best_R, snapshot=...)`` progress hook, where
        ``snapshot`` is an optional zero-argument in-progress-result builder (or
        ``None``) for a live full-plot preview. When supplied (e.g. by the GUI)
        the caller is assumed to render its own progress, so the built-in
        progress bar/report is suppressed.
    progress : {"auto", True, False}, optional
        Built-in progress reporting when no ``callback`` is given. ``"auto"``
        (default) shows a live bar and a final summary report (error, retrieved
        and transform-limited FWHM, iterations, time) in a terminal or notebook,
        and stays silent when output is non-interactive (piped/CI/tests).
        ``True`` forces the report on; ``False`` disables it.
    **kwargs
        Solver-specific options forwarded to the algorithm constructor (e.g.
        ``maxiters``, ``alpha`` for COPRA; ``material``, ``thickness``,
        ``npoints`` for dispersive propagation — paired with ``omega0`` above).

    Returns
    -------
    RetrievalResult
    """
    try:
        cls = ALGORITHMS[algorithm.lower()]
    except AttributeError, KeyError:
        raise ValueError(
            f"unknown algorithm {algorithm!r}; available: "
            f"{', '.join(sorted(ALGORITHMS))}"
        ) from None
    # Dispersive solvers build their dispersion model from ``omega0`` in the
    # *constructor*; run()'s ``omega0`` only records the carrier on the result. So an
    # explicit ``omega0`` must reach the constructor too, or dispersive retrieval would
    # silently use omega0=0 (the wrong carrier). Solvers without an ``omega0`` argument
    # are left untouched.
    if omega0 is not None and "omega0" in algorithm_params(algorithm.lower()):
        kwargs = {**kwargs, "omega0": omega0}
    solver = cls(**kwargs)

    # Install the built-in terminal/notebook progress reporter only when the
    # caller has not supplied its own callback (the GUI does, and renders its
    # own progress + summary).
    reporter = None
    run_callback = callback
    if callback is None:
        reporter = make_reporter(progress, total=getattr(solver, "maxiters", None))
        if reporter is not None:
            run_callback = reporter.callback

    t0 = time.perf_counter()
    result = solver.run(
        trace,
        omega,
        delays,
        interaction,
        guess=guess,
        weights=weights,
        rng=rng,
        omega0=omega0,
        callback=run_callback,
    )
    if reporter is not None:
        reporter.finish(result, time.perf_counter() - t0)
    return result
