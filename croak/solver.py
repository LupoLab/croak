"""Shared retrieval-solver base class and helpers.

Defines :class:`Retriever`, the common base for every solver registered in
:data:`croak.retrieve.ALGORITHMS`. It standardises the
``run`` entry point — trace normalisation, grid construction, initial-guess
resolution and weight handling — so every algorithm shares an identical calling
convention and return type (:class:`~croak.result.RetrievalResult`).
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .grid import Grid
from .interactions import get_interaction
from .metrics import compute_mu, compute_mu_per_freq
from .pulses import Pulse, random_gaussian_spectrum
from .result import RetrievalResult

__all__ = [
    "Retriever",
    "resolve_guess",
    "build_weights",
    "spectral_target_amplitude",
    "assemble_result",
    "adapt_callback",
]


def resolve_guess(
    grid: Grid,
    guess: ArrayLike | Pulse | None,
    *,
    rng: np.random.Generator | None = None,
) -> NDArray[np.complex128]:
    """Turn a flexible ``guess`` argument into a complex spectrum.

    Parameters
    ----------
    grid : Grid
        Time/frequency grid.
    guess : None, array_like or Pulse
        ``None`` draws a random Gaussian-spectrum initial guess; an array is
        used directly as the spectrum; a :class:`~croak.pulses.Pulse` contributes
        its spectrum.
    rng : numpy.random.Generator, optional
        Random generator for the ``None`` case.

    Returns
    -------
    numpy.ndarray
        A complex spectrum of length ``grid.n`` (centred order).
    """
    if guess is None:
        ew = random_gaussian_spectrum(grid, 2.5e-15, rng=rng)
    elif isinstance(guess, Pulse):
        ew = guess.spectrum().astype(complex)
    else:
        ew = np.asarray(guess, dtype=complex)
        if ew.shape != (grid.n,):
            raise ValueError(
                f"guess spectrum must have shape ({grid.n},), got {ew.shape}"
            )
        ew = ew.copy()
    # Normalise to O(1) peak amplitude. Retrieval is scale-invariant (the metric
    # factor mu absorbs the overall scale), and an O(1) amplitude keeps the
    # gradient-based solvers well conditioned.
    peak = np.max(np.abs(ew))
    return ew / peak if peak > 0 else ew


def build_weights(n: int, weights: ArrayLike | None) -> NDArray[np.float64]:
    """Return a length-``n`` per-frequency weight vector (uniform if ``None``)."""
    if weights is None:
        return np.ones(n)
    w = np.asarray(weights, dtype=float)
    if w.shape != (n,):
        raise ValueError(f"weights must have shape ({n},), got {w.shape}")
    return w


def spectral_target_amplitude(
    intensity: ArrayLike | None,
) -> NDArray[np.float64] | None:
    r"""Peak-normalised spectral amplitude from a measured spectral intensity.

    Shared by the gradient/LM solvers' spectral-match regularisation: converts a
    measured spectral **intensity** :math:`I(\omega)` (centred order, on the
    retrieval grid) to the peak-normalised amplitude :math:`\sqrt{I}` compared
    against the retrieved :math:`|\tilde E(\omega)|`. Returns ``None`` when no
    spectrum is supplied.
    """
    if intensity is None:
        return None
    s = np.sqrt(np.clip(np.asarray(intensity, dtype=float), 0.0, None))
    peak = float(np.max(s)) if s.size else 0.0
    return s / peak if peak > 0 else s


def adapt_callback(callback):
    """Return ``callback`` wrapped so it always accepts a ``snapshot=`` keyword.

    The documented progress signature is
    ``callback(iteration, R, best_R, snapshot=None)``, but a three-argument
    callback used to work with the projection solvers because they had no
    in-progress result to offer and never passed the keyword. Now that every
    solver can supply one, calling such a callback would fail with a
    ``TypeError`` deep inside the iteration.

    Rather than make the shorter form an error, drop the snapshot for callbacks
    that cannot receive it. That also removes a pre-existing trap in the other
    direction: the same three-argument callback already failed with
    ``lbfgs-ad``, so which forms were legal depended on the solver.

    Parameters
    ----------
    callback : callable or None
        A user progress callback, or ``None``.

    Returns
    -------
    callable or None
        A callable accepting ``snapshot=``, or ``None`` if none was given.
    """
    if callback is None:
        return None
    try:
        inspect.signature(callback).bind(0, 0.0, 0.0, snapshot=None)
    except TypeError:
        # Cannot take the snapshot: call it with the three positional arguments.
        def adapted(iteration, R, best_R, snapshot=None) -> None:
            callback(iteration, R, best_R)

        return adapted
    except ValueError:
        # No introspectable signature (some builtins/C callables): assume the
        # documented contract rather than silently dropping the snapshot.
        return callback
    return callback


def assemble_result(
    *,
    spectrum: NDArray[np.complex128],
    t_sim: NDArray[np.float64],
    grid: Grid,
    t_meas: NDArray[np.float64],
    delays: NDArray[np.float64],
    weights: NDArray[np.float64],
    interaction: str,
    algorithm: str,
    errors: Sequence[float],
    R_omega: bool = False,
    thickness: float | None = None,
    tau0: float = 0.0,
    smear_scale: float | None = None,
    smear_scale_delta: float | None = None,
    omega0: float = 0.0,
) -> RetrievalResult:
    r"""Assemble a :class:`~croak.result.RetrievalResult` from a candidate solution.

    Every retriever, at convergence, takes its retrieved ``spectrum`` and the
    corresponding **unscaled** simulated trace ``t_sim`` and computes the optimal
    intensity scale :math:`\mu`, the scaled trace and the FROG error :math:`R`
    identically. This helper is that shared assembly, so the final result and any
    in-progress *snapshot* (for the GUI's live preview) are built the same way —
    a single source of truth for the ``mu``/error maths.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Retrieved complex spectrum :math:`\tilde E(\omega)` (centred order).
    t_sim : numpy.ndarray
        Unscaled simulated trace :math:`T_\mathrm{sim}`, shape ``(Nomega, Ndelay)``
        (i.e. before the optimal :math:`\mu` scaling is applied).
    grid : Grid
        Retrieval time/frequency grid.
    t_meas : numpy.ndarray
        Normalised measured trace, same shape as ``t_sim``.
    delays : numpy.ndarray
        Delay axis (s).
    weights : numpy.ndarray
        Per-frequency weights (length ``Nomega``).
    interaction : str
        ``"shg"``/``"sd"``/``"pg"`` (normalised via
        :func:`~croak.interactions.get_interaction`).
    algorithm : str
        Algorithm name recorded on the result.
    errors : sequence of float
        FROG error per iteration so far (copied onto the result).
    R_omega : bool, optional
        Per-frequency adaptive scaling (a :math:`\mu` vector) instead of a single
        scalar.
    thickness : float or None, optional
        Fitted slab thickness (m), or ``None`` when not fitted.
    tau0 : float, optional
        Fitted delay-zero offset (s); ``0.0`` when not fitted.
    smear_scale : float or None, optional
        Fitted geometric-smearing multiplier, or ``None`` when not fitted. For
        a split fit this is the gate-shape (``p``) channel.
    smear_scale_delta : float or None, optional
        Fitted delay-channel multiplier for a split smearing fit
        (``fit_smearing_split``), or ``None`` for a joint fit.
    omega0 : float, optional
        Carrier angular frequency (rad/s) recorded for the wavelength axes.

    Returns
    -------
    RetrievalResult
    """
    # Optimal intensity scale mu minimising ||T_meas - mu*T_sim||_w; a scalar
    # (global) or one value per frequency (R_omega). The FROG error R is the
    # weighted RMS residual normalised by MN*max(T_meas*w)^2 (== metrics denom).
    wmat = np.broadcast_to(weights[:, None], t_sim.shape)
    denom = t_meas.size * np.max(t_meas * weights[:, None]) ** 2
    mu = (
        compute_mu_per_freq(t_meas, t_sim, weights, freq_axis=0)
        if R_omega
        else compute_mu(t_meas, t_sim, wmat)
    )
    mu_b = mu[:, None] if isinstance(mu, np.ndarray) else mu
    sim_trace = mu_b * t_sim
    resid = (t_meas - mu_b * t_sim) * weights[:, None]
    error = float(np.sqrt(np.sum(resid**2) / denom))
    return RetrievalResult(
        spectrum=spectrum,
        grid=grid,
        delays=delays,
        interaction=get_interaction(interaction).name,
        algorithm=algorithm,
        error=error,
        errors=list(errors),
        trace=sim_trace,
        mu=mu,
        thickness=thickness,
        tau0=tau0,
        smear_scale=smear_scale,
        smear_scale_delta=smear_scale_delta,
        omega0=omega0,
    )


#: Large finite objective value returned in place of a non-finite one: NLopt's
#: line search reads it as a very bad step and backtracks, where a bare NaN
#: would abort the whole optimization with an opaque ``nlopt.runtime_error``.
NONFINITE_PENALTY = 1e10


def guard_nonfinite(
    total: float,
    grad_values: NDArray[np.float64] | None,
    grad_out: NDArray[np.float64] | None,
    state: dict,
) -> float | None:
    """Trap a non-finite objective/gradient during an NLopt line search.

    A quasi-Newton line search occasionally probes an iterate where the jitted
    objective or its gradient evaluates to NaN/Inf (observed for roughly one in
    six random starts on large dispersive grids). NLopt aborts on such values;
    returning :data:`NONFINITE_PENALTY` with a zeroed gradient instead makes
    the line search backtrack and the optimization continue.

    Parameters
    ----------
    total : float
        The objective value just computed.
    grad_values : numpy.ndarray or None
        The gradient values to check (``None`` skips the gradient check).
    grad_out : numpy.ndarray or None
        NLopt's output gradient buffer, zeroed when the guard trips.
    state : dict
        Mutable per-solve state; the guard warns once per solve (key
        ``"warned"``).

    Returns
    -------
    float or None
        :data:`NONFINITE_PENALTY` when the guard trips, else ``None`` (the
        caller proceeds normally).
    """
    finite = bool(np.isfinite(total)) and (
        grad_values is None or bool(np.isfinite(grad_values).all())
    )
    if finite:
        return None
    if not state.get("warned"):
        warnings.warn(
            "objective or gradient became non-finite during the line search; "
            "returning a large finite penalty so the optimizer backtracks",
            RuntimeWarning,
            stacklevel=2,
        )
        state["warned"] = True
    if grad_out is not None and grad_out.size > 0:
        grad_out[:] = 0.0
    return NONFINITE_PENALTY


def split_smear_value(smear) -> tuple[float, float | None]:
    """Normalise a fitted smearing value to ``(scale_p, scale_delta or None)``.

    The augmented unpacking (:func:`croak._jax_pulse.augment`) yields a scalar
    for a joint smearing fit and a pair (length-2 array or tuple) for a split
    fit; retrievers use this to map either form onto the ``smear_scale`` /
    ``smear_scale_delta`` result fields.
    """
    arr = np.asarray(smear, dtype=float)
    if arr.ndim == 0:
        return float(arr), None
    return float(arr[0]), float(arr[1])


class Retriever:
    """Base class for retrieval algorithms.

    Subclasses implement :meth:`_solve`. Algorithm and forward-model
    hyperparameters are set on the constructor; :meth:`run` takes the
    measurement and returns a :class:`~croak.result.RetrievalResult`.
    """

    #: Algorithm name recorded in the result.
    name: str = "retriever"

    def __init__(self, *, maxiters: int = 100, verbose: bool = False):
        """Store hyperparameters shared by all retrievers (iterations, verbosity)."""
        self.maxiters = int(maxiters)
        self.verbose = bool(verbose)

    def run(
        self,
        trace: ArrayLike,
        omega: ArrayLike,
        delays: ArrayLike,
        interaction: str,
        *,
        guess: ArrayLike | Pulse | None = None,
        weights: ArrayLike | None = None,
        rng: np.random.Generator | None = None,
        omega0: float | None = None,
        callback=None,
    ) -> RetrievalResult:
        """Retrieve a pulse from a measured FROG trace.

        Parameters
        ----------
        trace : array_like
            Measured trace intensity, shape ``(Nomega, Ndelay)``. Copied and
            normalised internally.
        omega : array_like
            Centred angular-frequency axis (rad/s).
        delays : array_like
            Delay axis (s).
        interaction : str
            ``"shg"``, ``"sd"`` or ``"pg"``.
        guess : None, array_like or Pulse, optional
            Initial guess (see :func:`resolve_guess`).
        weights : array_like, optional
            Per-frequency weights (length ``Nomega``); uniform if omitted.
        rng : numpy.random.Generator, optional
            Random generator for a random initial guess.
        omega0 : float, optional
            Carrier angular frequency (rad/s) recorded on the result for
            wavelength axes; defaults to the solver's ``omega0`` (dispersive
            solvers) or ``0``.
        callback : callable, optional
            Called as ``callback(iteration, R, best_R, snapshot=...)`` after each
            iteration (or objective evaluation), e.g. for live GUI progress.
            ``snapshot`` is an optional zero-argument builder of the current
            in-progress :class:`~croak.result.RetrievalResult` (for a live
            full-plot preview); solvers that cannot cheaply build one pass
            ``None``. A callback may ignore it.

        Returns
        -------
        RetrievalResult
        """
        trace = np.asarray(trace, dtype=float)
        omega = np.asarray(omega, dtype=float)
        delays = np.asarray(delays, dtype=float)
        if trace.shape != (omega.size, delays.size):
            raise ValueError(
                f"trace shape {trace.shape} does not match "
                f"(Nomega, Ndelay) = ({omega.size}, {delays.size})"
            )
        rng = np.random.default_rng() if rng is None else rng
        grid = Grid.from_omega(omega)
        t_meas = trace / trace.max()
        ew0 = resolve_guess(grid, guess, rng=rng)
        w = build_weights(grid.n, weights)
        result = self._solve(
            grid, t_meas, delays, interaction, ew0, w, rng, adapt_callback(callback)
        )
        result.omega0 = (
            getattr(self, "omega0", 0.0) if omega0 is None else float(omega0)
        )
        return result

    def _solve(
        self,
        grid: Grid,
        trace: NDArray[np.float64],
        delays: NDArray[np.float64],
        interaction: str,
        ew0: NDArray[np.complex128],
        weights: NDArray[np.float64],
        rng: np.random.Generator,
        callback=None,
    ) -> RetrievalResult:
        raise NotImplementedError


#: Default regularisation weights ``(reg_spectrum, reg_amp)`` per solver family.
#:
#: The two families do not minimise the same objective, so the same number does
#: not mean the same thing to them. The gradient solvers minimise
#: :math:`R + \lambda P`; the Levenberg-Marquardt solvers append the penalty as
#: residual rows, hence minimise :math:`R^2 + \lambda' P`. Matching the
#: stationary conditions pairs them as :math:`\lambda' = 2 R \lambda`, i.e. at
#: the :math:`R \approx 2\times10^{-4}` of a good dispersive TG-FROG retrieval
#: the LM weight is some three orders of magnitude smaller.
#:
#: That is not a cosmetic difference. On the 2 fs, 9.5 um dispersive trace
#: (2026-08-25), sweeping the LM weight over four decades:
#:
#: =====================  ========  ========  =========  ======
#: LM (spectrum, amp)     R (%)     eps       dFWHM (%)  rough.
#: =====================  ========  ========  =========  ======
#: 1e-2, 3e-2 (as LBFGS)  0.0237    0.305     +1.11      1.00
#: 1e-5, 3e-5 (mapped)    0.0193    0.112     -0.11      1.01
#: 0, 0                   0.0174    0.466     +3.12      2.57
#: =====================  ========  ========  =========  ======
#:
#: Both failure modes are visible: the gradient weights over-regularise LM (a
#: worse pulse *and* a worse trace error), and no penalty at all overfits (the
#: best trace error of any arm, the worst pulse -- 2.6x the truth's spectral
#: roughness and +3.1% on duration). The measured optimum, 1e-5, sits within
#: 30% of the 2 R lambda prediction, which is the evidence that the mapping is
#: the right rule rather than a coincidence of one trace.
#:
#: So the LM entry scales with the trace error you actually reach: multiply it
#: by :math:`R/2\times10^{-4}` on a noisier measurement.
REG_DEFAULTS: dict[str, tuple[float, float]] = {
    "gradient": (1e-2, 3e-2),
    "lm": (1e-5, 3e-5),
}

#: Solvers whose penalty enters as residual rows (the ``R^2`` objective).
_LM_SOLVERS = frozenset({"lm", "lm-optx"})


def reg_family(solver: str) -> str:
    """Which objective ``solver`` minimises: ``"gradient"`` (R) or ``"lm"`` (R^2).

    Regularisation weights are only comparable within a family.
    """
    return "lm" if str(solver).lower() in _LM_SOLVERS else "gradient"


def reg_defaults(solver: str) -> tuple[float, float]:
    """``(reg_spectrum, reg_amp)`` suited to ``solver``; see :data:`REG_DEFAULTS`."""
    return REG_DEFAULTS[reg_family(solver)]


def resolve_reg(
    reg_spectrum: float | None, reg_amp: float | None, family: str = "gradient"
) -> tuple[float, float]:
    """Resolve ``(reg_spectrum, reg_amp)``, filling ``None`` with the defaults.

    ``None`` means "use the settled production weights for this solver
    family" (:data:`REG_DEFAULTS`); an explicit number — including ``0`` to
    disable a penalty — is passed through unchanged. The spectral-match term
    additionally requires a ``spectrum_target`` at run time and is inert
    without one, so the non-zero default is safe for retrievals that have no
    independent spectrum.
    """
    spec_default, amp_default = REG_DEFAULTS[family]
    return (
        spec_default if reg_spectrum is None else float(reg_spectrum),
        amp_default if reg_amp is None else float(reg_amp),
    )
