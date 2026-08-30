r"""Trace-error metrics for nonlinear-process pulse retrieval (FROG, TDP, …).

All retrieval algorithms compare a
measured trace ``T_meas`` against a simulated trace ``T_sim`` (both intensity)
through three quantities:

* the optimal scale factor :math:`\mu` minimising the weighted residual,
* the residual sum of squares :math:`r`,
* the normalised, dimensionless trace error :math:`R` (the standard "FROG
  error", but the same metric for any PNPS trace).

Traces may be in either orientation for the global metrics (which sum over all
elements); :func:`compute_mu_per_freq` is the exception and takes an explicit
frequency axis.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "compute_mu",
    "compute_mu_per_freq",
    "compute_r",
    "compute_R",
    "trace_error",
    "frog_error",
]


def compute_mu(t_meas: ArrayLike, t_sim: ArrayLike, weights: ArrayLike) -> float:
    r"""Optimal global scale factor :math:`\mu`.

    Minimises :math:`\sum (T_\mathrm{meas}-\mu T_\mathrm{sim})^2 w^2`, giving
    :math:`\mu = \sum(T_\mathrm{meas}T_\mathrm{sim}w^2)/\sum(T_\mathrm{sim}^2 w^2)`.
    """
    t_meas = np.asarray(t_meas, dtype=float)
    t_sim = np.asarray(t_sim, dtype=float)
    w2 = np.asarray(weights, dtype=float) ** 2
    return float(np.sum(t_meas * t_sim * w2) / np.sum(t_sim * t_sim * w2))


def compute_mu_per_freq(
    t_meas: ArrayLike,
    t_sim: ArrayLike,
    weights: ArrayLike,
    *,
    freq_axis: int = 0,
) -> NDArray[np.float64]:
    """Per-frequency scale factors (for the ``Rω`` adaptive-scaling mode).

    Each factor is the per-row least-squares optimum
    ``sum(t_sim * t_meas) / sum(t_sim**2)``. The weights multiply numerator and
    denominator identically and therefore *cancel*: the factors are
    weight-independent. Their only effect is that a zero-weight frequency slice
    has a vanishing denominator and falls back to ``1.0``, so rows excluded
    from the fit are left unscaled rather than scaled by a meaningless ratio.

    Parameters
    ----------
    t_meas, t_sim : array_like
        Measured and simulated traces (2-D).
    weights : array_like
        Per-frequency weight vector. Only its zero pattern matters (see above).
    freq_axis : int, optional
        Axis corresponding to frequency (default ``0``).

    Returns
    -------
    numpy.ndarray
        One scale factor per frequency slice (``1.0`` where the denominator
        vanishes).
    """
    t_meas = np.asarray(t_meas, dtype=float)
    t_sim = np.asarray(t_sim, dtype=float)
    w = np.asarray(weights, dtype=float)
    other = 1 - freq_axis
    # w2 cancels between num and den; it is kept solely so that zero-weight
    # rows produce den == 0 and take the 1.0 fallback below.
    w2 = w**2
    num = np.sum(t_sim * t_meas, axis=other) * w2
    den = np.sum(t_sim * t_sim, axis=other) * w2
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 1.0)


def compute_r(
    t_meas: ArrayLike,
    t_sim: ArrayLike,
    weights: ArrayLike,
    mu: float | NDArray[np.float64],
) -> float:
    r"""Weighted residual sum of squares.

    :math:`r = \sum[(T_\mathrm{meas}-\mu T_\mathrm{sim})w]^2`.

    ``mu`` is the optimal scale: a scalar (global) or a per-frequency column
    vector broadcast against the trace (the ``Rω`` mode).
    """
    t_meas = np.asarray(t_meas, dtype=float)
    t_sim = np.asarray(t_sim, dtype=float)
    w = np.asarray(weights, dtype=float)
    d = (t_meas - mu * t_sim) * w
    return float(np.sum(d * d))


def compute_R(r: float, t_meas: ArrayLike, weights: ArrayLike) -> float:
    r"""Normalised FROG error :math:`R = \sqrt{r/(MN\,\max(T_\mathrm{meas}w)^2)}`.

    Dimensionless and comparable across trace sizes and intensity scales;
    typically ``0.001``–``0.05`` for a good retrieval.
    """
    t_meas = np.asarray(t_meas, dtype=float)
    w = np.asarray(weights, dtype=float)
    size = t_meas.size
    maxval = np.max(t_meas * w)
    return float(np.sqrt(r / (size * maxval**2)))


def frog_error(
    t_meas: ArrayLike, t_sim: ArrayLike, weights: ArrayLike | None = None
) -> float:
    """Return the FROG error ``R`` via :func:`compute_mu` / ``r`` / ``R``.

    Parameters
    ----------
    t_meas, t_sim : array_like
        Measured and simulated traces.
    weights : array_like, optional
        Weight matrix (same shape as the traces) or per-frequency vector
        (broadcast along axis 0). Defaults to uniform weights.
    """
    t_meas = np.asarray(t_meas, dtype=float)
    t_sim = np.asarray(t_sim, dtype=float)
    if weights is None:
        w = np.ones_like(t_meas)
    else:
        w = np.asarray(weights, dtype=float)
        if w.ndim == 1:
            w = np.broadcast_to(w[:, None], t_meas.shape)
    mu = compute_mu(t_meas, t_sim, w)
    r = compute_r(t_meas, t_sim, w, mu)
    return compute_R(r, t_meas, w)


#: General name for the normalised trace error (the "FROG error" generalised to
#: any nonlinear-process trace). Alias of :func:`frog_error`.
trace_error = frog_error
