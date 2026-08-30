r"""JAX/autodiff twin of :mod:`croak.metrics` plus the solver objective pieces.

These are JAX clones of the trace-error metrics in :mod:`croak.metrics`, written
so that the FROG error (and the least-squares residual vector) is a
differentiable function of the pulse parameters. They back
:class:`~croak.lbfgs_ad.LBFGSAD` (which minimises the scalar :func:`frog_error`)
and :class:`~croak.lm.LM` (which drives :func:`residual_vector` to zero), and are
checked against the numpy metrics in the tests.

Conventions match :mod:`croak.metrics` and :mod:`croak.lbfgs`:

* ``t_meas`` / ``t_sim`` are ``(Nomega, Ndelay)`` intensity traces;
* ``weights`` is the per-frequency weight **vector** of length ``Nomega``
  (broadcast along the delay axis), as passed by the solvers;
* the scalar error is :math:`R=\sqrt{\sum\mathrm{resid}^2/\mathrm{denom}}` with
  :math:`\mathrm{denom}=MN\,\max(T_\mathrm{meas}w)^2` — identical to
  :func:`croak.metrics.compute_R` and to the value minimised in
  :class:`croak.lbfgs.LBFGS`.

The second-difference smoothness penalties mirror
:func:`croak.pulses._second_diff_penalty` so regularised retrieval matches the
analytic-gradient solver.
"""

from __future__ import annotations

import jax.numpy as jnp

__all__ = [
    "mu_global",
    "mu_per_freq",
    "frog_error",
    "residual_vector",
    "amplitude_penalty",
    "phase_penalty",
    "spectral_penalty",
]


def mu_global(
    t_meas: jnp.ndarray, t_sim: jnp.ndarray, weights: jnp.ndarray
) -> jnp.ndarray:
    r"""Optimal global scale factor :math:`\mu` (JAX clone of :func:`compute_mu`).

    Mirrors :func:`croak.metrics.compute_mu`. ``weights`` is a per-frequency
    vector broadcast along the delay axis.
    """
    w2 = weights[:, None] ** 2
    return jnp.sum(t_meas * t_sim * w2) / jnp.sum(t_sim * t_sim * w2)


def mu_per_freq(
    t_meas: jnp.ndarray, t_sim: jnp.ndarray, weights: jnp.ndarray
) -> jnp.ndarray:
    """Per-frequency scale factors (JAX clone of :func:`compute_mu_per_freq`).

    Mirrors :func:`croak.metrics.compute_mu_per_freq` with ``freq_axis=0``
    (sum over delays). As there, the weights cancel between numerator and
    denominator — the factors are weight-independent, and a zero-weight row
    falls back to ``1.0``. Returns one factor per frequency row.
    """
    # w2 cancels between num and den; kept so zero-weight rows hit the fallback.
    w2 = weights**2
    num = jnp.sum(t_sim * t_meas, axis=1) * w2
    den = jnp.sum(t_sim * t_sim, axis=1) * w2
    return jnp.where(den > 0, num / jnp.where(den > 0, den, 1.0), 1.0)


def _denom(t_meas: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """Normalising denominator ``MN * max(T_meas * w)**2`` (see :func:`compute_R`)."""
    return t_meas.size * jnp.max(t_meas * weights[:, None]) ** 2


def _mu(
    t_meas: jnp.ndarray, t_sim: jnp.ndarray, weights: jnp.ndarray, *, R_omega: bool
) -> jnp.ndarray:
    """Scale factor broadcast to ``(Nomega, Ndelay)`` (scalar or per-frequency)."""
    if R_omega:
        return mu_per_freq(t_meas, t_sim, weights)[:, None]
    return mu_global(t_meas, t_sim, weights)


def frog_error(
    t_meas: jnp.ndarray,
    t_sim: jnp.ndarray,
    weights: jnp.ndarray,
    *,
    R_omega: bool = False,
) -> jnp.ndarray:
    r"""Normalised FROG error :math:`R` minimised by the gradient solvers.

    Identical to the value computed in :class:`croak.lbfgs.LBFGS` (and to
    :func:`croak.metrics.frog_error`): the optimal scale ``mu`` is applied, the
    weighted residual is formed, and :math:`R=\sqrt{\sum\mathrm{resid}^2/
    \mathrm{denom}}` is returned.

    Parameters
    ----------
    R_omega : bool, optional
        Use per-frequency adaptive scaling factors instead of a single global
        ``mu``.
    """
    mu = _mu(t_meas, t_sim, weights, R_omega=R_omega)
    resid = (t_meas - mu * t_sim) * weights[:, None]
    return jnp.sqrt(jnp.sum(resid**2) / _denom(t_meas, weights))


def residual_vector(
    t_meas: jnp.ndarray,
    t_sim: jnp.ndarray,
    weights: jnp.ndarray,
    *,
    R_omega: bool = False,
) -> jnp.ndarray:
    r"""Flattened least-squares residual for :class:`~croak.lm.LM`.

    Returns :math:`(T_\mathrm{meas}-\mu T_\mathrm{sim})\,w/\sqrt{\mathrm{denom}}`
    flattened to 1-D, so that its Euclidean norm equals the FROG error
    :func:`frog_error`. Levenberg–Marquardt drives this vector to zero.
    """
    mu = _mu(t_meas, t_sim, weights, R_omega=R_omega)
    resid = (t_meas - mu * t_sim) * weights[:, None]
    return (resid / jnp.sqrt(_denom(t_meas, weights))).ravel()


# ---------------------------------------------------------------------------
# Second-difference smoothness penalties (JAX twin of pulses._second_diff_penalty)
# ---------------------------------------------------------------------------
def _second_diff(a: jnp.ndarray, *, normalize_by_max: bool) -> jnp.ndarray:
    """``sum((a[i+1]-2a[i]+a[i-1])**2)`` with the same normalisation as numpy."""
    n = a.size
    if n < 3:
        return jnp.asarray(0.0)
    d = a[2:] - 2.0 * a[1:-1] + a[:-2]
    s = jnp.sum(d * d)
    if normalize_by_max:
        eps = jnp.finfo(jnp.float64).eps
        return s / ((n - 2) * jnp.maximum(jnp.max(a) ** 2, eps))
    return s / (n - 2)


def amplitude_penalty(amp_params: jnp.ndarray) -> jnp.ndarray:
    """Amplitude smoothness penalty (max-normalised), as in :class:`ArrayPulse`."""
    return _second_diff(amp_params, normalize_by_max=True)


def phase_penalty(phase_params: jnp.ndarray) -> jnp.ndarray:
    """Phase smoothness penalty (mean squared curvature), as in :class:`ArrayPulse`."""
    return _second_diff(phase_params, normalize_by_max=False)


def spectral_penalty(amp_params: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    r"""Relative spectral-amplitude mismatch ``‖a - target‖² / ‖target‖²``.

    JAX twin of :meth:`croak.pulses.ArrayPulse.spectral_penalty`. ``target`` is
    the target spectral amplitude over the optimised support (same length as
    ``amp_params``).
    """
    d = amp_params - target
    eps = jnp.finfo(jnp.float64).eps
    return jnp.sum(d * d) / jnp.maximum(jnp.sum(target * target), eps)
