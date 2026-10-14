"""Match a model trace to the selected delay-origin convention.

The native forward model places zero delay at gate–probe coincidence.
Preprocessing can instead centre the measured trace on its delay-marginal
peak. These origins can differ for an asymmetric trace. With
``origin="marginal_peak"``, centre the model by the same parabolic peak rule,
including the row-gain weighting when measured data are supplied. This is a
coordinate convention, distinct from fitting an unknown delay offset.

The translation is periodic on a uniform delay grid. Check that the trace
has negligible support at its ends. The parabolic peak position is
differentiated while its discrete peak index is held fixed; the map is only
piecewise differentiable when competing peaks exchange order. See
:doc:`/explanation/retrieval_theory` for gains and centring assumptions.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["DELAY_ORIGINS", "marginal_peak_delay_jax", "recentre_trace", "recentring"]

#: ``"coincidence"``: delay zero at gate--probe coincidence (the model's native
#: convention; the default). ``"marginal_peak"``: the model trace is shifted so its
#: delay-marginal peak sits at delay zero, matching the data's preprocessing.
DELAY_ORIGINS = ("coincidence", "marginal_peak")


def marginal_peak_delay_jax(delays: jnp.ndarray, marginal: jnp.ndarray) -> jnp.ndarray:
    """Estimate the marginal-peak delay with a local parabola.

    Parameters
    ----------
    delays : jax.Array
        Uniform delay samples in seconds, with at least three points.
    marginal : jax.Array
        One marginal value per delay sample.

    Returns
    -------
    jax.Array
        Scalar peak delay in seconds. The fitted vertex is clamped to the
        neighbouring samples. For an edge maximum, the current implementation
        uses the adjacent interior delay sample; avoid clipped traces. A
        non-concave interior triplet uses its central sample. The peak index
        is not differentiated; the vertex location within that branch is.
    """
    n = marginal.shape[0]
    i = jnp.argmax(marginal)
    i = jnp.clip(i, 1, n - 2)
    y0, y1, y2 = marginal[i - 1], marginal[i], marginal[i + 1]
    d = delays[i + 1] - delays[i]
    curv = y0 - 2.0 * y1 + y2
    # vertex of the parabola through (-1, y0), (0, y1), (1, y2), in units of d
    vertex = jnp.where(
        curv < 0.0, 0.5 * (y0 - y2) / jnp.where(curv < 0.0, curv, -1.0), 0.0
    )
    vertex = jnp.clip(vertex, -1.0, 1.0)
    at_edge = (jnp.argmax(marginal) == 0) | (jnp.argmax(marginal) == n - 1)
    return delays[i] + jnp.where(at_edge, 0.0, vertex * d)


def recentre_trace(
    trace: jnp.ndarray, delays: jnp.ndarray, row_weights: jnp.ndarray | None = None
) -> jnp.ndarray:
    """Shift a trace so its estimated delay-marginal peak sits at zero.

    Parameters
    ----------
    trace : jax.Array
        Trace with shape ``(Nomega, Ndelay)``.
    delays : jax.Array
        Uniform delay samples in seconds, with at least three points.
    row_weights : jax.Array or None, optional
        Length-``Nomega`` intensity gains applied only when forming the marginal.
        Pass the row gains mapping the prediction onto the measured trace when
        the data have a different spectral response; see :func:`recentring`.

    Returns
    -------
    jax.Array
        Translated trace, with its original shape. The Fourier translation is
        periodic, so check support at both delay boundaries. The supplied row
        gains set the centring position but are not applied to the returned
        intensity. This low-level function assumes a uniform delay axis.
    """
    delays = jnp.asarray(delays)
    weighted = trace if row_weights is None else trace * row_weights[:, None]
    tau_p = marginal_peak_delay_jax(delays, jnp.sum(weighted, axis=0))
    n = delays.shape[0]
    d = delays[1] - delays[0]
    f = jnp.fft.fftfreq(n, d=d)
    # T_new(tau) = T(tau + tau_p): multiply the transform by exp(+2 pi i f tau_p)
    spec = jnp.fft.fft(trace, axis=1) * jnp.exp(2j * jnp.pi * f * tau_p)[None, :]
    return jnp.real(jnp.fft.ifft(spec, axis=1))


def recentring(
    trace_fn,
    delays: ArrayLike,
    origin: str = "coincidence",
    t_meas: ArrayLike | None = None,
):
    r"""Wrap a trace function so its output honours ``origin``.

    Parameters
    ----------
    trace_fn : callable
        ``trace_fn(*args) -> (Nomega, Ndelay)`` array (e.g. the ``trace_param`` of
        :func:`croak.forward_jax.make_param_trace_fn`).
    delays : array_like
        The (uniform) delay axis the trace is evaluated on (s).
    origin : {"coincidence", "marginal_peak"}
        ``"coincidence"`` returns ``trace_fn`` unchanged.
    t_meas : array_like, optional
        The measured trace on the same grid. When given, the model's marginal is
        formed with the per-row least-squares factors
        :math:`\mu_\omega = \langle T_{meas}, T\rangle_\omega /
        \langle T, T\rangle_\omega`
        that map the model rows onto the data rows, so the two marginals carry
        the same row weighting whatever generation-response exponent the data
        were corrected with. Recommended; the factors are recomputed at every
        evaluation from the current model, and differentiated through.

    Returns
    -------
    callable
    """
    if origin not in DELAY_ORIGINS:
        raise ValueError(f"delay_origin must be one of {DELAY_ORIGINS}, got {origin!r}")
    if origin == "coincidence":
        return trace_fn
    delays_j = jnp.asarray(np.asarray(delays, dtype=float))
    if delays_j.shape[0] > 2:
        steps = np.diff(np.asarray(delays, dtype=float))
        if not np.allclose(steps, steps[0], rtol=1e-6, atol=0.0):
            raise ValueError(
                "delay_origin='marginal_peak' needs a uniform delay axis (the "
                "retrieval grid is; pass the regridded delays)"
            )

    tm = None if t_meas is None else jnp.asarray(np.asarray(t_meas, dtype=float))

    def wrapped(*args, **kwargs):
        trace = trace_fn(*args, **kwargs)
        weights = None
        if tm is not None:
            num = jnp.sum(tm * trace, axis=1)
            den = jnp.sum(trace * trace, axis=1)
            weights = jnp.where(den > 0.0, num / jnp.where(den > 0.0, den, 1.0), 0.0)
            # The factors depend (smoothly) on the trace and the gradient must
            # flow through them: cutting it (stop_gradient) makes the objective
            # and its gradient inconsistent, and the L-BFGS line search then
            # fails outright near convergence (NLopt "runtime_error", FROG r38)
            # -- an accidental early stop, not a converged retrieval.
        return recentre_trace(trace, delays_j, weights)

    return wrapped


def marginal_peak_delay_np(delays: ArrayLike, trace: ArrayLike) -> float:
    """NumPy convenience: the model trace's marginal-peak delay (s)."""
    return float(
        marginal_peak_delay_jax(
            jnp.asarray(np.asarray(delays, dtype=float)),
            jnp.asarray(np.sum(np.asarray(trace, dtype=float), axis=0)),
        )
    )


def _check_grad_flows() -> NDArray[np.float64]:  # pragma: no cover - dev aid
    delays = jnp.linspace(-5e-15, 5e-15, 41)
    trace = jnp.exp(-(((delays - 0.3e-15) / 1e-15) ** 2))[None, :] * jnp.ones((3, 1))
    g = jax.grad(lambda t: jnp.sum(recentre_trace(t, delays)[:, 20]))(trace)
    return np.asarray(g)
