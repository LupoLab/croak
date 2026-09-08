"""The delay origin of a model trace: gate--probe coincidence, or the marginal peak.

A measured trace does not know where its delay zero is. The preprocessing
(:func:`croak.preprocess.regrid`) therefore centres the *data* on their
delay-marginal peak, to sub-sample precision. The forward model, on the other
hand, places delay zero at gate--probe coincidence. For a delay-symmetric trace
the two conventions agree; for the traces of a tightly apertured single-cycle
instrument they do not -- coherent collection makes the trace asymmetric and
puts its marginal peak tens of attoseconds from coincidence, growing with the
chirp accumulated in the medium -- and a retrieval with the delay zero held at
coincidence can pay for the misalignment only by broadening the pulse (FROG
N101/N102: the whole of the "+10 % single-cycle over-report" at the production
hole).

Two remedies exist. Fitting a delay offset ``tau0`` (``fit_tau0``) works for
transform-limited pulses but is not benign for chirped ones: the chirp itself
displaces the marginal peak, ``tau0`` then wanders by hundreds of attoseconds
and trades against the retrieved chirp. This module implements the other:
**centre the model trace on its own marginal peak, by the same sub-sample
parabolic rule the preprocessing applies to the data**, at every evaluation.
Both sides then share one convention, for any pulse, with no free parameter.

The shift is applied to the computed trace by an exact Fourier translation
along the (uniform) delay axis, so it costs two FFTs per evaluation and is
differentiable in both the trace values and the estimated peak position; the
peak *index* is found with ``argmax`` (no gradient, correctly -- the vertex of
the parabola through the three samples carries it).
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
    """Sub-sample delay of the marginal's peak (JAX port of ``marginal_peak_delay``).

    Parabola through the peak sample and its two neighbours, vertex clamped to the
    bracketing samples; the peak sample is used unchanged at either end of the axis
    or when the three points do not curve downward. The ``argmax`` carries no
    gradient; the vertex position does.
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
    r"""Shift ``trace`` so its delay-marginal peak sits at zero.

    Exact Fourier translation along the delay axis, which must be uniform (the
    retrieval grid is). The shift is periodic on the axis, so a trace that has
    decayed to zero at both ends -- the usual case -- is translated without artefact.

    ``row_weights`` (``(Nomega,)``) weights the rows before the marginal is formed.
    The data's marginal is taken *after* the generation-response scaling and the
    per-frequency factors, so for the two conventions to coincide the model's
    marginal must carry the same row weighting: pass the per-row least-squares
    factors :math:`\\mu_\\omega` that map the model rows onto the data rows (see
    :func:`recentring`). Without it, an asymmetric trace's marginal peak depends on
    how the rows are weighted and the two sides drift apart by up to tens of
    attoseconds when the response exponent is far from the model's own.
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
        :math:`\\mu_\\omega = \\langle T_{meas}, T\\rangle_\\omega /
        \\langle T, T\\rangle_\\omega`
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
