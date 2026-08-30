r"""JAX-accelerated COPRA (``copra-jax``).

A drop-in twin of :class:`croak.copra.COPRA` that runs the per-iteration work on
the JAX forward/adjoint kernel (:func:`croak.forward_jax.make_forward_adjoint_fns`)
instead of the per-delay NumPy loops. The algorithm is byte-for-byte the same as
:func:`croak.copra.run_copra` — same two modes, same step sizes, same
local→global switch and convergence tests — only the two stage bodies are
vectorised:

* the **global** step does one ``vmap`` forward (keeping the ``(test, gate)``
  tape) and one ``vmap`` adjoint over all delays inside a single ``jax.jit``;
* the **local** sweep, which is inherently sequential (each delay updates
  :math:`\tilde E` for the next), is expressed as a ``jax.lax.scan`` over the
  delay-index permutation. The permutation is drawn host-side with the same
  NumPy ``rng`` as the reference, so the two implementations track each other.

The outer iteration loop (mode switching, best-error tracking, callbacks,
abstol/reltol stops) stays in Python, exactly as :class:`croak.lbfgs_hand.LBFGSHand`
keeps its loop in Python and delegates the numerics to jitted JAX.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from . import metrics_jax as mj
from .forward_jax import make_forward_adjoint_fns
from .grid import Grid
from .interactions import get_interaction
from .result import RetrievalResult
from .solver import Retriever, assemble_result

__all__ = ["COPRAJax", "run_copra_jax"]

Complex = NDArray[np.complex128]

#: Look-back window for the relative-improvement stop.
#: See :data:`croak.copra._COPRA_PATIENCE`.
_COPRA_PATIENCE = 10
_TINY = float(np.finfo(float).tiny)


def run_copra_jax(
    signal_tape,
    adjoint,
    delays: NDArray[np.float64],
    t_meas: NDArray[np.float64],
    ew0: Complex,
    weights: NDArray[np.float64],
    *,
    maxiters: int = 100,
    alpha: float = 0.25,
    reltol: float = 0.0,
    abstol: float = 0.0,
    verbose: bool = False,
    rng: np.random.Generator | None = None,
    callback=None,
    fixed_amplitude: NDArray[np.float64] | None = None,
    R_omega: bool = False,
    smeared_trace_fn=None,
    snapshot_fn=None,
    stall_patience: int = 5,
) -> tuple[
    Complex, list[float], NDArray[np.float64], NDArray[np.float64] | float, float
]:
    """JAX clone of :func:`croak.copra.run_copra`.

    ``signal_tape`` / ``adjoint`` are the pair from
    :func:`~croak.forward_jax.make_forward_adjoint_fns`. Returns
    ``(spectrum, errors, trace, mu, best_R)`` as numpy, matching ``run_copra``.

    ``snapshot_fn``, when given, is ``(ew, t_sim, errors) -> RetrievalResult``:
    it turns the current iterate into an in-progress result, which is handed to
    ``callback`` as its ``snapshot=`` argument so the GUI can draw a live full
    plot. The sweep already computes both arguments each iteration for its own
    error bookkeeping, so a snapshot costs no extra forward pass.
    """
    delays_j = jnp.asarray(np.asarray(delays, dtype=float))
    m = int(delays_j.shape[0])
    tm = jnp.asarray(np.asarray(t_meas, dtype=float))
    wj = jnp.asarray(np.asarray(weights, dtype=float))
    wzero = wj == 0.0
    # Normalising denominator MN * max(T_meas * w)**2 (== metrics.compute_R denom).
    denom = float(t_meas.size * np.max(t_meas * np.asarray(weights)[:, None]) ** 2)
    fixed_amp = (
        None
        if fixed_amplitude is None
        else jnp.asarray(np.asarray(fixed_amplitude, dtype=float))
    )

    def impose(ew):
        """Phase-only projection (identity unless an amplitude is fixed)."""
        if fixed_amp is None:
            return ew
        return fixed_amp * jnp.exp(1j * jnp.angle(ew))

    def scale(t_sim):
        if R_omega:
            return mj.mu_per_freq(tm, t_sim, wj)
        return mj.mu_global(tm, t_sim, wj)

    def mu_b_of(mu):
        return mu[:, None] if R_omega else mu

    def project(psi, measured):
        """Amplitude projection (JAX clone of :func:`croak.copra._project`)."""
        amp = jnp.abs(psi)
        target = jnp.sqrt(jnp.maximum(measured, 0.0))
        nz = amp > 0
        safe = jnp.where(nz, amp, 1.0)
        return jnp.where(nz, psi / safe * target, target.astype(psi.dtype))

    # ---- optional smeared global stage -------------------------------------
    # COPRA's LOCAL step is a per-delay projection: it replaces each delay's
    # signal amplitude while keeping its phase. Geometrical smearing makes the
    # trace an incoherent mixture over (p, theta), so there is no single signal
    # whose modulus is the model trace, and the projection has nothing to act
    # on. That half of the algorithm is structurally closed to it.
    #
    # The GLOBAL step is not. It is a descent on the objective, and a descent
    # only needs a differentiable model. Supplying `smeared_trace_fn` (any
    # ew -> trace map, e.g. from `make_param_trace_fn(..., smearing=k)`) swaps
    # the global stage onto the smeared model with an AD gradient, leaving the
    # local sweep on the unsmeared projection. The result is a hybrid: local
    # projection for its convergence speed, global descent for the physics the
    # projection cannot express.
    #
    # Note this replaces the global stage's hand-rolled two-stage descent
    # (signal first, then spectrum) with a single gradient step, because the
    # intermediate "descent in the signal" is exactly what smearing makes
    # ill-defined. COPRA's step-size heuristic, eta = alpha r / |grad|^2, is
    # kept, so the stage is COPRA's in structure and step control but not a
    # literal clone of the unsmeared one.
    # Bound to a local so the closures below see a non-optional callable: they
    # are only ever reached under `smeared_trace_fn is not None`, but that guard
    # sits at the call sites, where a type checker cannot carry it into a body
    # defined here.
    _smeared_trace = cast("Callable[[Any], Any]", smeared_trace_fn)

    def _smeared_r(ew):
        t = _smeared_trace(ew)
        mu_b = mu_b_of(scale(t))
        return jnp.sum(((tm - mu_b * t) * wj[:, None]) ** 2)

    def smeared_global_step(ew):
        # Real parameterisation: the objective is real-valued and non-holomorphic
        # in ew, so differentiate the stacked (Re, Im) vector rather than relying
        # on a complex-gradient convention.
        def obj(ri):
            return _smeared_r(ri[0] + 1j * ri[1])

        ri = jnp.stack([jnp.real(ew), jnp.imag(ew)])
        r, g = jax.value_and_grad(obj)(ri)
        denom_g = jnp.maximum(jnp.sum(g**2), _TINY)
        step = alpha * r / denom_g
        out = ri - step * g
        return impose(out[0] + 1j * out[1])

    @jax.jit
    def eval_error(ew):
        """Full forward at ``ew`` -> (t_sim, mu, R), matching forward_all+errors.

        When a smeared model is supplied it is used here, so the reported trace
        error, the convergence tests and the best-iterate bookkeeping all refer
        to the model actually being fitted rather than to the unsmeared one the
        local sweep projects with.
        """
        if smeared_trace_fn is not None:
            t_sim = smeared_trace_fn(ew)
        else:
            psis = jax.vmap(lambda tau: signal_tape(ew, tau)[0])(delays_j)
            t_sim = jnp.abs(psis.T) ** 2  # (Nomega, Ndelay)
        mu = scale(t_sim)
        resid = (tm - mu_b_of(mu) * t_sim) * wj[:, None]
        R = jnp.sqrt(jnp.sum(resid**2) / denom)
        return t_sim, mu, R

    def _delay_grad(ew, tau, j, mu):
        """Per-delay projected gradient (clone of copra.per_delay_gradient)."""
        psi, test, gate = signal_tape(ew, tau)
        t_sim_j = jnp.abs(psi) ** 2
        # Guard the per-frequency rescale against degenerate scales: in Rω mode
        # ``mu`` is zero on frequency rows whose measured trace is identically zero
        # while the model still has signal there, so ``tm / mu`` is 0/0 = NaN and
        # would poison the adjoint for every frequency. A unit denominator there
        # leaves the (~0) measured target, driving |psi| to zero. Mirrors
        # :func:`croak.copra.per_delay_gradient`.
        mu_safe = jnp.where(mu > 0.0, mu, 1.0)
        measured = tm[:, j] / mu_safe
        # zero-weight bins: target the model so they contribute nothing.
        measured = jnp.where(wzero, t_sim_j, measured)
        sprime = project(psi, measured)
        grad = 2.0 * adjoint(psi - sprime, test, gate, tau)
        zm = jnp.sum(jnp.abs(sprime - psi) ** 2)
        gn = jnp.sum(jnp.abs(grad) ** 2)
        return grad, zm, gn

    @jax.jit
    def init_max_grad(ew, mu):
        """Max |grad|**2 across delays at a fixed ``ew`` (initial current_max_grad)."""
        idx = jnp.arange(m)
        gns = jax.vmap(lambda tau, j: _delay_grad(ew, tau, j, mu)[2])(delays_j, idx)
        return jnp.max(gns)

    @jax.jit
    def local_sweep(ew, perm, mu, prev_max_grad):
        """One random-order delay sweep via lax.scan (clone of the local mode)."""

        def body(carry, j):
            ew_c, cmax = carry
            grad, zm, gn = _delay_grad(ew_c, delays_j[j], j, mu)
            cmax = jnp.maximum(cmax, gn)
            gamma = zm / jnp.maximum(jnp.maximum(cmax, prev_max_grad), _TINY)
            ew_c = impose(ew_c - gamma * grad)
            return (ew_c, cmax), None

        (ew, cmax), _ = jax.lax.scan(body, (ew, jnp.asarray(0.0)), perm)
        return ew, cmax

    @jax.jit
    def global_step(ew):
        """One joint signal+spectrum descent over all delays (clone of global mode)."""
        psis, tests, gates = jax.vmap(lambda tau: signal_tape(ew, tau))(delays_j)
        psi_all = psis.T  # (Nomega, Ndelay)
        t_sim = jnp.abs(psi_all) ** 2
        mu = scale(t_sim)
        mu_b = mu_b_of(mu)
        # 1) descent in the signal.
        diff_meas = tm - mu_b * t_sim
        r = jnp.sum(((diff_meas) * wj[:, None]) ** 2)
        grad_r = -4.0 * mu_b * diff_meas * (wj**2)[:, None] * psi_all
        eta_r = alpha * r / jnp.maximum(jnp.sum(jnp.abs(grad_r) ** 2), _TINY)
        psi_target = psi_all - eta_r * grad_r
        # 2) descent in Ew towards the updated signal.
        diffs = psi_all - psi_target  # (Nomega, Ndelay)
        ew_bars = jax.vmap(lambda d, te, ga, tau: adjoint(d, te, ga, tau))(
            diffs.T, tests, gates, delays_j
        )
        grad_ew = 2.0 * jnp.sum(ew_bars, axis=0)
        z_total = jnp.sum(jnp.abs(diffs) ** 2)
        eta_z = alpha * z_total / jnp.maximum(jnp.sum(jnp.abs(grad_ew) ** 2), _TINY)
        return impose(ew - eta_z * grad_ew)

    # ---- iteration (Python control flow, jitted numerics) ------------------
    rng = np.random.default_rng() if rng is None else rng
    ew = impose(jnp.asarray(np.asarray(ew0), dtype=complex))
    _t_sim, mu, R = eval_error(ew)
    current_max_grad = init_max_grad(ew, mu)
    R = float(R)

    best_R = R
    best_ew = ew
    best_log: list[float] = []
    err_log: list[float] = []
    stalled = 0
    mode = "local"

    if verbose:
        print(f"COPRA(jax): initial R = {R:.6e}")

    for it in range(maxiters):
        if stalled >= stall_patience:
            mode = "global"
        prev_max_grad = current_max_grad

        if mode == "local":
            perm = jnp.asarray(rng.permutation(m))
            ew, current_max_grad = local_sweep(ew, perm, mu, prev_max_grad)
        else:
            ew = (
                smeared_global_step(ew)
                if smeared_trace_fn is not None
                else global_step(ew)
            )
            current_max_grad = jnp.asarray(0.0)
        _t_sim, mu, R = eval_error(ew)
        R = float(R)

        if R < best_R:
            best_R, best_ew, stalled = R, ew, 0
        else:
            stalled += 1
        err_log.append(R)
        best_log.append(best_R)
        if verbose:
            mark = "*" if R == best_R else " "
            print(f"  iter {it + 1:4d}: R = {R:.6e} {mark}")
        if callback is not None:
            if snapshot_fn is None:
                callback(it + 1, R, best_R)
            else:
                # Bind the iterate as default arguments: `ew`/`_t_sim` are
                # rebound every iteration, and a closure over the *names* would
                # hand a deferred caller whatever the sweep had reached by then.
                callback(
                    it + 1,
                    R,
                    best_R,
                    snapshot=lambda e=ew, t=_t_sim: snapshot_fn(e, t, list(err_log)),
                )

        if abstol > 0.0 and best_R <= abstol:
            if verbose:
                print(f"COPRA(jax): reached abstol ({abstol:.2e}) at iter {it + 1}")
            break
        if reltol > 0.0 and it >= _COPRA_PATIENCE:
            past = best_log[it - _COPRA_PATIENCE]
            if past - best_R <= reltol * max(past, _TINY):
                if verbose:
                    print(
                        f"COPRA(jax): best error plateaued (reltol {reltol:.2e}) "
                        f"at iter {it + 1}"
                    )
                break

    # Final trace / scale / error from the best spectrum.
    t_sim, final_mu, final_R = eval_error(best_ew)
    final_mu_b = mu_b_of(final_mu)
    sim_trace = np.asarray(final_mu_b * t_sim, dtype=float)
    return (
        np.asarray(best_ew, dtype=complex),
        err_log,
        sim_trace,
        np.asarray(final_mu, dtype=float),
        float(final_R),
    )


class COPRAJax(Retriever):
    """JAX-accelerated COPRA retriever (``copra-jax``).

    Same parameters and behaviour as :class:`croak.copra.COPRA` (``maxiters``,
    ``alpha``, ``reltol``/``abstol``, the dispersive-slab
    ``material``/``thickness``/``npoints``/``omega0``/``quadrature`` options,
    ``phase_only`` and ``R_omega``); see that class for the full description. The
    per-iteration numerics run on the JAX forward/adjoint kernel (``vmap``/``jit``
    global step, ``lax.scan`` local sweep). See :func:`run_copra_jax`.

    Notes
    -----
    Like :class:`croak.copra.COPRA`, it takes none of the gradient-solver
    regularisation penalties (``reg_amp``/``reg_phase``/``reg_spectrum``/
    ``reg_time``) or the B-spline phase basis.
    """

    name = "copra-jax"

    def __init__(
        self,
        *,
        maxiters: int = 100,
        alpha: float = 0.25,
        reltol: float = 0.0,
        abstol: float = 0.0,
        material: str | None = None,
        thickness: float = 0.0,
        npoints: int = 20,
        omega0: float = 0.0,
        quadrature: str = "gausslegendre",
        phase_only: bool = False,
        R_omega: bool = False,
        verbose: bool = False,
        smearing=None,
        stall_patience: int = 5,
    ):
        super().__init__(maxiters=maxiters, verbose=verbose)
        self.alpha = float(alpha)
        self.smearing = smearing
        #: Iterations without improvement before the local projection sweep
        #: hands over to the global descent. Sets how much of a run is
        #: projection and how much is gradient descent, which is the whole
        #: difference between the two stages -- and, with a smeared model,
        #: the difference between the stage that can represent the physics
        #: and the one that cannot. Geib's value is 5; a very large value
        #: never leaves the local sweep.
        self.stall_patience = int(stall_patience)
        self.reltol = float(reltol)
        self.abstol = float(abstol)
        self.material = material
        self.thickness = float(thickness)
        self.npoints = int(npoints)
        self.omega0 = float(omega0)
        self.quadrature = quadrature
        self.phase_only = bool(phase_only)
        self.R_omega = bool(R_omega)

    def _solve(
        self, grid: Grid, trace, delays, interaction, ew0, weights, rng, callback=None
    ) -> RetrievalResult:
        """Run the JAX COPRA loop and package the best-error spectrum."""
        signal_tape, adjoint = make_forward_adjoint_fns(
            grid.omega,
            delays,
            interaction,
            material=self.material,
            thickness=self.thickness,
            npoints=self.npoints,
            omega0=self.omega0,
            quadrature=self.quadrature,
        )
        fixed_amp = np.abs(ew0) if self.phase_only else None
        # Smeared model for the global stage only (see run_copra_jax). Built
        # once here so its JAX tracing is shared across iterations.
        smeared_fn = None
        if self.smearing is not None:
            from .forward_jax import make_param_trace_fn

            _tf = make_param_trace_fn(
                grid.omega,
                delays,
                interaction,
                material=self.material,
                thickness=self.thickness,
                npoints=self.npoints,
                omega0=self.omega0,
                quadrature=self.quadrature,
                smearing=self.smearing,
                normalize=False,
            )
            # Bind the fixed thickness and delay zero, leaving the single-argument
            # trace map run_copra_jax expects.
            smeared_fn = lambda e: _tf(e, self.thickness, 0.0)  # noqa: E731 - see above

        def snapshot(ew_now, t_now, errors_so_far) -> RetrievalResult:
            """Assemble an in-progress result from the current iterate.

            Routed through the shared :func:`~croak.solver.assemble_result`, so a
            live snapshot and the final result compute mu and the error the same
            way. Only built when the caller asked for previews.
            """
            return assemble_result(
                spectrum=np.asarray(ew_now, dtype=complex),
                t_sim=np.asarray(t_now, dtype=float),
                grid=grid,
                t_meas=trace,
                delays=delays,
                weights=weights,
                interaction=interaction,
                algorithm=self.name,
                errors=errors_so_far,
                R_omega=self.R_omega,
                thickness=self.thickness,
                omega0=self.omega0,
            )

        spectrum, errors, sim_trace, mu, best_R = run_copra_jax(
            signal_tape,
            adjoint,
            np.asarray(delays, dtype=float),
            trace,
            ew0,
            weights,
            smeared_trace_fn=smeared_fn,
            stall_patience=self.stall_patience,
            maxiters=self.maxiters,
            alpha=self.alpha,
            reltol=self.reltol,
            abstol=self.abstol,
            verbose=self.verbose,
            rng=rng,
            callback=callback,
            snapshot_fn=snapshot,
            fixed_amplitude=fixed_amp,
            R_omega=self.R_omega,
        )
        return RetrievalResult(
            spectrum=spectrum,
            grid=grid,
            delays=delays,
            interaction=get_interaction(interaction).name,
            algorithm=self.name,
            error=best_R,
            errors=errors,
            trace=sim_trace,
            mu=mu,
        )
