r"""L-BFGS retrieval with **hand-derived** gradients, jitted in JAX (``lbfgs-hand``).

A drop-in twin of :class:`croak.lbfgs.LBFGS` and :class:`croak.lbfgs_ad.LBFGSAD`:
same pulse parameterisation, same NLopt ``LD_LBFGS`` optimiser, same options
(phase-only, ``R_omega`` adaptive scaling, amplitude / phase / spectral
regularisation, dispersive or thin propagation).

The gradient is the **same analytic Wirtinger adjoint** as :class:`croak.lbfgs.LBFGS`
(``adjoint_single`` → ``ew_vjp``), but expressed entirely in JAX and evaluated by a
single :func:`jax.jit`-compiled function that ``vmap``\\s the forward + adjoint over
all delays — **no automatic differentiation is used anywhere**. This gives the
hand-derived gradients the same vectorisation/JIT speedup that makes ``lbfgs-ad``
faster than the pure-NumPy ``lbfgs`` (the gap is the trace evaluation, not the
gradient maths — the two gradients are identical, see
``tests/test_ad_vs_analytic.py``).

It is thus a fast, AD-free reference path, and shares the JAX forward/adjoint core
(:func:`croak.forward_jax.make_forward_adjoint_fns`) intended to also accelerate
COPRA.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import nlopt
import numpy as np
from numpy.typing import NDArray

from . import metrics_jax as mj
from ._jax_pulse import make_param_grad_fns, make_spectrum_fn
from .forward_jax import make_forward_adjoint_fns
from .result import RetrievalResult
from .solver import Retriever, assemble_result, spectral_target_amplitude

__all__ = ["LBFGSHand"]

Complex = NDArray[np.complex128]


class LBFGSHand(Retriever):
    """NLopt L-BFGS solver with jitted JAX **hand-derived** (non-AD) gradients.

    Parameters mirror :class:`croak.lbfgs.LBFGS` exactly (``maxiters``,
    ``material``/``thickness``/``npoints``/``omega0``/``quadrature``,
    ``phase_only``, ``R_omega``, ``reg_amp``/``reg_phase``/``reg_spectrum`` +
    ``spectrum_target``, ``reltol``/``abstol``/``targeterr``, ``verbose``); see
    that class for the full description. The forward model, FROG error and its
    gradient are all evaluated with the JAX twins, the gradient using the
    explicit Wirtinger adjoint (not :func:`jax.grad`).

    Notes
    -----
    Like :class:`croak.lbfgs.LBFGS` — and unlike :class:`croak.lbfgs_ad.LBFGSAD` —
    it does not expose the B-spline phase basis (``phase_basis``/``n_nodes``) or
    the temporal penalty (``reg_time``/``time_window``).
    """

    name = "lbfgs-hand"

    def __init__(
        self,
        *,
        maxiters: int = 300,
        material: str | None = None,
        thickness: float = 0.0,
        npoints: int = 20,
        omega0: float = 0.0,
        quadrature: str = "gausslegendre",
        phase_only: bool = False,
        R_omega: bool = False,
        reg_amp: float = 0.0,
        reg_phase: float = 0.0,
        reg_spectrum: float = 0.0,
        spectrum_target=None,
        reltol: float = 1e-4,
        abstol: float = 1e-8,
        targeterr: float = 0.0,
        verbose: bool = False,
    ):
        """Configure the jitted hand-gradient L-BFGS retriever (see class docstring)."""
        super().__init__(maxiters=maxiters, verbose=verbose)
        self.material = material
        self.thickness = float(thickness)
        self.npoints = int(npoints)
        self.omega0 = float(omega0)
        self.quadrature = quadrature
        self.phase_only = bool(phase_only)
        self.R_omega = bool(R_omega)
        self.reg_amp = float(reg_amp)
        self.reg_phase = float(reg_phase)
        self.reg_spectrum = float(reg_spectrum)
        self._spectral_target = spectral_target_amplitude(spectrum_target)
        self.reltol = float(reltol)
        self.abstol = float(abstol)
        self.targeterr = float(targeterr)

    def _solve(
        self, grid, t_meas, delays, interaction, ew0, weights, rng=None, callback=None
    ) -> RetrievalResult:
        """Run NLopt L-BFGS on the jitted hand-gradient objective (module docs)."""
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
        spectrum_fn, u0, nidcs = make_spectrum_fn(
            np.abs(ew0), np.angle(ew0), phase_only=self.phase_only
        )
        pg = make_param_grad_fns(np.abs(ew0), np.angle(ew0), phase_only=self.phase_only)

        delays_j = jnp.asarray(np.asarray(delays, dtype=float))
        tm = jnp.asarray(t_meas)
        wj = jnp.asarray(weights)
        # Constant normaliser MN * max(T_meas * w)**2 (== metrics.compute_R denom).
        denom = float(t_meas.size * np.max(t_meas * weights[:, None]) ** 2)

        reg_amp, reg_phase, phase_only, R_omega = (
            self.reg_amp,
            self.reg_phase,
            self.phase_only,
            self.R_omega,
        )
        use_spec = (
            self.reg_spectrum > 0
            and self._spectral_target is not None
            and not phase_only
        )
        reg_spectrum = self.reg_spectrum
        s_sup = jnp.asarray(self._spectral_target) if use_spec else None

        def value_and_grad(u):
            """Return ``(total, ferr, grad)`` via the hand adjoint (no AD)."""
            ew = spectrum_fn(u)
            # Forward over all delays, keeping the (test, gate) tape for the adjoint.
            psis, tests, gates = jax.vmap(lambda tau: signal_tape(ew, tau))(delays_j)
            psi_all = psis.T  # (Nomega, Ndelay)
            t_sim = jnp.abs(psi_all) ** 2

            if R_omega:
                mu = mj.mu_per_freq(tm, t_sim, wj)[:, None]
            else:
                mu = mj.mu_global(tm, t_sim, wj)
            resid = (tm - mu * t_sim) * wj[:, None]
            ferr = jnp.sqrt(jnp.sum(resid**2) / denom)

            # Backprop the FROG-error cotangent through |psi|^2 then the forward
            # model: dR/dT_sim = -mu w resid / (R denom); dT_sim/dpsi* = psi.
            cot_t = jnp.where(
                ferr > 0,
                -mu * wj[:, None] * resid / (ferr * denom),
                jnp.zeros_like(t_sim),
            )
            psi_bar = cot_t * psi_all  # (Nomega, Ndelay)
            ew_bars = jax.vmap(lambda pb, te, ga, tau: adjoint(pb, te, ga, tau))(
                psi_bar.T, tests, gates, delays_j
            )
            ew_bar = jnp.sum(ew_bars, axis=0)
            grad = pg.ew_vjp(u, ew_bar)

            total = ferr
            if reg_amp > 0 and not phase_only:
                total = total + reg_amp * mj.amplitude_penalty(u[:nidcs])
                grad = grad + reg_amp * pg.amplitude_penalty_grad(u)
            if reg_phase > 0:
                phase_part = u if phase_only else u[nidcs:]
                total = total + reg_phase * mj.phase_penalty(phase_part)
                grad = grad + reg_phase * pg.phase_penalty_grad(u)
            if s_sup is not None:  # equivalent to use_spec; narrows s_sup to non-None
                total = total + reg_spectrum * mj.spectral_penalty(u[:nidcs], s_sup)
                grad = grad + reg_spectrum * pg.spectral_penalty_grad(u, s_sup)
            return total, ferr, grad

        value_and_grad = jax.jit(value_and_grad)

        @jax.jit
        def trace_only(ew):
            psis, _t, _g = jax.vmap(lambda tau: signal_tape(ew, tau))(delays_j)
            return jnp.abs(psis.T) ** 2

        err_log: list[float] = []

        def make_snapshot(u) -> RetrievalResult:
            """Assemble an in-progress result from current params (live preview).

            Recomputes the spectrum and simulated trace for the current parameter
            vector and feeds them through the shared :func:`assemble_result`, so a
            live snapshot and the final result are built identically. Reused for
            the final return below.
            """
            spectrum = np.asarray(spectrum_fn(jnp.asarray(u)), dtype=complex)
            t_sim = np.asarray(trace_only(jnp.asarray(spectrum)), dtype=float)
            return assemble_result(
                spectrum=spectrum,
                t_sim=t_sim,
                grid=grid,
                t_meas=t_meas,
                delays=delays,
                weights=weights,
                interaction=interaction,
                algorithm=self.name,
                errors=err_log,
                R_omega=self.R_omega,
                omega0=self.omega0,
            )

        def evaluate(u: NDArray[np.float64], grad: NDArray[np.float64] | None) -> float:
            total, ferr, g = value_and_grad(jnp.asarray(u))
            total, ferr = float(total), float(ferr)
            err_log.append(ferr)
            if self.verbose:
                print(f"  eval {len(err_log):4d}: R = {ferr:.6e}")
            if callback is not None:
                callback(
                    len(err_log),
                    ferr,
                    min(err_log),
                    snapshot=lambda: make_snapshot(u),
                )
            if grad is not None and grad.size > 0:
                grad[:] = np.asarray(g, dtype=float)
            return total

        opt = nlopt.opt(nlopt.LD_LBFGS, u0.size)
        opt.set_min_objective(evaluate)
        opt.set_maxeval(self.maxiters)
        opt.set_ftol_rel(self.reltol)
        opt.set_ftol_abs(self.abstol)
        if self.targeterr > 0:
            opt.set_stopval(self.targeterr)
        u = opt.optimize(u0)

        # Assemble the final result exactly as the live snapshots (shared
        # :func:`make_snapshot` / `assemble_result`).
        return make_snapshot(u)
