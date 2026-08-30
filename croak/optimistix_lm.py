r"""Levenberg–Marquardt retrieval via Optimistix (``lm-optx``).

A JAX-native twin of :class:`croak.lm.LM`: it drives the **same** flattened FROG
residual (:func:`croak.metrics_jax.residual_vector`, with the same amplitude /
phase / spectral regularisation rows appended so ``‖rows‖² == reg·penalty``) to
zero, over the same real parameter vector (:mod:`croak._jax_pulse`), with the same
JAX forward model (:mod:`croak.forward_jax`). The **only** difference from
:class:`~croak.lm.LM` is the optimiser: instead of SciPy ``least_squares`` driving
a host-side trust-region loop with a JAX Jacobian materialised and transferred
every iteration, Optimistix's :class:`optimistix.LevenbergMarquardt` runs the
whole damped Gauss–Newton loop **on-device** (each step jitted) with the Jacobian
handled by ``lineax``.

Because it is pure JAX it also sidesteps the MINPACK reentrancy crash that stops
:class:`croak.lm.LM` from using a genuine LM with the analytic JAX Jacobian (see
the :mod:`croak.lm` docstring): this is a true JAX-Jacobian Levenberg–Marquardt.
"""

from __future__ import annotations

import jax.numpy as jnp
import lineax as lx
import numpy as np
import optimistix as optx

from . import metrics_jax as mj
from ._jax_pulse import ExtraParamSpec, augment, ls_parameterisation
from ._optimistix import run_optx
from .focal import FocalMixture
from .forward_jax import make_param_trace_fn
from .result import RetrievalResult
from .smearing import SmearingKernel
from .solver import (
    Retriever,
    assemble_result,
    spectral_target_amplitude,
    split_smear_value,
)

__all__ = ["OptxLM"]

#: Linear solvers for the damped Gauss--Newton step, selected by
#: :class:`OptxLM`'s ``linear_solver``. Each Levenberg--Marquardt iteration solves
#: the linear least-squares problem ``[J; sqrt(lambda) I] d = [r; 0]``, whose normal
#: equations are ``(J^T J + lambda I) d = J^T r``.
#:
#: * ``"normal"`` forms ``J^T J + lambda I`` and takes its Cholesky factor. The Gram
#:   matrix costs ``O(m n^2)`` in one BLAS ``gemm`` — which Accelerate/MKL run at
#:   near-peak throughput — and the factorisation is only ``O(n^3)`` on the small
#:   ``n x n`` result.
#: * ``"qr"`` factorises the ``(m + n) x n`` damped operator directly. It is
#:   Optimistix's own default and is the textbook-stable route because it never
#:   forms ``J^T J``, but LAPACK's blocked Householder QR on a tall-skinny matrix is
#:   panel-bound rather than FLOP-bound and does not reach the same throughput.
#:
#: Both routes rely on ``lambda > 0`` to regularise: the FROG Jacobian is strongly
#: rank-deficient (typically only a few hundred of its singular values sit above the
#: float64 epsilon), so the *undamped* Gauss--Newton system is singular either way.
#: Squaring the condition number costs the normal-equations route more of that
#: margin, which is why ``"qr"`` is kept as an escape hatch.
_LINEAR_SOLVERS: dict[str, lx.AbstractLinearSolver] = {
    "normal": lx.Normal(lx.Cholesky()),
    "qr": lx.QR(),
}


class OptxLM(Retriever):
    r"""Optimistix Levenberg–Marquardt solver with a JAX Jacobian.

    The flattened residual, parameterisation and options match
    :class:`croak.lm.LM`; only the optimiser differs (Optimistix's on-device
    :class:`optimistix.LevenbergMarquardt` in place of SciPy ``least_squares``).
    Because it is pure JAX it runs a genuine LM with the analytic Jacobian and so
    sidesteps the MINPACK reentrancy crash described in the :mod:`croak.lm`
    docstring.

    Parameters
    ----------
    maxiters : int, optional
        Maximum number of solver steps (default ``300``).
    material, thickness, npoints, omega0, quadrature
        Dispersive-slab parameters; see :class:`croak.forward.ForwardModel`.
    smearing : SmearingKernel or None, optional
        Geometric time-smearing kernel of a non-collinear BOXCARS geometry
        (default ``None`` = off). PG (TG) and SD only. See :mod:`croak.smearing`
        and :doc:`/howto/geometric_smearing`.
    focal : FocalMixture or None, optional
        Chromatic focal mixture (:mod:`croak.focal`) in place of ``smearing``:
        the focal average is carried explicitly, each node applying its own
        wavelength-dependent amplitude filter, where the reduced kernel
        evaluates a single :math:`|A|^6` weight at the carrier. Mutually
        exclusive with ``smearing``, and incompatible with ``fit_smearing``
        (an explicit mixture has no kernel width to scale).

        Only worth its cost for a broadband pulse. The reduced kernel's widths
        scale as :math:`\sigma\propto\lambda`, so freezing them at the carrier
        is a -16%/+24% spread of width across a 1 fs deep-UV band and costs a
        mean 12% of retrieved duration there, against 3.5% at 2 fs. The mixture
        needs one field build per focal node rather than per Gauss-Hermite
        node, so it is one to two orders of magnitude slower.
        Phase parameterisation; ``"bspline"`` uses ``n_nodes`` cubic-spline
        control points (implies phase-only). See
        :func:`croak._jax_pulse.make_spline_phase_basis`.
    n_nodes : int, optional
        Number of B-spline control points when ``phase_basis="bspline"``.
    R_omega : bool, optional
        Use per-frequency adaptive scaling factors.
    reg_amp, reg_phase, reg_spectrum, spectrum_target
        Smoothness / spectral-match penalties, appended as extra residual rows
        exactly as in :class:`croak.lm.LM`. See :doc:`/howto/regularisation`.
    linear_solver : {"normal", "qr"}, optional
        How to solve the damped Gauss--Newton system each iteration (default
        ``"normal"``). ``"normal"`` forms ``J^T J + lambda I`` and takes its
        Cholesky factor; ``"qr"`` factorises the stacked ``[J; sqrt(lambda) I]``
        operator directly, which is Optimistix's own default. ``"normal"`` is
        roughly 2x faster on a full-size pointwise retrieval because the Gram
        matrix is a single BLAS ``gemm`` while the tall-skinny QR is panel-bound.
        It squares the condition number, so switch to ``"qr"`` if a badly scaled
        problem stalls or returns a non-finite result. See
        :doc:`/explanation/algorithms`.
    reltol, abstol : float, optional
        Convergence tolerances (defaults ``1e-4`` / ``1e-8``). ``abstol`` is the
        Optimistix ``atol``; ``reltol`` is passed as the Optimistix ``rtol`` *and*
        drives a relative-improvement plateau stop in :func:`croak._optimistix.run_optx`
        (stop once the best FROG error improves by less than ``reltol`` over the last
        few steps). The plateau stop is needed because Optimistix's own Cauchy
        termination is ``atol``-dominated near convergence and effectively ignores
        ``rtol`` — so without it ``reltol`` would have no practical effect.
    tau0, smear_scale : float, optional
        Starting (and held) centres for the delay-zero offset (s) and the
        smearing-width multiplier, exactly as in :class:`croak.lm.LM`.
    fit_thickness : bool, optional
        Also fit the dispersive-slab **thickness** as an extra parameter
        (default ``False``). Requires a dispersive slab (``material`` set,
        ``thickness > 0``) — PG/SD only. The prior ``thickness`` is the starting
        value. See :doc:`/howto/fitting_thickness_tau0`.
    fit_tau0 : bool, optional
        Also fit a **delay-zero offset** :math:`\tau_0` (default ``False``),
        correcting a mis-set experimental time zero (all interactions).
    fit_smearing : bool, optional
        Also fit the geometric-smearing **strength** as an extra parameter
        (default ``False``): a dimensionless multiplier on both kernel widths,
        equivalently the mask ratio ``d/D``. Requires ``smearing``. Do not enable
        it together with ``fit_thickness`` — the smearing width is a nuisance
        parameter that will happily absorb dispersion-model error. See
        :doc:`/howto/geometric_smearing`.
    fit_smearing_split : bool, optional
        Fit the gate-shape (``p``) and delay (``delta``) kernel widths as two
        independent multipliers instead of one joint one (default ``False``) —
        a diagnostic for which channel deviates from the geometric prediction.
        Requires ``fit_smearing``; the fitted values land in ``smear_scale``
        (p) and ``smear_scale_delta`` (delta) on the result.
    smear_scale_delta : float, optional
        Starting delay-channel multiplier for a split fit (default ``1.0``);
        used only when ``fit_smearing_split`` is set.
    polish : bool, optional
        Two-phase polish (default ``False``): first retrieve the pulse with the
        extra parameters held fixed, then free them for a joint final phase.
    verbose : bool, optional
        Print per-step progress.

    Notes
    -----
    There is no ``method`` argument: Optimistix Levenberg–Marquardt has no
    trust-region variants to select between.

    Optimistix Levenberg–Marquardt is **unconstrained**, so unlike SciPy's
    bounded ``trf`` it cannot hold the fitted thickness non-negative — it ignores
    the lower bound :func:`croak._jax_pulse.augment` supplies and relies on
    starting from a good prior (:math:`\delta_L = 0`), exactly as the SciPy
    ``method="lm"`` path does. Fit the thickness as a polish from a good prior.
    """

    name = "lm-optx"

    def __init__(
        self,
        *,
        maxiters: int = 300,
        material: str | None = None,
        thickness: float = 0.0,
        npoints: int = 20,
        omega0: float = 0.0,
        quadrature: str = "gausslegendre",
        smearing: SmearingKernel | None = None,
        focal: FocalMixture | None = None,
        phase_only: bool = False,
        phase_basis: str = "pointwise",
        n_nodes: int = 20,
        R_omega: bool = False,
        reg_amp: float = 0.0,
        reg_phase: float = 0.0,
        reg_spectrum: float = 0.0,
        spectrum_target=None,
        tau0: float = 0.0,
        smear_scale: float = 1.0,
        smear_scale_delta: float = 1.0,
        fit_thickness: bool = False,
        fit_tau0: bool = False,
        fit_smearing: bool = False,
        fit_smearing_split: bool = False,
        polish: bool = False,
        linear_solver: str = "normal",
        reltol: float = 1e-4,
        abstol: float = 1e-8,
        verbose: bool = False,
    ):
        """Configure the Optimistix Levenberg–Marquardt retriever (see :class:`LM`)."""
        super().__init__(maxiters=maxiters, verbose=verbose)
        if linear_solver not in _LINEAR_SOLVERS:
            raise ValueError(
                f"unknown linear_solver {linear_solver!r}; "
                f"expected one of {sorted(_LINEAR_SOLVERS)}"
            )
        self.linear_solver = str(linear_solver)
        self.material = material
        self.thickness = float(thickness)
        self.npoints = int(npoints)
        self.omega0 = float(omega0)
        self.quadrature = quadrature
        self.smearing = smearing
        self.focal = focal
        self.phase_only = bool(phase_only)
        self.phase_basis = str(phase_basis)
        self.n_nodes = int(n_nodes)
        self.R_omega = bool(R_omega)
        self.reg_amp = float(reg_amp)
        self.reg_phase = float(reg_phase)
        self.reg_spectrum = float(reg_spectrum)
        self._spectral_target = spectral_target_amplitude(spectrum_target)
        self.tau0 = float(tau0)
        self.smear_scale = float(smear_scale)
        self.smear_scale_delta = float(smear_scale_delta)
        self.fit_thickness = bool(fit_thickness)
        self.fit_tau0 = bool(fit_tau0)
        self.fit_smearing = bool(fit_smearing)
        self.fit_smearing_split = bool(fit_smearing_split)
        self.polish = bool(polish)
        self.reltol = float(reltol)
        self.abstol = float(abstol)

    def _solve(
        self, grid, t_meas, delays, interaction, ew0, weights, rng=None, callback=None
    ) -> RetrievalResult:
        """Drive the flattened residual to zero with Optimistix LM (module docs)."""
        # Forward model differentiable in the extra parameters (thickness/tau0);
        # with both flags off it reduces exactly to the standard trace map.
        trace_param = make_param_trace_fn(
            grid.omega,
            delays,
            interaction,
            material=self.material,
            thickness=self.thickness,
            npoints=self.npoints,
            omega0=self.omega0,
            quadrature=self.quadrature,
            smearing=self.smearing,
            focal=self.focal,
            fit_thickness=self.fit_thickness,
            fit_tau0=self.fit_tau0,
            fit_smearing=self.fit_smearing,
        )
        spectrum_fn, u0_pulse, nidcs, phase_only = ls_parameterisation(
            ew0,
            grid.omega,
            phase_only=self.phase_only,
            phase_basis=self.phase_basis,
            n_nodes=self.n_nodes,
        )

        # Extra fitted scalars (slab thickness / delay-zero), appended to u.
        scale_tau = (
            float(np.median(np.abs(np.diff(delays)))) if delays.size > 1 else 1.0
        )
        spec = ExtraParamSpec(
            fit_thickness=self.fit_thickness,
            thickness0=self.thickness,
            fit_tau0=self.fit_tau0,
            tau0_0=self.tau0,
            scale_tau=scale_tau,
            fit_smearing=self.fit_smearing,
            smear_scale0=self.smear_scale,
            fit_smearing_split=self.fit_smearing_split,
            smear_delta0=self.smear_scale_delta,
        )
        aug = augment(u0_pulse, spec)

        tm = jnp.asarray(t_meas)
        wj = jnp.asarray(weights)
        reg_amp, reg_phase, R_omega = (
            self.reg_amp,
            self.reg_phase,
            self.R_omega,
        )
        eps = float(np.finfo(np.float64).eps)
        # Spectral-match rows: sqrt(reg/‖s‖²)·(a − s) so ‖rows‖² = reg·P_spec.
        use_spec = (
            self.reg_spectrum > 0
            and self._spectral_target is not None
            and not phase_only
        )
        if use_spec:
            s_sup = jnp.asarray(self._spectral_target)
            spec_scale = jnp.sqrt(
                self.reg_spectrum / jnp.maximum(jnp.sum(s_sup * s_sup), eps)
            )

        def make_residual(unpack):
            """Build the flattened residual for a given parameter unpacking."""

            def residual(u):
                u_pulse, thickness, tau0, smear = unpack(u)
                ew = spectrum_fn(u_pulse)
                t_sim = trace_param(ew, thickness, tau0, smear)
                parts = [mj.residual_vector(tm, t_sim, wj, R_omega=R_omega)]
                # Penalty rows act on the pulse block only.
                if reg_amp > 0 and not phase_only:
                    a = u_pulse[:nidcs]
                    d = a[2:] - 2.0 * a[1:-1] + a[:-2]
                    scale = jnp.sqrt(
                        reg_amp / ((a.size - 2) * jnp.maximum(jnp.max(a) ** 2, eps))
                    )
                    parts.append(scale * d)
                if reg_phase > 0:
                    p = u_pulse if phase_only else u_pulse[nidcs:]
                    d = p[2:] - 2.0 * p[1:-1] + p[:-2]
                    parts.append(jnp.sqrt(reg_phase / (p.size - 2)) * d)
                if use_spec:
                    parts.append(spec_scale * (u_pulse[:nidcs] - s_sup))
                return jnp.concatenate(parts)

            return residual

        nfrog = t_meas.size  # the FROG residual block; its 2-norm equals R

        def make_snapshot(u_pulse, thickness, tau0, smear) -> RetrievalResult:
            """Assemble an in-progress result from current params (live preview).

            Recomputes the spectrum and simulated trace for the current parameter
            vector and feeds them through the shared :func:`assemble_result`, so a
            live snapshot and the final result are built identically. Reused for
            the final return below.
            """
            smear_p, smear_d = split_smear_value(smear)
            spectrum = np.asarray(spectrum_fn(jnp.asarray(u_pulse)), dtype=complex)
            t_sim = np.asarray(
                trace_param(jnp.asarray(spectrum), thickness, tau0, smear),
                dtype=float,
            )
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
                thickness=thickness if self.fit_thickness else None,
                tau0=tau0 if self.fit_tau0 else 0.0,
                smear_scale=smear_p if self.fit_smearing else None,
                smear_scale_delta=smear_d if self.fit_smearing else None,
                omega0=self.omega0,
            )

        err_log: list[float] = []

        def run_lm_optx(unpack, u_start):
            """Run one Optimistix LM pass (accumulating into the shared ``err_log``)."""
            residual = make_residual(unpack)

            def fn(u, args):
                """Optimistix residual; aux is the bare FROG error R."""
                r = residual(u)
                return r, jnp.linalg.norm(r[:nfrog])

            def snapshot_from_y(y) -> RetrievalResult:
                return make_snapshot(*unpack(y))

            solver = optx.LevenbergMarquardt(
                rtol=self.reltol,
                atol=self.abstol,
                linear_solver=_LINEAR_SOLVERS[self.linear_solver],
            )
            u_pass, _ = run_optx(
                solver,
                fn,
                u_start,
                maxiters=self.maxiters,
                reltol=self.reltol,
                callback=callback,
                make_snapshot=snapshot_from_y,
                err_log=err_log,
                verbose=self.verbose,
            )
            return u_pass

        # Optimistix LM is unconstrained, so (like SciPy method="lm") the
        # thickness lower bound from `aug` cannot be enforced — we start from the
        # prior (delta_L = 0) and rely on the polish / a good warm start.
        def pulse_unpack(u):
            # Extras held at their centres, matching augment's unpack: a warm-started
            # or explicitly-set tau0/smear_scale still shifts the model here.
            smear0 = (
                np.array([self.smear_scale, self.smear_scale_delta])
                if self.fit_smearing_split
                else self.smear_scale
            )
            return u, float(self.thickness), self.tau0, smear0

        if spec.any and self.polish:
            # Phase 1: pulse with extras fixed. Phase 2: free the extras jointly.
            u_pulse = run_lm_optx(pulse_unpack, u0_pulse)
            u_seed = np.concatenate([u_pulse, np.zeros(aug.u0.size - u_pulse.size)])
            u = run_lm_optx(aug.unpack, u_seed)
        elif spec.any:
            u = run_lm_optx(aug.unpack, aug.u0)
        else:
            u = run_lm_optx(pulse_unpack, u0_pulse)

        # Resolve the final pulse vector and fitted extras, then assemble exactly
        # as the live snapshots (shared :func:`make_snapshot` / `assemble_result`).
        if spec.any:
            u_pulse_final = u[: aug.n_pulse]
            thickness_fit, tau0_fit, smear_fit = aug.physical(u)
        else:
            u_pulse_final = u
            thickness_fit, tau0_fit, smear_fit = (
                self.thickness,
                self.tau0,
                self.smear_scale,
            )

        return make_snapshot(u_pulse_final, thickness_fit, tau0_fit, smear_fit)
