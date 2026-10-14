r"""Retrieve by nonlinear least squares with SciPy (``lm``).

The default ``method="trf"`` uses trust-region reflective least squares with a
JAX Jacobian. It is distinct from Levenberg–Marquardt. Explicit ``method="lm"``
selects MINPACK Levenberg–Marquardt with a finite-difference Jacobian; ``dogbox``
is also available. The numeric Jacobian on the MINPACK route avoids a known
JAX callback interoperability problem.

The normalised trace residual has Euclidean norm :math:`R`. Regularisation
is appended as residual rows, so the squared objective is
:math:`R^2 + \sum_k\lambda_k P_k`, rather than the gradient solvers'
:math:`R + \sum_k\lambda_k P_k`. Equal penalty weights therefore do not define
equal regularised problems. See :doc:`/explanation/algorithms`.

The model supports thin local SHG/PG/SD and dispersive PG/SD, with optional
spatial models, phase constraints and fitted scalar parameters as documented
below. Bounds depend on the selected least-squares method.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from . import metrics_jax as mj
from ._jax_pulse import ExtraParamSpec, augment, ls_parameterisation
from .focal import FocalMixture
from .forward_jax import make_param_trace_fn
from .result import RetrievalResult
from .smearing import SmearingKernel
from .solver import (
    Retriever,
    assemble_result,
    resolve_reg,
    spectral_target_amplitude,
    split_smear_value,
)

__all__ = ["LM"]

Complex = NDArray[np.complex128]


class LM(Retriever):
    r"""Retrieve with SciPy nonlinear least squares and a JAX Jacobian by default.

    Parameters
    ----------
    maxiters : int, optional
        Maximum number of residual evaluations (``max_nfev``).
    method : {"trf", "lm", "dogbox"}, optional
        SciPy ``least_squares`` method. Default ``"trf"`` uses the analytic JAX
        Jacobian; ``"lm"`` falls back to a numeric Jacobian (see the module
        docstring for why).
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

        Resolve wavelength-dependent spatial filtering when the achromatic
        approximation is inadequate. Cost depends on the focal and collection
        quadrature sizes; check convergence for the chosen pulse and geometry.
    phase_only : bool, optional
        Hold the initial spectral amplitude fixed and retrieve phase only.
    phase_basis : {"pointwise", "bspline"}, optional
        Phase parameterisation. ``"pointwise"`` (default) optimises one phase
        value per frequency bin. ``"bspline"`` optimises the ``n_nodes`` control
        points of a cubic B-spline over the spectral support — a strong
        low-dimensional regulariser that implies phase-only retrieval. See
        :func:`croak._jax_pulse.make_spline_phase_basis`.
    n_nodes : int, optional
        Number of B-spline control points when ``phase_basis="bspline"``
        (default ``20``); ignored otherwise.
    R_omega : bool, optional
        Use per-frequency adaptive scaling factors.
    reg_amp : float or None, optional
        Amplitude second-difference smoothness weight, added as extra residual
        rows. ``None`` (default) resolves to the default LM-family weight
        ``3e-5`` (:data:`croak.solver.REG_DEFAULTS`; the residual-rows
        objective needs far smaller weights than the gradient solvers); pass
        ``0`` to disable.
    reg_phase : float, optional
        Phase second-difference smoothness weight (default ``0``).
    reg_spectrum : float or None, optional
        Spectral-match weight pulling the retrieved spectral amplitude towards
        ``spectrum_target`` (full mode only; added as extra residual rows).
        ``None`` (default) resolves to the default LM-family ``1e-5`` and is
        inert without a ``spectrum_target``; pass ``0`` to disable.
    spectrum_target : array_like or None, optional
        Measured spectral **intensity** on the retrieval grid (e.g.
        ``TraceData.Iomega``) for ``reg_spectrum``; ``None`` disables it.
    reltol, abstol : float, optional
        Convergence tolerances (defaults ``1e-4`` / ``1e-8``, matching the
        reference LM solver). ``reltol`` is the relative cost-reduction
        tolerance (SciPy ``ftol``); ``abstol`` bounds termination by the step
        and gradient (SciPy ``xtol`` and ``gtol``).
    tau0 : float, optional
        Starting delay-zero offset :math:`\tau_0` in seconds (default ``0``) —
        the centre ``fit_tau0`` optimises around, e.g. the previous fit's value
        when warm-starting. Held at this value when ``fit_tau0`` is off.
    smear_scale : float, optional
        Starting multiplier on the smearing-kernel widths (default ``1.0`` = the
        kernel as supplied) — the centre ``fit_smearing`` optimises around. Held
        at this value when ``fit_smearing`` is off.
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
        parameter that can absorb dispersion-model error. See
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
        Print per-evaluation progress.
    """

    name = "lm"

    def __init__(
        self,
        *,
        maxiters: int = 300,
        method: str = "trf",
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
        reg_amp: float | None = None,
        reg_phase: float = 0.0,
        reg_spectrum: float | None = None,
        spectrum_target=None,
        tau0: float = 0.0,
        smear_scale: float = 1.0,
        smear_scale_delta: float = 1.0,
        fit_thickness: bool = False,
        fit_tau0: bool = False,
        fit_smearing: bool = False,
        fit_smearing_split: bool = False,
        polish: bool = False,
        reltol: float = 1e-4,
        abstol: float = 1e-8,
        verbose: bool = False,
    ):
        """Configure the Levenberg–Marquardt retriever (see the class docstring)."""
        super().__init__(maxiters=maxiters, verbose=verbose)
        self.method = str(method)
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
        self.reg_spectrum, self.reg_amp = resolve_reg(reg_spectrum, reg_amp, "lm")
        self.reg_phase = float(reg_phase)
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
        err_log: list[float] = []

        def make_snapshot(u_pulse, thickness, tau0, smear) -> RetrievalResult:
            """Assemble an in-progress result from current params (live preview).

            Recomputes the spectrum and simulated trace for the current parameter
            vector and feeds them through the shared :func:`assemble_result` — so
            a live snapshot and the final result are built identically. Reused for
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

        def run_lm(unpack, u_start, lower_bounds):
            """Run one ``least_squares`` pass; logs into ``err_log``."""
            residual_jit = jax.jit(make_residual(unpack))
            jac_jit = jax.jit(jax.jacfwd(make_residual(unpack)))

            def residual_np(u):
                r = np.asarray(residual_jit(jnp.asarray(u)), dtype=float)
                ferr = float(np.linalg.norm(r[:nfrog]))
                err_log.append(ferr)
                if self.verbose:
                    print(f"  eval {len(err_log):4d}: R = {ferr:.6e}")
                if callback is not None:
                    callback(
                        len(err_log),
                        ferr,
                        min(err_log),
                        snapshot=lambda: make_snapshot(*unpack(u)),
                    )
                return r

            def jac_np(u):
                return np.asarray(jac_jit(jnp.asarray(u)), dtype=float)

            # MINPACK ("lm") segfaults when its Fortran callback evaluates the
            # JAX/XLA Jacobian program, so for that method we hand SciPy a numeric
            # ("2-point") Jacobian of the JAX residual; every other method
            # ("trf"/"dogbox") uses the analytic JAX Jacobian. MINPACK also has no
            # bound support, so for "lm" we drop the thickness lower bound and
            # rely on starting from a good prior.
            if self.method == "lm":
                jac_arg = "2-point"
                bounds = (-np.inf, np.inf)
            else:
                jac_arg = jac_np
                bounds = (
                    (lower_bounds, np.inf)
                    if np.any(np.isfinite(lower_bounds))
                    else (-np.inf, np.inf)
                )

            res = least_squares(
                residual_np,
                u_start,
                # scipy's stub types jac as str-only; it accepts a callable Jacobian
                # at runtime (used for the analytic JAX Jacobian on trf/dogbox).
                jac=jac_arg,  # pyright: ignore[reportArgumentType]
                method=self.method,
                bounds=bounds,
                max_nfev=self.maxiters,
                ftol=self.reltol,
                xtol=self.abstol,
                gtol=self.abstol,
            )
            return res.x

        if self.method == "lm":
            warnings.warn(
                "LM method='lm' uses a numeric Jacobian (the JAX Jacobian "
                "crashes MINPACK's Fortran callback); pass method='trf' for the "
                "analytic JAX Jacobian.",
                RuntimeWarning,
                stacklevel=2,
            )

        no_bounds = np.full(u0_pulse.size, -np.inf)

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
            u_pulse = run_lm(pulse_unpack, u0_pulse, no_bounds)
            u_seed = np.concatenate([u_pulse, np.zeros(aug.u0.size - u_pulse.size)])
            u = run_lm(aug.unpack, u_seed, aug.lower_bounds)
        elif spec.any:
            u = run_lm(aug.unpack, aug.u0, aug.lower_bounds)
        else:
            u = run_lm(pulse_unpack, u0_pulse, no_bounds)

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
