"""L-BFGS retrieval with **automatic-differentiation** gradients (``lbfgs-ad``).

The most feature-complete solver in the package. It shares the NLopt ``LD_LBFGS``
optimiser and the amplitude/phase parameterisation with :class:`croak.lbfgs.LBFGS`,
but takes its gradient from :func:`jax.value_and_grad` of the JAX forward model and
error metric (:mod:`croak.forward_jax`, :mod:`croak.metrics_jax`) rather than from
the hand-derived adjoint (:meth:`croak.forward.ForwardModel.adjoint_single` +
:meth:`croak.pulses.Pulse.ew_vjp`).

Differentiating the model automatically is what makes the extras possible, and
they are only available here and on the other autodiff solvers:

* ``smearing`` — the geometric time-smearing kernel (:mod:`croak.smearing`),
* ``fit_thickness`` / ``fit_tau0`` / ``fit_smearing`` — extra fitted scalars
  alongside the field,
* ``phase_basis="bspline"`` — the low-dimensional spline phase basis,
* ``reg_time`` / ``time_window`` — the temporal energy penalty.

Everything the plain ``lbfgs`` supports (phase-only, ``R_omega`` adaptive scaling,
amplitude/phase/spectral regularisation, thin or dispersive propagation) works
here too, so the two can also be run head-to-head on the same trace as a
cross-check of the hand-derived gradient.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import nlopt
import numpy as np
from numpy.typing import NDArray

from . import metrics_jax as mj
from ._jax_pulse import (
    ExtraParamSpec,
    augment,
    build_parameterisation,
    time_window_mask,
)
from .focal import FocalMixture
from .forward_jax import make_param_trace_fn
from .result import RetrievalResult
from .smearing import SmearingKernel
from .solver import (
    Retriever,
    assemble_result,
    guard_nonfinite,
    resolve_reg,
    spectral_target_amplitude,
    split_smear_value,
)

__all__ = ["LBFGSAD"]

Complex = NDArray[np.complex128]


class LBFGSAD(Retriever):
    r"""NLopt L-BFGS solver with JAX automatic-differentiation gradients.

    The forward model and FROG error are evaluated with the JAX twins
    (:mod:`croak.forward_jax`, :mod:`croak.metrics_jax`) and the gradient comes
    from :func:`jax.value_and_grad`. That makes this the most capable solver in
    the package: geometric smearing, the fitted extras (``fit_thickness``,
    ``fit_tau0``, ``fit_smearing``), the B-spline phase basis and the temporal
    penalty are all reachable here and not from the analytic-gradient solvers.

    The optimiser, parameterisation and objective otherwise match
    :class:`croak.lbfgs.LBFGS`, so running ``lbfgs`` and ``lbfgs-ad`` on the same
    trace also serves as a cross-check of the hand-derived gradient.

    Parameters
    ----------
    maxiters : int, optional
        Maximum number of objective evaluations (default ``300``).
    material, thickness, npoints, omega0, quadrature
        Dispersive-slab parameters, identical to :class:`croak.lbfgs.LBFGS`;
        ``material=None`` (default) is a thin medium. See
        :class:`croak.forward.ForwardModel`.
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
        crossing beams -- focal-spot evolution, beamlet walk-through, Gouy
        phase -- as a smooth per-depth factor on the generated signal; needed
        only when the slab is not much thinner than the beams' Rayleigh range.
        See :func:`croak.forward_jax.make_param_trace_fn`.
    phase_only : bool, optional
        Retrieve only the spectral phase, holding the guess amplitude fixed.
    phase_basis : {"pointwise", "bspline"}, optional
        Phase parameterisation. ``"pointwise"`` (default) optimises one phase
        value per frequency bin. ``"bspline"`` optimises the ``n_nodes`` control
        points of a cubic B-spline over the spectral support, a strong
        low-dimensional regulariser that implies phase-only retrieval (the
        amplitude is held at the guess). See
        :func:`croak._jax_pulse.make_spline_phase_basis`.
    n_nodes : int, optional
        Number of B-spline control points when ``phase_basis="bspline"``
        (default ``20``); ignored otherwise.
    R_omega : bool, optional
        Use per-frequency adaptive scaling factors instead of a single scalar.
    reg_amp : float or None, optional
        Amplitude second-difference smoothness weight. ``None`` (default)
        resolves to the settled production weight ``0.03``
        (:data:`croak.solver.REG_DEFAULTS`); pass ``0`` to disable.
    reg_phase : float, optional
        Phase second-difference smoothness weight (default ``0``).
        See :doc:`/howto/regularisation`.
    reg_spectrum : float or None, optional
        Weight of the spectral-match penalty pulling the retrieved spectral
        amplitude towards ``spectrum_target`` (full mode only). ``None``
        (default) resolves to the settled ``0.01`` and is inert without a
        ``spectrum_target``; pass ``0`` to disable.
    spectrum_target : array_like or None, optional
        Measured spectral **intensity** on the retrieval grid (centred order,
        e.g. ``TraceData.Iomega``) used by ``reg_spectrum``; ``None`` disables it.
    reg_time : float, optional
        Weight of a **temporal** penalty on the fraction of pulse energy falling
        outside ``time_window`` (``0`` = off). Suppresses spurious satellite
        pulses and pedestals far from the main feature. Not available in
        :class:`croak.lbfgs.LBFGS`.
    time_window : tuple of float or None, optional
        ``(t_lo, t_hi)`` in seconds bounding the region kept free of penalty for
        ``reg_time``. ``None`` (default) uses ``(delays.min(), delays.max())``.
        The window edges are Planck-tapered. See
        :func:`croak._jax_pulse.time_window_mask`.
    reltol, abstol : float, optional
        NLopt relative / absolute function tolerances (defaults ``1e-4`` /
        ``1e-8``).
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
        ``thickness > 0``) — only meaningful for PG/SD, where the depth-resolved
        interaction makes the trace depend on thickness beyond a pure input
        spectral phase. The prior ``thickness`` is the starting value. See
        :doc:`/howto/fitting_thickness_tau0`.
    fit_tau0 : bool, optional
        Also fit a **delay-zero offset** :math:`\tau_0` (default ``False``),
        correcting a mis-set experimental time zero. Works for every
        interaction; enters the forward model as ``exp(iω(τ − τ₀))``.
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
        Only meaningful when ``fit_thickness`` or ``fit_tau0`` is set.
    targeterr : float, optional
        Stop early when the FROG error drops below this value. Two
        robustness guards apply during optimization: a non-finite objective or
        gradient is replaced by a large finite penalty so the line search
        backtracks (:func:`croak.solver.guard_nonfinite`), and an internal
        NLopt failure (line-search breakdown, roundoff limit) returns the best
        iterate found instead of raising; both warn once.
    verbose : bool, optional
        Print per-evaluation progress.
    """

    name = "lbfgs-ad"

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
        depth_weight=None,
        phase_only: bool = False,
        phase_basis: str = "pointwise",
        n_nodes: int = 20,
        R_omega: bool = False,
        reg_amp: float | None = None,
        reg_phase: float = 0.0,
        reg_spectrum: float | None = None,
        spectrum_target=None,
        reg_time: float = 0.0,
        time_window: tuple[float, float] | None = None,
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
        targeterr: float = 0.0,
        verbose: bool = False,
    ):
        """Configure the JAX-AD L-BFGS retriever; see the class docstring."""
        super().__init__(maxiters=maxiters, verbose=verbose)
        self.material = material
        self.thickness = float(thickness)
        self.npoints = int(npoints)
        self.omega0 = float(omega0)
        self.quadrature = quadrature
        self.smearing = smearing
        self.focal = focal
        self.depth_weight = depth_weight
        self.phase_only = bool(phase_only)
        self.phase_basis = str(phase_basis)
        self.n_nodes = int(n_nodes)
        self.R_omega = bool(R_omega)
        self.reg_spectrum, self.reg_amp = resolve_reg(reg_spectrum, reg_amp, "gradient")
        self.reg_phase = float(reg_phase)
        self._spectral_target = spectral_target_amplitude(spectrum_target)
        self.reg_time = float(reg_time)
        self.time_window = time_window
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
        self.targeterr = float(targeterr)

    def _solve(
        self, grid, t_meas, delays, interaction, ew0, weights, rng=None, callback=None
    ) -> RetrievalResult:
        # Forward model differentiable in the extra parameters (thickness/tau0);
        # with both flags off it reduces exactly to the standard trace map.
        trace_param = make_param_trace_fn(
            grid.omega,
            delays,
            interaction,
            depth_weight=self.depth_weight,
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
        # Temporal-penalty mask (None when disabled); window defaults to the
        # measurement delay range.
        time_mask = None
        if self.reg_time > 0:
            lo, hi = self.time_window or (float(delays.min()), float(delays.max()))
            time_mask = time_window_mask(grid.t, lo, hi, grid.dt)
        par = build_parameterisation(
            ew0,
            grid.omega,
            phase_only=self.phase_only,
            phase_basis=self.phase_basis,
            n_nodes=self.n_nodes,
            spectral_target=self._spectral_target,
            time_mask=time_mask,
        )
        spectrum_fn, u0_pulse, pen = par.spectrum_fn, par.u0, par.penalties

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
        reg_amp, reg_phase, reg_spectrum, reg_time, R_omega = (
            self.reg_amp,
            self.reg_phase,
            self.reg_spectrum,
            self.reg_time,
            self.R_omega,
        )

        def make_objective(unpack):
            """Build the (total, R) objective for a given parameter unpacking."""

            def objective(u):
                u_pulse, thickness, tau0, smear = unpack(u)
                ew = spectrum_fn(u_pulse)
                t_sim = trace_param(ew, thickness, tau0, smear)
                ferr = mj.frog_error(tm, t_sim, wj, R_omega=R_omega)
                total = ferr
                # Penalties self-gate by mode/basis (return 0 where N/A); they act
                # on the pulse block only, so the extra scalars are unpenalised.
                if reg_amp > 0:
                    total = total + reg_amp * pen.amplitude(u_pulse)
                if reg_phase > 0:
                    total = total + reg_phase * pen.phase(u_pulse)
                if reg_spectrum > 0:
                    total = total + reg_spectrum * pen.spectral(u_pulse)
                if reg_time > 0:
                    total = total + reg_time * pen.temporal(u_pulse)
                return total, ferr

            return objective

        err_log: list[float] = []
        nan_state: dict = {}

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

        def run_lbfgs(unpack, u_start, lower_bounds):
            """Run one NLopt L-BFGS pass over ``u_start``; logs into ``err_log``.

            The best iterate is tracked so that an internal NLopt failure
            (line-search breakdown, roundoff limit) salvages the progress made
            instead of aborting the retrieval.
            """
            value_and_grad = jax.jit(
                jax.value_and_grad(make_objective(unpack), has_aux=True)
            )
            best_err = np.inf
            best_u = np.array(u_start, dtype=float, copy=True)

            def evaluate(u, grad):
                (total, ferr), g = value_and_grad(jnp.asarray(u))
                g_np = np.asarray(g, dtype=float)
                # A line-search step can probe an iterate where the jitted
                # objective goes non-finite; trap it so NLopt backtracks
                # instead of aborting (see solver.guard_nonfinite).
                penalty = guard_nonfinite(float(total), g_np, grad, nan_state)
                if penalty is not None:
                    return penalty
                ferr = float(ferr)
                err_log.append(ferr)
                nonlocal best_err, best_u
                if ferr < best_err:
                    best_err = ferr
                    best_u = np.array(u, dtype=float, copy=True)
                if self.verbose:
                    print(f"  eval {len(err_log):4d}: R = {ferr:.6e}")
                if callback is not None:
                    callback(
                        len(err_log),
                        ferr,
                        min(err_log),
                        snapshot=lambda: make_snapshot(*unpack(u)),
                    )
                if grad is not None and grad.size > 0:
                    grad[:] = g_np
                return float(total)

            opt = nlopt.opt(nlopt.LD_LBFGS, u_start.size)
            opt.set_min_objective(evaluate)
            opt.set_maxeval(self.maxiters)
            opt.set_ftol_rel(self.reltol)
            opt.set_ftol_abs(self.abstol)
            if np.any(np.isfinite(lower_bounds)):
                opt.set_lower_bounds(lower_bounds)
            if self.targeterr > 0:
                opt.set_stopval(self.targeterr)
            try:
                return opt.optimize(u_start)
            except (nlopt.RoundoffLimited, nlopt.runtime_error) as err:
                # NLopt's internal line search can fail outright (observed for
                # ~1 in 6 random starts on large dispersive grids); salvage the
                # best iterate rather than aborting the whole retrieval.
                warnings.warn(
                    f"NLopt stopped early ({type(err).__name__}); returning "
                    f"the best iterate found (R = {best_err:.3e})",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return best_u

        # Pulse-only unpack (extras held at their prior): used both when nothing
        # extra is fitted and as phase 1 of the two-phase polish.
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
            u_pulse = run_lbfgs(pulse_unpack, u0_pulse, no_bounds)
            u_seed = np.concatenate([u_pulse, np.zeros(aug.u0.size - u_pulse.size)])
            u = run_lbfgs(aug.unpack, u_seed, aug.lower_bounds)
        elif spec.any:
            u = run_lbfgs(aug.unpack, aug.u0, aug.lower_bounds)
        else:
            u = run_lbfgs(pulse_unpack, u0_pulse, no_bounds)

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
