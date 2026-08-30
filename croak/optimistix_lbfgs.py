"""L-BFGS retrieval via Optimistix (``lbfgs-optx``).

A JAX-native twin of :class:`croak.lbfgs_ad.LBFGSAD`: it minimises the **same**
scalar objective (FROG error plus the same amplitude / phase / spectral
penalties) over the same real parameter vector (:mod:`croak._jax_pulse`), with the
same JAX forward model (:mod:`croak.forward_jax`) and JAX autodiff gradient. The
**only** difference is the optimiser: instead of NLopt's ``LD_LBFGS`` driving a
host-side loop (with the gradient transferred device→host every evaluation),
Optimistix's :class:`optimistix.LBFGS` runs the limited-memory BFGS loop
**on-device** (each step jitted).

It is the natural counterpart to :class:`~croak.optimistix_lm.OptxLM` for
head-to-head comparison against the NLopt-backed ``lbfgs-ad``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import optimistix as optx

from . import metrics_jax as mj
from ._jax_pulse import build_parameterisation
from ._optimistix import run_optx
from .focal import FocalMixture
from .forward_jax import make_trace_fn
from .result import RetrievalResult
from .smearing import SmearingKernel
from .solver import Retriever, assemble_result, spectral_target_amplitude

__all__ = ["OptxLBFGS"]


class OptxLBFGS(Retriever):
    r"""Optimistix limited-memory BFGS solver with JAX autodiff gradients.

    The objective, parameterisation and options match
    :class:`croak.lbfgs_ad.LBFGSAD`; only the optimiser differs (Optimistix's
    on-device :class:`optimistix.LBFGS` in place of NLopt's host-side
    ``LD_LBFGS``).

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
        Smoothness / spectral-match penalties, identical to
        :class:`croak.lbfgs_ad.LBFGSAD`. See :doc:`/howto/regularisation`.
    reltol, abstol : float, optional
        Convergence tolerances (defaults ``1e-4`` / ``1e-8``). ``abstol`` is the
        Optimistix ``atol``; ``reltol`` is passed as the Optimistix ``rtol`` *and*
        drives a relative-improvement plateau stop in :func:`croak._optimistix.run_optx`
        (Optimistix's own Cauchy termination is ``atol``-dominated and effectively
        ignores ``rtol``, so the plateau stop is what makes ``reltol`` take effect).
    targeterr : float, optional
        Stop early when the FROG error drops below this value.
    verbose : bool, optional
        Print per-step progress.

    Notes
    -----
    Unlike :class:`croak.lbfgs_ad.LBFGSAD`, the temporal penalty
    (``reg_time`` / ``time_window``) is not exposed here.
    """

    name = "lbfgs-optx"

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
        reltol: float = 1e-4,
        abstol: float = 1e-8,
        targeterr: float = 0.0,
        verbose: bool = False,
    ):
        """Configure the Optimistix L-BFGS retriever (mirrors :class:`LBFGSAD`)."""
        super().__init__(maxiters=maxiters, verbose=verbose)
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
        self.reltol = float(reltol)
        self.abstol = float(abstol)
        self.targeterr = float(targeterr)

    def _solve(
        self, grid, t_meas, delays, interaction, ew0, weights, rng=None, callback=None
    ) -> RetrievalResult:
        """Minimise the scalar objective with Optimistix L-BFGS (module docs)."""
        trace_fn = make_trace_fn(
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
        )
        par = build_parameterisation(
            ew0,
            grid.omega,
            phase_only=self.phase_only,
            phase_basis=self.phase_basis,
            n_nodes=self.n_nodes,
            spectral_target=self._spectral_target,
        )
        spectrum_fn, u0, pen = par.spectrum_fn, par.u0, par.penalties

        tm = jnp.asarray(t_meas)
        wj = jnp.asarray(weights)
        reg_amp, reg_phase, reg_spectrum, R_omega = (
            self.reg_amp,
            self.reg_phase,
            self.reg_spectrum,
            self.R_omega,
        )

        def fn(u, args):
            """Total objective (FROG error + penalties); aux is the bare error R."""
            ew = spectrum_fn(u)
            t_sim = trace_fn(ew)
            ferr = mj.frog_error(tm, t_sim, wj, R_omega=R_omega)
            total = ferr
            if reg_amp > 0:
                total = total + reg_amp * pen.amplitude(u)
            if reg_phase > 0:
                total = total + reg_phase * pen.phase(u)
            if reg_spectrum > 0:
                total = total + reg_spectrum * pen.spectral(u)
            return total, ferr

        err_log: list[float] = []

        def make_snapshot(u) -> RetrievalResult:
            """Assemble an in-progress result from current params (live preview).

            Recomputes the spectrum and simulated trace for the current parameter
            vector and feeds them through the shared :func:`assemble_result`, so a
            live snapshot and the final result are built identically. Reused for
            the final return below.
            """
            spectrum = np.asarray(spectrum_fn(jnp.asarray(u)), dtype=complex)
            t_sim = np.asarray(trace_fn(jnp.asarray(spectrum)), dtype=float)
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

        solver = optx.LBFGS(rtol=self.reltol, atol=self.abstol)
        u, _ = run_optx(
            solver,
            fn,
            u0,
            maxiters=self.maxiters,
            targeterr=self.targeterr,
            reltol=self.reltol,
            callback=callback,
            make_snapshot=make_snapshot,
            err_log=err_log,
            verbose=self.verbose,
        )

        return make_snapshot(u)
