"""L-BFGS retrieval via NLopt with hand-derived gradients.

Minimises the FROG error :math:`R(u)` over a pulse parameterisation ``u``
(:class:`~croak.pulses.ArrayPulse`) using NLopt's ``LD_LBFGS``. The gradient is
analytic: the FROG-error cotangent on the trace is propagated through
:meth:`~croak.forward.ForwardModel.adjoint_single` and then through the pulse's
:meth:`~croak.pulses.Pulse.ew_vjp` — no automatic differentiation.

Optional features: per-frequency adaptive scaling (``R_omega``), phase-only
retrieval, and second-difference smoothness regularisation of the spectral
amplitude and phase.
"""

from __future__ import annotations

import warnings

import nlopt
import numpy as np
from numpy.typing import NDArray

from .forward import ForwardModel
from .metrics import compute_mu, compute_mu_per_freq
from .pulses import ArrayPulse
from .result import RetrievalResult
from .solver import (
    Retriever,
    assemble_result,
    guard_nonfinite,
    resolve_reg,
    spectral_target_amplitude,
)

__all__ = ["LBFGS"]

Complex = NDArray[np.complex128]


class LBFGS(Retriever):
    """NLopt L-BFGS solver with analytic gradients.

    Parameters
    ----------
    maxiters : int, optional
        Maximum number of objective evaluations (default ``300``).
    material : str or None, optional
        Dispersive slab material; ``None`` (default) is a thin medium.
    thickness : float, optional
        Slab thickness (m).
    npoints : int, optional
        Depth quadrature points (default ``20``).
    omega0 : float, optional
        Carrier angular frequency (rad/s) for the dispersion.
    quadrature : str, optional
        Quadrature method.
    phase_only : bool, optional
        Retrieve only the spectral phase, holding the guess amplitude fixed.
    R_omega : bool, optional
        Use per-frequency adaptive scaling factors.
    reg_amp : float or None, optional
        Amplitude second-difference smoothness weight. ``None`` (default)
        resolves to the settled production weight ``0.03``
        (:data:`croak.solver.REG_DEFAULTS`); pass ``0`` to disable.
    reg_phase : float, optional
        Phase second-difference smoothness weight (default ``0``).
    reg_spectrum : float or None, optional
        Weight of the spectral-match penalty pulling the retrieved spectral
        amplitude towards ``spectrum_target`` (full mode only). ``None``
        (default) resolves to the settled ``0.01`` and is inert without a
        ``spectrum_target``; pass ``0`` to disable.
    spectrum_target : array_like or None, optional
        Measured spectral **intensity** on the retrieval grid (centred order,
        e.g. ``TraceData.Iomega``) used by ``reg_spectrum``; ``None`` disables
        the penalty.
    reltol, abstol : float, optional
        NLopt relative/absolute function tolerances (defaults ``1e-4`` / ``1e-8``).
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

    name = "lbfgs"

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
        reg_amp: float | None = None,
        reg_phase: float = 0.0,
        reg_spectrum: float | None = None,
        spectrum_target=None,
        reltol: float = 1e-4,
        abstol: float = 1e-8,
        targeterr: float = 0.0,
        verbose: bool = False,
    ):
        """Configure the L-BFGS retriever (see the class docstring for parameters)."""
        super().__init__(maxiters=maxiters, verbose=verbose)
        self.material = material
        self.thickness = float(thickness)
        self.npoints = int(npoints)
        self.omega0 = float(omega0)
        self.quadrature = quadrature
        self.phase_only = bool(phase_only)
        self.R_omega = bool(R_omega)
        self.reg_spectrum, self.reg_amp = resolve_reg(reg_spectrum, reg_amp, "gradient")
        self.reg_phase = float(reg_phase)
        # Peak-normalised target amplitude |E(ω)| from the measured spectral
        # intensity, compared against the retrieved amplitude (None = disabled).
        self._spectral_target = spectral_target_amplitude(spectrum_target)
        self.reltol = float(reltol)
        self.abstol = float(abstol)
        self.targeterr = float(targeterr)

    def _solve(
        self, grid, t_meas, delays, interaction, ew0, weights, rng=None, callback=None
    ) -> RetrievalResult:
        model = ForwardModel(
            grid.omega,
            delays,
            interaction,
            material=self.material,
            thickness=self.thickness,
            npoints=self.npoints,
            omega0=self.omega0,
            quadrature=self.quadrature,
        )
        pulse = ArrayPulse.from_spectrum(grid, ew0, phase_only=self.phase_only)
        n, m = model.n, model.m
        wmat = np.broadcast_to(weights[:, None], (n, m))
        denom = t_meas.size * np.max(t_meas * weights[:, None]) ** 2

        psi_all = np.empty((n, m), dtype=complex)
        t_sim = np.empty((n, m))
        mu_buf = np.ones(n)
        err_log: list[float] = []

        def make_snapshot(ew_now: Complex) -> RetrievalResult:
            """Assemble an in-progress result from the current spectrum (live preview).

            Uses a *fresh* trace buffer and read-only forward evaluations so it
            never disturbs the shared gradient buffers (``psi_all``/``t_sim``/
            ``pulse``) that :func:`evaluate` reuses after the callback. Feeds the
            shared :func:`assemble_result`, identical to the final return below.
            """
            ts = np.empty((n, m))
            for j, tau in enumerate(delays):
                ts[:, j] = np.abs(model.signal_single(ew_now, tau)) ** 2
            return assemble_result(
                spectrum=np.array(ew_now, dtype=complex),
                t_sim=ts,
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

        nan_state: dict = {}
        best_err = np.inf
        best_u = pulse.get_params().copy()

        def evaluate(u: NDArray[np.float64], grad: NDArray[np.float64] | None) -> float:
            pulse.set_params(u)
            ew = pulse.spectrum()
            for j, tau in enumerate(delays):
                psi = model.signal_single(ew, tau)
                psi_all[:, j] = psi
                t_sim[:, j] = np.abs(psi) ** 2

            if self.R_omega:
                mu_buf[:] = compute_mu_per_freq(t_meas, t_sim, weights, freq_axis=0)
            else:
                mu_buf[:] = compute_mu(t_meas, t_sim, wmat)

            resid = (t_meas - mu_buf[:, None] * t_sim) * weights[:, None]
            ferr = float(np.sqrt(np.sum(resid**2) / denom))
            total = ferr
            penalty = guard_nonfinite(total, None, grad, nan_state)
            if penalty is not None:
                return penalty
            if self.reg_amp > 0:
                total += self.reg_amp * pulse.amplitude_penalty(u)
            if self.reg_phase > 0:
                total += self.reg_phase * pulse.phase_penalty(u)
            if self.reg_spectrum > 0 and self._spectral_target is not None:
                total += self.reg_spectrum * pulse.spectral_penalty(
                    u, self._spectral_target
                )

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
                    snapshot=lambda: make_snapshot(ew),
                )

            if grad is not None and grad.size > 0:
                # cotangent on T_sim: dR/dT_sim = -mu w resid / (R denom)
                if ferr > 0:
                    cot_t = -mu_buf[:, None] * weights[:, None] * resid / (ferr * denom)
                else:
                    cot_t = np.zeros_like(t_sim)
                ew_bar = np.zeros(n, dtype=complex)
                for j, tau in enumerate(delays):
                    model.signal_single(ew, tau, record=True)
                    # d/dpsi* of |psi|^2 = psi; asarray pins the complex dtype that
                    # the real-valued cot_t makes pyright lose.
                    psi_bar = np.asarray(cot_t[:, j] * psi_all[:, j], dtype=complex)
                    ew_bar += model.adjoint_single(psi_bar, tau)
                grad[:] = pulse.ew_vjp(ew_bar)
                if self.reg_amp > 0:
                    grad[:] += self.reg_amp * pulse.amplitude_penalty_grad(u)
                if self.reg_phase > 0:
                    grad[:] += self.reg_phase * pulse.phase_penalty_grad(u)
                if self.reg_spectrum > 0 and self._spectral_target is not None:
                    grad[:] += self.reg_spectrum * pulse.spectral_penalty_grad(
                        u, self._spectral_target
                    )
                penalty = guard_nonfinite(total, grad, grad, nan_state)
                if penalty is not None:
                    return penalty
            return total

        u0 = pulse.get_params()
        opt = nlopt.opt(nlopt.LD_LBFGS, u0.size)
        opt.set_min_objective(evaluate)
        opt.set_maxeval(self.maxiters)
        opt.set_ftol_rel(self.reltol)
        opt.set_ftol_abs(self.abstol)
        if self.targeterr > 0:
            opt.set_stopval(self.targeterr)
        try:
            u = opt.optimize(u0)
        except (nlopt.RoundoffLimited, nlopt.runtime_error) as err:
            # Salvage the best iterate on an internal NLopt failure instead of
            # aborting the retrieval (see croak.lbfgs_ad for the twin case).
            warnings.warn(
                f"NLopt stopped early ({type(err).__name__}); returning the "
                f"best iterate found (R = {best_err:.3e})",
                RuntimeWarning,
                stacklevel=2,
            )
            u = best_u

        # Assemble the final result exactly as the live snapshots (shared
        # :func:`make_snapshot` / `assemble_result`).
        pulse.set_params(u)
        return make_snapshot(pulse.spectrum())
