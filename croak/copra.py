r"""COPRA step-based retrieval.

Implements the Common Pulse Retrieval Algorithm (COPRA), reformulated so that
the frequency-domain signal
:math:`\psi(\omega)` produced by :class:`~croak.forward.ForwardModel` is the
working quantity. The single reverse-mode adjoint kernel of the forward model
provides every gradient, so the same engine serves both the thin-medium and the
dispersive case (the only difference is the forward model the engine is handed).

COPRA alternates two modes:

* **local** — sweep the delays in random order, taking a per-delay gradient step
  on :math:`\tilde E` towards the amplitude-projected signal;
* **global** — once the local mode stalls, take one joint gradient-descent step
  in the signal and then in :math:`\tilde E` over all delays at once.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .forward import ForwardModel
from .grid import Grid
from .interactions import get_interaction
from .metrics import compute_mu, compute_mu_per_freq, compute_R, compute_r
from .result import RetrievalResult
from .solver import Retriever, assemble_result

__all__ = ["COPRA", "run_copra"]

Complex = NDArray[np.complex128]

#: Look-back window (iterations) for the COPRA relative-improvement stop. The
#: error is noisy and the local→global switch needs a few stalled iterations, so
#: ``reltol`` only triggers once the *best* error has plateaued over this window.
_COPRA_PATIENCE = 10


def _project(psi: Complex, measured: NDArray[np.float64]) -> Complex:
    r"""Amplitude projection: replace :math:`|\psi|` with :math:`\sqrt{measured}`.

    Negative ``measured`` values (e.g. from noisy preprocessing) are clamped to
    zero; where ``|psi| = 0`` the phase is taken as zero.
    """
    amp = np.abs(psi)
    target = np.sqrt(np.maximum(measured, 0.0))
    out = np.empty_like(psi)
    nz = amp > 0
    out[nz] = psi[nz] / amp[nz] * target[nz]
    out[~nz] = target[~nz]
    return out


def run_copra(
    model: ForwardModel,
    grid: Grid,
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
    snapshot_fn=None,
) -> tuple[
    Complex, list[float], NDArray[np.float64], NDArray[np.float64] | float, float
]:
    """Run the COPRA iteration. Returns ``(spectrum, errors, trace, mu, best_R)``.

    ``snapshot_fn``, when given, is ``(ew, t_sim, errors) -> RetrievalResult``:
    it turns the current iterate into an in-progress result, handed to
    ``callback`` as its ``snapshot=`` argument so the GUI can draw a live full
    plot. Both arguments are already computed each iteration for the error
    bookkeeping, so a snapshot costs no extra forward pass.

    If ``fixed_amplitude`` is given, the spectral amplitude is held fixed to it
    after every update (phase-only retrieval): ``E ← fixed_amplitude · e^{iφ}``.

    If ``R_omega`` is set, the optimal scale ``mu`` is computed per frequency
    (one factor per spectral row) instead of as a single global value, matching
    the ``Rω`` adaptive-scaling mode of the gradient solvers. The returned
    ``mu`` is then a length-``Nomega`` vector.

    Parameters
    ----------
    model : ForwardModel
        Configured forward model (thin or dispersive).
    grid : Grid
        Time/frequency grid.
    t_meas : numpy.ndarray
        Measured trace ``(Nomega, Ndelay)``, normalised to unit maximum.
    ew0 : numpy.ndarray
        Initial complex spectrum (centred order).
    weights : numpy.ndarray
        Per-frequency weights (length ``Nomega``).
    maxiters : int, optional
        Maximum number of iterations.
    alpha : float, optional
        Global-mode step-size scaling.
    reltol : float, optional
        Relative-improvement convergence tolerance (``0`` disables). Iteration
        stops once the best FROG error has improved by less than ``reltol``
        (relative) over the last :data:`_COPRA_PATIENCE` iterations — i.e. the
        run has plateaued (including after the global-mode switch).
    abstol : float, optional
        Absolute target FROG error (``0`` disables): stop once the best error
        drops to or below ``abstol``.
    verbose : bool, optional
        Print per-iteration FROG error.
    """
    delays = model.delays
    n, m = model.n, model.m
    wmat = np.broadcast_to(weights[:, None], (n, m))
    w2 = weights**2
    wzero = weights == 0.0

    ew = ew0.astype(complex).copy()
    t_sim = np.empty((n, m))
    psi_all = np.empty((n, m), dtype=complex)

    def impose(spectrum: Complex) -> Complex:
        """Project onto the fixed amplitude (phase-only mode); identity otherwise."""
        if fixed_amplitude is None:
            return spectrum
        return (fixed_amplitude * np.exp(1j * np.angle(spectrum))).astype(complex)

    ew = impose(ew)

    def forward_all() -> None:
        for j, tau in enumerate(delays):
            psi = model.signal_single(ew, tau)
            psi_all[:, j] = psi
            t_sim[:, j] = np.abs(psi) ** 2

    def scale(t_sim_: NDArray[np.float64]) -> NDArray[np.float64] | float:
        """Optimal scale: a length-``n`` vector (Rω) or a scalar (global)."""
        if R_omega:
            return compute_mu_per_freq(t_meas, t_sim_, weights, freq_axis=0)
        return compute_mu(t_meas, t_sim_, wmat)

    def errors() -> tuple[NDArray[np.float64] | float, float]:
        mu = scale(t_sim)
        # Rω returns a per-frequency vector (broadcast as a column); global a scalar.
        mu_b = mu[:, None] if isinstance(mu, np.ndarray) else mu
        R = compute_R(compute_r(t_meas, t_sim, wmat, mu_b), t_meas, wmat)
        return mu, R

    forward_all()
    mu, R = errors()

    def per_delay_gradient(j: int, tau: float) -> tuple[Complex, float, float]:
        """Projected per-delay step: returns (grad_Ew, Zm, |grad|**2)."""
        psi = model.signal_single(ew, tau, record=True)
        psi_all[:, j] = psi
        t_sim[:, j] = np.abs(psi) ** 2
        # Rescale the measured intensity into the simulated signal's units before
        # the amplitude projection. In Rω mode ``mu`` is a per-frequency vector and
        # is exactly zero on any frequency row whose *measured* trace is identically
        # zero across all delays (out-of-band rows) while the model still has signal
        # there (see compute_mu_per_freq). There ``t_meas / mu`` is 0/0 = NaN, which
        # the FFT-based adjoint then smears across every frequency and destroys the
        # whole retrieval. Guard the degenerate rows with a unit denominator; since
        # the measured intensity there is ~0, the target is ~0 and the projection
        # correctly drives |psi| towards zero at those frequencies.
        mu_safe = np.where(mu > 0.0, mu, 1.0)
        measured = t_meas[:, j] / mu_safe
        if np.any(wzero):  # zero-weight bins: target the model so they contribute 0
            measured = measured.copy()
            measured[wzero] = t_sim[wzero, j]
        sprime = _project(psi, measured)
        # Zm = sum |sprime - psi|^2 ; dZm/dpsi* = psi - sprime
        grad = 2.0 * model.adjoint_single(psi - sprime, tau)
        zm = float(np.sum(np.abs(sprime - psi) ** 2))
        return grad, zm, float(np.sum(np.abs(grad) ** 2))

    # initial maximum gradient norm across delays
    current_max_grad = 0.0
    for j, tau in enumerate(delays):
        _, _, gn = per_delay_gradient(j, tau)
        current_max_grad = max(current_max_grad, gn)

    best_R = R
    best_ew = ew.copy()
    best_log: list[float] = []  # best_R after each iteration (for the reltol stop)
    stalled = 0
    mode = "local"
    err_log: list[float] = []
    rng = np.random.default_rng() if rng is None else rng

    if verbose:
        print(f"COPRA: initial R = {R:.6e}")

    for it in range(maxiters):
        if stalled >= 5:
            mode = "global"
        prev_max_grad = current_max_grad
        current_max_grad = 0.0

        if mode == "local":
            for j in rng.permutation(m):
                grad, zm, gn = per_delay_gradient(int(j), float(delays[int(j)]))
                current_max_grad = max(current_max_grad, gn)
                gamma = zm / max(current_max_grad, prev_max_grad, np.finfo(float).tiny)
                ew = impose(ew - gamma * grad)
            # recompute the full trace so R / best-tracking / stall detection use
            # the *true* error of the swept spectrum (not the incremental one,
            # whose columns were computed with stale Ew during the sweep)
            forward_all()
            mu, R = errors()
        else:
            forward_all()
            mu, R = errors()
            mu_b = mu[:, None] if isinstance(mu, np.ndarray) else mu
            # 1) descent in the signal: grad_r psi = -4 mu (T_meas - mu T_sim) w^2 psi
            grad_r = -4.0 * mu_b * (t_meas - mu_b * t_sim) * (w2[:, None]) * psi_all
            eta_r = (
                alpha
                * compute_r(t_meas, t_sim, wmat, mu_b)
                / max(float(np.sum(np.abs(grad_r) ** 2)), np.finfo(float).tiny)
            )
            # astype(complex): psi_all is complex, but the real-valued mu_b/weights
            # in grad_r make pyright infer a float array for the difference.
            psi_target = (psi_all - eta_r * grad_r).astype(complex)
            # 2) descent in Ew towards the updated signal
            grad_ew = np.zeros(n, dtype=complex)
            z_total = 0.0
            for j, tau in enumerate(delays):
                model.signal_single(ew, tau, record=True)
                grad_ew += 2.0 * model.adjoint_single(
                    psi_all[:, j] - psi_target[:, j], tau
                )
                z_total += float(np.sum(np.abs(psi_target[:, j] - psi_all[:, j]) ** 2))
            eta_z = (
                alpha
                * z_total
                / max(float(np.sum(np.abs(grad_ew) ** 2)), np.finfo(float).tiny)
            )
            ew = impose(ew - eta_z * grad_ew)
            forward_all()
            mu, R = errors()

        if R < best_R:
            best_R, best_ew, stalled = R, ew.copy(), 0
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
                # Copy both: `t_sim` is filled in place by `forward_all`, and
                # `ew` is rebound every iteration, so a deferred caller would
                # otherwise be handed whatever the sweep had reached by then.
                ew_now, t_now = ew.copy(), t_sim.copy()
                callback(
                    it + 1,
                    R,
                    best_R,
                    snapshot=lambda e=ew_now, t=t_now: snapshot_fn(e, t, list(err_log)),
                )

        # convergence: absolute target, or a plateau of the best error
        if abstol > 0.0 and best_R <= abstol:
            if verbose:
                print(f"COPRA: reached abstol ({abstol:.2e}) at iter {it + 1}")
            break
        if reltol > 0.0 and it >= _COPRA_PATIENCE:
            past = best_log[it - _COPRA_PATIENCE]
            if past - best_R <= reltol * max(past, np.finfo(float).tiny):
                if verbose:
                    print(
                        f"COPRA: best error plateaued (reltol {reltol:.2e}) "
                        f"at iter {it + 1}"
                    )
                break

    # final trace and error from the best spectrum (consistent with best_ew)
    for j, tau in enumerate(delays):
        t_sim[:, j] = np.abs(model.signal_single(best_ew, tau)) ** 2
    final_mu = scale(t_sim)
    final_mu_b = final_mu[:, None] if isinstance(final_mu, np.ndarray) else final_mu
    final_R = compute_R(compute_r(t_meas, t_sim, wmat, final_mu_b), t_meas, wmat)
    return best_ew, err_log, final_mu_b * t_sim, final_mu, final_R


class COPRA(Retriever):
    """COPRA step-based retrieval solver (optionally with dispersive propagation).

    The same engine handles the thin medium and the dispersive case — set
    ``material``/``thickness`` to propagate the field through a slab (depth
    integral over ``npoints`` quadrature nodes). With ``material=None`` (the
    default) it is the plain thin-medium algorithm.

    Parameters
    ----------
    maxiters : int, optional
        Maximum iterations (default ``100``).
    alpha : float, optional
        Global-mode step-size scaling (default ``0.25``).
    reltol : float, optional
        Relative-improvement convergence tolerance (default ``0`` = run all
        ``maxiters``); see :func:`run_copra`.
    abstol : float, optional
        Absolute target FROG error (default ``0`` = off); see :func:`run_copra`.
    material : str or None, optional
        Slab material for dispersive propagation (e.g. ``"SiO2"``); ``None`` for
        a thin medium.
    thickness : float, optional
        Slab thickness (m).
    npoints : int, optional
        Number of depth quadrature points (default ``20``).
    omega0 : float, optional
        Carrier angular frequency (rad/s) used to build the dispersion.
    quadrature : str, optional
        Quadrature method (``"gausslegendre"`` or ``"uniform"``).
    phase_only : bool, optional
        Retrieve only the spectral phase, holding the guess amplitude fixed.
    R_omega : bool, optional
        Use per-frequency adaptive scaling factors instead of a single global
        ``mu`` (the ``Rω`` mode).
    verbose : bool, optional
        Print per-iteration progress.
    """

    name = "copra"

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
    ):
        """Configure the COPRA retriever (see the class docstring for parameters)."""
        super().__init__(maxiters=maxiters, verbose=verbose)
        self.alpha = float(alpha)
        self.reltol = float(reltol)
        self.abstol = float(abstol)
        self.material = material
        self.thickness = float(thickness)
        self.npoints = int(npoints)
        self.omega0 = float(omega0)
        self.quadrature = quadrature
        self.phase_only = bool(phase_only)
        self.R_omega = bool(R_omega)

    def _build_model(self, grid, delays, interaction) -> ForwardModel:
        return ForwardModel(
            grid.omega,
            delays,
            interaction,
            material=self.material,
            thickness=self.thickness,
            npoints=self.npoints,
            omega0=self.omega0,
            quadrature=self.quadrature,
        )

    def _solve(
        self, grid, trace, delays, interaction, ew0, weights, rng, callback=None
    ) -> RetrievalResult:
        model = self._build_model(grid, delays, interaction)
        fixed_amp = np.abs(ew0) if self.phase_only else None

        def snapshot(ew_now, t_now, errors_so_far) -> RetrievalResult:
            """Assemble an in-progress result from the current iterate.

            Routed through the shared :func:`~croak.solver.assemble_result` so a
            live snapshot and the final result compute mu and the error the same
            way.
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

        spectrum, errors, sim_trace, mu, best_R = run_copra(
            model,
            grid,
            trace,
            ew0,
            weights,
            maxiters=self.maxiters,
            alpha=self.alpha,
            reltol=self.reltol,
            abstol=self.abstol,
            verbose=self.verbose,
            rng=rng,
            callback=callback,
            fixed_amplitude=fixed_amp,
            R_omega=self.R_omega,
            snapshot_fn=snapshot,
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
