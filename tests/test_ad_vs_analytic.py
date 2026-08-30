"""Verify the hand-derived analytic gradients with JAX automatic differentiation.

The analytic L-BFGS gradient (assembled from
:meth:`croak.forward.ForwardModel.adjoint_single` and
:meth:`croak.pulses.Pulse.ew_vjp`, exactly as in :class:`croak.lbfgs.LBFGS`) is
compared against :func:`jax.grad` of the *same* FROG-error objective built on the
JAX forward model (:mod:`croak.forward_jax`) and metrics (:mod:`croak.metrics_jax`).

This is an independent check of the adjoints — complementary to, and not a
replacement for, the finite-difference tests in ``test_forward.py``,
``test_interactions.py`` and ``test_pulses.py``, which remain. AD has no
step-size/truncation error, so the agreement here is to ~1e-6 (vs ~1e-4 for FD).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from croak import metrics_jax as mj
from croak._jax_pulse import make_param_grad_fns, make_spectrum_fn
from croak.forward import ForwardModel
from croak.forward_jax import make_forward_adjoint_fns, make_trace_fn
from croak.grid import Grid
from croak.maths import wlfreq
from croak.metrics import compute_mu, compute_mu_per_freq
from croak.pulses import ArrayPulse, gaussian_pulse


def _analytic_grad(
    model,
    pulse,
    t_meas,
    weights,
    *,
    R_omega,
    reg_amp,
    reg_phase,
    reg_spectrum=0.0,
    spec_target=None,
):
    """FROG error and its analytic gradient, exactly as :class:`croak.lbfgs.LBFGS`."""
    u = pulse.get_params()
    ew = pulse.spectrum()
    n, m = model.n, model.m
    wmat = np.broadcast_to(weights[:, None], (n, m))
    denom = t_meas.size * np.max(t_meas * weights[:, None]) ** 2

    psi_all = np.empty((n, m), complex)
    t_sim = np.empty((n, m))
    for j, tau in enumerate(model.delays):
        psi = model.signal_single(ew, tau)
        psi_all[:, j] = psi
        t_sim[:, j] = np.abs(psi) ** 2

    if R_omega:
        mu = compute_mu_per_freq(t_meas, t_sim, weights, freq_axis=0)[:, None]
    else:
        mu = compute_mu(t_meas, t_sim, wmat)
    resid = (t_meas - mu * t_sim) * weights[:, None]
    ferr = float(np.sqrt(np.sum(resid**2) / denom))

    cot_t = -mu * weights[:, None] * resid / (ferr * denom)
    ew_bar = np.zeros(n, complex)
    for j, tau in enumerate(model.delays):
        model.signal_single(ew, tau, record=True)
        ew_bar += model.adjoint_single(cot_t[:, j] * psi_all[:, j], tau)
    grad = pulse.ew_vjp(ew_bar)
    if reg_amp > 0:
        grad = grad + reg_amp * pulse.amplitude_penalty_grad(u)
    if reg_phase > 0:
        grad = grad + reg_phase * pulse.phase_penalty_grad(u)
    if reg_spectrum > 0 and spec_target is not None:
        grad = grad + reg_spectrum * pulse.spectral_penalty_grad(u, spec_target)
    return ferr, grad


def _ad_grad(
    grid,
    delays,
    interaction,
    ew0,
    t_meas,
    weights,
    *,
    model_kw,
    phase_only,
    R_omega,
    reg_amp,
    reg_phase,
    reg_spectrum=0.0,
    spec_target=None,
):
    """FROG error and its AD gradient, exactly as :class:`croak.lbfgs_ad.LBFGSAD`."""
    trace_fn = make_trace_fn(grid.omega, delays, interaction, **model_kw)
    spectrum_fn, u0, nidcs = make_spectrum_fn(
        np.abs(ew0), np.angle(ew0), phase_only=phase_only
    )
    tmeas = jnp.asarray(t_meas)
    wj = jnp.asarray(weights)
    s_sup = None if spec_target is None else jnp.asarray(spec_target)

    def objective(u):
        # Returns the regularised total (differentiated) plus the bare FROG
        # error as aux, matching the value/grad split in croak.lbfgs / lbfgs_ad.
        t_sim = trace_fn(spectrum_fn(u))
        ferr = mj.frog_error(tmeas, t_sim, wj, R_omega=R_omega)
        total = ferr
        if reg_amp > 0 and not phase_only:
            total = total + reg_amp * mj.amplitude_penalty(u[:nidcs])
        if reg_phase > 0:
            phase_part = u if phase_only else u[nidcs:]
            total = total + reg_phase * mj.phase_penalty(phase_part)
        if reg_spectrum > 0 and s_sup is not None and not phase_only:
            total = total + reg_spectrum * mj.spectral_penalty(u[:nidcs], s_sup)
        return total, ferr

    (_, ferr), grad = jax.value_and_grad(objective, has_aux=True)(jnp.asarray(u0))
    return float(ferr), np.asarray(grad)


def _hand_jax_grad(
    grid,
    delays,
    interaction,
    ew0,
    t_meas,
    weights,
    *,
    model_kw,
    phase_only,
    R_omega,
    reg_amp,
    reg_phase,
    reg_spectrum=0.0,
    spec_target=None,
):
    """FROG error and hand-derived gradient (see :class:`croak.lbfgs_hand.LBFGSHand`).

    Exercises the JAX forward/adjoint twin and parameter-space VJP directly,
    backpropagating the FROG-error cotangent with **no automatic differentiation**.
    """
    signal_tape, adjoint = make_forward_adjoint_fns(
        grid.omega, delays, interaction, **model_kw
    )
    spectrum_fn, u0, nidcs = make_spectrum_fn(
        np.abs(ew0), np.angle(ew0), phase_only=phase_only
    )
    pg = make_param_grad_fns(np.abs(ew0), np.angle(ew0), phase_only=phase_only)
    delays_j = jnp.asarray(np.asarray(delays, dtype=float))
    tm = jnp.asarray(t_meas)
    wj = jnp.asarray(weights)
    denom = float(t_meas.size * np.max(t_meas * weights[:, None]) ** 2)
    s_sup = None if spec_target is None else jnp.asarray(spec_target)

    u = jnp.asarray(u0)
    ew = spectrum_fn(u)
    psis, tests, gates = jax.vmap(lambda tau: signal_tape(ew, tau))(delays_j)
    psi_all = psis.T
    t_sim = jnp.abs(psi_all) ** 2
    mu = (
        mj.mu_per_freq(tm, t_sim, wj)[:, None]
        if R_omega
        else mj.mu_global(tm, t_sim, wj)
    )
    resid = (tm - mu * t_sim) * wj[:, None]
    ferr = jnp.sqrt(jnp.sum(resid**2) / denom)
    cot_t = -mu * wj[:, None] * resid / (ferr * denom)
    psi_bar = cot_t * psi_all
    ew_bars = jax.vmap(lambda pb, te, ga, tau: adjoint(pb, te, ga, tau))(
        psi_bar.T, tests, gates, delays_j
    )
    grad = pg.ew_vjp(u, jnp.sum(ew_bars, axis=0))
    if reg_amp > 0 and not phase_only:
        grad = grad + reg_amp * pg.amplitude_penalty_grad(u)
    if reg_phase > 0:
        grad = grad + reg_phase * pg.phase_penalty_grad(u)
    if reg_spectrum > 0 and s_sup is not None and not phase_only:
        grad = grad + reg_spectrum * pg.spectral_penalty_grad(u, s_sup)
    return float(ferr), np.asarray(grad)


CASES = [
    # (interaction, dispersive, phase_only, reg_amp, reg_phase, reg_spectrum)
    ("pg", False, False, 0.0, 0.0, 0.0),
    ("shg", False, False, 0.0, 0.0, 0.0),
    ("sd", False, False, 0.0, 0.0, 0.0),
    ("pg", True, False, 0.0, 0.0, 0.0),
    ("pg", False, True, 0.0, 0.0, 0.0),
    ("pg", False, False, 0.05, 0.02, 0.0),
    ("pg", False, False, 0.0, 0.0, 0.3),  # spectral-match only
    ("shg", False, False, 0.05, 0.02, 0.3),  # all three penalties together
]


@pytest.mark.parametrize(
    "interaction,dispersive,phase_only,reg_amp,reg_phase,reg_spectrum", CASES
)
def test_ad_matches_analytic_gradient(
    interaction, dispersive, phase_only, reg_amp, reg_phase, reg_spectrum
):
    g = Grid(64, dt=0.4e-15)
    ew_true = gaussian_pulse(g, 2.5e-15, phases=[3e-30])
    delays = np.linspace(-15e-15, 15e-15, 41)

    model_kw = {}
    if dispersive:
        model_kw = dict(
            material="SiO2", thickness=12e-6, npoints=12, omega0=wlfreq(800e-9)
        )

    model = ForwardModel(g.omega, delays, interaction, **model_kw)
    t_meas = model.trace(ew_true, normalize=True)
    weights = np.ones(g.n)

    # a perturbed guess so the gradient is non-trivial
    ew0 = gaussian_pulse(g, 1.8e-15, phases=[1e-30])
    pulse = ArrayPulse.from_spectrum(g, ew0, phase_only=phase_only)

    # spectral-match target: peak-normalised amplitude of the true spectrum
    spec_target = None
    if reg_spectrum > 0:
        spec_target = np.abs(ew_true) / np.max(np.abs(ew_true))

    R_omega = False
    fa, ga = _analytic_grad(
        model,
        pulse,
        t_meas,
        weights,
        R_omega=R_omega,
        reg_amp=reg_amp,
        reg_phase=reg_phase,
        reg_spectrum=reg_spectrum,
        spec_target=spec_target,
    )
    fb, gb = _ad_grad(
        g,
        delays,
        interaction,
        ew0,
        t_meas,
        weights,
        model_kw=model_kw,
        phase_only=phase_only,
        R_omega=R_omega,
        reg_amp=reg_amp,
        reg_phase=reg_phase,
        reg_spectrum=reg_spectrum,
        spec_target=spec_target,
    )
    fc, gc = _hand_jax_grad(
        g,
        delays,
        interaction,
        ew0,
        t_meas,
        weights,
        model_kw=model_kw,
        phase_only=phase_only,
        R_omega=R_omega,
        reg_amp=reg_amp,
        reg_phase=reg_phase,
        reg_spectrum=reg_spectrum,
        spec_target=spec_target,
    )

    assert fa == pytest.approx(fb, rel=1e-9)
    assert np.allclose(ga, gb, rtol=1e-6, atol=1e-8)
    # The jitted hand gradient is the *same* maths as the numpy analytic one,
    # so it agrees far more tightly than the AD comparison (only FFT rounding).
    assert fa == pytest.approx(fc, rel=1e-9)
    assert np.allclose(ga, gc, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("interaction", ["pg", "shg"])
def test_ad_matches_analytic_gradient_r_omega(interaction):
    """The per-frequency-scaling (``R_omega``) gradient also matches AD."""
    g = Grid(64, dt=0.4e-15)
    ew_true = gaussian_pulse(g, 2.5e-15, phases=[3e-30])
    delays = np.linspace(-15e-15, 15e-15, 41)
    model = ForwardModel(g.omega, delays, interaction)
    t_meas = model.trace(ew_true, normalize=True)
    weights = np.ones(g.n)
    ew0 = gaussian_pulse(g, 1.8e-15, phases=[1e-30])
    pulse = ArrayPulse.from_spectrum(g, ew0)

    fa, ga = _analytic_grad(
        model, pulse, t_meas, weights, R_omega=True, reg_amp=0.0, reg_phase=0.0
    )
    fb, gb = _ad_grad(
        g,
        delays,
        interaction,
        ew0,
        t_meas,
        weights,
        model_kw={},
        phase_only=False,
        R_omega=True,
        reg_amp=0.0,
        reg_phase=0.0,
    )
    fc, gc = _hand_jax_grad(
        g,
        delays,
        interaction,
        ew0,
        t_meas,
        weights,
        model_kw={},
        phase_only=False,
        R_omega=True,
        reg_amp=0.0,
        reg_phase=0.0,
    )
    assert fa == pytest.approx(fb, rel=1e-9)
    assert np.allclose(ga, gb, rtol=1e-6, atol=1e-8)
    assert fa == pytest.approx(fc, rel=1e-9)
    assert np.allclose(ga, gc, rtol=1e-8, atol=1e-10)
