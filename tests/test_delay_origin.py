"""The model's delay origin: marginal-peak re-centring and the tau0 bound.

The preprocessing centres a measured trace on its delay-marginal peak; the
forward model puts delay zero at gate--probe coincidence. ``delay_origin=
"marginal_peak"`` re-centres every model trace the same way the data were, so
an asymmetric trace no longer leaves a delay offset for the pulse to absorb.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from croak.delay_origin import (
    DELAY_ORIGINS,
    marginal_peak_delay_jax,
    recentre_trace,
    recentring,
)
from croak.forward import maketrace
from croak.grid import Grid
from croak.maths import wlfreq
from croak.lbfgs_ad import LBFGSAD
from croak.preprocess import marginal_peak_delay
from croak.pulses import gaussian_pulse


@pytest.fixture
def setup():
    g = Grid(128, dt=0.3e-15)
    ew = gaussian_pulse(g, 2.5e-15)
    delays = np.linspace(-15e-15, 15e-15, 87)
    return g, ew, delays


def _shift(trace, delays, d):
    """Exact Fourier translation T(tau) -> T(tau - d) (peak moves to +d)."""
    f = np.fft.fftfreq(delays.size, d=delays[1] - delays[0])
    return np.real(np.fft.ifft(np.fft.fft(trace, axis=1) * np.exp(-2j * np.pi * f * d)))


def test_jax_peak_matches_the_preprocessing_rule():
    """The JAX vertex estimate reproduces preprocess.marginal_peak_delay."""
    tau = np.linspace(-10e-15, 10e-15, 81)
    for true in (0.37e-15, -1.1e-15, 2.04e-15):
        marg = np.exp(-(((tau - true) / 3e-15) ** 2)) * (1 + 0.1 * np.sin(tau * 1e15))
        ref = marginal_peak_delay(tau, marg)
        got = float(marginal_peak_delay_jax(jnp.asarray(tau), jnp.asarray(marg)))
        assert got == pytest.approx(ref, abs=1e-19)


def test_recentre_puts_the_marginal_peak_at_zero_and_is_differentiable(setup):
    """A trace shifted by any sub-sample amount is brought back to peak = 0."""
    g, ew, delays = setup
    base = maketrace(g.omega, delays, ew, "shg")
    dtau = delays[1] - delays[0]
    for d in (0.13 * dtau, -0.61 * dtau, 1.7 * dtau):
        shifted = _shift(base, delays, d)
        assert abs(marginal_peak_delay(delays, shifted.sum(0)) - d) < 0.05 * dtau
        back = np.asarray(recentre_trace(jnp.asarray(shifted), jnp.asarray(delays)))
        assert abs(marginal_peak_delay(delays, back.sum(0))) < 0.02 * dtau
        # the re-centred trace is the original, to the precision of the peak estimate
        assert np.max(np.abs(back - base)) < 5e-3 * base.max()
    grad = jax.grad(lambda t: jnp.sum(recentre_trace(t, jnp.asarray(delays)) ** 2))(
        jnp.asarray(base)
    )
    assert np.all(np.isfinite(np.asarray(grad)))


def test_recentring_wrapper_validates_and_passes_through(setup):
    g, ew, delays = setup
    fn = lambda: jnp.asarray(maketrace(g.omega, delays, ew, "shg"))  # noqa: E731
    assert recentring(fn, delays, "coincidence") is fn
    assert set(DELAY_ORIGINS) == {"coincidence", "marginal_peak"}
    with pytest.raises(ValueError, match="delay_origin"):
        recentring(fn, delays, "peak")
    with pytest.raises(ValueError, match="uniform"):
        recentring(fn, np.concatenate([delays[:40], delays[41:]]), "marginal_peak")


def test_lbfgs_ad_marginal_peak_origin_absorbs_a_data_offset(setup):
    """Data centred on their marginal peak retrieve as well as un-offset data.

    The data are built with a +0.4-step delay offset and then centred on their
    marginal peak, as the preprocessing does. With the coincidence origin the
    model is misaligned by the (sub-sample) residual of that centring; with the
    marginal-peak origin both sides use the same rule.
    """
    g, ew, delays = setup
    dtau = delays[1] - delays[0]
    data = _shift(maketrace(g.omega, delays, ew, "shg"), delays, 0.4 * dtau)
    # the preprocessing's centring: translate the AXIS by the estimated peak
    axis = delays - marginal_peak_delay(delays, data.sum(0))
    near = gaussian_pulse(g, 2.0e-15)
    res = LBFGSAD(maxiters=300, delay_origin="marginal_peak").run(
        data, g.omega, axis, "shg", guess=near
    )
    # marginal_peak asks for a uniform axis (it is) and reports a centred trace
    assert res.error < 2e-3
    assert abs(marginal_peak_delay(axis, np.asarray(res.trace).sum(0))) < 0.05 * dtau
    with pytest.raises(ValueError, match="delay_origin"):
        LBFGSAD(delay_origin="nope")


def test_lbfgs_ad_tau0_bound_is_respected(setup):
    """A bounded tau0 fit stays inside the box; unbounded finds the true offset."""
    g, ew, delays = setup
    tau0_true = 1.5e-15
    trace = maketrace(g.omega, delays - tau0_true, ew, "shg")
    free = LBFGSAD(maxiters=300, fit_tau0=True).run(trace, g.omega, delays, "shg", guess=ew)
    bound = 0.5e-15
    boxed = LBFGSAD(maxiters=300, fit_tau0=True, tau0_bound=bound).run(
        trace, g.omega, delays, "shg", guess=ew
    )
    assert free.tau0 == pytest.approx(tau0_true, abs=0.2e-15)
    assert abs(boxed.tau0) <= bound * (1 + 1e-6)
    assert boxed.error > free.error
    with pytest.raises(ValueError, match="tau0_bound"):
        LBFGSAD(tau0_bound=-1.0)
