"""Tests for :mod:`croak.pulses`: generators, parameterisations, VJPs, penalties."""

import numpy as np
import pytest

from croak.grid import Grid
from croak.maths import fwhm as measure_fwhm
from croak.pulses import (
    ArrayPulse,
    ComplexPulse,
    gaussian_pulse,
    random_gaussian_spectrum,
    sech_pulse,
)


@pytest.fixture
def grid():
    return Grid(256, dt=0.5e-15)


def test_gaussian_pulse_fwhm_and_norm(grid):
    f = 8e-15
    ew = gaussian_pulse(grid, f)
    assert np.max(np.abs(ew) ** 2) == pytest.approx(1.0)
    et = grid.ifft(ew)
    # ``f`` is the *intensity* FWHM. Compare widths in fs so pytest.approx's
    # default absolute tolerance (1e-12) cannot silently pass a sub-fs error on
    # these 1e-15-scale values (the bug the old rel-only check masked).
    intensity_fwhm = measure_fwhm(grid.t, np.abs(et) ** 2)
    field_fwhm = measure_fwhm(grid.t, np.abs(et))
    assert intensity_fwhm / 1e-15 == pytest.approx(f / 1e-15, rel=2e-3)
    # the field-amplitude Gaussian is sqrt(2) wider than its intensity
    assert field_fwhm / 1e-15 == pytest.approx(np.sqrt(2.0) * f / 1e-15, rel=2e-3)


def test_sech_pulse_fwhm(grid):
    f = 10e-15
    ew = sech_pulse(grid, f)
    et = grid.ifft(ew)
    # intensity FWHM in fs (see test_gaussian_pulse_fwhm_and_norm on the units)
    assert measure_fwhm(grid.t, np.abs(et) ** 2) / 1e-15 == pytest.approx(
        f / 1e-15, rel=3e-3
    )


def test_gdd_chirp_broadens_pulse(grid):
    f = 8e-15
    tl = grid.ifft(gaussian_pulse(grid, f))
    chirped = grid.ifft(gaussian_pulse(grid, f, phases=[30e-30]))  # GDD = 30 fs^2
    w_tl = measure_fwhm(grid.t, np.abs(tl) ** 2)
    w_ch = measure_fwhm(grid.t, np.abs(chirped) ** 2)
    assert w_ch > 1.5 * w_tl


def test_random_spectrum_shape(grid):
    ew = random_gaussian_spectrum(grid, 3e-15, rng=np.random.default_rng(0))
    assert ew.shape == (grid.n,) and np.iscomplexobj(ew)


# -- parameterisation round trips ------------------------------------------
def test_complex_pulse_roundtrip(grid):
    ew = gaussian_pulse(grid, 6e-15, phases=[100e-30])
    p = ComplexPulse(grid, ew)
    assert p.nparams == 2 * grid.n
    assert np.allclose(p.spectrum(), ew)
    u = p.get_params()
    p.set_params(u)
    assert np.allclose(p.spectrum(), ew)


def test_array_pulse_roundtrip_and_phase_only(grid):
    ew = gaussian_pulse(grid, 6e-15, phases=[100e-30])
    p = ArrayPulse.from_spectrum(grid, ew)
    assert np.allclose(p.spectrum(), ew, atol=1e-12)
    po = ArrayPulse.from_spectrum(grid, ew, phase_only=True)
    assert po.nparams == grid.n
    assert np.allclose(po.spectrum(), ew, atol=1e-12)


def _vjp_vs_fd(pulse, w, h=1e-7):
    u0 = pulse.get_params()

    def loss(u):
        pulse.set_params(u)
        return float(np.real(np.sum(np.conj(w) * pulse.spectrum())))

    pulse.set_params(u0)
    grad = pulse.ew_vjp(0.5 * w)
    fd = np.array(
        [
            (loss(u0 + h * np.eye(len(u0))[k]) - loss(u0 - h * np.eye(len(u0))[k]))
            / (2 * h)
            for k in range(len(u0))
        ]
    )
    pulse.set_params(u0)
    return grad, fd


@pytest.mark.parametrize("phase_only", [False, True])
def test_array_pulse_vjp(grid, phase_only):
    small = Grid(12, dt=1e-15)
    ew = gaussian_pulse(small, 6e-15) + 0.05
    p = ArrayPulse.from_spectrum(small, ew, phase_only=phase_only)
    rng = np.random.default_rng(1)
    w = rng.standard_normal(small.n) + 1j * rng.standard_normal(small.n)
    grad, fd = _vjp_vs_fd(p, w)
    assert np.allclose(grad, fd, atol=1e-5, rtol=1e-5)


def test_complex_pulse_vjp():
    small = Grid(12, dt=1e-15)
    ew = gaussian_pulse(small, 6e-15) + 0.05
    p = ComplexPulse(small, ew)
    rng = np.random.default_rng(2)
    w = rng.standard_normal(small.n) + 1j * rng.standard_normal(small.n)
    grad, fd = _vjp_vs_fd(p, w)
    assert np.allclose(grad, fd, atol=1e-6, rtol=1e-6)


def test_penalty_grads_vs_fd():
    small = Grid(16, dt=1e-15)
    ew = gaussian_pulse(small, 6e-15) + 0.05
    p = ArrayPulse.from_spectrum(small, ew)
    u0 = p.get_params()
    h = 1e-6
    for penalty, grad_fn in [
        (p.amplitude_penalty, p.amplitude_penalty_grad),
        (p.phase_penalty, p.phase_penalty_grad),
    ]:
        g = grad_fn(u0)
        fd = np.array(
            [
                (
                    penalty(u0 + h * np.eye(len(u0))[k])
                    - penalty(u0 - h * np.eye(len(u0))[k])
                )
                / (2 * h)
                for k in range(len(u0))
            ]
        )
        assert np.allclose(g, fd, atol=1e-5, rtol=1e-4)


def test_spectral_penalty_grad_vs_fd():
    small = Grid(16, dt=1e-15)
    ew = gaussian_pulse(small, 6e-15) + 0.05
    p = ArrayPulse.from_spectrum(small, ew)
    u0 = p.get_params()
    # a distinct full-grid target amplitude
    target = np.abs(gaussian_pulse(small, 4e-15)) + 0.02
    h = 1e-6
    g = p.spectral_penalty_grad(u0, target)
    fd = np.array(
        [
            (
                p.spectral_penalty(u0 + h * np.eye(len(u0))[k], target)
                - p.spectral_penalty(u0 - h * np.eye(len(u0))[k], target)
            )
            / (2 * h)
            for k in range(len(u0))
        ]
    )
    assert np.allclose(g, fd, atol=1e-6, rtol=1e-5)
    # phase block is untouched by the amplitude-only spectral penalty
    assert np.allclose(g[p.nidcs :], 0.0)


def test_spectral_penalty_phase_only_is_noop():
    small = Grid(16, dt=1e-15)
    ew = gaussian_pulse(small, 6e-15) + 0.05
    p = ArrayPulse.from_spectrum(small, ew, phase_only=True)
    u0 = p.get_params()
    target = np.abs(ew)
    assert p.spectral_penalty(u0, target) == 0.0
    assert np.allclose(p.spectral_penalty_grad(u0, target), 0.0)


def test_complex_pulse_inherits_neutral_bounds_and_penalties(grid):
    p = ComplexPulse(grid, gaussian_pulse(grid, 5e-15))
    lo, hi = p.bounds()
    assert lo.shape == hi.shape == (p.nparams,)
    assert np.all(lo == -np.inf) and np.all(hi == np.inf)
    u = p.get_params()
    # the base class provides "no regularisation" for the gradients and the phase
    assert p.phase_penalty(u) == 0.0
    np.testing.assert_array_equal(p.phase_penalty_grad(u), np.zeros(p.nparams))
    np.testing.assert_array_equal(p.amplitude_penalty_grad(u), np.zeros(p.nparams))


def test_complex_pulse_amplitude_penalty_measures_roughness(grid):
    smooth = ComplexPulse(grid, gaussian_pulse(grid, 5e-15))
    rough_ew = gaussian_pulse(grid, 5e-15) * (1.0 + 0.2 * (-1.0) ** np.arange(grid.n))
    rough = ComplexPulse(grid, rough_ew)
    assert smooth.amplitude_penalty(smooth.get_params()) >= 0.0
    # a bin-to-bin alternation is pure curvature: far rougher than the Gaussian
    assert rough.amplitude_penalty(rough.get_params()) > 100 * smooth.amplitude_penalty(
        smooth.get_params()
    )


def test_phase_only_array_pulse_has_no_amplitude_penalty(grid):
    amp = np.abs(gaussian_pulse(grid, 5e-15))
    p = ArrayPulse(grid, amp, phase_only=True)
    u = np.random.default_rng(0).standard_normal(p.nparams)
    assert p.amplitude_penalty(u) == 0.0
    np.testing.assert_array_equal(p.amplitude_penalty_grad(u), np.zeros(p.nparams))


def test_second_difference_penalty_is_zero_below_three_points():
    from croak.pulses import _second_diff_penalty, _second_diff_penalty_grad

    a = np.array([1.0, 2.0])
    assert _second_diff_penalty(a) == 0.0
    np.testing.assert_array_equal(_second_diff_penalty_grad(a), np.zeros(2))


def test_second_difference_penalty_grad_of_an_all_zero_amplitude():
    from croak.pulses import _second_diff_penalty_grad

    # max(a) = 0: the epsilon floor guards the division and the max-term is skipped
    np.testing.assert_array_equal(_second_diff_penalty_grad(np.zeros(5)), np.zeros(5))


def test_pulse_base_class_defaults_to_no_amplitude_penalty(grid):
    from croak.pulses import Pulse

    class Constant(Pulse):
        """A one-parameter pulse overriding nothing but its size."""

        nparams = 1

    assert Constant().amplitude_penalty(np.zeros(1)) == 0.0
