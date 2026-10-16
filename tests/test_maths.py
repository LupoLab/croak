"""Tests for the numerical helpers in :mod:`croak.maths`."""

import numpy as np
import pytest

from croak import maths
from croak.constants import C


def test_wlfreq_is_self_inverse():
    lam = 800e-9
    omega = maths.wlfreq(lam)
    assert omega == pytest.approx(2 * np.pi * C / lam)
    assert maths.wlfreq(omega) == pytest.approx(lam)


def test_gauss_fwhm_matches_definition():
    x = np.linspace(-10, 10, 4001)
    f = 3.0
    y = maths.gauss(x, fwhm=f)
    assert y.max() == pytest.approx(1.0)
    assert maths.fwhm(x, y) == pytest.approx(f, rel=1e-3)


def test_gauss_requires_one_width():
    with pytest.raises(ValueError):
        maths.gauss(np.array([0.0]))
    with pytest.raises(ValueError):
        maths.gauss(np.array([0.0]), sigma=1.0, fwhm=1.0)


def test_sech_intensity_fwhm():
    x = np.linspace(-30, 30, 8001)
    f = 4.0
    amp = maths.sech(x, fwhm=f)
    assert maths.fwhm(x, amp**2) == pytest.approx(f, rel=1e-3)


def test_moment_centroid_and_variance():
    x = np.linspace(-20, 20, 20001)
    y = maths.gauss(x, sigma=2.0, x0=1.5)
    assert maths.moment(x, y, 1) == pytest.approx(1.5, abs=1e-6)
    variance = maths.moment(x, y, 2) - maths.moment(x, y, 1) ** 2
    assert variance == pytest.approx(4.0, rel=1e-4)


def test_derivative_orders():
    # d/dx sin = cos, d2/dx2 sin = -sin
    assert maths.derivative(np.sin, 0.3, 1) == pytest.approx(np.cos(0.3), abs=1e-6)
    assert maths.derivative(np.sin, 0.3, 2) == pytest.approx(-np.sin(0.3), abs=1e-5)
    with pytest.raises(ValueError):
        maths.derivative(np.sin, 0.0, 3)


def test_sech_requires_exactly_one_width():
    with pytest.raises(ValueError, match="exactly one of `tau` or `fwhm`"):
        maths.sech([0.0], fwhm=1.0, tau=1.0)
    with pytest.raises(ValueError, match="exactly one of `tau` or `fwhm`"):
        maths.sech([0.0])


def test_sech_fwhm_and_tau_agree():
    x = np.linspace(-5.0, 5.0, 11)
    tau = 1.3
    fwhm = 2.0 * np.log(1.0 + np.sqrt(2.0)) * tau
    np.testing.assert_allclose(maths.sech(x, fwhm=fwhm), maths.sech(x, tau=tau))


def test_fwhm_is_nan_when_the_level_is_never_reached():
    # level > 1 puts the threshold above the peak: nothing is "above" it
    assert np.isnan(maths.fwhm([0.0, 1.0, 2.0], [0.0, 1.0, 0.0], level=1.5))


def test_fwhm_is_nan_when_the_right_side_never_drops():
    x = np.arange(5.0)
    assert np.isnan(maths.fwhm(x, [0.0, 0.2, 1.0, 0.9, 0.8]))


def test_derivative_with_an_explicit_step():
    # a quadratic is differentiated exactly by central differences at any step
    assert maths.derivative(lambda x: x**2, 3.0, dx=0.5) == pytest.approx(6.0)
    assert maths.derivative(lambda x: x**2, 3.0, order=2, dx=0.5) == pytest.approx(2.0)
