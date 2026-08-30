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
