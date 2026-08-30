"""Numerical helpers.

This module provides the small set of mathematical utilities the retrieval core
needs to be self-contained: pulse envelope shapes (:func:`gauss`, :func:`sech`),
distribution statistics (:func:`fwhm`, :func:`moment`), a numerical derivative,
and the angular-frequency/wavelength conversion :func:`wlfreq`.

All quantities are SI unless stated otherwise.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import C

__all__ = [
    "wlfreq",
    "fwhm_to_sigma",
    "gauss",
    "sech",
    "fwhm",
    "moment",
    "derivative",
    "planck_taper",
]


def wlfreq(x: ArrayLike) -> NDArray[np.float64]:
    r"""Convert between angular frequency and wavelength.

    Implements :math:`2\pi c / x`, which maps angular frequency
    :math:`\omega` (rad/s) to vacuum wavelength :math:`\lambda` (m) and vice
    versa. The function is its own inverse.

    Parameters
    ----------
    x : array_like
        Angular frequency (rad/s) or wavelength (m).

    Returns
    -------
    numpy.ndarray
        Wavelength (m) or angular frequency (rad/s).
    """
    return 2.0 * np.pi * C / np.asarray(x, dtype=float)


def fwhm_to_sigma(fwhm: float, power: float = 2.0) -> float:
    r"""Convert a full width at half maximum to a Gaussian ``sigma``.

    For a (hyper-)Gaussian :math:`\exp[-\tfrac12 (x/\sigma)^{p}]`, the FWHM and
    width parameter are related by
    :math:`\sigma = \mathrm{FWHM} / (2\,(2\ln 2)^{1/p})`.

    Parameters
    ----------
    fwhm : float
        Full width at half maximum.
    power : float, optional
        Exponent ``p`` of the (hyper-)Gaussian. ``2`` (default) is an ordinary
        Gaussian.
    """
    return fwhm / (2.0 * (2.0 * np.log(2.0)) ** (1.0 / power))


def gauss(
    x: ArrayLike,
    sigma: float | None = None,
    *,
    fwhm: float | None = None,
    x0: float = 0.0,
    power: float = 2.0,
) -> NDArray[np.float64]:
    r"""Evaluate a (hyper-)Gaussian.

    Computes :math:`\exp[-\tfrac12 ((x-x_0)/\sigma)^{p}]`. Exactly one of
    ``sigma`` or ``fwhm`` must be given.

    Parameters
    ----------
    x : array_like
        Sample points.
    sigma : float, optional
        Standard-deviation-like width parameter.
    fwhm : float, optional
        Full width at half maximum (converted to ``sigma`` internally).
    x0 : float, optional
        Centre of the distribution.
    power : float, optional
        Exponent of the (hyper-)Gaussian. ``2`` (default) is an ordinary
        Gaussian.
    """
    # Exactly one of sigma/fwhm; the structure lets the type checker prove that
    # ``sigma`` is non-None at the return.
    if fwhm is not None:
        if sigma is not None:
            raise ValueError("provide exactly one of `sigma` or `fwhm`")
        sigma = fwhm_to_sigma(fwhm, power=power)
    elif sigma is None:
        raise ValueError("provide exactly one of `sigma` or `fwhm`")
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * ((x - x0) / sigma) ** power)


def sech(
    x: ArrayLike,
    *,
    fwhm: float | None = None,
    tau: float | None = None,
    x0: float = 0.0,
) -> NDArray[np.float64]:
    r"""Evaluate a hyperbolic-secant amplitude profile.

    Computes :math:`\operatorname{sech}((x-x_0)/\tau)`. Exactly one of ``tau``
    or ``fwhm`` must be given. The intensity :math:`\operatorname{sech}^2` has
    full width at half maximum :math:`\mathrm{FWHM} = 2\,\tau\,\operatorname{arcsech}
    (1/\sqrt2) = 2\,\tau\,\ln(1+\sqrt2)`, so ``tau = fwhm / (2 ln(1+sqrt 2))``.

    Parameters
    ----------
    x : array_like
        Sample points.
    fwhm : float, optional
        Intensity FWHM.
    tau : float, optional
        Characteristic width.
    x0 : float, optional
        Centre of the distribution.
    """
    # Exactly one of tau/fwhm; structured so ``tau`` is provably non-None below.
    if fwhm is not None:
        if tau is not None:
            raise ValueError("provide exactly one of `tau` or `fwhm`")
        tau = float(fwhm / (2.0 * np.log(1.0 + np.sqrt(2.0))))
    elif tau is None:
        raise ValueError("provide exactly one of `tau` or `fwhm`")
    x = np.asarray(x, dtype=float)
    return 1.0 / np.cosh((x - x0) / tau)


def moment(x: ArrayLike, y: ArrayLike, n: int = 1) -> float:
    r"""Return the ``n``-th moment of a distribution.

    Computes :math:`\sum_i x_i^n\, y_i / \sum_i y_i`, the weighted average of
    :math:`x^n` with weights ``y``. The first moment (``n=1``) is the centroid.

    Parameters
    ----------
    x : array_like
        Coordinate values.
    y : array_like
        Non-negative weights (e.g. an intensity distribution).
    n : int, optional
        Moment order (default ``1``).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.sum(x**n * y) / np.sum(y))


def fwhm(x: ArrayLike, y: ArrayLike, *, level: float = 0.5) -> float:
    """Estimate the full width at ``level`` of the maximum by linear interpolation.

    Finds the outermost crossings of ``level * max(y)`` on either side of the
    peak and returns their separation. Assumes a single dominant peak.

    Parameters
    ----------
    x : array_like
        Monotonically increasing coordinate.
    y : array_like
        Signal values.
    level : float, optional
        Fraction of the maximum at which the width is measured (``0.5`` for the
        conventional FWHM).

    Returns
    -------
    float
        The full width. Returns ``nan`` if the threshold is not crossed on both
        sides of the peak.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    threshold = level * np.max(y)
    above = y >= threshold
    if not np.any(above):
        return float("nan")

    peak = int(np.argmax(y))

    def crossing(lo: int, hi: int) -> float:
        # linear interpolation of the x value where y crosses `threshold`
        # between indices lo and hi (y[lo] and y[hi] straddle the threshold).
        frac = (threshold - y[lo]) / (y[hi] - y[lo])
        return x[lo] + frac * (x[hi] - x[lo])

    # walk left from the peak to the first sample below threshold
    left = peak
    while left > 0 and y[left] >= threshold:
        left -= 1
    if y[left] >= threshold:
        return float("nan")
    xl = crossing(left, left + 1)

    # walk right from the peak
    right = peak
    n = len(y)
    while right < n - 1 and y[right] >= threshold:
        right += 1
    if y[right] >= threshold:
        return float("nan")
    xr = crossing(right, right - 1)

    return abs(xr - xl)


def planck_taper(
    x: ArrayLike, left0: float, left1: float, right1: float, right0: float
) -> NDArray[np.float64]:
    """Planck-taper window defined by four corner points.

    The window is ``0`` for ``x <= left0``, rises smoothly to ``1`` across
    ``[left0, left1]``, stays ``1`` over ``[left1, right1]``, falls back to ``0``
    across ``[right1, right0]``, and is ``0`` for ``x >= right0``. The smooth
    transition is the Planck-taper of arXiv:1003.2939 eq. (7).

    Parameters
    ----------
    x : array_like
        Sample points.
    left0, left1, right1, right0 : float
        Corner positions (must satisfy ``left0 <= left1 <= right1 <= right0``).

    Returns
    -------
    numpy.ndarray
        Window values in ``[0, 1]``.
    """
    x = np.asarray(x, dtype=float)
    span = right0 - left0
    x0 = 0.5 * (right0 + left0)
    xc = x - x0
    eps_left = abs(left1 - left0) / span
    eps_right = abs(right0 - right1) / span
    x1 = -span / 2
    x2 = -span / 2 * (1 - 2 * eps_left)
    x3 = span / 2 * (1 - 2 * eps_right)
    x4 = span / 2

    out = np.zeros_like(xc)
    out[(xc >= x2) & (xc <= x3)] = 1.0
    rising = (xc > x1) & (xc < x2)
    if np.any(rising):
        xr = xc[rising]
        z = (x2 - x1) / (xr - x1) + (x2 - x1) / (xr - x2)
        out[rising] = 1.0 / (1.0 + np.exp(np.clip(z, -700, 700)))
    falling = (xc > x3) & (xc < x4)
    if np.any(falling):
        xf = xc[falling]
        z = (x3 - x4) / (xf - x3) + (x3 - x4) / (xf - x4)
        out[falling] = 1.0 / (1.0 + np.exp(np.clip(z, -700, 700)))
    return out


def derivative(f, x: float, order: int = 1, *, dx: float | None = None) -> float:
    r"""Numerical derivative of a scalar function via central finite differences.

    Used for low-order dispersion quantities (e.g. the group delay
    :math:`\beta_1 = d\beta/d\omega`). For ``order`` 1 and 2 this uses the
    standard second-order-accurate central stencils.

    Parameters
    ----------
    f : callable
        Scalar function of a scalar argument.
    x : float
        Point at which to evaluate the derivative.
    order : int, optional
        Derivative order (``1`` or ``2``).
    dx : float, optional
        Step size. Defaults to a scale-aware value based on ``x``.

    Returns
    -------
    float
        The estimated derivative.
    """
    if dx is None:
        # Step that roughly balances truncation and round-off: ~eps**(1/3) for a
        # first derivative, ~eps**(1/4) for a second derivative.
        scale = abs(x) + 1.0
        dx = scale * (6e-6 if order == 1 else 1.2e-4)
    if order == 1:
        return (f(x + dx) - f(x - dx)) / (2.0 * dx)
    if order == 2:
        return (f(x + dx) - 2.0 * f(x) + f(x - dx)) / dx**2
    raise ValueError("only derivative orders 1 and 2 are supported")
