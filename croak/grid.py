r"""Time/frequency grid and Fourier transforms.

A :class:`Grid` holds matched, centred time
and angular-frequency axes and provides the *physics-convention* Fourier
transform pair used throughout croak:

* forward (time → frequency): :math:`\tilde E(\omega) = \int E(t)\,e^{+i\omega t}\,dt`
* inverse (frequency → time):
  :math:`E(t) = \tfrac{1}{2\pi}\int \tilde E(\omega)\,e^{-i\omega t}\,d\omega`

discretised as :math:`\tilde E \approx \Delta t \sum E\,e^{+i\omega t}` and
:math:`E \approx \tfrac{\Delta\omega}{2\pi}\sum \tilde E\,e^{-i\omega t}`.

These :meth:`Grid.fft`/:meth:`Grid.ifft` operate on **centred** arrays (zero
frequency in the middle). The forward *model* (see :mod:`croak.forward`) instead
works in raw DFT-bin order; the unshifted, unnormalised transforms it uses are
exposed as the module-level :func:`raw_fft`/:func:`raw_ifft`.

All FFTs are computed with :mod:`scipy.fft`.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import scipy.fft
from numpy.typing import ArrayLike, NDArray

from .constants import C

__all__ = ["Grid", "raw_fft", "raw_ifft", "gridparams_omega", "gridparams_lambda"]


def raw_fft(x: NDArray[np.complexfloating]) -> NDArray[np.complex128]:
    """Unnormalised, unshifted forward DFT (``scipy.fft.fft``).

    Used by the forward model as the spectrum→time transform in DFT-bin order.
    """
    # np.asarray: scipy.fft's @_dispatchable stubs do not expose the ndarray return.
    return np.asarray(scipy.fft.fft(x))


def raw_ifft(x: NDArray[np.complexfloating]) -> NDArray[np.complex128]:
    """Unitary-inverse (``1/N``) DFT (``scipy.fft.ifft``).

    Companion to :func:`raw_fft`; the time→frequency transform in DFT-bin order.
    """
    return np.asarray(scipy.fft.ifft(x))


class Grid:
    r"""A matched, centred time/angular-frequency grid.

    Parameters
    ----------
    n : int
        Number of grid points.
    dt : float, optional
        Time step (s). Provide exactly one of ``dt`` or ``domega``.
    domega : float, optional
        Angular-frequency step (rad/s). The other step follows from
        :math:`\Delta\omega = 2\pi/(\Delta t\,N)`.

    Attributes
    ----------
    omega : numpy.ndarray
        Centred angular-frequency axis (rad/s), relative to the carrier.
    t : numpy.ndarray
        Centred time axis (s).
    dt, domega, df : float
        Time step, angular-frequency step, and ordinary-frequency step
        ``domega / (2*pi)``.
    n : int
        Number of points.
    """

    def __init__(self, n: int, *, dt: float | None = None, domega: float | None = None):
        """Build a centred ``n``-point grid from exactly one of ``dt``/``domega``."""
        # Exactly one of dt/domega must be given; the structure (rather than the
        # terser ``(dt is None) == (domega is None)``) lets the type checker see
        # that the other value is non-None where it is used below.
        if dt is None:
            if domega is None:
                raise ValueError("provide exactly one of `dt` or `domega`")
            dt = 2.0 * np.pi / (domega * n)
        elif domega is not None:
            raise ValueError("provide exactly one of `dt` or `domega`")
        else:
            domega = 2.0 * np.pi / (dt * n)

        # Build centred axes such that the DFT zero-bin (the element ifftshift
        # maps to index 0) sits at value 0. This is what makes an unshifted FFT
        # of a centred array come out with the correct phase reference.
        omega = np.arange(1, n + 1, dtype=float) * domega
        omega -= scipy.fft.ifftshift(omega)[0]
        t = np.arange(1, n + 1, dtype=float) * dt
        t -= scipy.fft.ifftshift(t)[0]

        self.n = int(n)
        self.dt = float(dt)
        self.domega = float(domega)
        self.df = float(domega / (2.0 * np.pi))
        self.omega = omega
        self.t = t

    @classmethod
    def from_omega(cls, omega: ArrayLike) -> Grid:
        """Construct a :class:`Grid` matching a uniformly spaced ``omega`` axis."""
        omega = np.asarray(omega, dtype=float)
        domega = float(omega[1] - omega[0])
        return cls(len(omega), domega=domega)

    # -- Fourier transforms (centred, physics convention) -------------------
    def fft(self, et: ArrayLike) -> NDArray[np.complex128]:
        r"""Forward transform time → frequency (centred), scaled by ``dt``."""
        et = np.asarray(et, dtype=complex)
        out = scipy.fft.fftshift(scipy.fft.ifft(scipy.fft.ifftshift(et)))
        return out * (self.dt * self.n)

    def ifft(self, ew: ArrayLike) -> NDArray[np.complex128]:
        r"""Inverse transform frequency → time (centred), scaled by ``df``."""
        ew = np.asarray(ew, dtype=complex)
        out = scipy.fft.fftshift(scipy.fft.fft(scipy.fft.ifftshift(ew)))
        return out * self.df

    # -- Utilities ----------------------------------------------------------
    def tshift(self, et: ArrayLike, tau: float) -> NDArray[np.complex128]:
        r"""Shift a time-domain field by ``tau`` via a spectral phase ramp.

        Applies :math:`E(t) \to E(t-\tau)` using
        :math:`\tilde E \to \tilde E\,e^{+i\omega\tau}` in the raw
        (DFT-bin) domain.
        """
        et = np.asarray(et, dtype=complex)
        omega_bin = scipy.fft.ifftshift(self.omega)
        spec = raw_ifft(et) * np.exp(1j * omega_bin * tau)
        return raw_fft(spec)


def gridparams_omega(
    trange: float,
    *,
    omega_min: float,
    omega_max: float,
    omega0: float | None = None,
) -> tuple[int, float, float]:
    """Grid parameters spanning a time range and frequency range.

    Picks the time step from the Nyquist limit of the largest frequency offset
    from the carrier and the number of points from the requested time window
    (rounded up to an FFT-friendly length).

    Parameters
    ----------
    trange : float
        Required temporal window (s).
    omega_min, omega_max : float
        Extremes of the absolute angular-frequency content (rad/s).
    omega0 : float, optional
        Carrier angular frequency; defaults to the midpoint.

    Returns
    -------
    tuple
        ``(n, dt, omega0)``.
    """
    omega_min, omega_max = sorted((omega_min, omega_max))
    if omega0 is None:
        omega0 = 0.5 * (omega_min + omega_max)
    maxshift = max(abs(omega_min - omega0), abs(omega_max - omega0))
    dt = np.pi / maxshift
    # cast: next_fast_len returns an int at runtime, but its scipy stub declares None.
    n = cast(int, scipy.fft.next_fast_len(int(np.ceil(trange / dt))))
    return n, dt, omega0


def gridparams_lambda(
    trange: float,
    *,
    lambda_min: float,
    lambda_max: float,
    omega0: float | None = None,
) -> tuple[int, float, float]:
    """As :func:`gridparams_omega`, but specified by a wavelength range (m)."""
    omega_min = 2.0 * np.pi * C / lambda_max
    omega_max = 2.0 * np.pi * C / lambda_min
    return gridparams_omega(
        trange, omega_min=omega_min, omega_max=omega_max, omega0=omega0
    )
