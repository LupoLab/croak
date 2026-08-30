r"""Pulse representations and test-pulse generators.

Two groups of functionality live here:

* **Generators** — :func:`gaussian_pulse`, :func:`sech_pulse` and
  :func:`random_gaussian_spectrum` build complex spectra :math:`\tilde E(\omega)`
  (centred order) for tests, examples and retrieval initial guesses. Chirp and
  higher-order phase are added as a Taylor expansion of the spectral phase.

* **Parameterisations** — :class:`ComplexPulse` and :class:`ArrayPulse` expose a
  real parameter vector ``u`` to the gradient-based (L-BFGS) solver, together
  with the hand-derived vector–Jacobian product :meth:`Pulse.ew_vjp` that maps a
  spectral cotangent back to ``u``, and optional spectral-smoothness penalties.

The carrier (centre wavelength) is *not* encoded in the baseband pulse shape: a
pulse lives on the grid's centred ``omega`` axis, and the carrier enters
retrieval only through the dispersive model's ``omega0``. "Different
wavelengths" therefore means the same baseband pulse with a different ``omega0``
in :class:`~croak.forward.ForwardModel`.
"""

from __future__ import annotations

from math import factorial

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .grid import Grid
from .maths import gauss

__all__ = [
    "gaussian_pulse",
    "sech_pulse",
    "random_gaussian_spectrum",
    "Pulse",
    "ComplexPulse",
    "ArrayPulse",
]

Complex = NDArray[np.complex128]


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------
def _apply_taylor_phase(grid: Grid, ew: Complex, phases: ArrayLike) -> Complex:
    r"""Multiply ``ew`` by a Taylor spectral phase ``exp(i phi)``.

    The phase is :math:`\phi = \sum_k p_k\,\omega^{k+2}/(k+2)!`. The first
    coefficient is the group-delay dispersion (coefficient of
    :math:`\omega^2/2`), the second is third-order dispersion, and so on.
    """
    phi = np.zeros(grid.n)
    for k, p in enumerate(np.atleast_1d(np.asarray(phases, dtype=float))):
        order = k + 2
        phi += p * grid.omega**order / factorial(order)
    return ew * np.exp(1j * phi)


def _normalize(ew: Complex) -> Complex:
    """Scale a spectrum so that ``max(|ew|**2) == 1``."""
    return ew / np.sqrt(np.max(np.abs(ew) ** 2))


def gaussian_pulse(grid: Grid, fwhm: float, *, phases: ArrayLike = ()) -> Complex:
    """Transform-limited Gaussian pulse with optional spectral-phase chirp.

    Parameters
    ----------
    grid : Grid
        Time/frequency grid.
    fwhm : float
        Intensity FWHM of the (unchirped) temporal pulse (s).
    phases : sequence of float, optional
        Taylor spectral-phase coefficients ``[GDD, TOD, ...]`` (s², s³, …).

    Returns
    -------
    numpy.ndarray
        Complex spectrum (centred order), normalised to unit peak spectral
        intensity.
    """
    # ``fwhm`` is the *intensity* FWHM. The intensity I(t) = |E(t)|^2 is the
    # Gaussian of that FWHM, so build I(t) with :func:`gauss` and take the field
    # as its square root. (Setting the field directly to a FWHM Gaussian would
    # make the field-amplitude FWHM equal ``fwhm`` and the intensity FWHM a
    # factor sqrt(2) narrower — the long-standing bug this avoids.)
    et = np.sqrt(gauss(grid.t, fwhm=fwhm)).astype(complex)
    ew = grid.fft(et)
    return _normalize(_apply_taylor_phase(grid, ew, phases))


def sech_pulse(grid: Grid, fwhm: float, *, phases: ArrayLike = ()) -> Complex:
    r"""Transform-limited :math:`\operatorname{sech}^2` pulse with optional chirp.

    Parameters as in :func:`gaussian_pulse`; ``fwhm`` is the intensity FWHM.
    """
    tau = fwhm / (2.0 * np.log(1.0 + np.sqrt(2.0)))
    et = (1.0 / np.cosh(grid.t / tau)).astype(complex)
    ew = grid.fft(et)
    return _normalize(_apply_taylor_phase(grid, ew, phases))


def random_gaussian_spectrum(
    grid: Grid, fwhm: float = 2.5e-15, *, rng: np.random.Generator | None = None
) -> Complex:
    """Random initial-guess spectrum: a Gaussian envelope with random phase.

    Builds a Gaussian temporal envelope of the given FWHM, multiplies by a
    uniform random temporal phase, and transforms to the spectral domain.

    Parameters
    ----------
    grid : Grid
        Time/frequency grid.
    fwhm : float, optional
        Temporal FWHM of the envelope (s).
    rng : numpy.random.Generator, optional
        Random generator (defaults to a fresh default generator).
    """
    rng = np.random.default_rng() if rng is None else rng
    # ``fwhm`` is the intensity FWHM (consistent with :func:`gaussian_pulse`): the
    # envelope amplitude is the square root of a FWHM Gaussian intensity.
    amp = np.sqrt(gauss(grid.t, fwhm=fwhm))
    et = amp * np.exp(2j * np.pi * rng.random(grid.n))
    return grid.fft(et)


# ---------------------------------------------------------------------------
# Parameterisations
# ---------------------------------------------------------------------------
class Pulse:
    """Abstract real parameterisation of a complex spectrum for optimisation.

    Subclasses map between a real parameter vector ``u`` and the complex
    spectrum, and provide the hand-derived adjoint :meth:`ew_vjp`.
    """

    grid: Grid

    @property
    def nparams(self) -> int:
        """Number of real parameters in ``u``."""
        raise NotImplementedError

    def get_params(self) -> NDArray[np.float64]:
        """Return the current parameter vector."""
        raise NotImplementedError

    def set_params(self, u: ArrayLike) -> None:
        """Set the pulse from a parameter vector."""
        raise NotImplementedError

    def spectrum(self) -> Complex:
        """Return the complex spectrum (centred order)."""
        raise NotImplementedError

    def ew_vjp(self, ew_bar: Complex) -> NDArray[np.float64]:
        r"""Map a spectral cotangent ``ew_bar`` to the parameter gradient ``dL/du``.

        ``ew_bar`` is the Wirtinger cotangent :math:`\partial L/\partial\tilde E^*`.
        """
        raise NotImplementedError

    def bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(lower, upper)`` parameter bounds."""
        inf = np.full(self.nparams, np.inf)
        return -inf, inf

    # optional spectral-smoothness regularisation (default: none)
    def amplitude_penalty(self, u: ArrayLike) -> float:
        """Spectral-amplitude smoothness penalty (``0`` by default)."""
        return 0.0

    def amplitude_penalty_grad(self, u: ArrayLike) -> NDArray[np.float64]:
        """Gradient of :meth:`amplitude_penalty` (zeros by default)."""
        return np.zeros(self.nparams)

    def phase_penalty(self, u: ArrayLike) -> float:
        """Spectral-phase smoothness penalty (``0`` by default)."""
        return 0.0

    def phase_penalty_grad(self, u: ArrayLike) -> NDArray[np.float64]:
        """Gradient of :meth:`phase_penalty` (zeros by default)."""
        return np.zeros(self.nparams)


class ComplexPulse(Pulse):
    """Full complex spectrum parameterised by ``u = [Re E; Im E]`` (length ``2N``).

    The natural representation for step-based algorithms and a fully general,
    unconstrained parameterisation for L-BFGS.
    """

    def __init__(self, grid: Grid, ew: ArrayLike):
        """Wrap a complex spectrum ``ew`` on ``grid`` as free real parameters."""
        self.grid = grid
        self._ew: Complex = np.asarray(ew, dtype=complex).copy()

    @property
    def nparams(self) -> int:
        """Number of real parameters (``2N``: real and imaginary parts)."""
        return 2 * self.grid.n

    def get_params(self):
        """Return ``u = [Re E; Im E]``."""
        return np.concatenate([self._ew.real, self._ew.imag])

    def set_params(self, u):
        """Set the spectrum from ``u = [Re E; Im E]``."""
        u = np.asarray(u, dtype=float)
        n = self.grid.n
        self._ew = u[:n] + 1j * u[n:]

    def spectrum(self):
        """Return the complex spectrum (centred order)."""
        return self._ew.copy()

    def ew_vjp(self, ew_bar):
        """Map a spectral cotangent to ``dL/du`` (real then imaginary blocks)."""
        # For z = x + iy, real loss: dL/dx = 2 Re(z_bar), dL/dy = 2 Im(z_bar).
        return np.concatenate([2.0 * np.real(ew_bar), 2.0 * np.imag(ew_bar)])

    def amplitude_penalty(self, u):
        """Second-difference smoothness penalty on the spectral amplitude."""
        u = np.asarray(u, dtype=float)
        n = self.grid.n
        amp = np.hypot(u[:n], u[n:])
        return _second_diff_penalty(amp)


class ArrayPulse(Pulse):
    r"""Amplitude + spectral phase parameterisation.

    Parameters ``u`` are the spectral amplitude followed by the spectral phase
    over an optional support window: ``u = [A[support]; phi[support]]`` (or just
    the phase when ``phase_only=True``). This is the standard L-BFGS FROG
    parameterisation; ``phase_only`` retrieves the phase with a fixed amplitude.

    Parameters
    ----------
    grid : Grid
        Time/frequency grid.
    amplitude : array_like
        Spectral amplitude :math:`|\tilde E|` (length ``N``).
    phase : array_like, optional
        Spectral phase (length ``N``); defaults to zero.
    support : tuple of int, optional
        ``(start, stop)`` half-open index range of the optimised window;
        defaults to the full grid.
    phase_only : bool, optional
        Optimise only the phase, keeping ``amplitude`` fixed.
    """

    def __init__(
        self,
        grid: Grid,
        amplitude: ArrayLike,
        phase: ArrayLike | None = None,
        *,
        support: tuple[int, int] | None = None,
        phase_only: bool = False,
    ):
        """Initialise from amplitude/phase arrays over an optional support window."""
        self.grid = grid
        n = grid.n
        self.amp = np.asarray(amplitude, dtype=float).copy()
        self.phase = (
            np.zeros(n) if phase is None else np.asarray(phase, dtype=float).copy()
        )
        self.start, self.stop = (0, n) if support is None else support
        self.nidcs = self.stop - self.start
        self.phase_only = bool(phase_only)

    @classmethod
    def from_spectrum(
        cls,
        grid: Grid,
        ew: ArrayLike,
        *,
        support: tuple[int, int] | None = None,
        phase_only: bool = False,
    ) -> ArrayPulse:
        """Build an :class:`ArrayPulse` from a complex spectrum."""
        ew = np.asarray(ew, dtype=complex)
        return cls(
            grid, np.abs(ew), np.angle(ew), support=support, phase_only=phase_only
        )

    @property
    def nparams(self) -> int:
        """Number of real parameters (phase only, or amplitude + phase)."""
        return self.nidcs if self.phase_only else 2 * self.nidcs

    def _sl(self) -> slice:
        return slice(self.start, self.stop)

    def get_params(self):
        """Return the parameter vector over the support window."""
        if self.phase_only:
            return self.phase[self._sl()].copy()
        return np.concatenate([self.amp[self._sl()], self.phase[self._sl()]])

    def set_params(self, u):
        """Set amplitude/phase over the support window from ``u``."""
        u = np.asarray(u, dtype=float)
        if self.phase_only:
            self.phase[self._sl()] = u
        else:
            self.amp[self._sl()] = u[: self.nidcs]
            self.phase[self._sl()] = u[self.nidcs :]

    def spectrum(self):
        """Return the complex spectrum ``A exp(i phi)``."""
        return self.amp * np.exp(1j * self.phase)

    def ew_vjp(self, ew_bar):
        """Map a spectral cotangent to ``dL/du`` (amplitude/phase blocks)."""
        s = self._sl()
        phi = self.phase[s]
        a = self.amp[s]
        d = ew_bar[s]
        cos, sin = np.cos(phi), np.sin(phi)
        du_phase = 2.0 * a * (cos * np.imag(d) - sin * np.real(d))
        if self.phase_only:
            return du_phase
        du_amp = 2.0 * (cos * np.real(d) + sin * np.imag(d))
        return np.concatenate([du_amp, du_phase])

    # -- regularisation -----------------------------------------------------
    def amplitude_penalty(self, u):
        """Second-difference amplitude smoothness penalty (``0`` if phase-only)."""
        if self.phase_only:
            return 0.0
        return _second_diff_penalty(np.asarray(u, dtype=float)[: self.nidcs])

    def amplitude_penalty_grad(self, u):
        """Gradient of :meth:`amplitude_penalty` (phase block zero)."""
        g = np.zeros(self.nparams)
        if not self.phase_only:
            g[: self.nidcs] = _second_diff_penalty_grad(
                np.asarray(u, dtype=float)[: self.nidcs]
            )
        return g

    def phase_penalty(self, u):
        """Second-difference phase smoothness penalty (mean squared curvature)."""
        u = np.asarray(u, dtype=float)
        phi = u if self.phase_only else u[self.nidcs :]
        return _second_diff_penalty(phi, normalize_by_max=False)

    def phase_penalty_grad(self, u):
        """Gradient of :meth:`phase_penalty`."""
        u = np.asarray(u, dtype=float)
        g = np.zeros(self.nparams)
        offset = 0 if self.phase_only else self.nidcs
        phi = u if self.phase_only else u[self.nidcs :]
        g[offset:] = _second_diff_penalty_grad(phi, normalize_by_max=False)
        return g

    # -- spectral-match regularisation --------------------------------------
    def spectral_penalty(self, u, target: ArrayLike) -> float:
        r"""Relative spectral-amplitude mismatch to a target spectrum.

        Penalises the squared difference between the retrieved spectral
        amplitude :math:`|\tilde E(\omega)|` (the amplitude parameters) and a
        target amplitude ``target`` over the optimised support:

        .. math:: P = \lVert a - s\rVert^2 / \max(\lVert s\rVert^2, \varepsilon),

        a dimensionless relative mismatch. Pulls a full-mode retrieval towards a
        measured independent spectrum (and softly suppresses amplitude outside
        the measured band). Returns ``0`` in phase-only mode (no amplitude
        parameters). ``target`` is the full-grid amplitude :math:`s` (length
        ``N``), sliced here to the optimisation support.
        """
        if self.phase_only:
            return 0.0
        a = np.asarray(u, dtype=float)[: self.nidcs]
        s = np.asarray(target, dtype=float)[self._sl()]
        d = a - s
        denom = max(float(np.sum(s * s)), np.finfo(float).eps)
        return float(np.sum(d * d) / denom)

    def spectral_penalty_grad(self, u, target: ArrayLike) -> NDArray[np.float64]:
        """Gradient of :meth:`spectral_penalty` (amplitude block; phase block zero)."""
        g = np.zeros(self.nparams)
        if self.phase_only:
            return g
        a = np.asarray(u, dtype=float)[: self.nidcs]
        s = np.asarray(target, dtype=float)[self._sl()]
        denom = max(float(np.sum(s * s)), np.finfo(float).eps)
        g[: self.nidcs] = 2.0 * (a - s) / denom
        return g


# ---------------------------------------------------------------------------
# Second-difference smoothness penalty (shared)
# ---------------------------------------------------------------------------
def _second_diff_penalty(
    a: NDArray[np.float64], *, normalize_by_max: bool = True
) -> float:
    r"""``sum((a[i+1]-2a[i]+a[i-1])**2)`` normalised for a dimensionless measure.

    Amplitude penalty divides by ``(N-2)*max(a)**2``; the phase penalty
    (``normalize_by_max=False``) divides only by ``(N-2)`` (mean squared
    curvature in rad²).
    """
    n = a.size
    if n < 3:
        return 0.0
    d = a[2:] - 2.0 * a[1:-1] + a[:-2]
    s = float(np.sum(d * d))
    if normalize_by_max:
        return s / ((n - 2) * max(np.max(a) ** 2, np.finfo(float).eps))
    return s / (n - 2)


def _second_diff_penalty_grad(
    a: NDArray[np.float64], *, normalize_by_max: bool = True
) -> NDArray[np.float64]:
    """Gradient of :func:`_second_diff_penalty` (max(a) treated as constant)."""
    n = a.size
    g = np.zeros(n)
    if n < 3:
        return g
    d = a[2:] - 2.0 * a[1:-1] + a[:-2]  # length n-2, indexed i=1..n-2
    # 5-point stencil: dP/da[k] = 2 (d[k-1] - 2 d[k] + d[k+1]) with d=0 outside.
    grad = np.zeros(n)
    grad[2:] += d
    grad[1:-1] += -2.0 * d
    grad[:-2] += d
    if not normalize_by_max:
        return 2.0 / (n - 2) * grad
    amax = float(np.max(a))
    amax2 = max(amax**2, np.finfo(float).eps)
    out = 2.0 / ((n - 2) * amax2) * grad
    # Extra term from differentiating the max(a)**2 in the denominator.
    if amax**2 > np.finfo(float).eps and amax > 0:
        sumd2 = float(np.sum(d * d))
        out[int(np.argmax(a))] += -2.0 * sumd2 / ((n - 2) * amax**3)
    return out
