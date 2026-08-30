r"""Interactive residual-dispersion tuning of a retrieved pulse.

Applies Taylor (GDD/TOD/FOD) and material dispersion to a retrieved complex
spectrum and provides auto-tuners that maximise the temporal peak power (i.e.
compress the pulse).

Sign convention matches :func:`croak.pulses.gaussian_pulse`: a *positive* GDD
adds the phase :math:`+\tfrac12\,\mathrm{GDD}\,\omega^2`, so applying a
material's GDD and then its negative cancels exactly, and ``material_gdd`` is
positive for normal dispersion.

Chirped-mirror dispersion is data-dependent; :data:`MIRRORS` is an empty,
extensible registry (use :func:`register_mirror`).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import differential_evolution, direct, dual_annealing, minimize

from . import materials
from .constants import C
from .maths import derivative, wlfreq

__all__ = [
    "apply_dispersion",
    "dispersion_phase",
    "material_gdd",
    "material_tod",
    "auto_taylor",
    "auto_material",
    "register_mirror",
    "unregister_mirror",
    "OPTIMIZERS",
    "MIRRORS",
    "MIRROR_AMPLITUDES",
]

#: Available global/local optimisers for the dispersion auto-tuners.
OPTIMIZERS = ("differential_evolution", "dual_annealing", "direct", "nelder-mead")

Complex = NDArray[np.complex128]

#: A per-bounce spectral function ``f(omega_abs) -> ndarray`` (phase or amplitude).
MirrorFn = Callable[[NDArray[np.float64]], NDArray[np.float64]]

#: Registry of chirped-mirror dispersion data, keyed by name. Each entry is a
#: callable ``phase(omega_abs) -> ndarray`` giving the per-bounce spectral phase.
MIRRORS: dict[str, MirrorFn] = {}

#: Optional per-bounce *amplitude* (reflectivity) registry, keyed by the same
#: name as :data:`MIRRORS`. Each entry is a callable ``r(omega_abs) -> ndarray``
#: returning the amplitude reflectivity (0–1), with ``r = 1`` outside its valid
#: band so it is a no-op there. A name present here has its reflectivity applied
#: by :func:`apply_dispersion` (removal divides by it); a name absent here is
#: treated as lossless (phase-only) — the convention for chirped compressors.
MIRROR_AMPLITUDES: dict[str, MirrorFn] = {}

#: Upper bound on the net per-mirror amplitude gain. Removing a mirror divides
#: the spectrum by ``r(ω)^N``; where ``R`` is small (anti-reflection dips or
#: out-of-band noise) this would amplify unboundedly, so the total amplitude
#: factor is clamped to ``[1/_MIRROR_AMPLITUDE_CAP, _MIRROR_AMPLITUDE_CAP]``.
_MIRROR_AMPLITUDE_CAP = 1e3


def register_mirror(
    name: str, phase_fn: MirrorFn, amplitude_fn: MirrorFn | None = None
) -> None:
    """Register a mirror's per-bounce spectral-phase (and optional amplitude).

    Parameters
    ----------
    name : str
        Registry key used in ``mirror_bounces``.
    phase_fn : callable
        ``phase(omega_abs) -> ndarray``, the per-bounce spectral phase (rad).
    amplitude_fn : callable, optional
        ``r(omega_abs) -> ndarray``, the per-bounce amplitude reflectivity
        (0–1), returning 1 outside its band. When given, :func:`apply_dispersion`
        applies it (removal — a negative bounce count — divides by it). When
        omitted the mirror is lossless (phase-only) and any stale amplitude under
        ``name`` is dropped.
    """
    MIRRORS[name] = phase_fn
    if amplitude_fn is not None:
        MIRROR_AMPLITUDES[name] = amplitude_fn
    else:
        MIRROR_AMPLITUDES.pop(name, None)


def unregister_mirror(name: str) -> None:
    """Remove a mirror's phase and amplitude functions from the registries."""
    MIRRORS.pop(name, None)
    MIRROR_AMPLITUDES.pop(name, None)


def _beta_abs(material: str, omega_abs: float) -> float:
    """Absolute propagation constant β(ω) = ω/c · n(λ)."""
    n = materials.refractive_index(material)
    return omega_abs / C * float(n(wlfreq(omega_abs)))


def apply_dispersion(
    omega: ArrayLike,
    omega0: float,
    ew: ArrayLike,
    *,
    gdd: float = 0.0,
    tod: float = 0.0,
    fod: float = 0.0,
    material_thicknesses: dict[str, float] | None = None,
    mirror_bounces: dict[str, int] | None = None,
) -> Complex:
    r"""Apply Taylor + material + mirror dispersion to a spectrum (returns a copy).

    Parameters
    ----------
    omega : array_like
        Centred angular-frequency grid (rad/s).
    omega0 : float
        Carrier angular frequency (rad/s).
    ew : array_like
        Complex spectrum (centred order).
    gdd, tod, fod : float, optional
        Taylor coefficients in SI (s², s³, s⁴). Phase added:
        ``GDD·ω²/2 + TOD·ω³/6 + FOD·ω⁴/24``.
    material_thicknesses : dict, optional
        ``{material_name: thickness_m}`` of material to propagate through.
    mirror_bounces : dict, optional
        ``{mirror_name: n_bounces}`` for registered mirrors. A *negative* count
        back-propagates (removes) the mirror. Mirrors that also have a registered
        amplitude (:data:`MIRROR_AMPLITUDES`) have their reflectivity applied too
        (removal divides by it, clamped by ``_MIRROR_AMPLITUDE_CAP``).

    Returns
    -------
    numpy.ndarray
        The dispersed complex spectrum.
    """
    out = np.asarray(ew, dtype=complex)
    phi = dispersion_phase(
        omega,
        omega0,
        gdd=gdd,
        tod=tod,
        fod=fod,
        material_thicknesses=material_thicknesses,
        mirror_bounces=mirror_bounces,
    )
    out = out * np.exp(1j * phi)
    amp = _mirror_amplitude(omega, omega0, mirror_bounces)
    if amp is not None:
        out = out * amp
    return np.asarray(out, dtype=np.complex128)


def dispersion_phase(
    omega: ArrayLike,
    omega0: float,
    *,
    gdd: float = 0.0,
    tod: float = 0.0,
    fod: float = 0.0,
    material_thicknesses: dict[str, float] | None = None,
    mirror_bounces: dict[str, int] | None = None,
) -> NDArray[np.float64]:
    r"""Spectral phase that :func:`apply_dispersion` adds (Taylor/material/mirror).

    This is exactly the argument of the ``exp(i·φ)`` factor applied to the
    spectrum, with the same sign convention; useful for plotting the *applied*
    dispersion separately from the pulse's own retrieved phase.
    """
    omega = np.asarray(omega, dtype=float)
    phi = gdd * omega**2 / 2 + tod * omega**3 / 6 + fod * omega**4 / 24
    if material_thicknesses:
        for material, thickness in material_thicknesses.items():
            phi = phi + materials.beta(material, omega, omega0) * thickness
    if mirror_bounces:
        omega_abs = omega + omega0
        for name, bounces in mirror_bounces.items():
            if name in MIRRORS and bounces:
                phi = phi + MIRRORS[name](omega_abs) * bounces
    return phi


def _mirror_amplitude(
    omega: ArrayLike, omega0: float, mirror_bounces: dict[str, int] | None
) -> NDArray[np.float64] | None:
    r"""Net per-spectrum amplitude factor from mirrors that carry reflectivity.

    Returns :math:`\prod_k r_k(\omega)^{N_k}` over mirrors with a registered
    amplitude, where ``N_k`` is the signed bounce count (negative removes, i.e.
    divides by ``r``). The product is clamped to
    ``[1/_MIRROR_AMPLITUDE_CAP, _MIRROR_AMPLITUDE_CAP]`` so a small ``r`` (an
    anti-reflection dip or out-of-band noise) cannot blow the amplitude up.
    Returns ``None`` when no mirror contributes amplitude (the common phase-only
    case), so the caller can skip the multiplication entirely.
    """
    if not mirror_bounces:
        return None
    omega = np.asarray(omega, dtype=float)
    omega_abs = omega + omega0
    amp: NDArray[np.float64] | None = None
    for name, bounces in mirror_bounces.items():
        r_fn = MIRROR_AMPLITUDES.get(name)
        if r_fn is None or not bounces:
            continue
        factor = np.asarray(r_fn(omega_abs), dtype=float) ** bounces
        amp = factor if amp is None else amp * factor
    if amp is None:
        return None
    return np.clip(amp, 1.0 / _MIRROR_AMPLITUDE_CAP, _MIRROR_AMPLITUDE_CAP)


def material_gdd(material: str, omega0: float, thickness: float) -> float:
    r"""GDD (fs²) of ``thickness`` of ``material`` at the carrier ``omega0``."""
    beta2 = derivative(lambda w: _beta_abs(material, w), float(omega0), 2)
    return beta2 * thickness / 1e-30


def material_tod(material: str, omega0: float, thickness: float) -> float:
    r"""TOD (fs³) of ``thickness`` of ``material`` at the carrier ``omega0``.

    Estimated from the third derivative of β via two second-derivative samples.
    """
    dw = (abs(omega0) + 1.0) * 1e-4
    b2p = derivative(lambda w: _beta_abs(material, w), float(omega0) + dw, 2)
    b2m = derivative(lambda w: _beta_abs(material, w), float(omega0) - dw, 2)
    beta3 = (b2p - b2m) / (2 * dw)
    return beta3 * thickness / 1e-45


def _peak_power(grid, ew: Complex) -> float:
    """Absolute temporal peak power (the spectrum's amplitude/energy is fixed)."""
    return float(np.max(np.abs(grid.ifft(ew)) ** 2))


def _optimize(func, bounds, method: str, *, x0=None, simplex_steps=None):
    """Minimise ``func`` over ``bounds`` with the chosen scipy optimiser.

    ``bounds`` is a sequence of ``(lo, hi)`` pairs. Global methods
    (differential evolution, dual annealing, DIRECT) explore the whole box;
    ``"nelder-mead"`` is a local polish. ``x0`` (default the all-zero
    "no change" point) seeds the search so the optimum can never be worse than
    leaving the dispersion untouched.
    """
    bounds = [tuple(b) for b in bounds]
    n = len(bounds)
    if x0 is None:
        x0 = np.zeros(n)
    x0 = np.asarray(x0, dtype=float)
    method = method.lower().replace(" ", "_").replace("-", "_")

    if method in ("differential_evolution", "de"):
        # x0 puts the no-change baseline in the initial population, so DE keeps a
        # solution at least as good as doing nothing.
        res = differential_evolution(
            func, bounds, x0=x0, tol=1e-10, polish=True, rng=0, maxiter=1000
        )
        x = np.atleast_1d(res.x)
    elif method in ("dual_annealing", "annealing"):
        res = dual_annealing(func, bounds, x0=x0, maxiter=1000, rng=0)
        x = np.atleast_1d(res.x)
    elif method == "direct":
        res = direct(func, bounds, maxiter=5000)
        x = np.atleast_1d(res.x)
    else:  # local Nelder–Mead from x0
        if simplex_steps is None:
            simplex_steps = [0.25 * (hi - lo) for lo, hi in bounds]
        simplex = np.vstack(
            [x0] + [x0 + np.eye(n)[i] * simplex_steps[i] for i in range(n)]
        )
        res = minimize(
            func,
            x0,
            method="Nelder-Mead",
            options={
                "maxiter": 5000,
                "xatol": 1e-2,
                "fatol": 1e-16,
                "initial_simplex": simplex,
            },
        )
        x = np.atleast_1d(res.x)

    # never return something worse than the no-change baseline
    return x if func(x) <= func(x0) else x0


def auto_taylor(
    grid,
    ew: ArrayLike,
    omega0: float,
    *,
    bounds=((-10000.0, 10000.0), (-100000.0, 100000.0), (-1000000.0, 1000000.0)),
    method: str = "differential_evolution",
) -> tuple[float, float, float]:
    """Find ``(GDD, TOD, FOD)`` in SI that maximise the absolute peak power.

    Parameters
    ----------
    grid, ew, omega0
        Grid, complex spectrum and carrier frequency.
    bounds : sequence of (lo, hi)
        Search bounds for ``(GDD, TOD, FOD)`` in fs² / fs³ / fs⁴ (use the
        slider bounds).
    method : str
        One of :data:`OPTIMIZERS`.

    Returns
    -------
    tuple
        ``(GDD, TOD, FOD)`` in SI (s², s³, s⁴).
    """
    ew = np.asarray(ew, dtype=complex)
    omega = grid.omega

    def neg_peak(x):
        mod = apply_dispersion(
            omega, omega0, ew, gdd=x[0] * 1e-30, tod=x[1] * 1e-45, fod=x[2] * 1e-60
        )
        return -_peak_power(grid, mod)

    x = _optimize(neg_peak, bounds, method)
    return x[0] * 1e-30, x[1] * 1e-45, x[2] * 1e-60


def auto_material(
    grid,
    ew: ArrayLike,
    omega0: float,
    material: str,
    *,
    bounds: tuple[float, float] = (-20e-3, 20e-3),
    method: str = "differential_evolution",
) -> float:
    """Find the ``material`` thickness (m) maximising the absolute peak power."""
    ew = np.asarray(ew, dtype=complex)
    omega = grid.omega

    def neg_peak(x):
        mod = apply_dispersion(
            omega, omega0, ew, material_thicknesses={material: float(x[0])}
        )
        return -_peak_power(grid, mod)

    x = _optimize(neg_peak, [bounds], method)
    return float(x[0])
