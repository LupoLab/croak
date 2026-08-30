r"""Optical material dispersion (Sellmeier refractive index).

The dispersive-propagation model needs a registry of
refractive-index data and the derived propagation constant
:math:`\beta(\omega)`.

Each material maps to a callable ``n(wavelength_m)`` returning the (real)
refractive index. :func:`beta` builds the slab propagation constant on a
frequency grid, referenced to the group velocity at ``omega0`` so that a pulse
propagates without an overall group delay — matching
the dispersive forward model.

Materials come from three sources, resolved in this order: runtime-registered
callables (:func:`register_material`), built-in **tabulated** datasets shipped
with the package (e.g. ``SiO2-Franta``, loaded lazily through
:func:`make_tabulated_material`), and built-in **Sellmeier** formulae. Sellmeier
coefficients are standard literature values (references in comments).

Tabulated data is experimental, so its refractive index is represented by a
generalised-cross-validation *smoothing* spline rather than a raw interpolant:
the dispersion quantities (:func:`beta1`, :func:`croak.dispersion.material_gdd`,
:func:`croak.dispersion.material_tod`) differentiate :math:`n(\lambda)`, and
finite differences of an interpolant through noisy samples amplify that noise
catastrophically in the second and third derivatives. The smoothing spline
regularises the noise once, at the representation layer, so the existing
finite-difference machinery is left untouched.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import make_smoothing_spline

from .constants import C
from .maths import derivative, wlfreq

__all__ = [
    "refractive_index",
    "beta",
    "beta1",
    "register_material",
    "unregister_material",
    "make_tabulated_material",
    "read_nk_csv",
    "MATERIALS",
]

# Refractive-index Sellmeier formulae as functions of wavelength in micrometres.
# Each returns the refractive index n(λ_µm). Kept in µm because that is the unit
# in which the literature coefficients are expressed.
_SELLMEIER_UM: dict[str, Callable[[NDArray[np.float64]], NDArray[np.float64]]] = {
    # Fused silica — J. Opt. Soc. Am. 55, 1205-1208 (1965)
    "SiO2": lambda u: np.sqrt(
        1.0
        + 0.6961663 / (1.0 - (0.0684043 / u) ** 2)
        + 0.4079426 / (1.0 - (0.1162414 / u) ** 2)
        + 0.8974794 / (1.0 - (9.896161 / u) ** 2)
    ),
    # BK7 — SCHOTT catalogue (refractiveindex.info)
    "BK7": lambda u: np.sqrt(
        1.0
        + 1.03961212 / (1.0 - 0.00600069867 / u**2)
        + 0.231792344 / (1.0 - 0.0200179144 / u**2)
        + 1.01046945 / (1.0 - 103.560653 / u**2)
    ),
    # Calcium fluoride — Appl. Opt. 41, 5275-5281 (2002)
    "CaF2": lambda u: np.sqrt(
        1.0
        + 0.443749998 / (1.0 - 0.00178027854 / u**2)
        + 0.444930066 / (1.0 - 0.00788536061 / u**2)
        + 0.150133991 / (1.0 - 0.0124119491 / u**2)
        + 8.85319946 / (1.0 - 2752.28175 / u**2)
    ),
    # Barium fluoride — J. Phys. Chem. Ref. Data 9, 161-289 (1980)
    "BaF2": lambda u: np.sqrt(
        1.0
        + 0.33973
        + 0.81070 / (1.0 - (0.10065 / u) ** 2)
        + 0.19652 / (1.0 - (29.87 / u) ** 2)
        + 4.52469 / (1.0 - (53.82 / u) ** 2)
    ),
    # Magnesium fluoride — Handbook of Optics
    "MgF2": lambda u: np.sqrt(
        1.0
        + 0.27620
        + 0.60967 / (1.0 - (0.08636 / u) ** 2)
        + 0.0080 / (1.0 - (18.0 / u) ** 2)
        + 2.14973 / (1.0 - (25.0 / u) ** 2)
    ),
}

# Built-in tabulated materials shipped as CSV data files under ``data/materials``
# (same packaging mechanism as the chirped-mirror data; see ``mirrors.py``). Each
# maps a material name to its filename; the file is loaded and turned into a
# smoothing-spline ``n(wavelength_m)`` lazily on first use (see
# :func:`_load_tabulated`), so importing this module does no I/O.
_DATA_DIR = Path(__file__).parent / "data" / "materials"
_TABULATED_FILES: dict[str, str] = {
    # Franta et al. broadband fused-silica optical constants (24.8 nm – 125 µm),
    # extending well beyond the Malitson Sellmeier range (0.21–6.7 µm).
    "SiO2-Franta": "FrantaSiO2.csv",
}

#: Names of the materials available in the dispersion registry (built-in
#: Sellmeier formulae plus built-in tabulated datasets).
MATERIALS: tuple[str, ...] = tuple(_SELLMEIER_UM) + tuple(_TABULATED_FILES)

# Runtime-registered materials (e.g. refractiveindex.info pages or custom data),
# each a callable ``n(wavelength_m) -> ndarray``. Consulted by
# :func:`refractive_index` and so transparently usable by :func:`beta`,
# :func:`dispersion.material_gdd` and the dispersion auto-tuners.
_CUSTOM: dict[str, Callable[[ArrayLike], NDArray[np.float64]]] = {}


def register_material(
    name: str, n_func: Callable[[ArrayLike], NDArray[np.float64]]
) -> None:
    """Register a custom ``n(wavelength_m)`` callable under ``name``.

    The callable must accept a vacuum wavelength array in metres and return the
    real refractive index; values outside its validity range should be returned
    as ``nan`` so :func:`beta` masks them. Registered materials are resolved
    case-insensitively and take precedence over the built-in Sellmeier set.
    """
    _CUSTOM[name] = n_func


def unregister_material(name: str) -> None:
    """Remove a previously registered custom material (no-op if absent)."""
    for key in [k for k in _CUSTOM if k.lower() == name.lower()]:
        del _CUSTOM[key]


def read_nk_csv(path: Path | str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read the refractive index from a stacked ``wl,n`` / ``wl,k`` CSV.

    The file format (as exported by some optical-constants databases, e.g.
    Franta et al.) stacks two blank-line-separated sections: a ``wl,n``
    refractive-index table followed by a ``wl,k`` extinction-coefficient table,
    each with a wavelength column in micrometres. Only the real refractive index
    ``n`` is read; the extinction ``k`` is ignored (the propagation model is
    lossless).

    Parameters
    ----------
    path : pathlib.Path or str
        Path to the CSV file.

    Returns
    -------
    (lam_m, n)
        Vacuum wavelength in metres and the real refractive index, in file
        order (ascending wavelength).

    Raises
    ------
    ValueError
        If the file contains no ``wl,n`` section.
    """
    lam_um: list[float] = []
    n_vals: list[float] = []
    section: str | None = None
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        header = line.lower().replace(" ", "")
        if header in ("wl,n", "wl,k"):
            section = header  # switch active section; subsequent rows belong to it
            continue
        if section == "wl,n":
            w, v = line.split(",")
            lam_um.append(float(w))
            n_vals.append(float(v))
    if not lam_um:
        raise ValueError(f"no 'wl,n' section found in {path}")
    # the wavelength column is in micrometres; convert to metres
    return np.asarray(lam_um, dtype=float) * 1e-6, np.asarray(n_vals, dtype=float)


def make_tabulated_material(
    lam_m: ArrayLike, n: ArrayLike
) -> tuple[Callable[[ArrayLike], NDArray[np.float64]], tuple[float, float]]:
    r"""Build a smooth ``n(wavelength_m)`` callable from tabulated samples.

    Intended for experimental refractive-index data. The index is represented by
    a *smoothing* spline (``scipy.interpolate.make_smoothing_spline``, with the
    smoothing penalty chosen automatically by generalised cross-validation) so
    that the derived dispersion quantities — which differentiate
    :math:`n(\lambda)` up to third order — are not corrupted by measurement
    noise. The fit is performed in ``log``-wavelength, because broadband optical
    data is typically (near-)logarithmically sampled; this gives uniform
    fractional resolution across many decades of wavelength.

    Parameters
    ----------
    lam_m : array_like
        Vacuum wavelength samples in metres (any order; duplicates are dropped).
    n : array_like
        Real refractive index at each wavelength.

    Returns
    -------
    (n_of_lambda_m, wl_range_m)
        ``n_of_lambda_m(lam_m)`` returns the smoothed refractive index, with
        ``nan`` outside the sampled range so :func:`beta` masks it.
        ``wl_range_m`` is ``(min_m, max_m)``.

    Raises
    ------
    ValueError
        If fewer than four valid samples remain after cleaning.
    """
    lam = np.asarray(lam_m, dtype=float)
    n_arr = np.asarray(n, dtype=float)
    # keep finite, positive-wavelength samples and sort ascending in wavelength
    finite = np.isfinite(lam) & np.isfinite(n_arr) & (lam > 0.0)
    lam, n_arr = lam[finite], n_arr[finite]
    order = np.argsort(lam)
    lam, n_arr = lam[order], n_arr[order]
    # make_smoothing_spline needs a strictly increasing abscissa
    keep = np.concatenate(([True], np.diff(lam) > 0.0))
    lam, n_arr = lam[keep], n_arr[keep]
    if lam.size < 4:
        raise ValueError("need at least 4 valid (wavelength, n) samples")

    lo, hi = float(lam[0]), float(lam[-1])
    spline = make_smoothing_spline(np.log(lam), n_arr)

    def n_of_lambda_m(query_m: ArrayLike) -> NDArray[np.float64]:
        q = np.asarray(query_m, dtype=float)
        in_range = (q >= lo) & (q <= hi)
        # clip before evaluating so the spline is never extrapolated (splines
        # diverge rapidly outside their knots); out-of-range entries become nan
        clipped = np.clip(q, lo, hi)
        value = np.asarray(spline(np.log(clipped)), dtype=float)
        return np.where(in_range, value, np.nan)

    return n_of_lambda_m, (lo, hi)


def _load_tabulated(
    material: str,
) -> Callable[[ArrayLike], NDArray[np.float64]] | None:
    """Lazily build and cache a built-in tabulated material's ``n`` callable.

    Returns the callable for a name in :data:`_TABULATED_FILES` (case-insensitive),
    caching it via :func:`register_material` so the CSV is read and the spline
    fitted only once; returns ``None`` for any other name.
    """
    for name, filename in _TABULATED_FILES.items():
        if name.lower() == material.lower():
            lam_m, n_vals = read_nk_csv(_DATA_DIR / filename)
            n_func, _ = make_tabulated_material(lam_m, n_vals)
            register_material(name, n_func)  # cache for subsequent lookups
            return n_func
    return None


def refractive_index(material: str) -> Callable[[ArrayLike], NDArray[np.float64]]:
    """Return a callable ``n(wavelength_m)`` for ``material``.

    Parameters
    ----------
    material : str
        Material name (case-insensitive); a built-in Sellmeier or tabulated
        material (see :data:`MATERIALS`) or one added via
        :func:`register_material`.

    Returns
    -------
    callable
        Function mapping vacuum wavelength in metres to the real refractive
        index. Non-finite results (e.g. at Sellmeier poles, or outside a
        registered or tabulated material's validity range) are returned as-is;
        callers working on a frequency grid should mask them.
    """
    for key, fn in _CUSTOM.items():
        if key.lower() == material.lower():
            return fn
    tabulated = _load_tabulated(material)
    if tabulated is not None:
        return tabulated
    key = _resolve(material)
    sell = _SELLMEIER_UM[key]

    def n(wavelength_m: ArrayLike) -> NDArray[np.float64]:
        um = np.asarray(wavelength_m, dtype=float) * 1e6
        return np.real(sell(um))

    n.__doc__ = f"Refractive index of {key} as a function of wavelength (m)."
    return n


def _resolve(material: str) -> str:
    for key in _SELLMEIER_UM:
        if key.lower() == material.lower():
            return key
    available = ", ".join((*MATERIALS, *_CUSTOM))
    raise ValueError(f"unknown material {material!r}; available: {available}")


def beta1(material: str, omega0: float) -> float:
    r"""Group delay per unit length :math:`\beta_1 = d\beta/d\omega` at ``omega0``.

    Computed by numerically differentiating :math:`\beta(\omega) =
    \omega\, n(\lambda(\omega))/c`.
    """
    n = refractive_index(material)

    def beta_scalar(w: float) -> float:
        return w * float(n(wlfreq(w))) / C

    return derivative(beta_scalar, float(omega0), order=1)


def beta(
    material: str | None,
    omega: ArrayLike,
    omega0: float,
) -> NDArray[np.float64]:
    r"""Propagation constant on a frequency grid, referenced to ``omega0``.

    Returns :math:`\beta(\omega) = (\omega+\omega_0)\,n/c - \beta_1(\omega_0)\,\omega`,
    where ``omega`` is the centred grid (relative to ``omega0``) and the group
    delay term keeps the pulse centred in the time window. Non-finite entries
    (outside the Sellmeier validity range) are set to zero.

    Parameters
    ----------
    material : str or None
        Material name, or ``None`` for a dispersionless medium (returns zeros).
    omega : array_like
        Centred angular-frequency grid (rad/s), relative to ``omega0``.
    omega0 : float
        Reference (carrier) angular frequency (rad/s).

    Returns
    -------
    numpy.ndarray
        Real propagation constant of the same length as ``omega``.
    """
    omega = np.asarray(omega, dtype=float)
    if material is None:
        return np.zeros_like(omega)
    n = refractive_index(material)
    omega_abs = omega + omega0
    with np.errstate(divide="ignore", invalid="ignore"):
        b = omega_abs / C * n(wlfreq(omega_abs))
        b = b - beta1(material, omega0) * omega
    b[~np.isfinite(b)] = 0.0
    return np.real(b)
