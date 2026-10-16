r"""Optical material dispersion: the refractive index and the propagation constant.

The dispersive-propagation model needs a registry of refractive-index data and the
derived propagation constant :math:`\beta(\omega)`.

Each material maps to a callable ``n(wavelength_m)`` returning the (real)
refractive index. :func:`beta` builds the slab propagation constant on a
frequency grid, referenced to the group velocity at ``omega0`` so that a pulse
propagates without an overall group delay — matching the dispersive forward model.

Materials are resolved in this order: runtime-registered callables
(:func:`register_material`), then the built-in materials of :data:`MATERIALS`.
The built-in materials come from `optmat <https://optmat.readthedocs.io>`_, a
curated library of optical constants: each name uses optmat's recommended model of
that material, whose sources, validity range and reason for the choice optmat
records (:func:`material_model` returns it). Its indices are absolute (relative to
vacuum), as the propagation model needs.

Experimental data of your own can be registered through
:func:`make_tabulated_material`, which represents the index by a
generalised-cross-validation *smoothing* spline rather than a raw interpolant: the
dispersion quantities (:func:`beta1`, :func:`croak.dispersion.material_gdd`,
:func:`croak.dispersion.material_tod`) differentiate :math:`n(\lambda)`, and
finite differences of an interpolant through noisy samples amplify that noise
catastrophically in the second and third derivatives. The smoothing spline
regularises the noise once, at the representation layer, so the
finite-difference machinery is unchanged.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from optmat.solid.models.base import Model
from optmat.solid.models.registry import default_model
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
    "material_model",
    "canonical_name",
    "read_nk_csv",
    "MATERIALS",
    "DESCRIPTIONS",
]


@dataclass(frozen=True)
class _Builtin:
    """A built-in material: the optmat material it is, and how to describe it."""

    optmat_id: str
    axis: str | None
    description: str


# Each built-in name is optmat's recommended model of that material
# (optmat.solid.models.registry.default_model). A uniaxial crystal uses its
# ordinary index: a window or plate is cut with the optic axis along its normal,
# so a beam at normal incidence sees n_o in either polarisation. croak's MgF2
# before optmat was the ordinary ray too (within 4e-4).
_BUILTINS: dict[str, _Builtin] = {
    "SiO2": _Builtin("SiO2", None, "fused silica"),
    "BK7": _Builtin("N-BK7", None, "SCHOTT N-BK7"),
    "CaF2": _Builtin("CaF2", None, "CaF₂"),
    "BaF2": _Builtin("BaF2", None, "BaF₂"),
    "MgF2": _Builtin("MgF2", "o", "MgF₂ (ordinary ray)"),
    "Al2O3": _Builtin("Al2O3", "o", "sapphire (ordinary ray)"),
    "Si": _Builtin("Si", None, "crystalline silicon (opaque below 1.1 µm)"),
}

# Names earlier croak versions offered, kept so saved sessions and scripts still
# resolve. SiO2-Franta was Franta et al.'s tabulated fused silica; optmat's SiO2
# model is Franta's universal dispersion model refitted to absolute refractometry
# (optmat TC-399), so the two names are now the same material.
_ALIASES: dict[str, str] = {"SiO2-Franta": "SiO2"}

#: Names of the built-in materials, each optmat's recommended model of it.
MATERIALS: tuple[str, ...] = tuple(_BUILTINS)

#: A short description of each built-in material, by name (for labels and help).
DESCRIPTIONS: dict[str, str] = {name: b.description for name, b in _BUILTINS.items()}

# Runtime-registered materials (e.g. a gas at a pressure, or custom data),
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
    case-insensitively and take precedence over the built-in materials.
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
    noise (calibration errors in the data remain). The fit is performed in
    ``log``-wavelength, because broadband optical data is typically
    (near-)logarithmically sampled; this gives uniform fractional resolution across many
    decades of wavelength.

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


def canonical_name(material: str) -> str:
    """Return the current name of a built-in material given by an earlier name.

    Parameters
    ----------
    material : str
        A material name, any case.

    Returns
    -------
    str
        The :data:`MATERIALS` name that ``material`` refers to (``"SiO2"`` for
        ``"sio2-franta"``), or ``material`` unchanged if it names no built-in
        material (a registered one, say).

    Examples
    --------
    >>> canonical_name("SiO2-Franta")
    'SiO2'
    >>> canonical_name("MyGlass")
    'MyGlass'
    """
    lowered = material.lower()
    for old, new in _ALIASES.items():
        if old.lower() == lowered:
            return new
    for name in _BUILTINS:
        if name.lower() == lowered:
            return name
    return material


@functools.cache
def material_model(material: str) -> Model:
    """Return the optmat model behind a built-in material.

    Parameters
    ----------
    material : str
        A name in :data:`MATERIALS` (or an earlier name of one), any case.

    Returns
    -------
    optmat.solid.models.base.Model
        optmat's recommended model of the material, with its ``name``,
        ``citation``, ``wavelength_range`` (m) and ``identity`` for provenance.

    Raises
    ------
    ValueError
        If ``material`` is not a built-in material.

    Examples
    --------
    >>> material_model("BK7").name
    'schott_2023/sellmeier'
    """
    name = canonical_name(material)
    if name not in _BUILTINS:
        raise ValueError(
            f"unknown material {material!r}; available: {', '.join(MATERIALS)}"
        )
    builtin = _BUILTINS[name]
    return default_model(builtin.optmat_id, axis=builtin.axis)


def _optmat_index(model: Model) -> Callable[[ArrayLike], NDArray[np.float64]]:
    """Wrap an optmat model as croak's ``n(wavelength_m)`` callable."""

    def n(wavelength_m: ArrayLike) -> NDArray[np.float64]:
        lam = np.asarray(wavelength_m, dtype=float)
        # A centred frequency grid can reach ω ≤ 0 (λ ≤ 0 or ∞), where optmat
        # refuses to evaluate: no frequency of a model lies there. Those points
        # become nan, which beta() masks, as it masks every unphysical entry.
        physical = np.isfinite(lam) & (lam > 0.0)
        out = np.full(lam.shape, np.nan)
        # warn=False: the grid spans far beyond any model's stated range, where
        # the pulse carries no energy, so a range warning would fire on every
        # call and mean nothing; the model is extrapolated there. The real part
        # is taken because the propagation model is lossless.
        out[physical] = np.real(model.refractive_index(lam[physical], warn=False))
        return out

    n.__doc__ = (
        f"Refractive index of {model.material} (optmat {model.name}) as a function "
        "of wavelength (m)."
    )
    return n


def refractive_index(material: str) -> Callable[[ArrayLike], NDArray[np.float64]]:
    """Return a callable ``n(wavelength_m)`` for ``material``.

    Parameters
    ----------
    material : str
        Material name (case-insensitive): one added via :func:`register_material`,
        or a built-in material (see :data:`MATERIALS`).

    Returns
    -------
    callable
        Function mapping vacuum wavelength in metres to the real refractive
        index. A built-in material gives ``nan`` at a non-positive or non-finite
        wavelength and is extrapolated beyond its model's stated range; a
        registered material returns what its callable returns. Callers working on
        a frequency grid should mask non-finite values.

    Raises
    ------
    ValueError
        If ``material`` is neither registered nor built in.
    """
    for key, fn in _CUSTOM.items():
        if key.lower() == material.lower():
            return fn
    if canonical_name(material) not in _BUILTINS:
        available = ", ".join((*MATERIALS, *_CUSTOM))
        raise ValueError(f"unknown material {material!r}; available: {available}")
    return _optmat_index(material_model(material))


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
    (a non-positive frequency, or outside a registered material's range) are set
    to zero.

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
        Real propagation constant (rad/m) of the same length as ``omega``.
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
