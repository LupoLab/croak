"""Apply a saved :class:`DispersionParams` to a retrieved pulse (headless).

The dispersion stage compensates residual dispersion on the *already-retrieved*
pulse: Taylor terms, propagation through built-in materials / a refractiveindex
material / a gas, and the beam-path mirror stack. The GUI does this interactively
against live registry entries; here we resolve the persisted parameters into the
same :func:`croak.dispersion.apply_dispersion` call, registering the gas / RI
material / mirrors under transient keys and cleaning them up afterwards so no
module-global state leaks.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from .. import dispersion, gases, io, materials, mirrors, refractive_db
from ..grid import Grid
from ..result import RetrievalResult
from .params import BeamPathMirror, DispersionParams

__all__ = ["compose_dispersion", "has_dispersion", "transfer_function"]

# fs^n → s^n for the Taylor coefficients (GDD fs², TOD fs³, FOD fs⁴).
_GDD_SI = 1e-30
_TOD_SI = 1e-45
_FOD_SI = 1e-60


def has_dispersion(p: DispersionParams) -> bool:
    """Whether ``p`` applies any dispersion (else composition is a no-op)."""
    return bool(
        p.gdd_fs2
        or p.tod_fs3
        or p.fod_fs4
        or (p.ri_page and p.ri_thickness_mm)
        or (p.gas_name and p.gas_path_cm)
        or any(p.material_thickness_mm.values())
        or any(m.bounces for m in p.mirrors)
    )


def _resolve_mirror(spec: BeamPathMirror):
    """Resolve a mirror spec to ``(phase_fn, amplitude_fn | None)``, or ``None``.

    Reuses the same built-in/custom loaders as the GUI mirror rows. Returns
    ``None`` for an inactive row (no name and no custom file).
    """
    if spec.custom_file:
        mat = io.load_csv(spec.custom_file)
        lam = mat[:, spec.custom_lam_col] * io.unit_to_si(spec.custom_unit)
        vals = mat[:, spec.custom_val_col]
        if spec.custom_datatype == "gdd":
            mode, vals = "gdd", vals * 1e-30  # fs² → s²
        else:
            mode = "phase"
        phase_fn, _rng = mirrors.mirror_from_arrays(lam, vals, mode=mode)
        return phase_fn, None  # custom mirrors are phase-only
    if spec.name in mirrors.BUILTIN_COATINGS:
        phase_fn, amp_fn, _rng = mirrors.load_coating(spec.name, spec.polarization)
        return phase_fn, amp_fn
    if spec.name in mirrors.BUILTIN_MIRRORS:
        phase_fn, amp_fn, _rng = mirrors.load_builtin(spec.name)
        return phase_fn, amp_fn
    return None


@contextmanager
def _registered(p: DispersionParams):
    """Register the gas / RI material / mirrors, yield ``(materials, bounces)``.

    Yields the ``material_thicknesses`` (m, by registry name) and ``mirror_bounces``
    (signed, by transient key) that :func:`croak.dispersion.apply_dispersion`
    consumes, then unregisters everything it added on exit.
    """
    mat_keys: list[str] = []
    mirror_keys: list[str] = []
    try:
        thicknesses: dict[str, float] = {
            name: mm * 1e-3 for name, mm in p.material_thickness_mm.items() if mm
        }
        # refractiveindex.info material (optional ``ridb`` extra)
        if p.ri_page and p.ri_thickness_mm:
            if not refractive_db.available():
                raise RuntimeError(
                    "this session uses a refractiveindex.info material but the "
                    "'ridb' extra is not installed; run `uv sync --extra ridb`"
                )
            n_func, _rng = refractive_db.make_material(p.ri_shelf, p.ri_book, p.ri_page)
            key = f"RI:{p.ri_book}/{p.ri_page}"
            materials.register_material(key, n_func)
            mat_keys.append(key)
            thicknesses[key] = p.ri_thickness_mm * 1e-3
        # active gas: pressure baked into n(λ), path length used as thickness
        if p.gas_name and p.gas_path_cm:
            n_func = gases.gas_refractive_index(p.gas_name, p.gas_pressure_bar)
            key = "_session_gas"
            materials.register_material(key, n_func)
            mat_keys.append(key)
            thicknesses[key] = p.gas_path_cm * 1e-2  # cm → m
        # beam-path mirror stack
        bounces: dict[str, int] = {}
        for i, spec in enumerate(p.mirrors):
            if not spec.bounces:
                continue
            resolved = _resolve_mirror(spec)
            if resolved is None:
                continue
            phase_fn, amp_fn = resolved
            key = f"_session_mirror_{i}"
            dispersion.register_mirror(key, phase_fn, amp_fn)
            mirror_keys.append(key)
            sign = -1 if spec.direction == "remove" else 1
            bounces[key] = sign * int(spec.bounces)
        yield thicknesses, bounces
    finally:
        for key in mat_keys:
            materials.unregister_material(key)
        for key in mirror_keys:
            dispersion.unregister_mirror(key)


def compose_dispersion(p: DispersionParams, result: RetrievalResult) -> RetrievalResult:
    """Apply the persisted dispersion compensation to ``result``'s spectrum.

    Parameters
    ----------
    p : DispersionParams
        The stage-4 settings (Taylor terms, materials, gas, RI material, mirrors).
    result : RetrievalResult
        The retrieved pulse to compensate.

    Returns
    -------
    RetrievalResult
        A copy of ``result`` with the dispersed spectrum. When ``p`` applies no
        dispersion the input is returned unchanged.

    Raises
    ------
    RuntimeError
        If a refractiveindex.info material is requested but the ``ridb`` extra is
        not installed.
    """
    if not has_dispersion(p):
        return result
    with _registered(p) as (thicknesses, bounces):
        ew = dispersion.apply_dispersion(
            result.grid.omega,
            result.omega0,
            result.spectrum,
            gdd=p.gdd_fs2 * _GDD_SI,
            tod=p.tod_fs3 * _TOD_SI,
            fod=p.fod_fs4 * _FOD_SI,
            material_thicknesses=thicknesses or None,
            mirror_bounces=bounces or None,
        )
    return replace(result, spectrum=ew)


def transfer_function(
    p: DispersionParams, grid: Grid, omega0: float
) -> NDArray[np.complex128]:
    r"""Build the spectral transfer function ``H(ω) = a(ω)·exp(iφ_prop(ω))`` of ``p``.

    Propagation between two beamline points is the deterministic linear map
    :math:`\tilde E'(\omega) = H(\omega)\,\tilde E(\omega)`. This resolves the
    persisted dispersion settings — exactly the ones :func:`compose_dispersion`
    applies — into that transfer function by applying them to an all-ones spectrum,
    so the phase (Taylor + material + gas + mirror) and any mirror reflectivity
    match the dispersion stage bit-for-bit. The result is the ``propagation``
    argument of the :mod:`croak.uncertainty` estimators and
    :func:`croak.covariance.covariance_uncertainty`, which propagate the retrieval
    uncertainty to that point. Returns the identity (all-ones) when ``p`` applies no
    dispersion.

    Parameters
    ----------
    p : DispersionParams
        The dispersion-stage settings (Taylor terms, materials, gas, RI material,
        mirrors).
    grid : Grid
        The retrieval grid (its centred ``omega`` axis defines ``H``).
    omega0 : float
        Carrier angular frequency (rad/s).

    Returns
    -------
    numpy.ndarray
        Complex transfer function on ``grid.omega`` (length ``grid.n``).

    Raises
    ------
    RuntimeError
        If a refractiveindex.info material is requested but the ``ridb`` extra is
        not installed.
    """
    ones = np.ones(grid.omega.shape, dtype=np.complex128)
    if not has_dispersion(p):
        return ones
    with _registered(p) as (thicknesses, bounces):
        return np.asarray(
            dispersion.apply_dispersion(
                grid.omega,
                omega0,
                ones,
                gdd=p.gdd_fs2 * _GDD_SI,
                tod=p.tod_fs3 * _TOD_SI,
                fod=p.fod_fs4 * _FOD_SI,
                material_thicknesses=thicknesses or None,
                mirror_bounces=bounces or None,
            ),
            dtype=np.complex128,
        )
