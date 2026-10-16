"""Tests for :mod:`croak.session.dispersion` (persisted dispersion → transfer function).

Every route resolves the saved :class:`~croak.session.params.DispersionParams`
into the registry-based :func:`croak.apply_dispersion` call and must leave the
material and mirror registries exactly as it found them.
"""

from __future__ import annotations

import numpy as np
import pytest

import croak
from croak import dispersion, materials, mirrors
from croak.result import RetrievalResult
from croak.session.dispersion import (
    compose_dispersion,
    has_dispersion,
    transfer_function,
)
from croak.session.params import BeamPathMirror, DispersionParams

GRID = croak.Grid(128, dt=0.5e-15)
OMEGA0 = croak.maths.wlfreq(800e-9)


@pytest.fixture
def registries_untouched():
    """Assert the transient material/mirror registrations are cleaned up."""
    mats = dict(materials._CUSTOM)
    mirs = set(dispersion.MIRRORS)
    yield
    assert dict(materials._CUSTOM) == mats
    assert set(dispersion.MIRRORS) == mirs


def _custom_mirror_csv(tmp_path, values) -> str:
    """Write a two-column (wavelength nm, value) CSV spanning 700-900 nm."""
    lam_nm = np.linspace(700.0, 900.0, 101)
    path = tmp_path / "mirror.csv"
    np.savetxt(path, np.column_stack([lam_nm, values(lam_nm)]), delimiter=",")
    return str(path)


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        (DispersionParams(), False),
        (DispersionParams(gdd_fs2=1.0), True),
        (DispersionParams(tod_fs3=1.0), True),
        (DispersionParams(fod_fs4=1.0), True),
        # a gas without a path does nothing
        (DispersionParams(gas_name="He"), False),
        (DispersionParams(gas_name="He", gas_path_cm=1.0), True),
        (DispersionParams(material_thickness_mm={"SiO2": 0.0}), False),
        (DispersionParams(material_thickness_mm={"SiO2": 0.1}), True),
        (DispersionParams(mirrors=[BeamPathMirror(name="PC70")]), False),
        (DispersionParams(mirrors=[BeamPathMirror(name="PC70", bounces=1)]), True),
    ],
)
def test_has_dispersion_detects_each_active_term(params, expected):
    assert has_dispersion(params) is expected


def test_transfer_function_identity_without_dispersion():
    h = transfer_function(DispersionParams(), GRID, OMEGA0)
    assert h.dtype == np.complex128
    np.testing.assert_array_equal(h, np.ones(GRID.n))


def test_transfer_function_matches_taylor_phase(registries_untouched):
    """A pure-GDD session is the analytic phase exp(i·GDD·ω²/2)."""
    h = transfer_function(DispersionParams(gdd_fs2=40.0), GRID, OMEGA0)
    expected = np.exp(1j * 40e-30 * GRID.omega**2 / 2)
    # both are the same closed form; only rounding differs
    np.testing.assert_allclose(h, expected, rtol=1e-12, atol=1e-12)


def test_transfer_function_gas_matches_direct_registration(registries_untouched):
    """The gas row is the gas index at its pressure over the path length."""
    params = DispersionParams(gas_name="He", gas_pressure_bar=3.0, gas_path_cm=50)
    h = transfer_function(params, GRID, OMEGA0)
    n_func = croak.gases.gas_refractive_index("He", 3.0)
    materials.register_material("_test_gas", n_func)
    try:
        direct = croak.apply_dispersion(
            GRID.omega,
            OMEGA0,
            np.ones(GRID.n, complex),
            material_thicknesses={"_test_gas": 0.5},
        )
    finally:
        materials.unregister_material("_test_gas")
    # identical arithmetic on the same inputs
    np.testing.assert_allclose(h, direct, rtol=1e-12, atol=1e-12)
    assert not np.allclose(h, 1.0)


@pytest.mark.parametrize("datatype", ["gdd", "phase"])
def test_custom_mirror_add_and_remove_are_inverse(
    tmp_path, datatype, registries_untouched
):
    """A custom (phase-only) mirror: 'add' undoes 'remove', and |H| stays 1."""
    if datatype == "gdd":
        path = _custom_mirror_csv(tmp_path, lambda lam: -50.0 + 0.1 * (lam - 800.0))
    else:
        path = _custom_mirror_csv(tmp_path, lambda lam: 1e-4 * (lam - 800.0) ** 2)

    def row(direction):
        return BeamPathMirror(
            bounces=2,
            direction=direction,
            custom_file=path,
            custom_lam_col=0,
            custom_val_col=1,
            custom_unit="nm",
            custom_datatype=datatype,
        )

    h_add = transfer_function(DispersionParams(mirrors=[row("add")]), GRID, OMEGA0)
    h_rem = transfer_function(DispersionParams(mirrors=[row("remove")]), GRID, OMEGA0)
    # phase-only: unit modulus everywhere (rounding only)
    np.testing.assert_allclose(np.abs(h_add), 1.0, atol=1e-12)
    np.testing.assert_allclose(h_add * h_rem, 1.0, atol=1e-12)
    assert not np.allclose(h_add, 1.0)


def test_builtin_coating_carries_its_reflectivity(registries_untouched):
    """A coating row applies the coating's amplitude as well as its phase."""
    name = next(iter(mirrors.BUILTIN_COATINGS))
    spec = BeamPathMirror(name=name, polarization="s", bounces=3, direction="add")
    h = transfer_function(DispersionParams(mirrors=[spec]), GRID, OMEGA0)
    lam = croak.maths.wlfreq(GRID.omega + OMEGA0)
    inside = (lam > 700e-9) & (lam < 900e-9)
    # a lossy coating: reflectivity below one (rounding aside) and not unity
    assert np.all(np.abs(h[inside]) <= 1.0 + 1e-12)
    assert not np.allclose(np.abs(h[inside]), 1.0)


def test_inactive_and_unknown_mirror_rows_are_skipped(registries_untouched):
    """Zero-bounce rows and unresolvable names contribute nothing."""
    params = DispersionParams(
        gdd_fs2=10.0,
        mirrors=[
            BeamPathMirror(name="PC70", bounces=0),
            BeamPathMirror(name="no such mirror", bounces=4),
        ],
    )
    h = transfer_function(params, GRID, OMEGA0)
    gdd_only = transfer_function(DispersionParams(gdd_fs2=10.0), GRID, OMEGA0)
    np.testing.assert_array_equal(h, gdd_only)


def test_compose_dispersion_applies_the_transfer_function(registries_untouched):
    """compose_dispersion multiplies the spectrum by exactly transfer_function."""
    ew = croak.gaussian_pulse(GRID, 8e-15)
    result = RetrievalResult(
        spectrum=ew,
        grid=GRID,
        delays=np.zeros(1),
        interaction="shg",
        algorithm="copra",
        error=0.0,
        omega0=OMEGA0,
    )
    params = DispersionParams(gdd_fs2=30.0, gas_name="He", gas_path_cm=10.0)
    composed = compose_dispersion(params, result)
    h = transfer_function(params, GRID, OMEGA0)
    np.testing.assert_allclose(composed.spectrum, ew * h, rtol=1e-12, atol=1e-14)
