"""Tests for the gas dispersion model in :mod:`croak.gases`."""

import numpy as np
import pytest

from croak import dispersion, gases, materials
from croak.grid import Grid
from croak.maths import fwhm, wlfreq
from croak.pulses import gaussian_pulse


def test_gas_reference_refractive_index():
    # n−1 at the reference conditions (1 bar, 273.15 K) at 800 nm. Reference
    # values follow from the ported Sellmeier coefficients and match the
    # literature (air ≈ 2.86e-4, helium ≈ 3.48e-5 per bar near 800 nm).
    n_air = gases.gas_refractive_index("Air", 1.0, gases.GAS_REFERENCE_TEMPERATURE_K)
    n_he = gases.gas_refractive_index("Helium", 1.0, gases.GAS_REFERENCE_TEMPERATURE_K)
    assert float(n_air(800e-9)) - 1.0 == pytest.approx(2.861e-4, abs=2e-6)
    assert float(n_he(800e-9)) - 1.0 == pytest.approx(3.479e-5, abs=5e-7)
    # helium is far less dispersive than air
    assert float(n_he(800e-9)) - 1.0 < float(n_air(800e-9)) - 1.0


def test_gas_zero_pressure_is_vacuum():
    n = gases.gas_refractive_index("Air", 0.0)
    assert float(n(800e-9)) == pytest.approx(1.0, abs=1e-15)


def test_gas_pressure_scaling_linear():
    # χ₁ ∝ pressure (ideal gas), and n−1 ≈ χ₁/2, so n−1 doubles from 1 to 2 bar.
    t = gases.GAS_REFERENCE_TEMPERATURE_K
    n1 = gases.gas_refractive_index("Air", 1.0, t)
    n2 = gases.gas_refractive_index("Air", 2.0, t)
    assert float(n2(800e-9)) - 1.0 == pytest.approx(
        2.0 * (float(n1(800e-9)) - 1.0), rel=1e-3
    )


def test_gas_temperature_scaling_inverse():
    # n−1 ∝ 1/T at fixed pressure: doubling T halves it.
    t = gases.GAS_REFERENCE_TEMPERATURE_K
    n_t = gases.gas_refractive_index("Air", 1.0, t)
    n_2t = gases.gas_refractive_index("Air", 1.0, 2.0 * t)
    assert float(n_2t(800e-9)) - 1.0 == pytest.approx(
        0.5 * (float(n_t(800e-9)) - 1.0), rel=1e-3
    )


def test_gas_default_temperature_is_room():
    # The default temperature is room temperature: omitting it must give exactly
    # the same index as passing ROOM_TEMPERATURE_K explicitly, and a lower index
    # than at the colder 0 °C reference (warmer gas is less dense).
    n_default = gases.gas_refractive_index("Air", 1.0)
    n_room = gases.gas_refractive_index("Air", 1.0, gases.ROOM_TEMPERATURE_K)
    n_ref = gases.gas_refractive_index("Air", 1.0, gases.GAS_REFERENCE_TEMPERATURE_K)
    assert float(n_default(800e-9)) == float(n_room(800e-9))
    assert float(n_default(800e-9)) - 1.0 < float(n_ref(800e-9)) - 1.0


def test_gas_unknown_raises():
    with pytest.raises(ValueError):
        gases.gas_refractive_index("Plasma", 1.0)


@pytest.fixture
def air_key():
    """Register a high-pressure air material and clean it up afterwards."""
    key = "_test_air"
    materials.register_material(key, gases.gas_refractive_index("Air", 10.0))
    yield key
    materials.unregister_material(key)


def test_gas_normal_dispersion_positive_gdd(air_key):
    # Air has normal GVD (positive GDD) in the NIR, like a solid.
    omega0 = float(wlfreq(800e-9))
    gdd = dispersion.material_gdd(air_key, omega0, 1.0)  # over 1 m
    assert gdd > 0


def test_gas_gdd_scales_with_pressure():
    # GDD ∝ pressure (β ≈ ω/c·(1 + χ₁/2), χ₁ ∝ pressure).
    omega0 = float(wlfreq(800e-9))
    materials.register_material("_g1", gases.gas_refractive_index("Air", 1.0))
    materials.register_material("_g2", gases.gas_refractive_index("Air", 2.0))
    try:
        g1 = dispersion.material_gdd("_g1", omega0, 1.0)
        g2 = dispersion.material_gdd("_g2", omega0, 1.0)
        assert g2 == pytest.approx(2.0 * g1, rel=1e-2)
    finally:
        materials.unregister_material("_g1")
        materials.unregister_material("_g2")


def test_gas_apply_dispersion_broadens_and_roundtrips(air_key):
    # A gas path broadens a transform-limited pulse, and the negative path
    # exactly cancels it (the gas is just another material in the pipeline).
    g = Grid(256, dt=0.5e-15)
    omega0 = float(wlfreq(800e-9))
    ew = gaussian_pulse(g, 8e-15)
    tl = fwhm(g.t, np.abs(g.ifft(ew)) ** 2)

    chirped = dispersion.apply_dispersion(
        g.omega, omega0, ew, material_thicknesses={air_key: 1.0}
    )
    assert fwhm(g.t, np.abs(g.ifft(chirped)) ** 2) > tl

    recovered = dispersion.apply_dispersion(
        g.omega, omega0, chirped, material_thicknesses={air_key: -1.0}
    )
    assert np.allclose(recovered, ew, atol=1e-12)


def test_gas_auto_material_compensates(air_key):
    # Pre-chirp with a gas path, then let auto_material remove it.
    g = Grid(256, dt=0.5e-15)
    omega0 = float(wlfreq(800e-9))
    ew = gaussian_pulse(g, 8e-15)
    chirped = dispersion.apply_dispersion(
        g.omega, omega0, ew, material_thicknesses={air_key: 1.0}
    )
    length = dispersion.auto_material(g, chirped, omega0, air_key, bounds=(-3.0, 3.0))
    assert length == pytest.approx(-1.0, abs=0.1)
