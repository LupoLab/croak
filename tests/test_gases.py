"""Tests for the gas dispersion model in :mod:`croak.gases`."""

import numpy as np
import pytest
from optmat.conditions import State

from croak import dispersion, gases, materials
from croak.grid import Grid
from croak.maths import fwhm, wlfreq
from croak.pulses import gaussian_pulse

_ZERO_C = 273.15  # K


def test_gas_reference_refractive_index():
    # n−1 at 1 bar, 0 °C, 800 nm. Air: Börzsönyi et al. (2008) give 2.861e-4 at
    # these conditions; optmat's dry air is the sum of its constituents' models
    # (within 2e-6). Helium: 3.48e-5 at 1 atm in the literature, so 3.43e-5 at
    # 1 bar; optmat's ab initio fit is good to 0.1 ppm (within 5e-8).
    n_air = gases.gas_refractive_index("Air", 1.0, _ZERO_C)
    n_he = gases.gas_refractive_index("He", 1.0, _ZERO_C)
    assert float(n_air(800e-9)) - 1.0 == pytest.approx(2.861e-4, abs=2e-6)
    assert float(n_he(800e-9)) - 1.0 == pytest.approx(3.4287e-5, abs=5e-8)
    # helium is far less dispersive than air
    assert float(n_he(800e-9)) - 1.0 < float(n_air(800e-9)) - 1.0


@pytest.mark.parametrize("gas", gases.GASES)
def test_every_gas_is_optmats_model_at_the_real_gas_density(gas):
    # croak adds the bar → Pa conversion, the real part and the mask, nothing more.
    lam = np.array([400e-9, 800e-9, 1600e-9])
    model = gases.gas_model(gas)
    expected = model.refractive_index(lam, State(2e5, 300.0), warn=False)
    np.testing.assert_array_equal(
        gases.gas_refractive_index(gas, 2.0, 300.0)(lam), np.real(expected)
    )


def test_a_retired_gas_name_resolves():
    assert gases.canonical_name("Helium") == "He"
    assert gases.canonical_name("air") == "Air"
    assert gases.gas_model("helium") is gases.gas_model("He")


def test_gas_zero_pressure_is_vacuum():
    n = gases.gas_refractive_index("Air", 0.0)
    assert float(n(800e-9)) == 1.0
    assert np.isnan(n(np.array([-1e-6]))).all()


def test_an_unphysical_wavelength_is_nan():
    n = gases.gas_refractive_index("Ar", 1.0)(np.array([0.0, -1e-6, np.inf, 1e-6]))
    assert np.all(np.isnan(n[:3]))
    assert np.isfinite(n[3])


def test_gas_pressure_scaling_linear():
    # N ∝ pressure, and n−1 ≈ Nα/(2ε₀), so n−1 doubles from 1 to 2 bar; air's
    # compressibility (Z−1 ≈ −4e-4 per bar at 0 °C) and Lorentz–Lorenz's
    # local-field term (~n−1, 3e-4) each depart from that by a few 1e-4.
    n1 = gases.gas_refractive_index("Air", 1.0, _ZERO_C)
    n2 = gases.gas_refractive_index("Air", 2.0, _ZERO_C)
    assert float(n2(800e-9)) - 1.0 == pytest.approx(
        2.0 * (float(n1(800e-9)) - 1.0), rel=1e-3
    )


def test_gas_temperature_scaling_inverse():
    # n−1 ∝ 1/T at fixed pressure: doubling T halves it (to a few 1e-4, as above).
    n_t = gases.gas_refractive_index("Air", 1.0, _ZERO_C)
    n_2t = gases.gas_refractive_index("Air", 1.0, 2.0 * _ZERO_C)
    assert float(n_2t(800e-9)) - 1.0 == pytest.approx(
        0.5 * (float(n_t(800e-9)) - 1.0), rel=1e-3
    )


def test_gas_default_temperature_is_room():
    # The default temperature is room temperature: omitting it must give exactly
    # the same index as passing ROOM_TEMPERATURE_K explicitly, and a lower index
    # than at 0 °C (warmer gas is less dense).
    n_default = gases.gas_refractive_index("Air", 1.0)
    n_room = gases.gas_refractive_index("Air", 1.0, gases.ROOM_TEMPERATURE_K)
    n_cold = gases.gas_refractive_index("Air", 1.0, _ZERO_C)
    assert float(n_default(800e-9)) == float(n_room(800e-9))
    assert float(n_default(800e-9)) - 1.0 < float(n_cold(800e-9)) - 1.0


def test_gas_unknown_raises():
    with pytest.raises(ValueError, match="unknown gas 'Plasma'"):
        gases.gas_refractive_index("Plasma", 1.0)


@pytest.mark.parametrize(
    ("pressure", "temperature", "match"),
    [
        (-1.0, 293.15, "pressure must be finite and >= 0"),
        (np.nan, 293.15, "pressure must be finite and >= 0"),
        (1.0, 0.0, "temperature must be finite and > 0"),
        (1.0, np.inf, "temperature must be finite and > 0"),
    ],
)
def test_an_unphysical_state_raises(pressure, temperature, match):
    with pytest.raises(ValueError, match=match):
        gases.gas_refractive_index("Air", pressure, temperature)


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


def test_the_real_gas_density_matters_for_a_heavy_gas():
    # At 10 bar and 20 °C SF6 is 15 % denser than the ideal gas (Z ≈ 0.87).
    state = State(10e5, gases.ROOM_TEMPERATURE_K)
    ideal = gases.gas_model("SF6").refractive_index(800e-9, state, ideal=True)
    n = gases.gas_refractive_index("SF6", 10.0)(800e-9)
    assert (float(n) - 1.0) / (float(np.real(ideal)) - 1.0) == pytest.approx(
        1.15, abs=0.01
    )


def test_a_condensed_state_is_refused_when_the_gas_is_built():
    # SF6's vapour pressure at 20 °C is about 21 bar.
    with pytest.raises(ValueError, match="condensed"):
        gases.gas_refractive_index("SF6", 30.0)
