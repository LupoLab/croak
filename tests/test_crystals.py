"""Type I SHG crystal: indices, angles, propagation constants, depth resolution.

The crystal used throughout is a three-term Sellmeier pair with BBO-like
coefficients (Tamošauskas et al., Opt. Mater. Express 8, 1410 (2018), Eq. (1)),
written out here so the tests do not depend on a material registry. They are a
realistic negative uniaxial crystal; the physics tested does not depend on their
exact values.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from croak import crystals
from croak.constants import C
from croak.crystals import CrystalSetup, UniaxialCrystal
from croak.geometry import TwoBeamGeometry
from croak.grid import Grid
from croak.maths import wlfreq


def _sellmeier(b, c):
    def n(lam):
        l2 = (np.asarray(lam, dtype=float) * 1e6) ** 2
        return np.sqrt(
            1.0 + sum(bi * l2 / (l2 - ci) for bi, ci in zip(b, c, strict=True))
        )

    return n


BBO = UniaxialCrystal(
    "BBO-like",
    _sellmeier((0.90291, 0.83155, 0.76536), (0.003926, 0.018786, 60.01)),
    _sellmeier((1.151075, 0.21803, 0.656), (0.007142, 0.02259, 263.0)),
)
W0 = float(wlfreq(800e-9))
GRID = Grid(256, dt=3.0e-15)  # spans roughly 560-1400 nm around 800 nm


def _tuned(**kw) -> CrystalSetup:
    return CrystalSetup(BBO, math.radians(29.2), phase_match_wavelength=800e-9, **kw)


def _sinc_integral(phi):
    """(1/L) * integral_0^L exp(2 i phi z / L) dz, exactly."""
    return np.where(phi == 0, 1.0, (np.exp(2j * phi) - 1.0) / (2j * phi))


# -- indices and angles ---------------------------------------------------------
def test_extraordinary_index_limits():
    assert crystals.extraordinary_index(1.6, 1.5, 0.0) == pytest.approx(1.6)
    assert crystals.extraordinary_index(1.6, 1.5, np.pi / 2) == pytest.approx(1.5)
    # 1/n^2 is linear in sin^2(theta): halfway at 45 degrees
    n45 = crystals.extraordinary_index(1.6, 1.5, np.pi / 4)
    assert 1 / n45**2 == pytest.approx(0.5 / 1.6**2 + 0.5 / 1.5**2)


def test_dn_dtheta_matches_a_finite_difference():
    theta, h = 0.5, 1e-6
    fd = (
        crystals.extraordinary_index(1.66, 1.55, theta + h)
        - crystals.extraordinary_index(1.66, 1.55, theta - h)
    ) / (2 * h)
    # central difference error ~h^2 * n''' ~ 1e-12; float rounding ~1e-10
    assert crystals._dn_dtheta(1.66, 1.55, theta) == pytest.approx(fd, rel=1e-8)


def test_phase_matching_angle_of_800_nm():
    # Tamosauskas et al. cut their type I crystal at 29.2 degrees for SHG of 800 nm
    # and state their angles within 0.2 degrees of the measured ones (pp. 1413, 1416).
    theta = crystals.phase_matching_angle(BBO, 800e-9)
    assert math.degrees(theta) == pytest.approx(29.2, abs=0.2)
    omega = np.array([W0])
    dk = crystals._k_signal(BBO, 2 * omega, theta) - 2 * crystals._k_fundamental(
        BBO, omega, 0.0
    )
    assert abs(dk[0]) < 1e-6  # rad/m, against k ~ 2.6e7: brentq's xtol


def test_crossing_raises_the_phase_matching_angle():
    # k_z = k_o cos(alpha_int) < k_o, so the signal needs a smaller index: theta grows.
    collinear = crystals.phase_matching_angle(BBO, 800e-9)
    crossed = crystals.phase_matching_angle(BBO, 800e-9, math.radians(2.0))
    assert crossed > collinear


def test_phase_matching_angle_raises_without_a_solution():
    # Below ~410 nm BBO cannot phase-match type I SHG (its UV absorption edge).
    with pytest.raises(ValueError, match="cannot phase-match"):
        crystals.phase_matching_angle(BBO, 380e-9)


# -- the setup ------------------------------------------------------------------
def test_setup_needs_exactly_one_angle_specification():
    with pytest.raises(ValueError, match="exactly one of tilt"):
        CrystalSetup(BBO, 0.5)
    with pytest.raises(ValueError, match="exactly one of tilt"):
        CrystalSetup(BBO, 0.5, tilt=0.0, phase_match_wavelength=800e-9)


def test_setup_rejects_a_cut_angle_out_of_range():
    with pytest.raises(ValueError, match="cut_angle"):
        CrystalSetup(BBO, -0.1, tilt=0.0)


@pytest.mark.parametrize("polarization", [None, "vertical"])
def test_setup_with_a_geometry_needs_a_polarization(polarization):
    geometry = TwoBeamGeometry(5e-3, 0.1)
    with pytest.raises(ValueError, match="polarization"):
        CrystalSetup(BBO, 0.5, tilt=0.0, geometry=geometry, polarization=polarization)


def test_explicit_tilt_refracts_with_snell():
    setup = CrystalSetup(BBO, math.radians(29.0), tilt=math.radians(10.0))
    n = float(BBO.n_o(np.array([800e-9]))[0])
    gamma_int = math.asin(math.sin(math.radians(10.0)) / n)
    assert setup.internal_angle(W0) == pytest.approx(math.radians(29.0) + gamma_int)
    assert setup.external_tilt(W0) == math.radians(10.0)
    assert setup.path_length(10e-6, W0) == pytest.approx(10e-6 / math.cos(gamma_int))


def test_tuned_setup_solves_the_tilt_consistently():
    setup = _tuned()
    theta = setup.internal_angle(W0)
    gamma = setup.external_tilt(W0)
    # re-tilting by the solved external angle lands on the same internal angle
    again = CrystalSetup(BBO, math.radians(29.2), tilt=gamma)
    n_pm = float(BBO.n_o(np.array([800e-9]))[0])
    assert again.cut_angle + math.asin(math.sin(gamma) / n_pm) == pytest.approx(theta)
    assert setup.internal_tilt(W0) == pytest.approx(theta - math.radians(29.2))


def test_tuning_beyond_total_internal_reflection_raises():
    # a 0-degree cut tuned to 800 nm needs ~29 degrees inside: n sin(29) > 1? no,
    # but a 80-degree cut needs ~-51 degrees inside, which no external angle gives.
    setup = CrystalSetup(BBO, math.radians(80.0), phase_match_wavelength=800e-9)
    with pytest.raises(ValueError, match="total internal reflection"):
        setup.external_tilt(W0)


# -- propagation constants -------------------------------------------------------
def test_isotropic_material_shares_one_beta():
    prop = crystals.propagation("SiO2", GRID.omega, W0, 10e-6)
    np.testing.assert_array_equal(prop.beta_in, prop.beta_sig)
    np.testing.assert_array_equal(prop.sig_amp, 1.0)
    assert prop.path_length == 10e-6


def test_no_material_is_dispersionless():
    prop = crystals.propagation(None, GRID.omega, W0, 0.0)
    np.testing.assert_array_equal(prop.beta_in, 0.0)


def test_crystal_propagation_carries_the_exact_mismatch():
    # beta_sig(O1+O2) - beta_in(O1) - beta_in(O2) must equal the physical mismatch
    # k_s(w1+w2) - k_z(w1) - k_z(w2): the frame terms cancel exactly.
    setup = _tuned(geometry=TwoBeamGeometry(5e-3, 0.1), polarization="parallel")
    prop = crystals.propagation(setup, GRID.omega, W0, 10e-6)
    i1, i2 = 128 + 10, 128 - 25  # centred indices of two grid frequencies
    i_s = i1 + i2 - 128  # their sum on the centred grid
    model = prop.beta_sig[i_s] - prop.beta_in[i1] - prop.beta_in[i2]
    w1, w2 = GRID.omega[i1] + W0, GRID.omega[i2] + W0
    theta = setup.internal_angle(W0)
    sin_a = math.sin(setup.crossing_half_angle)
    physical = (
        crystals._k_signal(BBO, np.array([w1 + w2]), theta)
        - crystals._k_fundamental(BBO, np.array([w1]), sin_a)
        - crystals._k_fundamental(BBO, np.array([w2]), sin_a)
    )[0]
    # absolute rounding: three wavevectors ~2.6e7 rad/m at 1e-16 relative
    assert model == pytest.approx(physical, abs=1e-6 * 2.6e7 * 1e-6)
    # and at the carrier pair the tuned crystal is phase-matched
    assert prop.beta_sig[128] - 2 * prop.beta_in[128] == pytest.approx(0.0, abs=1e-6)


def test_crystal_prefactor_is_omega_over_n_normalised_at_the_carrier():
    setup = _tuned()
    prop = crystals.propagation(setup, GRID.omega, W0, 10e-6)
    assert prop.sig_amp[128] == pytest.approx(1.0)
    theta = setup.internal_angle(W0)
    ws = GRID.omega[140] + 2 * W0
    lam = wlfreq(ws)
    n_s = crystals.extraordinary_index(BBO.n_o(lam), BBO.n_e(lam), theta)
    lam0 = wlfreq(2 * W0)
    n_0 = crystals.extraordinary_index(BBO.n_o(lam0), BBO.n_e(lam0), theta)
    assert prop.sig_amp[140] == pytest.approx((ws / n_s) / (2 * W0 / n_0), rel=1e-12)


def test_crystal_propagation_zeroes_frequencies_without_light():
    # a grid wide enough to reach negative absolute frequency at its low edge
    grid = Grid(64, dt=0.05e-15)
    prop = crystals.propagation(_tuned(), grid.omega, W0, 10e-6)
    dark = grid.omega + W0 <= 0
    assert dark.any()
    assert np.all(prop.beta_in[dark] == 0.0)
    assert np.all(np.isfinite(prop.beta_in) & np.isfinite(prop.beta_sig))


# -- depth resolution ------------------------------------------------------------
@pytest.mark.parametrize("phi", [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 60.0])
def test_depth_rule_resolves_the_oscillation(phi):
    # The rule depth_points uses, n = ceil(phi) + 3, integrates exp(2 i phi x) over
    # [0, 1] to below 1e-6 (the accuracy the docstring promises).
    n = math.ceil(phi) + 3
    x, w = leggauss(n)
    x, w = 0.5 * (x + 1.0), 0.5 * w
    numeric = np.sum(w * np.exp(2j * phi * x))
    assert abs(numeric - _sinc_integral(np.array(phi))) < 1e-6


def test_depth_points_grow_with_thickness():
    setup = _tuned()
    assert crystals.depth_points(setup, GRID.omega, W0, 0.0) == 1
    thin = crystals.depth_points(setup, GRID.omega, W0, 10e-6)
    thick = crystals.depth_points(setup, GRID.omega, W0, 100e-6)
    assert 3 < thin < thick
    # the mismatch scales with L, so phi does too (up to the +3 and the ceiling)
    assert thick - 3 == pytest.approx(10 * (thin - 3), abs=10)


def test_depth_points_subsample_a_fine_grid():
    setup = _tuned()
    fine = Grid(1024, dt=3.0e-15)  # the same band as GRID, four times the points
    assert crystals.depth_points(setup, fine.omega, W0, 50e-6) == pytest.approx(
        crystals.depth_points(setup, GRID.omega, W0, 50e-6), abs=1
    )


# -- the neglected signal tilt ----------------------------------------------------
def test_tilt_phase_error_vanishes_out_of_the_principal_plane():
    geometry = TwoBeamGeometry(5e-3, 0.1)
    parallel = _tuned(geometry=geometry, polarization="parallel")
    assert crystals.tilt_phase_error(parallel, GRID.omega, W0, 100e-6) == 0.0
    assert crystals.tilt_phase_error(_tuned(), GRID.omega, W0, 100e-6) == 0.0
    perpendicular = _tuned(geometry=geometry, polarization="perpendicular")
    assert crystals.tilt_phase_error(perpendicular, GRID.omega, W0, 0.0) == 0.0


def test_tilt_phase_error_scales_with_thickness_and_angle():
    def error(separation, thickness):
        setup = _tuned(
            geometry=TwoBeamGeometry(separation, 0.1), polarization="perpendicular"
        )
        return crystals.tilt_phase_error(setup, GRID.omega, W0, thickness)

    base = error(3.5e-3, 10e-6)  # ~1 degree half-angle
    assert base > 0.0
    assert error(3.5e-3, 100e-6) == pytest.approx(10 * base, rel=1e-3)
    # linear in sin(alpha) (to the ~1e-3 the solved angle and path move with alpha)
    assert error(7.0e-3, 10e-6) == pytest.approx(2 * base, rel=2e-2)


def test_tilt_phase_error_matches_a_direct_estimate():
    # One pair, by hand: the grid's extreme pair dominates.
    geometry = TwoBeamGeometry(3.5e-3, 0.1)
    setup = _tuned(geometry=geometry, polarization="perpendicular")
    estimate = crystals.tilt_phase_error(setup, GRID.omega, W0, 100e-6)
    theta = setup.internal_angle(W0)
    alpha = geometry.crossing_half_angle
    half = float(np.max(np.abs(GRID.omega)))
    w1, w2 = W0 + half, W0 - half  # the widest pair still on the signal grid
    ws = w1 + w2
    lam = wlfreq(ws)
    n_o, n_e = BBO.n_o(np.array([lam])), BBO.n_e(np.array([lam]))
    n_s = crystals.extraordinary_index(n_o, n_e, theta)[0]
    dn = crystals._dn_dtheta(n_o, n_e, theta)[0]
    by_hand = ws / C * abs(dn) * (w1 - w2) * math.sin(alpha) / (ws * n_s) * 50e-6
    by_hand /= math.cos(setup.internal_tilt(W0))
    # the subsampled pair grid reaches the extreme pair to within a grid step
    assert estimate == pytest.approx(by_hand, rel=2e-2)
