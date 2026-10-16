"""Tests for :mod:`croak.dispersion`."""

import numpy as np
import pytest

from croak import dispersion, materials
from croak.grid import Grid
from croak.maths import fwhm, wlfreq
from croak.pulses import gaussian_pulse


@pytest.fixture
def setup():
    g = Grid(256, dt=0.5e-15)
    omega0 = wlfreq(800e-9)
    ew = gaussian_pulse(g, 8e-15)  # transform-limited
    return g, omega0, ew


def _temporal_fwhm(g, ew):
    return fwhm(g.t, np.abs(g.ifft(ew)) ** 2)


def test_apply_gdd_broadens(setup):
    g, omega0, ew = setup
    tl = _temporal_fwhm(g, ew)
    chirped = dispersion.apply_dispersion(g.omega, omega0, ew, gdd=60e-30)
    assert _temporal_fwhm(g, chirped) > 1.5 * tl


def test_phase_only_mirror_leaves_amplitude_unchanged(setup):
    """A mirror with no registered amplitude does not touch ``|E(ω)|``."""
    g, omega0, ew = setup
    dispersion.register_mirror("_t_phase", lambda w: 0.3 * (w - omega0) ** 2 * 1e-30)
    try:
        out = dispersion.apply_dispersion(
            g.omega, omega0, ew, mirror_bounces={"_t_phase": 1}
        )
        assert np.allclose(np.abs(out), np.abs(ew))
    finally:
        dispersion.unregister_mirror("_t_phase")


def test_mirror_amplitude_applied_and_divided_on_removal(setup):
    """A registered amplitude attenuates when added and amplifies when removed."""
    g, omega0, ew = setup
    r = 0.5  # flat amplitude reflectivity
    dispersion.register_mirror(
        "_t_amp",
        lambda w: np.zeros_like(w),  # phase-flat, to isolate amplitude
        lambda w: np.full_like(w, r),
    )
    try:
        added = dispersion.apply_dispersion(
            g.omega, omega0, ew, mirror_bounces={"_t_amp": 2}
        )
        removed = dispersion.apply_dispersion(
            g.omega, omega0, ew, mirror_bounces={"_t_amp": -2}
        )
        # 2 bounces add → ×r²; removing 2 → ÷r²
        assert np.allclose(np.abs(added), np.abs(ew) * r**2)
        assert np.allclose(np.abs(removed), np.abs(ew) / r**2)
    finally:
        dispersion.unregister_mirror("_t_amp")


def test_mirror_amplitude_gain_is_clamped(setup):
    """Dividing by a tiny reflectivity is bounded by the gain cap."""
    g, omega0, ew = setup
    tiny = 1e-6
    dispersion.register_mirror(
        "_t_clamp", lambda w: np.zeros_like(w), lambda w: np.full_like(w, tiny)
    )
    try:
        removed = dispersion.apply_dispersion(
            g.omega, omega0, ew, mirror_bounces={"_t_clamp": -3}
        )
        # |removed| = |ew|·amp ≤ |ew|·cap; multiplicative form avoids 0/0 at the
        # Gaussian's underflowed spectral wings.
        cap = dispersion._MIRROR_AMPLITUDE_CAP
        assert np.all(np.abs(removed) <= np.abs(ew) * cap * (1 + 1e-9))
        # the cap actually bites here (tiny r would otherwise give 1e18 gain)
        assert np.any(np.abs(removed) > np.abs(ew))
    finally:
        dispersion.unregister_mirror("_t_clamp")


def test_register_unregister_mirror_clears_both_registries():
    dispersion.register_mirror(
        "_t_reg", lambda w: np.zeros_like(w), lambda w: np.ones_like(w)
    )
    assert "_t_reg" in dispersion.MIRRORS
    assert "_t_reg" in dispersion.MIRROR_AMPLITUDES
    # re-registering without an amplitude drops the stale one
    dispersion.register_mirror("_t_reg", lambda w: np.zeros_like(w))
    assert "_t_reg" not in dispersion.MIRROR_AMPLITUDES
    dispersion.unregister_mirror("_t_reg")
    assert "_t_reg" not in dispersion.MIRRORS


def test_gdd_roundtrip(setup):
    g, omega0, ew = setup
    a = dispersion.apply_dispersion(g.omega, omega0, ew, gdd=60e-30, tod=200e-45)
    b = dispersion.apply_dispersion(g.omega, omega0, a, gdd=-60e-30, tod=-200e-45)
    assert np.allclose(b, ew, atol=1e-12)


def test_material_gdd_matches_apply(setup):
    g, omega0, ew = setup
    # applying +1 mm SiO2 should add a GDD matching material_gdd's report
    gdd_fs2 = dispersion.material_gdd("SiO2", omega0, 1e-3)
    assert gdd_fs2 > 0  # normal dispersion in the NIR
    mod = dispersion.apply_dispersion(
        g.omega, omega0, ew, material_thicknesses={"SiO2": 1e-3}
    )
    # measure the GDD of the *added* phase by a cubic fit near the peak
    Iw = np.abs(ew) ** 2
    mask = Iw / Iw.max() > 0.05
    added = np.unwrap((np.angle(mod) - np.angle(ew))[mask])
    pc = np.polyfit(g.omega[mask], added, 3)
    measured_gdd = 2 * pc[1] / 1e-30
    assert measured_gdd == pytest.approx(gdd_fs2, rel=0.1)


def test_auto_taylor_compensates_chirp(setup):
    g, omega0, ew = setup
    chirped = dispersion.apply_dispersion(g.omega, omega0, ew, gdd=80e-30)
    gdd, tod, fod = dispersion.auto_taylor(g, chirped, omega0)
    compensated = dispersion.apply_dispersion(
        g.omega, omega0, chirped, gdd=gdd, tod=tod, fod=fod
    )
    assert _temporal_fwhm(g, compensated) == pytest.approx(
        _temporal_fwhm(g, ew), rel=0.1
    )
    assert gdd == pytest.approx(-80e-30, rel=0.15)


def test_auto_material_compensates(setup):
    g, omega0, ew = setup
    # pre-chirp with some SiO2, then let auto_material remove it (negative thickness)
    chirped = dispersion.apply_dispersion(
        g.omega, omega0, ew, material_thicknesses={"SiO2": 1e-3}
    )
    thickness = dispersion.auto_material(g, chirped, omega0, "SiO2")
    assert thickness == pytest.approx(-1e-3, abs=2e-4)


@pytest.mark.parametrize("method", ["dual_annealing", "direct", "nelder-mead"])
def test_auto_material_with_every_optimizer(setup, method):
    g, omega0, ew = setup
    chirped = dispersion.apply_dispersion(
        g.omega, omega0, ew, material_thicknesses={"SiO2": 1e-3}
    )
    # A narrower box than the default keeps the global searches quick; the
    # tolerance is the same as the differential-evolution test's.
    thickness = dispersion.auto_material(
        g, chirped, omega0, "SiO2", bounds=(-3e-3, 3e-3), method=method
    )
    assert thickness == pytest.approx(-1e-3, abs=2e-4)


def test_optimize_never_returns_worse_than_the_seed():
    # The box excludes the seed x0 = 0, so every point the search can return is
    # worse than "no change": the seed comes back.
    x = dispersion._optimize(lambda x: float(x[0] ** 2), [(1.0, 2.0)], "direct")
    np.testing.assert_array_equal(x, [0.0])
    # an explicit seed is honoured the same way
    x = dispersion._optimize(
        lambda x: float((x[0] - 0.5) ** 2), [(1.0, 2.0)], "direct", x0=[0.4]
    )
    np.testing.assert_array_equal(x, [0.4])


def test_dispersion_phase_skips_unknown_and_zero_bounce_mirrors(setup):
    g, omega0, _ = setup
    plain = dispersion.dispersion_phase(g.omega, omega0, gdd=10e-30)
    dispersion.register_mirror("TestMirror", lambda w: np.ones_like(w))
    try:
        skipped = dispersion.dispersion_phase(
            g.omega,
            omega0,
            gdd=10e-30,
            mirror_bounces={"NoSuchMirror": 3, "TestMirror": 0},
        )
        bounced = dispersion.dispersion_phase(
            g.omega, omega0, gdd=10e-30, mirror_bounces={"TestMirror": 2}
        )
    finally:
        dispersion.unregister_mirror("TestMirror")
    np.testing.assert_array_equal(skipped, plain)
    # two bounces of a constant 1 rad per-bounce phase add exactly 2 rad
    np.testing.assert_allclose(bounced - plain, 2.0)


def test_material_tod_of_fused_silica(setup):
    _, omega0, _ = setup
    # Fused silica at 800 nm: ~27 fs^3 per mm (tabulated values 26-28 fs^3/mm);
    # the bracket allows for the choice of model.
    tod_fs3 = dispersion.material_tod("SiO2", omega0, 1e-3)
    assert 25.0 < tod_fs3 < 30.0
    # linear in thickness
    assert dispersion.material_tod("SiO2", omega0, 2e-3) == pytest.approx(2 * tod_fs3)


@pytest.mark.parametrize("material", ["SiO2", "BK7", "CaF2"])
@pytest.mark.parametrize("wavelength", [800e-9, 1030e-9])
def test_material_tod_matches_optmats_analytic_gvd(material, wavelength):
    # Regression: with 1e-4·ω steps the nested difference amplified the 6e-15
    # noise of optmat's fused silica into a 10 % error at 1030 nm. The reference
    # differentiates optmat's gvd, which is built on the exact dn/dω, once
    # (a 1e-3·ω step: truncation 1e-6); croak must agree to 1e-3.
    model = materials.material_model(material)
    omega0 = float(wlfreq(wavelength))
    h = 1e-3 * omega0
    beta3 = (
        model.gvd(wlfreq(omega0 + h), warn=False)
        - model.gvd(wlfreq(omega0 - h), warn=False)
    ) / (2 * h)
    expected_fs3 = float(beta3) * 1e-3 / 1e-45
    assert dispersion.material_tod(material, omega0, 1e-3) == pytest.approx(
        expected_fs3, rel=1e-3
    )


def test_auto_material_nelder_mead_respects_the_bounds(setup):
    """Regression: the Nelder-Mead polish was unbounded and returned -1.5 mm for a
    +-1 mm box when the optimum (-1.5 mm) lay outside it."""
    g, omega0, ew = setup
    chirped = dispersion.apply_dispersion(
        g.omega, omega0, ew, material_thicknesses={"SiO2": 1.5e-3}
    )
    thickness = dispersion.auto_material(
        g, chirped, omega0, "SiO2", bounds=(-1e-3, 1e-3), method="nelder-mead"
    )
    assert -1e-3 <= thickness <= 1e-3
    # the best point inside the box is its edge nearest the true optimum
    assert thickness == pytest.approx(-1e-3, abs=2e-4)
