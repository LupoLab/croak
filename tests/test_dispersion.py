"""Tests for :mod:`croak.dispersion`."""

import numpy as np
import pytest

from croak import dispersion
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
