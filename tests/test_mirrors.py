"""Tests for :mod:`croak.mirrors` and :mod:`croak.refractive_db`."""

import numpy as np
import pytest

from croak import dispersion, mirrors
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


@pytest.mark.parametrize("name", list(mirrors.BUILTIN_MIRRORS))
def test_builtin_loads_and_is_finite(name):
    phase_fn, amp_fn, (lo, hi) = mirrors.load_builtin(name)
    assert lo < hi
    lam = np.linspace(lo, hi, 2001)
    phi = phase_fn(wlfreq(lam))
    assert np.all(np.isfinite(phi))
    # application-ready amplitude is physical (0 < r ≤ 1) in band, and a no-op
    # (r = 1) outside the design band where the phase is tapered to zero
    r = amp_fn(wlfreq(lam))
    assert np.all((r > 0) & (r <= 1.001))
    assert amp_fn(wlfreq(np.array([10e-9]))) == pytest.approx(1.0)


@pytest.mark.parametrize("name", list(mirrors.BUILTIN_COATINGS))
@pytest.mark.parametrize("pol", ["s", "p"])
def test_coating_loads_with_physical_phase_and_amplitude(name, pol):
    phase_fn, amp_fn, (lo, hi) = mirrors.load_coating(name, pol)
    assert lo < hi
    lam = np.linspace(700e-9, 900e-9, 2001)  # near 800 nm, well in band
    phi = phase_fn(wlfreq(lam))
    r = amp_fn(wlfreq(lam))
    assert np.all(np.isfinite(phi))
    assert np.all((r > 0) & (r <= 1.0))
    # outside the data band the amplitude is a no-op (r = 1)
    assert amp_fn(wlfreq(np.array([10e-9]))) == pytest.approx(1.0)


def test_coating_normal_incidence_s_equals_p():
    """At 0° incidence the s and p data coincide (R_s == R_p, GDD_s == GDD_p)."""
    lam = np.linspace(400e-9, 1200e-9, 2000)
    w = wlfreq(lam)
    ps, as_, _ = mirrors.load_coating("Si @ 0°", "s")
    pp, ap, _ = mirrors.load_coating("Si @ 0°", "p")
    assert np.allclose(ps(w), pp(w))
    assert np.allclose(as_(w), ap(w))


def test_coating_oblique_s_differs_from_p():
    """At 45° the polarisations have genuinely different dispersion."""
    lam = np.linspace(400e-9, 1200e-9, 2000)
    w = wlfreq(lam)
    ps, _, _ = mirrors.load_coating("MgF2/Al @ 45°", "s")
    pp, _, _ = mirrors.load_coating("MgF2/Al @ 45°", "p")
    assert not np.allclose(ps(w), pp(w))


def test_coating_back_propagation_roundtrip(setup):
    """Removing N bounces then re-adding N recovers the spectrum exactly."""
    g, omega0, ew = setup
    phase_fn, amp_fn, _ = mirrors.load_coating("MgF2/Al @ 45°", "s")
    dispersion.register_mirror("_t_coat", phase_fn, amp_fn)
    try:
        removed = dispersion.apply_dispersion(
            g.omega, omega0, ew, mirror_bounces={"_t_coat": -3}
        )
        restored = dispersion.apply_dispersion(
            g.omega, omega0, removed, mirror_bounces={"_t_coat": 3}
        )
        assert np.allclose(restored, ew, atol=1e-10)
        # removal genuinely changed the field (phase and/or amplitude)
        assert not np.allclose(removed, ew)
    finally:
        dispersion.unregister_mirror("_t_coat")


def test_builtin_reflectivity_is_applied(setup):
    """A chirped mirror with R data attenuates the spectrum per bounce."""
    g, omega0, ew = setup
    phase_fn, amp_fn, (lo, hi) = mirrors.load_builtin("PC70")  # has R data
    # in band the per-bounce amplitude is below unity (real reflectivity)
    lam = np.linspace(lo, hi, 1001)
    assert np.median(amp_fn(wlfreq(lam))) < 1.0
    dispersion.register_mirror("_t_builtin", phase_fn, amp_fn)
    try:
        out = dispersion.apply_dispersion(
            g.omega, omega0, ew, mirror_bounces={"_t_builtin": 3}
        )
        # adding bounces attenuates (loss), so the spectrum shrinks somewhere
        assert np.any(np.abs(out) < np.abs(ew) - 1e-12)
    finally:
        dispersion.unregister_mirror("_t_builtin")


def test_unknown_coating_raises():
    with pytest.raises(ValueError, match="unknown coating"):
        mirrors.load_coating("NotACoating")


def test_coating_bad_polarisation_raises():
    with pytest.raises(ValueError, match="pol must be"):
        mirrors.load_coating("Si @ 0°", "x")


def test_builtin_mirror_is_anomalous():
    """A chirped compressor mirror provides negative (anomalous) in-band GDD."""
    phase_fn, _r_fn, (lo, hi) = mirrors.load_builtin("PC70")
    lam = np.linspace(lo, hi, 4001)
    w = wlfreq(lam)
    order = np.argsort(w)
    ws, ps = w[order], phase_fn(w)[order]
    d2 = np.gradient(np.gradient(ps, ws), ws)  # ≈ GDD(ω)
    mid = len(ws) // 2
    assert np.median(d2[mid - 200 : mid + 200]) < 0


def test_builtin_zero_outside_band():
    phase_fn, _r_fn, (lo, _hi) = mirrors.load_builtin("PC70")
    # 300 nm is below PC70's 400 nm taper edge → no phase contribution
    assert phase_fn(wlfreq(np.array([300e-9]))) == pytest.approx(0.0)


def test_mirror_from_gdd_array_compensates(setup):
    """A flat-GDD 'mirror' applied once cancels a matching Taylor chirp."""
    g, omega0, ew = setup
    chirped = dispersion.apply_dispersion(g.omega, omega0, ew, gdd=80e-30)
    lam = np.linspace(500e-9, 1300e-9, 4000)  # wide band, taper far from pulse
    gdd_data = np.full_like(lam, -80e-30)  # s² per bounce
    phase_fn, rng = mirrors.mirror_from_arrays(lam, gdd_data, mode="gdd")
    assert rng[0] < rng[1]
    dispersion.register_mirror("_test_gdd", phase_fn)
    try:
        comp = dispersion.apply_dispersion(
            g.omega, omega0, chirped, mirror_bounces={"_test_gdd": 1}
        )
        assert _temporal_fwhm(g, comp) == pytest.approx(_temporal_fwhm(g, ew), rel=0.15)
    finally:
        dispersion.MIRRORS.pop("_test_gdd", None)


def test_mirror_from_phase_array_compensates(setup):
    """A mirror whose phase is the negative chirp parabola compresses the pulse."""
    g, omega0, ew = setup
    chirped = dispersion.apply_dispersion(g.omega, omega0, ew, gdd=80e-30)
    lam = np.linspace(500e-9, 1300e-9, 4000)
    w = wlfreq(lam)
    phase = -0.5 * 80e-30 * (w - omega0) ** 2  # cancels +gdd·ω²/2
    phase_fn, _rng = mirrors.mirror_from_arrays(lam, phase, mode="phase")
    dispersion.register_mirror("_test_phase", phase_fn)
    try:
        comp = dispersion.apply_dispersion(
            g.omega, omega0, chirped, mirror_bounces={"_test_phase": 1}
        )
        assert _temporal_fwhm(g, comp) == pytest.approx(_temporal_fwhm(g, ew), rel=0.15)
    finally:
        dispersion.MIRRORS.pop("_test_phase", None)


def test_unknown_mirror_raises():
    with pytest.raises(ValueError, match="unknown mirror"):
        mirrors.load_builtin("NotAMirror")


# -- refractive_db (optional dependency) ------------------------------------
def test_refractive_db_smoke():
    pytest.importorskip("refractiveindex")
    from croak import refractive_db

    assert refractive_db.available()
    sid, _ = refractive_db.shelves()[0]
    bid, _ = refractive_db.books(sid)[0]
    pid, _ = refractive_db.pages(sid, bid)[0]
    n, rng = refractive_db.make_material(sid, bid, pid)
    assert rng is None or rng[0] < rng[1]
    if rng is not None:
        mid = 0.5 * (rng[0] + rng[1])
        # n() is array-in/array-out, so index the result rather than calling
        # float() on it — NumPy 2 rejects converting a size-1 array to a scalar.
        assert np.isfinite(n(np.array([mid]))).all()
        # outside the valid range the index is NaN, so beta() masks it
        assert not np.isfinite(n(np.array([rng[1] * 10]))).any()
