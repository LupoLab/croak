"""Tests for the dispersion model (Sellmeier + tabulated) in :mod:`croak.materials`."""

from pathlib import Path

import numpy as np
import pytest

from croak import materials
from croak.maths import wlfreq

# the built-in tabulated SiO2-Franta dataset shipped with the package
_FRANTA_CSV = Path(materials.__file__).parent / "data" / "materials" / "FrantaSiO2.csv"


def test_known_refractive_indices():
    # Reference values from refractiveindex.info for the same Sellmeier formulae.
    n_sio2 = materials.refractive_index("SiO2")
    assert float(n_sio2(1.0e-6)) == pytest.approx(1.4504, abs=2e-3)
    assert float(n_sio2(0.5e-6)) == pytest.approx(1.4623, abs=2e-3)

    n_caf2 = materials.refractive_index("CaF2")
    assert float(n_caf2(0.5e-6)) == pytest.approx(1.4365, abs=2e-3)


def test_case_insensitive_and_unknown():
    assert float(materials.refractive_index("sio2")(1e-6)) > 1.0
    with pytest.raises(ValueError):
        materials.refractive_index("unobtainium")


def test_beta_none_is_zero():
    omega = np.linspace(-1e15, 1e15, 64)
    assert np.all(materials.beta(None, omega, 2.4e15) == 0.0)


def test_beta_referenced_to_group_delay():
    # At omega = 0 (i.e. the carrier) the centred beta should vanish to first
    # order because the linear group-delay term is subtracted.
    omega0 = wlfreq(800e-9)
    omega = np.linspace(-0.3e15, 0.3e15, 257)
    b = materials.beta("SiO2", omega, omega0)
    assert np.all(np.isfinite(b))
    # derivative of beta at the carrier (centre index) ~ 0 after referencing
    centre = len(omega) // 2
    dbeta = np.gradient(b, omega)[centre]
    # beta1 itself is ~ n/c ~ 5e-9 s/m; a residual slope this small means the
    # linear group-delay term has been removed.
    assert abs(dbeta) < 1e-12


def test_beta_curvature_positive_normal_dispersion():
    # Fused silica has normal GVD (beta2 > 0) in the visible/NIR.
    omega0 = wlfreq(800e-9)
    omega = np.linspace(-0.2e15, 0.2e15, 257)
    b = materials.beta("SiO2", omega, omega0)
    beta2 = np.gradient(np.gradient(b, omega), omega)[len(omega) // 2]
    assert beta2 > 0


def test_register_custom_material():
    # a registered callable resolves through refractive_index and the dispersion
    # helpers exactly like a built-in (here we alias SiO2 under a new name).
    from croak import dispersion

    n_sio2 = materials.refractive_index("SiO2")
    materials.register_material("MyGlass", n_sio2)
    try:
        omega0 = wlfreq(800e-9)
        assert dispersion.material_gdd("MyGlass", omega0, 1e-3) == pytest.approx(
            dispersion.material_gdd("SiO2", omega0, 1e-3)
        )
    finally:
        materials.unregister_material("MyGlass")
    with pytest.raises(ValueError, match="unknown material"):
        materials.refractive_index("MyGlass")


def test_sio2_franta_registered_and_resolves():
    # the built-in tabulated material is listed and resolves case-insensitively
    assert "SiO2-Franta" in materials.MATERIALS
    n = materials.refractive_index("sio2-franta")
    # fused silica at 800 nm ~ 1.4535
    assert float(n(800e-9)) == pytest.approx(1.4535, abs=2e-3)


def test_sio2_franta_out_of_range_is_masked():
    n = materials.refractive_index("SiO2-Franta")
    # 200 µm is beyond the tabulated range (24.8 nm – 125 µm) → nan
    assert np.isnan(float(n(200e-6)))
    # ... and beta masks the non-finite entry to zero
    omega0 = wlfreq(800e-9)
    omega = np.array([0.0, 1e18])  # second point maps to a deep-UV, out-of-range λ
    b = materials.beta("SiO2-Franta", omega, omega0)
    assert np.all(np.isfinite(b))


def test_sio2_franta_dispersion_matches_sellmeier():
    """GDD/TOD from the tabulated data agree with the SiO2 Sellmeier model.

    The key invariant: differentiating the smoothing-spline ``n(λ)`` up to third
    order reproduces fused silica's known dispersion, confirming the derivatives
    are well conditioned. A raw interpolant of the same noisy data would not (see
    :func:`test_tabulated_smoothing_tames_derivative_noise`).
    """
    from croak import dispersion

    for lam0 in (400e-9, 800e-9, 1030e-9):
        omega0 = wlfreq(lam0)
        gdd_tab = dispersion.material_gdd("SiO2-Franta", omega0, 1e-3)
        gdd_ref = dispersion.material_gdd("SiO2", omega0, 1e-3)
        assert gdd_tab == pytest.approx(gdd_ref, rel=0.05)
        tod_tab = dispersion.material_tod("SiO2-Franta", omega0, 1e-3)
        tod_ref = dispersion.material_tod("SiO2", omega0, 1e-3)
        assert tod_tab == pytest.approx(tod_ref, rel=0.25)


def test_make_tabulated_material_clean_recovery_and_contract():
    """On clean Sellmeier samples the spline recovers n and GDD, masks outside
    the range, and rejects degenerate input."""
    from croak import dispersion

    lam = np.geomspace(250e-9, 1800e-9, 800)
    n_true = materials.refractive_index("SiO2")
    n_clean = np.asarray(n_true(lam), dtype=float)
    n_func, (lo, hi) = materials.make_tabulated_material(lam, n_clean)
    assert (lo, hi) == pytest.approx((250e-9, 1800e-9))
    assert float(n_func(800e-9)) == pytest.approx(float(n_true(800e-9)), abs=1e-4)
    assert np.isnan(float(n_func(100e-9)))  # below range

    materials.register_material("CleanSiO2", n_func)
    try:
        omega0 = wlfreq(800e-9)
        assert dispersion.material_gdd("CleanSiO2", omega0, 1e-3) == pytest.approx(
            dispersion.material_gdd("SiO2", omega0, 1e-3), rel=0.01
        )
    finally:
        materials.unregister_material("CleanSiO2")

    with pytest.raises(ValueError, match="at least 4"):
        materials.make_tabulated_material([1e-6, 2e-6], [1.5, 1.4])


def test_tabulated_smoothing_tames_derivative_noise():
    """A raw interpolant of the experimental data yields a meaningless GDD where
    the smoothing spline recovers the physical value."""
    from croak import dispersion

    omega0 = wlfreq(800e-9)
    gdd_ref = dispersion.material_gdd("SiO2", omega0, 1e-3)
    # shipped material (smoothing spline) ~ physical fused-silica GDD
    gdd_smooth = dispersion.material_gdd("SiO2-Franta", omega0, 1e-3)
    assert gdd_smooth == pytest.approx(gdd_ref, rel=0.05)

    # a raw piecewise-linear interpolant of the SAME data: its second derivative
    # is ~0 within a segment (or spikes across a knot) — either way meaningless.
    lam_m, n = materials.read_nk_csv(_FRANTA_CSV)

    def n_raw(query_m):
        q = np.asarray(query_m, dtype=float)
        return np.interp(q, lam_m, n, left=np.nan, right=np.nan)

    materials.register_material("FrantaRaw", n_raw)
    try:
        gdd_raw = dispersion.material_gdd("FrantaRaw", omega0, 1e-3)
    finally:
        materials.unregister_material("FrantaRaw")
    assert abs(gdd_raw - gdd_ref) > 0.5 * abs(gdd_ref)


def test_read_nk_csv_parses_franta_file():
    lam_m, n = materials.read_nk_csv(_FRANTA_CSV)
    assert lam_m.shape == n.shape
    assert np.all(np.diff(lam_m) > 0)  # ascending wavelength
    # the broadband range in metres (24.8 nm – 125 µm), wavelength column was µm
    assert lam_m.min() == pytest.approx(24.797e-9, rel=1e-3)
    assert lam_m.max() == pytest.approx(125.14e-6, rel=1e-3)
    # n at 800 nm ~ fused silica (the wl,k section must not leak in)
    i = int(np.argmin(np.abs(lam_m - 800e-9)))
    assert n[i] == pytest.approx(1.4535, abs=2e-3)
