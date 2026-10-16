"""Tests for the dispersion model (optmat and tabulated) in :mod:`croak.materials`."""

from pathlib import Path

import numpy as np
import pytest

from croak import materials
from croak.maths import wlfreq

# Franta et al.'s broadband fused-silica optical constants (24.8 nm – 125 µm), the
# measured data make_tabulated_material was written for; test data only.
_FRANTA_CSV = Path(__file__).parent / "data" / "FrantaSiO2.csv"


def test_known_refractive_indices():
    # Literature values (Malitson 1965; Daimon & Masumura 2002). The tolerance
    # covers optmat's absolute (vacuum) indices against the printed ones, which
    # are relative to air (3e-4), and the choice of model.
    n_sio2 = materials.refractive_index("SiO2")
    assert float(n_sio2(1.0e-6)) == pytest.approx(1.4504, abs=2e-3)
    assert float(n_sio2(0.5e-6)) == pytest.approx(1.4623, abs=2e-3)

    n_caf2 = materials.refractive_index("CaF2")
    assert float(n_caf2(0.5e-6)) == pytest.approx(1.4365, abs=2e-3)


@pytest.mark.parametrize("name", materials.MATERIALS)
def test_each_material_is_optmats_recommended_model(name):
    # croak adds nothing to optmat's index but the real part and the mask, so the
    # two agree exactly at physical wavelengths.
    lam = np.array([600e-9, 800e-9, 1500e-9])
    model = materials.material_model(name)
    np.testing.assert_array_equal(
        materials.refractive_index(name)(lam),
        np.real(model.refractive_index(lam, warn=False)),
    )
    assert model.status == "validated"
    assert name in materials.DESCRIPTIONS


def test_uniaxial_crystals_use_the_ordinary_ray():
    assert materials.material_model("MgF2").axis == "o"
    assert materials.material_model("Al2O3").axis == "o"


def test_an_unphysical_wavelength_is_nan():
    # A centred grid can reach ω ≤ 0: λ ≤ 0 or ∞. optmat refuses those; croak
    # returns nan there, which beta() masks.
    n = materials.refractive_index("SiO2")(np.array([-1e-6, 0.0, np.inf, np.nan]))
    assert np.all(np.isnan(n))


def test_beyond_its_range_a_material_is_extrapolated_without_a_warning():
    # N-BK7's catalogue formula covers 0.31-2.33 µm; the FFT grid goes far
    # beyond, so croak extrapolates silently (warnings are errors in the suite).
    n = materials.refractive_index("BK7")(np.array([3e-6, 5e-6]))
    assert np.all(np.isfinite(n))


def test_a_retired_name_resolves_to_its_material():
    lam = np.array([400e-9, 800e-9])
    np.testing.assert_array_equal(
        materials.refractive_index("SiO2-Franta")(lam),
        materials.refractive_index("SiO2")(lam),
    )
    assert materials.canonical_name("sio2-franta") == "SiO2"
    assert materials.canonical_name("caf2") == "CaF2"
    assert materials.canonical_name("MyGlass") == "MyGlass"
    assert "SiO2-Franta" not in materials.MATERIALS


def test_case_insensitive_and_unknown():
    assert float(materials.refractive_index("sio2")(1e-6)) > 1.0
    with pytest.raises(ValueError, match="unknown material 'unobtainium'"):
        materials.refractive_index("unobtainium")
    with pytest.raises(ValueError, match="unknown material 'unobtainium'"):
        materials.material_model("unobtainium")


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


@pytest.fixture
def franta_spline():
    """Register Franta's measured fused silica, as a smoothing spline."""
    lam_m, n = materials.read_nk_csv(_FRANTA_CSV)
    n_func, _ = materials.make_tabulated_material(lam_m, n)
    materials.register_material("FrantaSpline", n_func)
    yield n_func
    materials.unregister_material("FrantaSpline")


def test_franta_data_agree_with_optmats_fused_silica(franta_spline):
    """GDD/TOD of the smoothed measured data agree with optmat's SiO2.

    optmat's fused silica is Franta's model refitted to absolute refractometry, so
    the two must agree; and differentiating the smoothing-spline ``n(λ)`` up to
    third order must be well conditioned. A raw interpolant of the same noisy data
    would not be (see :func:`test_tabulated_smoothing_tames_derivative_noise`).
    The tolerances are the spline's: 5 % in GDD and 25 % in TOD.
    """
    from croak import dispersion

    for lam0 in (400e-9, 800e-9, 1030e-9):
        omega0 = wlfreq(lam0)
        gdd_tab = dispersion.material_gdd("FrantaSpline", omega0, 1e-3)
        gdd_ref = dispersion.material_gdd("SiO2", omega0, 1e-3)
        assert gdd_tab == pytest.approx(gdd_ref, rel=0.05)
        tod_tab = dispersion.material_tod("FrantaSpline", omega0, 1e-3)
        tod_ref = dispersion.material_tod("SiO2", omega0, 1e-3)
        assert tod_tab == pytest.approx(tod_ref, rel=0.25)
    # The refit moved the index by 0.8-3.4e-4 (optmat TC-399): within 5e-4.
    assert float(franta_spline(800e-9)) == pytest.approx(
        float(materials.refractive_index("SiO2")(800e-9)), abs=5e-4
    )


def test_a_tabulated_material_is_nan_out_of_range(franta_spline):
    # 200 µm is beyond the tabulated range (24.8 nm – 125 µm) → nan ...
    assert np.isnan(float(franta_spline(200e-6)))
    # ... and beta masks the non-finite entry to zero
    omega0 = wlfreq(800e-9)
    omega = np.array([0.0, 1e18])  # second point maps to a deep-UV, out-of-range λ
    b = materials.beta("FrantaSpline", omega, omega0)
    assert np.all(np.isfinite(b))


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


def test_tabulated_smoothing_tames_derivative_noise(franta_spline):
    """A raw interpolant of the experimental data yields a meaningless GDD where
    the smoothing spline recovers the physical value."""
    from croak import dispersion

    omega0 = wlfreq(800e-9)
    gdd_ref = dispersion.material_gdd("SiO2", omega0, 1e-3)
    # the smoothing spline of the data ~ physical fused-silica GDD
    gdd_smooth = dispersion.material_gdd("FrantaSpline", omega0, 1e-3)
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


def test_read_nk_csv_without_an_n_section_raises(tmp_path):
    # Only a k section (and blank lines): there is nothing to read the index from.
    path = tmp_path / "k_only.csv"
    path.write_text("wl,k\n0.5,0.1\n\n0.6,0.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no 'wl,n' section"):
        materials.read_nk_csv(path)
