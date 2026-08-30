"""Convergence tests for the AD L-BFGS solver (``lbfgs-ad``).

Mirrors ``test_lbfgs.py``: the JAX-autodiff solver must converge on the same
problems as the analytic-gradient :class:`croak.lbfgs.LBFGS`, and reach a
comparable final error (they minimise the identical objective with the identical
optimiser, differing only in gradient source).
"""

import numpy as np
import pytest

from croak.forward import maketrace
from croak.grid import Grid
from croak.lbfgs import LBFGS
from croak.lbfgs_ad import LBFGSAD
from croak.maths import wlfreq
from croak.processing import process_result
from croak.pulses import gaussian_pulse
from croak.smearing import square_boxcars_kernel


@pytest.fixture
def setup():
    g = Grid(128, dt=0.3e-15)
    ew = gaussian_pulse(g, 2.5e-15)
    delays = np.linspace(-15e-15, 15e-15, 87)
    return g, ew, delays


@pytest.mark.parametrize("interaction", ["pg", "shg"])
def test_lbfgs_ad_near_initial(setup, interaction):
    g, ew, delays = setup
    trace = maketrace(g.omega, delays, ew, interaction)
    near = gaussian_pulse(g, 1.5e-15)
    res = LBFGSAD(maxiters=300).run(trace, g.omega, delays, interaction, guess=near)
    assert res.error < 5e-3


@pytest.mark.parametrize("interaction", ["pg", "shg"])
def test_lbfgs_ad_random_initial(setup, interaction):
    g, ew, delays = setup
    trace = maketrace(g.omega, delays, ew, interaction)
    rng = np.random.default_rng(1)
    res = LBFGSAD(maxiters=500).run(
        trace, g.omega, delays, interaction, guess=None, rng=rng
    )
    assert res.error < 5e-2


def test_lbfgs_ad_chirped_phase_only(setup):
    g, _, delays = setup
    ew = gaussian_pulse(g, 2.5e-15, phases=[3e-30])
    trace = maketrace(g.omega, delays, ew, "pg")
    guess = gaussian_pulse(g, 2.5e-15, phases=[1e-30])
    res = LBFGSAD(maxiters=300, phase_only=True).run(
        trace, g.omega, delays, "pg", guess=guess
    )
    assert res.error < 1e-2


def test_lbfgs_ad_dispersive():
    g = Grid(128, dt=0.5e-15)
    omega0 = wlfreq(800e-9)
    ew = gaussian_pulse(g, 3.0e-15)
    delays = np.linspace(-20e-15, 20e-15, 80)
    trace = maketrace(
        g.omega,
        delays,
        ew,
        "pg",
        material="SiO2",
        thickness=10e-6,
        npoints=20,
        omega0=omega0,
    )
    near = gaussian_pulse(g, 2.0e-15)
    res = LBFGSAD(
        maxiters=400, material="SiO2", thickness=10e-6, npoints=20, omega0=omega0
    ).run(trace, g.omega, delays, "pg", guess=near)
    assert res.error < 5e-2


def test_lbfgs_ad_smeared_trace_needs_the_kernel():
    """A geometrically smeared trace only retrieves correctly with the kernel modelled.

    Without it, the free pulse has to absorb the instrument response and comes out
    substantially too long — the duration inflation the kernel exists to remove.
    """
    g = Grid(128, dt=0.4e-15)
    ew = gaussian_pulse(g, 2.5e-15)
    delays = np.linspace(-20e-15, 20e-15, 81)
    kernel = square_boxcars_kernel(
        "pg", hole_diameter=1e-3, hole_spacing=2.5e-3, wavelength=800e-9
    )
    trace = maketrace(g.omega, delays, ew, "pg", smearing=kernel)
    near = gaussian_pulse(g, 1.8e-15)

    modelled = LBFGSAD(maxiters=300, smearing=kernel).run(
        trace, g.omega, delays, "pg", guess=near
    )
    ignored = LBFGSAD(maxiters=300).run(trace, g.omega, delays, "pg", guess=near)

    assert modelled.error < 1e-4
    assert modelled.error < ignored.error
    fwhm_modelled = process_result(modelled).fwhm_retr
    fwhm_ignored = process_result(ignored).fwhm_retr
    assert fwhm_modelled == pytest.approx(2.5e-15, rel=0.02)
    assert fwhm_ignored > 1.5 * fwhm_modelled


def test_lbfgs_ad_fits_smearing_scale():
    """A known smearing strength is recovered as one extra scalar.

    The multiplier scales both kernel widths together — equivalently the mask ratio
    ``d/D``, the only physical lever on the effect — so a single free parameter is
    enough to calibrate the instrument response from the data.
    """
    g = Grid(64, dt=0.8e-15)
    ew = gaussian_pulse(g, 4.0e-15)
    delays = np.linspace(-25e-15, 25e-15, 41)
    kernel = square_boxcars_kernel(
        "pg", hole_diameter=1e-3, hole_spacing=2e-3, wavelength=800e-9
    )
    true_scale = 1.6
    trace = maketrace(g.omega, delays, ew, "pg", smearing=kernel.scaled(true_scale))
    guess = gaussian_pulse(g, 3.0e-15)

    fit = LBFGSAD(maxiters=400, smearing=kernel, fit_smearing=True).run(
        trace, g.omega, delays, "pg", guess=guess
    )
    no_fit = LBFGSAD(maxiters=400, smearing=kernel).run(
        trace, g.omega, delays, "pg", guess=guess
    )
    assert fit.smear_scale == pytest.approx(true_scale, rel=0.05)
    assert fit.error < no_fit.error
    # The kernel is held fixed unless asked for, so no scale is reported.
    assert no_fit.smear_scale is None


def test_lbfgs_ad_fit_smearing_requires_a_kernel():
    g = Grid(64, dt=0.8e-15)
    ew = gaussian_pulse(g, 4.0e-15)
    delays = np.linspace(-25e-15, 25e-15, 41)
    trace = maketrace(g.omega, delays, ew, "pg")
    with pytest.raises(ValueError, match="requires a smearing kernel"):
        LBFGSAD(maxiters=10, fit_smearing=True).run(
            trace, g.omega, delays, "pg", guess=ew
        )


def test_lbfgs_ad_smearing_rejects_shg():
    g = Grid(64, dt=0.5e-15)
    ew = gaussian_pulse(g, 3.0e-15)
    delays = np.linspace(-15e-15, 15e-15, 41)
    trace = maketrace(g.omega, delays, ew, "shg")
    kernel = square_boxcars_kernel(
        "pg", hole_diameter=1e-3, hole_spacing=0.5e-3, wavelength=800e-9
    )
    with pytest.raises(ValueError, match="three-arm"):
        LBFGSAD(maxiters=10, smearing=kernel).run(
            trace, g.omega, delays, "shg", guess=ew
        )


def test_lbfgs_ad_spectral_reg_improves_spectrum(setup):
    # Noisy trace so the amplitude is under-constrained (a clean PG trace already
    # determines |E(ω)|); the spectral-match penalty then pulls it toward truth.
    g, _, delays = setup
    ew = gaussian_pulse(g, 2.5e-15) + 0.6 * gaussian_pulse(g, 1.2e-15)
    ew = ew / np.max(np.abs(ew))
    I_true = np.abs(ew) ** 2
    clean = maketrace(g.omega, delays, ew, "pg")
    rng = np.random.default_rng(0)
    trace = np.clip(clean + 0.03 * rng.standard_normal(clean.shape), 0.0, None)
    near = gaussian_pulse(g, 4.0e-15)

    def spec_l2(res):
        a = np.abs(res.spectrum)
        a /= a.max()
        s = np.sqrt(I_true)
        s /= s.max()
        return float(np.linalg.norm(a - s))

    r0 = LBFGSAD(maxiters=400).run(trace, g.omega, delays, "pg", guess=near)
    r1 = LBFGSAD(maxiters=400, reg_spectrum=0.5, spectrum_target=I_true).run(
        trace, g.omega, delays, "pg", guess=near
    )
    assert r1.error < 5e-2  # near the noise floor
    assert spec_l2(r1) < spec_l2(r0)


@pytest.mark.parametrize("interaction", ["pg", "shg"])
def test_lbfgs_ad_matches_lbfgs(setup, interaction):
    """``lbfgs-ad`` reaches a final error comparable to analytic ``lbfgs``."""
    g, ew, delays = setup
    trace = maketrace(g.omega, delays, ew, interaction)
    near = gaussian_pulse(g, 1.5e-15)
    r_analytic = LBFGS(maxiters=300).run(
        trace, g.omega, delays, interaction, guess=near
    )
    r_ad = LBFGSAD(maxiters=300).run(trace, g.omega, delays, interaction, guess=near)
    # both drive the FROG error to (near) zero on this clean synthetic problem
    assert r_analytic.error < 5e-3
    assert r_ad.error < 5e-3


# ---------------------------------------------------------------------------
# Extra fitted parameters: delay-zero offset and slab thickness
# ---------------------------------------------------------------------------
def test_lbfgs_ad_fits_tau0(setup):
    """A known delay-zero offset is recovered, lowering the trace error."""
    g, ew, delays = setup
    tau0_true = 1.5e-15
    # Data whose true zero sits at +tau0_true (trace evaluated on shifted delays).
    trace = maketrace(g.omega, delays - tau0_true, ew, "shg")
    no_fit = LBFGSAD(maxiters=400).run(trace, g.omega, delays, "shg", guess=ew)
    fit = LBFGSAD(maxiters=400, fit_tau0=True).run(
        trace, g.omega, delays, "shg", guess=ew
    )
    assert fit.tau0 == pytest.approx(tau0_true, abs=0.2e-15)
    assert fit.error < no_fit.error
    assert fit.error < 1e-3


def test_lbfgs_ad_fits_thickness():
    """A known slab thickness is recovered in the strong-dispersion regime."""
    g = Grid(128, dt=2e-15)
    omega0 = wlfreq(800e-9)
    ew = gaussian_pulse(g, 8e-15)
    delays = np.linspace(-120e-15, 120e-15, 90)
    L_true = 400e-6
    trace = maketrace(
        g.omega,
        delays,
        ew,
        "pg",
        material="SiO2",
        thickness=L_true,
        npoints=20,
        omega0=omega0,
    )
    # The free pulse alone cannot absorb a 150 um thickness error.
    no_fit = LBFGSAD(
        maxiters=400, material="SiO2", thickness=250e-6, npoints=20, omega0=omega0
    ).run(trace, g.omega, delays, "pg", guess=ew, omega0=omega0)
    fit = LBFGSAD(
        maxiters=400,
        material="SiO2",
        thickness=250e-6,
        npoints=20,
        omega0=omega0,
        fit_thickness=True,
    ).run(trace, g.omega, delays, "pg", guess=ew, omega0=omega0)
    assert fit.thickness == pytest.approx(L_true, rel=0.05)
    assert fit.error < no_fit.error


def test_lbfgs_ad_two_phase_polish(setup):
    """The two-phase polish recovers the offset and populates result fields."""
    g, ew, delays = setup
    tau0_true = 1.0e-15
    trace = maketrace(g.omega, delays - tau0_true, ew, "shg")
    res = LBFGSAD(maxiters=300, fit_tau0=True, polish=True).run(
        trace, g.omega, delays, "shg", guess=ew
    )
    assert res.tau0 == pytest.approx(tau0_true, abs=0.2e-15)
    assert res.thickness is None  # thickness not fitted


def test_lbfgs_ad_holds_a_non_fitted_tau0_centre(setup):
    """A known delay-zero offset shifts the model even when it is not fitted.

    ``tau0`` is the centre the fit optimises around, so passing it without
    ``fit_tau0`` holds the model there — the same contract as ``thickness`` and
    ``smear_scale``.
    """
    g, ew, delays = setup
    tau0_true = 1.5e-15
    trace = maketrace(g.omega, delays - tau0_true, ew, "shg")
    # a near (not exact) guess: starting on the solution leaves NLopt nothing to
    # do and it stops with a round-off error, masking what is under test here
    near = gaussian_pulse(g, 2.0e-15)
    wrong = LBFGSAD(maxiters=200).run(trace, g.omega, delays, "shg", guess=near)
    held = LBFGSAD(maxiters=200, tau0=tau0_true).run(
        trace, g.omega, delays, "shg", guess=near
    )
    assert held.error < wrong.error
    assert held.error < 1e-3
    # not fitted, so nothing is reported back as a fitted value
    assert held.tau0 == 0.0


def test_lbfgs_ad_tau0_centre_warm_starts_the_fit(setup):
    """Seeding τ₀ at a previous fit's value converges in far fewer iterations."""
    g, ew, delays = setup
    tau0_true = 1.5e-15
    trace = maketrace(g.omega, delays - tau0_true, ew, "shg")
    near = gaussian_pulse(g, 2.0e-15)
    common = dict(maxiters=15, fit_tau0=True)
    cold = LBFGSAD(**common).run(trace, g.omega, delays, "shg", guess=near)
    warm = LBFGSAD(tau0=tau0_true, **common).run(
        trace, g.omega, delays, "shg", guess=near
    )
    assert warm.error < cold.error
    assert warm.tau0 == pytest.approx(tau0_true, abs=0.2e-15)


def test_lbfgs_ad_smear_scale_centre_warm_starts_the_fit():
    """Seeding the smearing multiplier starts the search from the last fit's value.

    Checked on a short budget, where starting at the previous strength is still
    ahead; given enough iterations both runs converge to the same answer, which is
    the point of a warm start — it saves iterations, not accuracy.
    """
    g = Grid(64, dt=0.8e-15)
    ew = gaussian_pulse(g, 4.0e-15)
    delays = np.linspace(-25e-15, 25e-15, 41)
    kernel = square_boxcars_kernel(
        "pg", hole_diameter=1e-3, hole_spacing=2e-3, wavelength=800e-9
    )
    true_scale = 1.6
    trace = maketrace(g.omega, delays, ew, "pg", smearing=kernel.scaled(true_scale))
    guess = gaussian_pulse(g, 3.0e-15)  # near, not exact (see the τ₀ test above)
    common = dict(maxiters=10, smearing=kernel, fit_smearing=True)
    cold = LBFGSAD(**common).run(trace, g.omega, delays, "pg", guess=guess)
    warm = LBFGSAD(smear_scale=true_scale, **common).run(
        trace, g.omega, delays, "pg", guess=guess
    )
    assert warm.error < cold.error
    assert warm.smear_scale == pytest.approx(true_scale, rel=0.05)


def test_lbfgs_ad_no_extras_leaves_result_fields_unset(setup):
    g, ew, delays = setup
    trace = maketrace(g.omega, delays, ew, "shg")
    near = gaussian_pulse(g, 1.5e-15)
    res = LBFGSAD(maxiters=100).run(trace, g.omega, delays, "shg", guess=near)
    assert res.thickness is None
    assert res.tau0 == 0.0


def test_lbfgs_ad_thickness_fit_shg_guard(setup):
    g, ew, delays = setup
    trace = maketrace(g.omega, delays, ew, "shg")
    with pytest.raises(ValueError, match="dispersive"):
        LBFGSAD(
            material="SiO2", thickness=10e-6, omega0=wlfreq(400e-9), fit_thickness=True
        ).run(trace, g.omega, delays, "shg", guess=ew)
