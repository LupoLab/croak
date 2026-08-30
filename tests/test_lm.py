"""Convergence tests for the Levenberg–Marquardt solver (``lm``).

The LM retriever drives the flattened FROG residual to zero with SciPy
``least_squares`` and a JAX Jacobian. The default ``method="trf"`` uses the
analytic JAX Jacobian; ``method="lm"`` falls back to a numeric Jacobian (the JAX
Jacobian crashes MINPACK's Fortran callback — see :mod:`croak.lm`). Both paths are
exercised here.
"""

import numpy as np
import pytest

from croak.forward import maketrace
from croak.grid import Grid
from croak.lm import LM
from croak.maths import wlfreq
from croak.pulses import gaussian_pulse


@pytest.fixture
def setup():
    g = Grid(128, dt=0.3e-15)
    ew = gaussian_pulse(g, 2.5e-15)
    delays = np.linspace(-15e-15, 15e-15, 87)
    return g, ew, delays


@pytest.mark.parametrize("interaction", ["pg", "shg"])
def test_lm_near_initial(setup, interaction):
    g, ew, delays = setup
    trace = maketrace(g.omega, delays, ew, interaction)
    near = gaussian_pulse(g, 1.5e-15)
    res = LM(maxiters=300).run(trace, g.omega, delays, interaction, guess=near)
    assert res.error < 5e-3


def test_lm_chirped_phase_only(setup):
    g, _, delays = setup
    ew = gaussian_pulse(g, 2.5e-15, phases=[3e-30])
    trace = maketrace(g.omega, delays, ew, "pg")
    guess = gaussian_pulse(g, 2.5e-15, phases=[1e-30])
    res = LM(maxiters=300, phase_only=True).run(
        trace, g.omega, delays, "pg", guess=guess
    )
    assert res.error < 1e-2


def test_lm_dispersive():
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
    res = LM(
        maxiters=400, material="SiO2", thickness=10e-6, npoints=20, omega0=omega0
    ).run(trace, g.omega, delays, "pg", guess=near)
    assert res.error < 5e-2


def test_lm_spectral_reg_improves_spectrum(setup):
    g, _, delays = setup
    ew = gaussian_pulse(g, 2.5e-15) + 0.6 * gaussian_pulse(g, 1.2e-15)
    ew = ew / np.max(np.abs(ew))
    trace = maketrace(g.omega, delays, ew, "pg")
    I_true = np.abs(ew) ** 2
    near = gaussian_pulse(g, 4.0e-15)

    def spec_l2(res):
        a = np.abs(res.spectrum)
        a /= a.max()
        s = np.sqrt(I_true)
        s /= s.max()
        return float(np.linalg.norm(a - s))

    r0 = LM(maxiters=400).run(trace, g.omega, delays, "pg", guess=near)
    r1 = LM(maxiters=400, reg_spectrum=0.3, spectrum_target=I_true).run(
        trace, g.omega, delays, "pg", guess=near
    )
    assert r1.error < 5e-3
    assert spec_l2(r1) < spec_l2(r0)


def test_lm_classic_method_numeric_jac(setup):
    """method='lm' (MINPACK, numeric Jacobian) also converges and warns."""
    g, ew, delays = setup
    trace = maketrace(g.omega, delays, ew, "pg")
    near = gaussian_pulse(g, 1.5e-15)
    with pytest.warns(RuntimeWarning, match="numeric Jacobian"):
        res = LM(maxiters=300, method="lm").run(
            trace, g.omega, delays, "pg", guess=near
        )
    assert res.error < 5e-3


# ---------------------------------------------------------------------------
# Extra fitted parameters: delay-zero offset and slab thickness
# ---------------------------------------------------------------------------
def test_lm_fits_tau0(setup):
    """A known delay-zero offset is recovered, lowering the trace error."""
    g, ew, delays = setup
    tau0_true = 1.5e-15
    trace = maketrace(g.omega, delays - tau0_true, ew, "shg")
    no_fit = LM(maxiters=300).run(trace, g.omega, delays, "shg", guess=ew)
    fit = LM(maxiters=300, fit_tau0=True).run(trace, g.omega, delays, "shg", guess=ew)
    assert fit.tau0 == pytest.approx(tau0_true, abs=0.2e-15)
    assert fit.error < no_fit.error
    assert fit.error < 1e-3


def test_lm_fits_thickness():
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
    no_fit = LM(
        maxiters=300, material="SiO2", thickness=250e-6, npoints=20, omega0=omega0
    ).run(trace, g.omega, delays, "pg", guess=ew, omega0=omega0)
    fit = LM(
        maxiters=300,
        material="SiO2",
        thickness=250e-6,
        npoints=20,
        omega0=omega0,
        fit_thickness=True,
    ).run(trace, g.omega, delays, "pg", guess=ew, omega0=omega0)
    assert fit.thickness == pytest.approx(L_true, rel=0.05)
    assert fit.error < no_fit.error


def test_lm_two_phase_polish_thickness():
    """The two-phase polish recovers the thickness even from a cold pulse start."""
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
    res = LM(
        maxiters=300,
        material="SiO2",
        thickness=250e-6,
        npoints=20,
        omega0=omega0,
        fit_thickness=True,
        polish=True,
    ).run(trace, g.omega, delays, "pg", guess=ew, omega0=omega0)
    assert res.thickness == pytest.approx(L_true, rel=0.05)


def test_lm_thickness_fit_shg_guard(setup):
    g, ew, delays = setup
    trace = maketrace(g.omega, delays, ew, "shg")
    with pytest.raises(ValueError, match="dispersive"):
        LM(
            material="SiO2", thickness=10e-6, omega0=wlfreq(400e-9), fit_thickness=True
        ).run(trace, g.omega, delays, "shg", guess=ew)
