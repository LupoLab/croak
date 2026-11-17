"""Two-beam delay smearing of SHG FROG (croak.smearing.shg_two_beam_kernel).

The reference is the reduction's own statement, evaluated by brute force: the
signal from transverse position x in the focus, weighted by the product of the
two beams' intensities there, sees the relative delay tau + 2 x sin(alpha) / c;
the detector sums over x.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from test_crystals import BBO, W0

from croak import crystals
from croak.constants import C
from croak.crystals import CrystalSetup
from croak.forward import ForwardModel
from croak.forward_jax import make_param_trace_fn
from croak.geometry import TwoBeamGeometry
from croak.grid import Grid
from croak.pulses import gaussian_pulse
from croak.smearing import SmearingKernel, check_kernel, shg_two_beam_kernel

# Beams 10 mm apart at f = 100 mm (2.86 degrees), 2.6 mm across: a 12.5 um focal
# waist and about 1.5 fs of rms delay smearing at 800 nm.
GEOMETRY = TwoBeamGeometry(
    beam_separation=10e-3, focal_length=0.1, beam_diameter=2.6e-3
)
GRID = Grid(128, dt=1.0e-15)
DELAYS = GRID.dt * np.arange(-40, 41)
THICKNESS = 30e-6


def _crystal(geometry=GEOMETRY):
    return CrystalSetup(
        BBO,
        math.radians(29.2),
        phase_match_wavelength=800e-9,
        geometry=geometry,
        polarization="perpendicular",
        delayed_beam="toward_axis",
    )


def test_kernel_width_is_the_spread_of_delay_across_two_tilted_beams():
    # Independent of the closed form: integrate over the crossing plane two
    # Gaussian beams at +-alpha, each of 1/e^2 waist radius w0 measured across its
    # own axis, weighting each point by the product of their intensities; each
    # beam's pulse front reaches x at x sin(alpha)/c, ahead or behind.
    alpha = math.atan(10e-3 / 0.2)
    w0 = 2 * 800e-9 * 0.1 / (math.pi * 2.6e-3)
    x = np.linspace(-8 * w0, 8 * w0, 200001)
    weight = np.exp(-2 * (x * math.cos(alpha)) ** 2 / w0**2) ** 2
    delay = 2 * x * math.sin(alpha) / C
    rms = math.sqrt(np.sum(weight * delay**2) / np.sum(weight))
    k = shg_two_beam_kernel(GEOMETRY, 800e-9)
    # the sum over 200001 points of a smooth Gaussian is exact to rounding;
    # abs=0 because the widths (~1e-15 s) are below approx's default abs tolerance
    assert k.sigma_delta == pytest.approx(rms, rel=1e-9, abs=0)
    assert (k.sigma_p, k.rho, k.npoints) == (0.0, 0.0, 1)
    # exactly the small-angle-free form lambda d / (sqrt 2 pi D c)
    closed = 800e-9 * 10e-3 / (math.sqrt(2) * math.pi * 2.6e-3 * C)
    assert k.sigma_delta == pytest.approx(closed, rel=1e-12, abs=0)
    # the spot, and so the width, scales with the wavelength
    assert shg_two_beam_kernel(GEOMETRY, 400e-9).sigma_delta == pytest.approx(
        k.sigma_delta / 2, rel=1e-12, abs=0
    )


def test_kernel_needs_the_beam_diameter_and_a_wavelength():
    with pytest.raises(ValueError, match="needs the beams' diameter"):
        shg_two_beam_kernel(TwoBeamGeometry(10e-3, 0.1), 800e-9)
    for bad in (0.0, -1.0, math.nan):
        with pytest.raises(ValueError, match="wavelength must be finite and positive"):
            shg_two_beam_kernel(GEOMETRY, bad)


def test_check_kernel_refuses_a_shape_parameter_for_shg_only():
    shaped = SmearingKernel(sigma_p=0.2e-15, sigma_delta=0.3e-15)
    with pytest.raises(ValueError, match="two beams"):
        check_kernel("shg", shaped)
    check_kernel("pg", shaped)
    check_kernel("shg", shg_two_beam_kernel(GEOMETRY, 800e-9))


def _brute_force(model_kw, ew, sigma):
    """Sum the unsmeared trace over focal positions, by Gauss-Hermite quadrature."""
    x, w = np.polynomial.hermite_e.hermegauss(40)
    w = w / w.sum()
    total = 0.0
    for xi, wi in zip(x, w, strict=True):
        # the relative delay at this position, sigma * xi with xi ~ N(0, 1)
        model = ForwardModel(GRID.omega, DELAYS + sigma * xi, "shg", **model_kw)
        total = total + wi * model.trace(ew, normalize=False)
    return total


@pytest.mark.parametrize("medium", ["thin", "crystal"])
def test_smeared_trace_is_the_average_over_the_focus(medium):
    ew = gaussian_pulse(GRID, 6e-15, phases=(0.0, 20e-45))
    kernel = shg_two_beam_kernel(GEOMETRY, 800e-9)
    if medium == "crystal":
        setup = _crystal()
        kw = dict(
            material=setup,
            thickness=THICKNESS,
            npoints=crystals.depth_points(setup, GRID.omega, W0, THICKNESS),
            omega0=W0,
        )
    else:
        kw = dict(omega0=W0)
    smeared = ForwardModel(GRID.omega, DELAYS, "shg", smearing=kernel, **kw).trace(
        ew, normalize=False
    )
    reference = _brute_force(kw, ew, kernel.sigma_delta)
    unsmeared = ForwardModel(GRID.omega, DELAYS, "shg", **kw).trace(ew, normalize=False)
    # The model convolves on the sampled delay axis by DFT (periodic) and the
    # reference shifts the delays exactly; the trace has decayed to ~1e-12 of its
    # peak at the edges of the +-40 fs axis, so the two agree to rounding.
    peak = reference.max()
    np.testing.assert_allclose(smeared, reference, rtol=0, atol=1e-9 * peak)
    # and the smearing is not negligible here
    assert np.abs(smeared - unsmeared).max() > 0.05 * peak


def test_numpy_and_jax_agree_with_a_crystal_and_smearing():
    ew = gaussian_pulse(GRID, 6e-15, phases=(10e-30,))
    setup = _crystal()
    npoints = crystals.depth_points(setup, GRID.omega, W0, THICKNESS)
    kernel = shg_two_beam_kernel(GEOMETRY, 800e-9)
    kw = dict(material=setup, npoints=npoints, omega0=W0, smearing=kernel)
    numpy_trace = ForwardModel(
        GRID.omega, DELAYS, "shg", thickness=THICKNESS, **kw
    ).trace(ew, normalize=False)
    jax_fn = make_param_trace_fn(GRID.omega, DELAYS, "shg", thickness=THICKNESS, **kw)
    jax_trace = np.asarray(jax_fn(ew, THICKNESS, 0.0))
    # same sums in the same order up to FFT rounding
    np.testing.assert_allclose(
        jax_trace, numpy_trace, rtol=0, atol=1e-10 * numpy_trace.max()
    )
    # the smearing scale widens the blur, as for PG/SD (fit_smearing)
    wider = np.asarray(jax_fn(ew, THICKNESS, 0.0, 1.5))
    expected = ForwardModel(
        GRID.omega,
        DELAYS,
        "shg",
        thickness=THICKNESS,
        **{**kw, "smearing": kernel.scaled(1.5)},
    ).trace(ew, normalize=False)
    np.testing.assert_allclose(wider, expected, rtol=0, atol=1e-10 * expected.max())


def test_jax_normalised_smeared_shg_trace_has_unit_peak():
    ew = gaussian_pulse(GRID, 6e-15)
    kernel = shg_two_beam_kernel(GEOMETRY, 800e-9)
    fn = make_param_trace_fn(GRID.omega, DELAYS, "shg", smearing=kernel, normalize=True)
    t = np.asarray(fn(ew, 0.0, 0.0))
    expected = ForwardModel(GRID.omega, DELAYS, "shg", smearing=kernel).trace(ew)
    assert t.max() == pytest.approx(1.0)
    np.testing.assert_allclose(t, expected, rtol=0, atol=1e-10)


def test_a_split_smearing_fit_needs_a_shape_channel():
    from croak.lbfgs_ad import LBFGSAD
    from croak.smearing import check_split_fit

    kernel = shg_two_beam_kernel(GEOMETRY, 800e-9)
    with pytest.raises(ValueError, match="needs a kernel with a shape channel"):
        check_split_fit(kernel, True)
    check_split_fit(kernel, False)
    check_split_fit(None, True)
    check_split_fit(SmearingKernel(sigma_p=0.2e-15, sigma_delta=0.3e-15), True)
    ew = gaussian_pulse(GRID, 6e-15)
    trace = ForwardModel(GRID.omega, DELAYS, "shg", smearing=kernel).trace(ew)
    solver = LBFGSAD(
        maxiters=2, smearing=kernel, fit_smearing=True, fit_smearing_split=True
    )
    with pytest.raises(ValueError, match="shape channel"):
        solver.run(trace, GRID.omega, DELAYS, "shg", guess=ew)


def test_retrieving_a_smeared_shg_trace_needs_the_kernel():
    # A 6 fs pulse smeared by 1.5 fs: retrieved with the kernel the trace is fitted
    # far better, and the pulse's duration recovered, than without it.
    import croak
    from croak.maths import fwhm

    ew = gaussian_pulse(GRID, 6e-15, phases=(0.0, 20e-45))
    kernel = shg_two_beam_kernel(GEOMETRY, 800e-9)
    trace = ForwardModel(GRID.omega, DELAYS, "shg", smearing=kernel).trace(ew)
    common = dict(algorithm="lbfgs-ad", maxiters=400, rng=np.random.default_rng(0))
    with_kernel = croak.retrieve(
        trace, GRID.omega, DELAYS, "shg", smearing=kernel, **common
    )
    without = croak.retrieve(trace, GRID.omega, DELAYS, "shg", **common)
    assert with_kernel.error < 0.2 * without.error

    def duration(spectrum):
        return fwhm(GRID.t, np.abs(GRID.ifft(spectrum)) ** 2)

    true = duration(ew)
    assert abs(duration(with_kernel.spectrum) - true) < 0.03 * true
    assert abs(duration(without.spectrum) - true) > 0.05 * true


@pytest.mark.parametrize(
    "algorithm", ["lbfgs-ad", "lm", "lm-optx", "cma-es", "warm-lbfgs"]
)
def test_every_extras_solver_refuses_a_split_fit_of_a_two_beam_sd_kernel(algorithm):
    # The two-beam SD kernel has sigma_p = 0 too; each solver that fits extras
    # refuses the split before it starts.
    from croak.retrieve import ALGORITHMS
    from croak.smearing import collinear_sd_kernel

    kernel = collinear_sd_kernel(
        hole_diameter=1e-3, hole_spacing=0.5e-3, wavelength=800e-9
    )
    assert kernel.sigma_p == 0.0
    ew = gaussian_pulse(GRID, 6e-15)
    trace = ForwardModel(GRID.omega, DELAYS, "sd", smearing=kernel).trace(ew)
    solver = ALGORITHMS[algorithm](
        maxiters=2, smearing=kernel, fit_smearing=True, fit_smearing_split=True
    )
    with pytest.raises(ValueError, match="shape channel"):
        solver.run(trace, GRID.omega, DELAYS, "sd", guess=ew)


def test_the_covariance_refuses_a_split_fit_of_a_two_beam_kernel():
    import croak
    from croak.covariance import parameter_covariance

    kernel = shg_two_beam_kernel(GEOMETRY, 800e-9)
    ew = gaussian_pulse(GRID, 6e-15)
    trace = ForwardModel(GRID.omega, DELAYS, "shg", smearing=kernel).trace(ew)
    res = croak.retrieve(
        trace,
        GRID.omega,
        DELAYS,
        "shg",
        algorithm="lbfgs-ad",
        maxiters=5,
        smearing=kernel,
    )
    with pytest.raises(ValueError, match="shape channel"):
        parameter_covariance(
            res,
            trace,
            noise=1e-3,
            smearing=kernel,
            fit_smearing=True,
            fit_smearing_split=True,
        )
