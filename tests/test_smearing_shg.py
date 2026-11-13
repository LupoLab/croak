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


def test_kernel_width_is_the_focal_rms_across_the_delay_gradient():
    k = shg_two_beam_kernel(GEOMETRY, 800e-9)
    waist = 2 * 800e-9 * 0.1 / (math.pi * 2.6e-3)
    alpha = math.atan(10e-3 / 0.2)
    # x rms under |E1|^2 |E2|^2 = exp(-4 x^2 / w0^2) is w0 / (2 sqrt 2); the delay
    # gradient is 2 sin(alpha) / c
    expected = 2 * math.sin(alpha) / C * waist / (2 * math.sqrt(2))
    assert k.sigma_delta == pytest.approx(expected, rel=1e-12)
    assert (k.sigma_p, k.rho, k.npoints) == (0.0, 0.0, 1)
    # the small-angle form, lambda d / (sqrt 2 pi D c), to O(alpha^2)
    small = 800e-9 * 10e-3 / (math.sqrt(2) * math.pi * 2.6e-3 * C)
    assert k.sigma_delta == pytest.approx(small, rel=2e-3)
    # the spot, and so the width, scales with the wavelength
    assert shg_two_beam_kernel(GEOMETRY, 400e-9).sigma_delta == pytest.approx(
        k.sigma_delta / 2
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
