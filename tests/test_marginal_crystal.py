"""The SHG frequency marginal through a Type I crystal, and the correction onto it.

The reference is the crystal forward model itself: summed over the grid's whole time
axis (a full period of delays), its trace's frequency marginal is the pair sum of
:func:`croak.crystals.shg_marginal` exactly (Parseval), up to the depth quadrature
(:func:`croak.crystals.depth_points`, below 1e-6 of the path per pair) and the pairs
that alias round the grid, which the linear sum leaves out and the FFT model keeps.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from test_crystals import BBO, W0

from croak import crystals, preprocess
from croak import marginal_checks as mc
from croak.crystals import CrystalSetup, PhaseMatching
from croak.forward import ForwardModel
from croak.geometry import TwoBeamGeometry
from croak.grid import Grid
from croak.maths import moment, wlfreq
from croak.pipeline import retrieve_from_tracedata
from croak.pulses import gaussian_pulse

GRID = Grid(64, dt=3.0e-15)  # about 560-1440 nm around 800 nm
GEOMETRY = TwoBeamGeometry(beam_separation=5e-3, focal_length=0.1)  # 1.43 degrees
SETUPS = {
    "collinear": {},
    "in-plane": {"geometry": GEOMETRY, "polarization": "parallel"},
    "principal-toward": {
        "geometry": GEOMETRY,
        "polarization": "perpendicular",
        "delayed_beam": "toward_axis",
    },
    "principal-away": {
        "geometry": GEOMETRY,
        "polarization": "perpendicular",
        "delayed_beam": "away_from_axis",
    },
}


def _setup(**kw) -> CrystalSetup:
    return CrystalSetup(BBO, math.radians(29.2), phase_match_wavelength=800e-9, **kw)


def _pulse(g=GRID):
    """A 12 fs Gaussian with 30 fs^2 of GDD, peak-normalised spectrum."""
    ew = gaussian_pulse(g, 12e-15, phases=(30e-30,))
    return ew / np.max(np.abs(ew))


def _full_period_trace(setup, thickness, ew, g=GRID):
    """The crystal model's trace over the grid's whole time axis (all delays)."""
    model = ForwardModel(
        g.omega,
        g.t,
        "shg",
        material=setup,
        thickness=thickness,
        npoints=crystals.depth_points(setup, g.omega, W0, thickness),
        omega0=W0,
    )
    return model.trace(ew, normalize=False)


def _on_grid(marginal, g=GRID):
    """The linear marginal's samples on the centred grid ``g.omega`` (around 2*w0)."""
    start = round(-g.omega[0] / g.domega)
    return marginal[start : start + g.n]


@pytest.mark.parametrize("name", SETUPS)
@pytest.mark.parametrize("thickness", [0.0, 20e-6, 100e-6])
def test_shg_marginal_is_the_crystal_models_delay_summed_trace(name, thickness):
    setup = _setup(**SETUPS[name])
    ew = _pulse()
    model = _full_period_trace(setup, thickness, ew).sum(axis=1)
    predicted = crystals.shg_marginal(
        GRID.omega + W0, np.abs(ew) ** 2, PhaseMatching(setup, thickness), W0
    )
    # Parseval over the N delays of a full period gives N times the pair sum; the
    # tolerance is the quadrature's 1e-6 per pair plus the aliased wings (measured
    # 4e-7 of the peak), so 2e-6.
    np.testing.assert_allclose(
        GRID.n * _on_grid(predicted), model, rtol=0, atol=2e-6 * model.max()
    )


def test_the_principal_plane_drift_changes_the_marginal():
    # The same crystal crossed out of and in its principal plane: the arms' delay
    # drift adds kappa (w1 - w2) to the mismatch. The pairs (w1, w2) and (w2, w1)
    # share a weight, so it enters the marginal at second order only: a 5e-4
    # change here, still 250 times the tolerance of the check above, which
    # therefore cannot pass by ignoring it.
    spectrum = np.abs(_pulse()) ** 2
    omega = GRID.omega + W0
    in_plane = _setup(**SETUPS["in-plane"])
    principal = _setup(**SETUPS["principal-toward"])
    assert crystals.propagation(principal, GRID.omega, W0, 100e-6).arm_shift != 0.0
    a = crystals.shg_marginal(omega, spectrum, PhaseMatching(in_plane, 100e-6), W0)
    b = crystals.shg_marginal(omega, spectrum, PhaseMatching(principal, 100e-6), W0)
    assert np.max(np.abs(a - b)) > 1e-4 * a.max()


def test_zero_thickness_is_the_autoconvolution_times_the_prefactor():
    setup = _setup()
    spectrum = np.abs(_pulse()) ** 2
    omega = GRID.omega + W0
    out = crystals.shg_marginal(omega, spectrum, PhaseMatching(setup, 0.0), W0)
    signal = 2.0 * omega[0] + GRID.domega * np.arange(2 * GRID.n - 1)
    amp = crystals.propagation(setup, signal - 2 * W0, W0, 0.0).sig_amp
    np.testing.assert_allclose(out, amp**2 * np.convolve(spectrum, spectrum))


def test_a_thick_crystal_narrows_the_predicted_marginal():
    spectrum = np.abs(_pulse()) ** 2
    omega = GRID.omega + W0
    thin = mc.predicted_marginal(
        omega, spectrum, "shg", crystal=PhaseMatching(_setup(), 1e-6), omega0=W0
    )
    thick = mc.predicted_marginal(
        omega, spectrum, "shg", crystal=PhaseMatching(_setup(), 200e-6), omega0=W0
    )
    assert thick.rms_width < 0.8 * thin.rms_width


def test_predicted_marginal_through_a_crystal_carries_its_moments():
    spectrum = np.abs(_pulse()) ** 2
    omega = GRID.omega + W0
    pm = PhaseMatching(_setup(**SETUPS["principal-toward"]), 50e-6)
    shape = crystals.shg_marginal(omega, spectrum, pm, W0)
    axis = 2.0 * omega[0] + GRID.domega * np.arange(shape.size)
    pred = mc.predicted_marginal(omega, spectrum, "SHG", crystal=pm, omega0=W0)
    np.testing.assert_allclose(pred.omega, axis)
    np.testing.assert_allclose(pred.marginal, shape / shape.max())
    assert pred.centroid == pytest.approx(moment(axis, shape, 1), rel=1e-12)
    assert pred.rms_width == pytest.approx(mc.rms_width(axis, shape), rel=1e-12)
    assert pred.width_bracket is None
    assert pred.centroid_is_hard and pred.width_is_hard
    assert pred.sigma_spectrum == pytest.approx(mc.rms_width(omega, spectrum))
    centroid = mc.predicted_centroid(omega, spectrum, "shg", crystal=pm, omega0=W0)
    assert centroid == pytest.approx(pred.centroid, rel=1e-12)


def test_a_spectrum_that_makes_no_shg_is_refused():
    with pytest.raises(ValueError, match="generates no SHG"):
        mc.predicted_marginal(
            GRID.omega + W0,
            np.zeros(GRID.n),
            "shg",
            crystal=PhaseMatching(_setup(), 10e-6),
            omega0=W0,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"interaction": "pg"}, "for SHG, not 'pg'"),
        ({"omega0": None}, "needs the pulse carrier"),
    ],
)
def test_a_crystal_prediction_needs_shg_and_the_carrier(kwargs, match):
    args = {"interaction": "shg", "omega0": W0} | kwargs
    spectrum = np.abs(_pulse()) ** 2
    pm = PhaseMatching(_setup(), 10e-6)
    for predict in (mc.predicted_marginal, mc.predicted_centroid):
        with pytest.raises(ValueError, match=match):
            predict(
                GRID.omega + W0,
                spectrum,
                args["interaction"],
                crystal=pm,
                omega0=args["omega0"],
            )


@pytest.mark.parametrize(
    ("omega", "spectrum", "match"),
    [
        (np.array([1.0]), np.array([1.0]), "at least 2 samples"),
        (np.arange(1.0, 5.0), np.ones(3), "same shape"),
        (np.array([1.0, 2.0, 4.0]), np.ones(3), "uniform and ascending"),
        (np.array([3.0, 2.0, 1.0]), np.ones(3), "uniform and ascending"),
    ],
)
def test_shg_marginal_rejects_a_bad_axis(omega, spectrum, match):
    with pytest.raises(ValueError, match=match):
        crystals.shg_marginal(omega * 1e15, spectrum, PhaseMatching(_setup(), 1e-6), W0)


@pytest.mark.parametrize("thickness", [-1e-6, math.nan, math.inf])
def test_phase_matching_rejects_a_bad_thickness(thickness):
    with pytest.raises(ValueError, match="finite and >= 0"):
        PhaseMatching(_setup(), thickness)


def test_the_correction_takes_out_a_spectral_response_and_keeps_the_crystal():
    # A crystal trace seen through a smooth instrument response R(omega): the
    # crystal-aware correction recovers the crystal trace (to the identity's
    # tolerance), where the thin-medium correction would also remove the
    # phase-matching filter.
    setup = _setup(**SETUPS["principal-toward"])
    pm = PhaseMatching(setup, 100e-6)
    ew = _pulse()
    trace = _full_period_trace(setup, pm.thickness, ew)
    response = np.exp(-(((GRID.omega - 0.3e15) / 1.5e15) ** 2))
    measured = trace * response[:, None]
    spectrum = np.abs(ew) ** 2
    out = mc.apply_marginal_correction(
        measured, spectrum, crystal=pm, omega=GRID.omega, omega0=W0
    )
    np.testing.assert_allclose(out / out.max(), trace / trace.max(), rtol=0, atol=1e-5)
    thin = mc.apply_marginal_correction(measured, spectrum)
    assert np.max(np.abs(thin / thin.max() - trace / trace.max())) > 1e-2


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"omega": None}, "needs the grid omega and omega0"),
        ({"omega0": None}, "needs the grid omega and omega0"),
        ({"omega": GRID.omega[:-1]}, "frequency rows"),
    ],
)
def test_a_crystal_correction_needs_the_grid(kwargs, match):
    args = {"omega": GRID.omega, "omega0": W0} | kwargs
    with pytest.raises(ValueError, match=match):
        mc.apply_marginal_correction(
            np.ones((GRID.n, 3)),
            np.ones(GRID.n),
            crystal=PhaseMatching(_setup(), 1e-6),
            omega=args["omega"],
            omega0=args["omega0"],
        )


# --- through load_and_clean and the retrieval -----------------------------------

PIPE_CRYSTAL = _setup(**SETUPS["principal-toward"])
PIPE_THICKNESS = 30e-6


def _measurement():
    """A crystal SHG trace through a tilted response, with its exact spectrum."""
    g = Grid(128, dt=1.5e-15)
    ew = gaussian_pulse(g, 10e-15, phases=(30e-30,))
    delays = 1.5e-15 * np.arange(-24, 25)
    model = ForwardModel(
        g.omega,
        delays,
        "shg",
        material=PIPE_CRYSTAL,
        thickness=PIPE_THICKNESS,
        npoints=crystals.depth_points(PIPE_CRYSTAL, g.omega, W0, PIPE_THICKNESS),
        omega0=W0,
    )
    trace = model.trace(ew, normalize=False)
    trace = trace * np.linspace(0.5, 1.5, g.n)[:, None]  # the instrument's tilt
    lam = wlfreq(g.omega + 2.0 * W0)
    trace = trace / (lam**2)[:, None]  # per unit wavelength, as measured
    order = np.argsort(lam)
    lam_spec = wlfreq(g.omega + W0)
    spec_order = np.argsort(lam_spec)
    # intensity per unit wavelength, from per unit frequency (|dw/dlam| ~ 1/lam^2)
    i_lam = np.abs(ew) ** 2 / lam_spec**2
    return dict(
        Ifrog=trace[order] / trace.max(),
        lam_frog=lam[order],
        scanaxis=delays,
        interaction="shg",
        lam_min=600e-9,
        lam_max=1200e-9,
        input_unit="delay",
        lam_spec=lam_spec[spec_order],
        Ilam_spec=i_lam[spec_order] / i_lam.max(),
    )


def test_load_and_clean_corrects_through_the_crystal_and_records_it():
    pm = PhaseMatching(PIPE_CRYSTAL, PIPE_THICKNESS)
    plain = preprocess.load_and_clean(**_measurement())
    corrected = preprocess.load_and_clean(
        **_measurement(), marginal_correct=True, marginal_crystal=pm
    )
    assert corrected.marginal_corrected
    assert corrected.marginal_crystal == pm
    assert plain.marginal_crystal is None
    assert plain.Iomega is not None
    expected = mc.apply_marginal_correction(
        plain.trace,
        plain.Iomega,
        crystal=pm,
        omega=plain.grid.omega,
        omega0=plain.omega0_pulse,
    )
    np.testing.assert_allclose(
        corrected.trace / corrected.trace.max(),
        expected / expected.max(),
        rtol=0,
        atol=1e-12,
    )


def test_load_and_clean_refuses_a_crystal_without_the_correction():
    with pytest.raises(ValueError, match="only with marginal_correct=True"):
        preprocess.load_and_clean(
            **_measurement(),
            marginal_crystal=PhaseMatching(PIPE_CRYSTAL, PIPE_THICKNESS),
        )


@pytest.fixture(scope="module")
def corrected_tracedata():
    return preprocess.load_and_clean(
        **_measurement(),
        marginal_correct=True,
        marginal_crystal=PhaseMatching(PIPE_CRYSTAL, PIPE_THICKNESS),
    )


def _retrieve(td, material, thickness):
    return retrieve_from_tracedata(
        td,
        algorithm="lbfgs-ad",
        material=material,
        thickness=thickness,
        npoints=crystals.depth_points(
            PIPE_CRYSTAL, td.grid.omega, td.omega0_pulse, PIPE_THICKNESS
        ),
        maxiters=3,
    )


def test_a_crystal_corrected_trace_retrieves_through_the_same_crystal(
    corrected_tracedata,
):
    res = _retrieve(corrected_tracedata, PIPE_CRYSTAL, PIPE_THICKNESS)
    assert np.all(np.isfinite(res.spectrum))


@pytest.mark.parametrize(
    ("material", "thickness", "got"),
    [
        (None, 0.0, "no crystal"),
        (_setup(), PIPE_THICKNESS, "another crystal set-up: pass the CrystalSetup"),
        (PIPE_CRYSTAL, 20e-6, "20 um"),
    ],
)
def test_a_crystal_corrected_trace_refuses_another_medium(
    corrected_tracedata, material, thickness, got
):
    with pytest.raises(ValueError, match=f"through 30 um of a crystal.*got {got}"):
        _retrieve(corrected_tracedata, material, thickness)


def test_a_thin_corrected_trace_refuses_a_crystal(corrected_tracedata):
    td = replace(corrected_tracedata, marginal_crystal=None)
    with pytest.raises(ValueError, match="onto the thin-medium marginal"):
        _retrieve(td, PIPE_CRYSTAL, PIPE_THICKNESS)


@pytest.mark.parametrize("n", [63, 81])
def test_the_correction_maps_the_marginal_onto_an_odd_grid(n):
    # On an odd grid the zero offset is not at n/2; the correction of an exact
    # crystal trace must leave it unchanged (to the identity's tolerance).
    g = Grid(n, dt=3.0e-15)
    setup = _setup(**SETUPS["principal-toward"])
    pm = PhaseMatching(setup, 50e-6)
    ew = _pulse(g)
    trace = _full_period_trace(setup, pm.thickness, ew, g)
    out = mc.apply_marginal_correction(
        trace, np.abs(ew) ** 2, crystal=pm, omega=g.omega, omega0=W0
    )
    np.testing.assert_allclose(out / out.max(), trace / trace.max(), atol=1e-5)


@pytest.mark.parametrize(
    "omega",
    [GRID.omega + W0, GRID.omega + 0.3 * GRID.domega, GRID.omega[: GRID.n // 2]],
    ids=["absolute", "off-sample", "half-grid"],
)
def test_a_crystal_correction_needs_the_centred_grid(omega):
    trace = np.ones((omega.size, 3))
    with pytest.raises(ValueError, match="centred grid"):
        mc.apply_marginal_correction(
            trace,
            np.ones(omega.size),
            crystal=PhaseMatching(_setup(), 1e-6),
            omega=omega,
            omega0=W0,
        )


@pytest.mark.parametrize("bad", ["spectrum", "omega"])
def test_shg_marginal_refuses_non_finite_input(bad):
    omega = GRID.omega + W0
    spectrum = np.abs(_pulse()) ** 2
    if bad == "spectrum":
        spectrum[10] = np.nan
    else:
        omega = omega.copy()
        omega[-1] = np.inf
    with pytest.raises(ValueError, match="must be finite"):
        crystals.shg_marginal(omega, spectrum, PhaseMatching(_setup(), 1e-6), W0)


def test_load_and_clean_takes_out_a_response_on_a_grid_with_another_carrier():
    # End to end: a crystal trace over a delay scan that holds it, seen through an
    # instrument response, regridded on a band whose carrier is not the
    # simulation's. Corrected through the crystal it matches the response-free
    # trace regridded the same way; the residual is the resampling's (measured
    # about 2e-4 of the peak), and uncorrected it is off by several per cent.
    g = Grid(256, dt=1.5e-15)  # fundamental axis stays at positive frequencies
    ew = gaussian_pulse(g, 10e-15, phases=(30e-30,))
    delays = 2.0e-15 * np.arange(-60, 61)
    model = ForwardModel(
        g.omega,
        delays,
        "shg",
        material=PIPE_CRYSTAL,
        thickness=PIPE_THICKNESS,
        npoints=crystals.depth_points(PIPE_CRYSTAL, g.omega, W0, PIPE_THICKNESS),
        omega0=W0,
    )
    clean = model.trace(ew, normalize=False)
    response = np.exp(-(((g.omega - 0.4e15) / 2.0e15) ** 2))[:, None]
    lam = wlfreq(g.omega + 2.0 * W0)
    clean = clean / (lam**2)[:, None]  # per unit wavelength, as measured
    order = np.argsort(lam)
    lam_spec = wlfreq(g.omega + W0)
    spec_order = np.argsort(lam_spec)
    i_lam = np.abs(ew) ** 2 / lam_spec**2
    common = dict(
        lam_frog=lam[order],
        scanaxis=delays,
        interaction="shg",
        lam_min=620e-9,
        lam_max=1150e-9,
        input_unit="delay",
        lam_spec=lam_spec[spec_order],
        Ilam_spec=i_lam[spec_order] / i_lam.max(),
    )
    reference = preprocess.load_and_clean(clean[order], **common).trace
    pm = PhaseMatching(PIPE_CRYSTAL, PIPE_THICKNESS)
    measured = (clean * response)[order]
    corrected = preprocess.load_and_clean(
        measured, **common, marginal_correct=True, marginal_crystal=pm
    )
    assert corrected.omega0_pulse != pytest.approx(W0, rel=1e-3)
    plain = preprocess.load_and_clean(measured, **common).trace

    def deviation(trace):
        return np.max(np.abs(trace / trace.max() - reference / reference.max()))

    assert deviation(corrected.trace) < 1e-3
    assert deviation(plain) > 2e-2


def test_a_crystal_corrected_trace_refuses_a_generation_envelope(corrected_tracedata):
    npoints = crystals.depth_points(
        PIPE_CRYSTAL,
        corrected_tracedata.grid.omega,
        corrected_tracedata.omega0_pulse,
        PIPE_THICKNESS,
    )
    with pytest.raises(ValueError, match="generation envelope"):
        retrieve_from_tracedata(
            corrected_tracedata,
            algorithm="lbfgs-ad",
            material=PIPE_CRYSTAL,
            thickness=PIPE_THICKNESS,
            npoints=npoints,
            depth_weight=np.ones(npoints),
            maxiters=3,
        )


def test_a_thin_corrected_trace_retrieves_without_a_crystal(corrected_tracedata):
    # a thin-medium correction and an ideal-SHG retrieval describe the same medium
    td = replace(corrected_tracedata, marginal_crystal=None)
    res = retrieve_from_tracedata(td, algorithm="lbfgs-ad", maxiters=3)
    assert np.all(np.isfinite(res.spectrum))
