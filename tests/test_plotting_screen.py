"""The screen data of the interactive retrieval views (:mod:`croak.plotting.screen`)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from croak import plotting
from croak.forward import maketrace
from croak.grid import Grid
from croak.maths import wlfreq
from croak.plotting import screen
from croak.processing import TruthPulse
from croak.pulses import gaussian_pulse
from croak.retrieve import retrieve


@pytest.fixture(scope="module")
def data():
    """A short PG retrieval with a known truth, as the panels' plot data."""
    g = Grid(96, dt=0.5e-15)
    omega0 = float(wlfreq(800e-9))
    ew = gaussian_pulse(g, 7e-15, phases=[20e-30])
    delays = np.linspace(-50e-15, 50e-15, 60)
    trace = maketrace(g.omega, delays, ew, "pg")
    res = retrieve(
        trace,
        g.omega,
        delays,
        "pg",
        algorithm="copra",
        maxiters=20,
        guess=ew,
        omega0=omega0,
    )
    truth = TruthPulse.from_spectrum(g, ew, omega0)
    return plotting.retrieval_plot_data(
        res,
        measured=trace,
        Iomega_meas=np.abs(ew) ** 2,
        lam_min=700e-9,
        lam_max=950e-9,
        truth=truth,
        flim=(0.7e15, 0.8e15),
    )


def test_screen_uniform_image_reverses_and_resamples():
    x, y, values = screen.uniform_image(
        [0.0, 1.0, 3.0], [1.0, 0.0], [[0.0, 1.0, 3.0], [0.0, 2.0, 6.0]]
    )
    # Resampled at the finest spacing (1), so no sample is lost.
    assert_allclose(x, [0.0, 1.0, 2.0, 3.0])
    assert_allclose(y, [0.0, 1.0])
    # Linear in x along each row: resampling moves no feature.
    assert_allclose(values, [[0.0, 2.0, 4.0, 6.0], [0.0, 1.0, 2.0, 3.0]])


def test_screen_uniform_image_caps_its_samples():
    x = np.concatenate([[0.0, 1e-6], np.linspace(1.0, 2.0, 5)])
    xe, _, values = screen.uniform_image(x, [0.0, 1.0], np.ones((2, x.size)))
    assert xe.size == screen.MAX_SAMPLES and values.shape == (2, screen.MAX_SAMPLES)


@pytest.mark.parametrize(
    ("x", "y", "values", "match"),
    [
        ([0.0, 1.0], [0.0, 1.0], [[1.0, 2.0]], "do not match"),
        ([0.0], [0.0], [[1.0]], "at least two samples"),
    ],
)
def test_screen_uniform_image_rejects_bad_shapes(x, y, values, match):
    with pytest.raises(ValueError, match=match):
        screen.uniform_image(x, y, values)


def test_screen_log_trace_spans_the_dynamic_range(data):
    view = screen.trace_map(data, "measured", db=True)
    assert view.levels == (-data.tracedb, 0.0) and view.unit == "dB"
    assert view.values.max() == pytest.approx(0.0)
    assert view.values.min() >= -data.tracedb
    assert view.title == "Measured (log)" and not view.signed
    assert view.yrange == pytest.approx((0.7, 0.8))  # flim in PHz
    assert view.xrange == pytest.approx((-50.0, 50.0))


def test_screen_linear_trace_shares_its_scale_and_shows_negatives(data):
    view = screen.trace_map(data, "retrieved", db=False)
    assert view.levels == (0.0, data.lin_vmax) and view.signed
    dipped = replace(data, measured=data.measured - 0.1 * data.measured.max())
    signed = screen.trace_map(dipped, "measured", db=False)
    assert signed.levels[0] < 0 < signed.levels[1]
    assert (
        screen.trace_map(replace(data, flim=None), "measured", db=True).yrange is None
    )
    with pytest.raises(ValueError, match="'measured' or 'retrieved'"):
        screen.trace_map(data, "truth", db=True)


def test_screen_residual_is_symmetric_and_optional(data):
    view = screen.residual_map(data)
    assert view is not None
    low, high = view.levels
    assert low == pytest.approx(-high) and view.colours == "bwr"
    empty = replace(data, processed=replace(data.processed, residual=None))
    assert screen.residual_map(empty) is None


def test_screen_spectrogram_is_in_the_wavelength_window(data):
    view = screen.spectrogram_map(data)
    assert view.yrange == pytest.approx((700.0, 950.0))
    assert view.levels[0] == 0.0 and view.levels[1] > 0


def test_screen_temporal_view_has_pulse_limit_truth_and_phases(data):
    view = screen.temporal_view(data)
    roles = [c.role for c in view.curves]
    assert roles == ["retrieved", "transform-limited", "truth", "phase", "truth-phase"]
    assert [c.right for c in view.curves] == [False, False, False, True, True]
    assert view.yrange is not None and view.yrange[0] == 0.0
    assert view.right_range is not None
    assert view.curves[2].name.startswith("truth (")


def test_screen_spectral_view_has_notes_and_warns_on_edge_energy(data):
    view = screen.spectral_view(data)
    roles = [c.role for c in view.curves]
    assert roles[:3] == ["retrieved", "measured", "truth"]
    assert "fit" in roles and "truth-phase" in roles
    (dispersion, _), (edge, warn) = view.notes
    assert dispersion.startswith("GDD:") and edge.startswith("Edge:")
    assert not warn
    leaky = replace(data, processed=replace(data.processed, edge_energy=0.5))
    assert screen.spectral_view(leaky).notes[1][1] is True


def test_screen_spectral_view_without_measurement_or_truth(data):
    bare = replace(
        data,
        truth=None,
        processed=replace(data.processed, Iw_meas=None),
        spectral_axis="wavelength",
    )
    view = screen.spectral_view(bare)
    assert [c.role for c in view.curves] == ["retrieved", "phase", "fit"]
    assert view.xlabel == "Wavelength (nm)"
    assert screen.temporal_view(bare).curves[-1].role == "phase"


def test_screen_convergence_view_marks_inner_handovers_only():
    view = screen.convergence_view([1.0, 0.5, 0.2, 0.1], boundaries=[0, 2, 4])
    assert view.log_y and view.marks == ((2.5, "boundary"),)
    assert_allclose(view.curves[0].x, [1, 2, 3, 4])


def test_screen_marginals_and_filter(data):
    freq = screen.frequency_marginal_view(data)
    delay = screen.delay_marginal_view(data)
    assert [c.role for c in freq.curves] == ["retrieved", "measured"]
    assert delay.xrange == pytest.approx((-50.0, 50.0))
    no_meas = replace(data, processed=replace(data.processed, tau_marg_meas=None))
    assert len(screen.delay_marginal_view(no_meas).curves) == 1
    filt = screen.filter_view(data)
    x = filt.curves[0].x
    assert np.all(np.diff(x) > 0)  # sorted by wavelength
    assert filt.xrange == pytest.approx((700.0, 950.0))


def test_screen_raw_trace_scales_by_the_trusted_band_and_crops_to_signal():
    lam = np.linspace(1100.0, 100.0, 11)  # descending, as some files store it
    delay = np.linspace(-20.0, 20.0, 5)
    trace = np.zeros((11, 5))
    trace[lam == 500.0] = 2.0  # the signal, in the trusted band
    trace[lam == 100.0] = 50.0  # amplified noise outside it
    trace[lam == 300.0, 0] = -0.5  # a negative left by background subtraction
    view = screen.raw_trace_map(delay, lam, trace, (190.0, 1000.0))
    assert view.levels == (-0.5, 2.0) and view.signed
    assert view.yrange == (500.0, 500.0)
    assert view.y[0] < view.y[-1]
    quiet = screen.raw_trace_map(delay, lam, np.zeros((11, 5)), (190.0, 1000.0))
    assert quiet.levels == (0.0, 1.0) and quiet.yrange is None


def test_screen_pulse_views_zoom_to_the_pulse_and_blank_its_wings():
    g = Grid(128, dt=0.5e-15)
    ew = gaussian_pulse(g, 7e-15, phases=[20e-30])
    temporal, spectral, gram = screen.pulse_field_views(g, ew, float(wlfreq(800e-9)))
    assert [c.role for c in temporal.curves] == ["retrieved", "phase"]
    lo, hi = temporal.xrange
    assert -40 < lo < 0 < hi < 40  # zoomed to a 7 fs pulse, not the whole window
    phase = temporal.curves[1].y
    assert np.isnan(phase[0]) and np.isfinite(phase[np.argmax(temporal.curves[0].y)])
    assert spectral.xlabel == "Wavelength (nm)"
    full = spectral.curves[0].x
    assert full.min() < spectral.xrange[0] < 800 < spectral.xrange[1] < full.max()
    # The even resampling of the uneven wavelength axis keeps the peak to 1 %.
    assert gram.levels == (0.0, 1.0)
    assert gram.values.max() == pytest.approx(1.0, abs=1e-2)
    assert gram.yrange is not None


def test_screen_pulse_views_tolerate_an_empty_field():
    g = Grid(64, dt=1e-15)
    temporal, spectral, gram = screen.pulse_field_views(
        g, np.zeros(g.n, dtype=complex), 2.35e15
    )
    assert temporal.xrange is None and spectral.xrange is None
    assert gram.yrange is None and gram.values.max() == 0.0


def test_screen_filter_views_cover_the_three_traces_in_both_scales(experiment):
    from croak import preprocess

    td = preprocess.load_and_clean(
        experiment.trace,
        experiment.lam_nm * 1e-9,
        experiment.delay_fs * 1e-15,
        experiment.interaction,
        lam_min=experiment.lam_min,
        lam_max=experiment.lam_max,
        input_unit="delay",
    )
    views = screen.filter_views(td, tracedb=20.0)
    assert list(views) == [key for key, *_ in screen.FILTER_PANELS]
    assert views["measured_db"].levels == (-20.0, 0.0)
    assert views["measured_db"].values.min() >= -20.0
    shared = plotting.filter_view_vmax(td)
    assert views["filtered_lin"].levels[1] == shared
    assert views["regridded_lin"].levels[1] == shared
    assert views["regridded_db"].xlabel == "Frequency (PHz)"
    assert views["filtered_lin"].xlabel == "Wavelength (nm)"
    assert all(v.ylabel == "Delay (fs)" for v in views.values())


def test_screen_simulated_views_show_the_file_density_on_a_unit_peak():
    lam = np.linspace(700e-9, 900e-9, 21)
    centre = np.exp(-(((lam - 800e-9) / 30e-9) ** 2))
    trace = np.outer(centre / lam**2, np.hanning(9))  # λ-density, as loaded
    data = {
        "lam": lam,
        "scanaxis": np.linspace(-20e-15, 20e-15, 9),
        "trace": trace,
        "lam_spec": lam,
        "Ilam_spec": centre,
    }
    linear, logarithmic, delay, spectral = screen.simulated_trace_views(data)
    # A unit peak before the even resampling, which on this coarse 21-sample
    # axis falls between samples: within 5 %.
    assert linear.levels == (0.0, 1.0)
    assert linear.values.max() == pytest.approx(1.0, abs=0.05)
    assert logarithmic.levels == (-screen.SIMULATED_DB, 0.0)
    assert linear.ylabel == "Frequency (PHz)" and linear.yrange is not None
    assert [c.name for c in spectral.curves] == ["trace marginal", "input"]
    assert delay.curves[0].y.max() == pytest.approx(1.0)
    blank = dict(data, trace=np.zeros_like(trace), Ilam_spec=None)
    linear, _, _, spectral = screen.simulated_trace_views(blank)
    assert linear.yrange is not None  # a blank trace shows its full range
    assert len(spectral.curves) == 1


def _estimate(method="parametric", *, profiles=True, thickness=False):
    from croak.uncertainty import UncertaintyResult

    rng = np.random.default_rng(1)
    t = np.linspace(-15e-15, 15e-15, 30)
    extra = {}
    if thickness:
        extra = {
            "thicknesses": rng.normal(10e-6, 1e-6, 40),
            "thickness_central": 10e-6,
            "thickness_sigma": 1e-6,
        }
    return UncertaintyResult(
        statistic="fwhm",
        point_estimate=5e-15,
        samples=5e-15 + 0.1e-15 * rng.standard_normal(40),
        interval_68=(4.9e-15, 5.1e-15),
        interval_95=(4.8e-15, 5.2e-15),
        interval_method="percentile",
        method=method,
        n_resamples=40,
        n_converged=40,
        frog_errors=rng.uniform(0.01, 0.02, 40),
        base_error=0.01,
        profiles=(
            np.exp(-((t / 3e-15) ** 2)) * rng.uniform(0.9, 1.1, (40, 1))
            if profiles
            else None
        ),
        t_profile=t if profiles else None,
        **extra,
    )


def test_screen_uncertainty_views_follow_the_composite_figure():
    hist, band = screen.uncertainty_views(_estimate(), truth=5.05e-15)
    assert hist.bars is not None and hist.bars.counts.sum() == 40
    assert len(hist.bars.edges) == 12 + 1  # a fifth of 40, at least 12 bins
    assert [s.name for s in hist.spans] == ["95 %", "68 %"]
    assert [role for _, role in hist.marks] == ["estimate", "truth-phase"]
    assert hist.title.startswith("5.00 ± ")
    assert [b.name for b in band.bands] == ["95 %", "68 %"]
    assert band.xrange is not None and band.xrange[0] < 0 < band.xrange[1]
    (only,) = screen.uncertainty_views(_estimate(profiles=False))
    assert only.marks == ((5.0, "estimate"),)


def test_screen_uncertainty_views_for_the_thickness_systematic():
    fwhm, error, band = screen.uncertainty_views(
        _estimate("thickness", thickness=True), truth=5e-15
    )
    assert fwhm.curves[0].points and error.curves[0].points
    assert [s.name for s in fwhm.spans] == ["L₀ ± σ_L", "68 %"]
    assert not fwhm.spans[1].vertical
    assert [role for _, role in fwhm.levels] == ["estimate", "truth-phase"]
    assert [s.name for s in error.spans] == ["L₀ ± σ_L"]
    assert band.title == "Temporal intensity"
    no_prior = replace(_estimate("thickness", thickness=True), thickness_sigma=None)
    fwhm, error, _ = screen.uncertainty_views(no_prior)
    assert [s.name for s in fwhm.spans] == ["68 %"] and error.spans == ()


def test_screen_uncertainty_band_of_nothing_has_no_window():
    flat = replace(_estimate(), profiles=np.zeros((40, 30)))
    _, band = screen.uncertainty_views(flat)
    assert band.xrange is None
