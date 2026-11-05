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
    assert_allclose(x, [0.0, 1.5, 3.0])
    assert_allclose(y, [0.0, 1.0])
    # Linear in x along each row: resampling moves no feature.
    assert_allclose(values, [[0.0, 3.0, 6.0], [0.0, 1.5, 3.0]])


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
