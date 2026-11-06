"""The display conversion: a screen view on wavelength or frequency, and dB range."""

from __future__ import annotations

import numpy as np
import pytest

from croak.plotting.display import (
    DB_RANGE_LIMITS,
    SPEED_OF_LIGHT_NM_PHZ,
    Display,
    displayed,
)
from croak.plotting.screen import (
    DB_UNIT,
    FREQUENCY_LABEL,
    WAVELENGTH_LABEL,
    BandView,
    CurveView,
    LineView,
    MapView,
    SpanView,
)

C = SPEED_OF_LIGHT_NM_PHZ
LAM = np.linspace(600.0, 1000.0, 81)  # nm
DELAY = np.linspace(-50.0, 50.0, 41)  # fs


def _gaussian_in_frequency(lam):
    """A spectrum that is a Gaussian in frequency, as a wavelength density."""
    nu = C / lam
    return np.exp(-(((nu - 0.375) / 0.03) ** 2)) * nu**2  # I_λ = I_ν ν²/c


def _line(xlabel=WAVELENGTH_LABEL, **extra):
    return LineView(
        curves=(
            CurveView(LAM, _gaussian_in_frequency(LAM), "measured", "spectrum"),
            CurveView(
                LAM, np.full(LAM.size, 2.0), "phase", "phase", True, density=False
            ),
        ),
        title="Spectrum",
        xlabel=xlabel,
        ylabel="Intensity",
        **extra,
    )


def _map(*, db: bool, ylabel=WAVELENGTH_LABEL, signed=False):
    column = _gaussian_in_frequency(LAM)
    image = np.outer(column / column.max(), np.exp(-((DELAY / 20.0) ** 2)))
    if signed:
        image = image - 0.1
    if db:
        image = np.clip(10 * np.log10(np.maximum(image, 1e-30)), -120.0, 0.0)
    return MapView(
        values=image,
        x=DELAY,
        y=LAM,
        levels=(-30.0, 0.0) if db else (float(image.min()), float(image.max())),
        title="Trace",
        xlabel="Delay (fs)",
        ylabel=ylabel,
        unit=DB_UNIT if db else "a.u.",
        signed=signed,
        yrange=(650.0, 950.0),
    )


@pytest.mark.parametrize(
    ("axis", "db_range", "match"),
    [
        ("log", 30.0, "axis must be one of"),
        ("native", DB_RANGE_LIMITS[0] - 1, "db_range must be within"),
        ("native", DB_RANGE_LIMITS[1] + 1, "db_range must be within"),
    ],
)
def test_display_refuses_an_unknown_axis_or_range(axis, db_range, match):
    with pytest.raises(ValueError, match=match):
        Display(axis, db_range)


@pytest.mark.parametrize(
    ("view", "display"),
    [
        (_line(), Display()),  # native keeps every view's own axis
        (_line(), Display(axis="wavelength")),  # already there
        (_line(xlabel="Time (fs)"), Display(axis="frequency")),  # not spectral
    ],
)
def test_display_leaves_a_line_view_that_needs_nothing(view, display):
    assert displayed(view, display) is view


def test_display_moves_a_spectrum_to_frequency_with_the_jacobian():
    shown = displayed(_line(), Display(axis="frequency"))
    assert shown.xlabel == FREQUENCY_LABEL
    spectrum, phase = shown.curves
    # increasing positions, ν = c/λ
    np.testing.assert_allclose(spectrum.x, C / LAM[::-1])
    assert np.all(np.diff(spectrum.x) > 0)
    # the λ² Jacobian makes it the Gaussian in frequency it was built from,
    # rescaled to the old peak (tolerance: float rounding only)
    expected = np.exp(-(((spectrum.x - 0.375) / 0.03) ** 2))
    old_peak = _gaussian_in_frequency(LAM).max()
    np.testing.assert_allclose(
        spectrum.y, expected / expected.max() * old_peak, rtol=1e-9
    )
    # a phase only moves
    np.testing.assert_array_equal(phase.y, np.full(LAM.size, 2.0))


def test_display_round_trip_returns_a_spectrum():
    there = displayed(_line(), Display(axis="frequency"))
    back = displayed(there, Display(axis="wavelength"))
    np.testing.assert_allclose(back.curves[0].x, LAM)
    np.testing.assert_allclose(back.curves[0].y, _line().curves[0].y, rtol=1e-9)


def test_display_moves_ranges_rules_spans_and_bands():
    view = _line(
        xrange=(700.0, 900.0),
        marks=((800.0, "measured"), (-1.0, "predicted")),
        spans=(
            SpanView(650.0, 750.0, "prior"),
            SpanView(0.0, 1.0, "prior", vertical=False),
            SpanView(-5.0, 750.0, "prior"),
        ),
        bands=(BandView(LAM, np.zeros(LAM.size), np.ones(LAM.size), "estimate"),),
    )
    shown = displayed(view, Display(axis="frequency"))
    assert shown.xrange == pytest.approx((C / 900.0, C / 700.0))
    assert shown.marks == ((pytest.approx(C / 800.0), "measured"),)  # x ≤ 0 dropped
    moved, across, unmovable = shown.spans
    assert (moved.low, moved.high) == pytest.approx((C / 750.0, C / 650.0))
    assert across is view.spans[1]  # an interval of y does not move
    assert unmovable is view.spans[2]  # it reaches zero: left as it was
    np.testing.assert_allclose(shown.bands[0].x, C / LAM[::-1])


def test_display_drops_a_window_reaching_zero():
    shown = displayed(_line(xrange=(0.0, 900.0)), Display(axis="frequency"))
    assert shown.xrange is None


def test_display_keeps_a_curve_with_nothing_to_scale():
    view = LineView(
        curves=(CurveView(LAM, np.zeros(LAM.size), "measured"),),
        title="Empty",
        xlabel=WAVELENGTH_LABEL,
        ylabel="",
    )
    shown = displayed(view, Display(axis="frequency"))
    np.testing.assert_array_equal(shown.curves[0].y, 0.0)
    nan = LineView(
        curves=(CurveView(LAM, np.full(LAM.size, np.nan), "measured"),),
        title="Blank",
        xlabel=WAVELENGTH_LABEL,
        ylabel="",
    )
    assert np.isnan(displayed(nan, Display(axis="frequency")).curves[0].y).all()


def test_display_sets_every_log_map_range():
    view = _map(db=True)
    shown = displayed(view, Display(db_range=60.0))
    assert shown.levels == (-60.0, 0.0)
    assert shown.values is view.values  # no axis change asked for
    linear = _map(db=False)
    assert displayed(linear, Display(db_range=60.0)) is linear


def test_display_moves_a_log_map_and_keeps_its_peak_at_zero():
    shown = displayed(_map(db=True), Display(axis="frequency", db_range=40.0))
    assert shown.ylabel == FREQUENCY_LABEL
    assert shown.levels == (-40.0, 0.0)
    assert shown.yrange == pytest.approx((C / 950.0, C / 650.0))
    # the peak is set before the resampling onto an even axis, which can fall
    # between samples: a few thousandths of a dB at this sampling
    assert np.nanmax(shown.values) == pytest.approx(0.0, abs=0.01)
    assert np.nanmin(shown.values) >= -120.0
    # even samples on the new axis
    steps = np.diff(shown.y)
    np.testing.assert_allclose(steps, steps[0], rtol=1e-9)
    # the frequency-Gaussian peaks at 0.375 PHz once the Jacobian is applied
    # (tolerance: one sample of the resampled axis)
    peak_row = int(np.argmax(shown.values[:, DELAY.size // 2]))
    assert shown.y[peak_row] == pytest.approx(0.375, abs=steps[0])


@pytest.mark.parametrize("signed", [False, True])
def test_display_moves_a_linear_map_and_keeps_its_scale(signed):
    view = _map(db=False, signed=signed)
    shown = displayed(view, Display(axis="frequency"))
    # the scale is kept before the resampling onto an even axis, which can fall
    # between samples: 0.1 % at this sampling
    peak = np.max(np.abs(view.values))
    assert np.nanmax(np.abs(shown.values)) == pytest.approx(peak, rel=1e-3)
    assert shown.levels == view.levels


def test_display_moves_a_map_whose_spectral_axis_is_x():
    view = _map(db=False)
    across = MapView(
        values=view.values.T,
        x=LAM,
        y=DELAY,
        levels=view.levels,
        title="Across",
        xlabel=WAVELENGTH_LABEL,
        ylabel="Delay (fs)",
        xrange=(650.0, 950.0),
    )
    shown = displayed(across, Display(axis="frequency"))
    assert shown.xlabel == FREQUENCY_LABEL
    assert shown.values.shape == (DELAY.size, shown.x.size)
    assert shown.xrange == pytest.approx((C / 950.0, C / 650.0))
    np.testing.assert_array_equal(shown.y, DELAY)


def test_display_leaves_a_blank_map_unscaled():
    view = MapView(
        values=np.zeros((LAM.size, DELAY.size)),
        x=DELAY,
        y=LAM,
        levels=(0.0, 1.0),
        title="Blank",
        xlabel="Delay (fs)",
        ylabel=WAVELENGTH_LABEL,
    )
    shown = displayed(view, Display(axis="frequency"))
    assert not shown.values.any()
