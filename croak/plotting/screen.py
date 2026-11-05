"""What the GUI's interactive retrieval views show, as plain data.

The matplotlib panels (:mod:`croak.plotting.panels`) draw a retrieval for a
figure; this module turns the same :class:`RetrievalPlotData` into the arrays an
interactive view draws: :class:`MapView` for an image and :class:`LineView` for
curves. Both use the panels' own helpers, so a curve on screen is the curve in
an exported figure. Nothing here imports Qt.

Two display choices differ from the figures. An image is resampled onto evenly
spaced axes (an interactive image is placed by its corners, a ``pcolormesh`` by
every edge), and a logarithmic trace shows the positive part only: negative
samples, from background subtraction, are visible on the linear maps, where the
signed colour map shows them in red.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..maths import wlfreq
from .panels import RetrievalPlotData, _filter_factor
from .pulse import (
    EDGE_ENERGY_WARN,
    _dispersion_text,
    _intensity_ylim,
    _phase_ylim,
    _spectral_curves,
    _spectral_xlim,
    _temporal_curves,
    _truth_intensity,
    _truth_spectral_phase,
    _truth_spectrum,
)
from .units import _TWOPI_PHZ

__all__ = [
    "CurveView",
    "LineView",
    "MapView",
    "convergence_view",
    "delay_marginal_view",
    "filter_view",
    "frequency_marginal_view",
    "raw_trace_map",
    "residual_map",
    "spectral_view",
    "spectrogram_map",
    "temporal_view",
    "trace_map",
    "uniform_image",
]

# Roles a view gives its curves; the GUI maps each to a colour and a line style.
ROLES = (
    "retrieved",
    "transform-limited",
    "measured",
    "truth",
    "phase",
    "fit",
    "truth-phase",
    "error",
    "predicted",
    "corrected",
    "faint",
    "boundary",
)


@dataclass(frozen=True)
class MapView:
    """An image on evenly spaced axes.

    Attributes
    ----------
    values : ndarray, shape (len(y), len(x))
        What is drawn, in the units of ``levels``.
    x, y : ndarray
        Increasing, evenly spaced sample positions.
    levels : tuple of float
        The colour scale's span.
    title, xlabel, ylabel, unit : str
        The view's title, axis labels and colour-bar label.
    colours : str
        The colour map's name.
    signed : bool
        Values below zero are meaningful (draw them in red, fading to white at 0).
    xrange, yrange : tuple of float or None
        The view's initial window (``None``: the whole image).
    """

    values: NDArray[np.float64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    levels: tuple[float, float]
    title: str
    xlabel: str
    ylabel: str
    unit: str = ""
    colours: str = "viridis"
    signed: bool = False
    xrange: tuple[float, float] | None = None
    yrange: tuple[float, float] | None = None


@dataclass(frozen=True)
class CurveView:
    """One curve of a :class:`LineView`; ``role`` decides its colour and style."""

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    role: str
    name: str = ""
    right: bool = False


@dataclass(frozen=True)
class LineView:
    """Curves against one x axis, with an optional right-hand axis.

    Attributes
    ----------
    curves : tuple of CurveView
        Left-axis curves first, then any on the right axis.
    title, xlabel, ylabel : str
        Labels.
    right_label : str or None
        The right-hand axis's label; ``None`` for none.
    xrange, yrange, right_range : tuple of float or None
        The initial window of each axis.
    log_y : bool
        A logarithmic left axis.
    notes : tuple of (str, bool)
        Short annotations, each with whether it is a warning.
    marks : tuple of (float, str)
        Vertical rules: x position and role (the role sets colour and style).
    """

    curves: tuple[CurveView, ...]
    title: str
    xlabel: str
    ylabel: str
    right_label: str | None = None
    xrange: tuple[float, float] | None = None
    yrange: tuple[float, float] | None = None
    right_range: tuple[float, float] | None = None
    log_y: bool = False
    notes: tuple[tuple[str, bool], ...] = field(default_factory=tuple)
    marks: tuple[tuple[float, str], ...] = field(default_factory=tuple)


def _evenly(axis: NDArray[np.float64]) -> NDArray[np.float64]:
    """``axis`` if its steps are equal (to 0.1 %), else an evenly spaced copy."""
    steps = np.diff(axis)
    if axis.size < 3 or np.allclose(steps, steps[0], rtol=1e-3, atol=0.0):
        return axis
    return np.linspace(axis[0], axis[-1], axis.size)


def uniform_image(
    x: ArrayLike, y: ArrayLike, values: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Put an image on increasing, evenly spaced axes for display.

    A decreasing axis is reversed with the image; an uneven one (a wavelength
    axis from an even frequency grid, an uneven delay scan) is resampled by
    linear interpolation, which moves no feature: each sample stays at its own
    position. The physics arrays are untouched; this is for drawing only.

    Parameters
    ----------
    x, y : array_like
        The sample positions, monotonic.
    values : array_like, shape (len(y), len(x))
        The image.

    Returns
    -------
    tuple of ndarray
        ``(x, y, values)`` on even, increasing axes.

    Raises
    ------
    ValueError
        If the shapes disagree or an axis has fewer than two samples.

    Examples
    --------
    >>> x, y, v = uniform_image([2.0, 1.0, 0.0], [0.0, 1.0], [[3, 2, 1], [6, 5, 4]])
    >>> x.tolist(), v[0].tolist()
    ([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    values = np.asarray(values, dtype=float)
    if values.shape != (y.size, x.size):
        raise ValueError(f"values {values.shape} do not match y×x {(y.size, x.size)}")
    if x.size < 2 or y.size < 2:
        raise ValueError("an image needs at least two samples along each axis")
    if x[0] > x[-1]:
        x, values = x[::-1], values[:, ::-1]
    if y[0] > y[-1]:
        y, values = y[::-1], values[::-1]
    xe, ye = _evenly(x), _evenly(y)
    if xe is not x:
        values = np.array([np.interp(xe, x, row) for row in values])
    if ye is not y:
        values = np.array([np.interp(ye, y, col) for col in values.T]).T
    return xe, ye, values


def _delay_window(data: RetrievalPlotData) -> tuple[float, float]:
    half = data.halfwidth / 1e-15
    return -half, half


def _frequency_window(data: RetrievalPlotData) -> tuple[float, float] | None:
    if data.flim is None:
        return None
    return data.flim[0] / 1e15, data.flim[1] / 1e15


def trace_map(data: RetrievalPlotData, which: str, *, db: bool) -> MapView:
    """Return the measured or retrieved trace against delay and signal frequency.

    Parameters
    ----------
    data : RetrievalPlotData
        The retrieval.
    which : {"measured", "retrieved"}
        Which trace.
    db : bool
        Logarithmic (``data.tracedb`` dynamic range below the peak), or linear
        on the scale the two linear traces share.

    Raises
    ------
    ValueError
        For any other ``which``.
    """
    if which not in ("measured", "retrieved"):
        raise ValueError(f"which must be 'measured' or 'retrieved', not {which!r}")
    trace = data.measured if which == "measured" else data.retrieved
    if db:
        peak = float(np.nanmax(trace)) or 1.0
        with np.errstate(divide="ignore"):
            shown = 10 * np.log10(np.maximum(trace, 0.0) / peak)
        shown = np.clip(shown, -data.tracedb, 0.0)
        levels, unit = (-data.tracedb, 0.0), "dB"
    else:
        shown, levels, unit = trace, (0.0, data.lin_vmax), "a.u."
    x, y, values = uniform_image(
        data.result.delays / 1e-15,
        (data.result.omega + data.omega0_trace) / _TWOPI_PHZ,
        shown,
    )
    low = float(np.nanmin(values))
    if not db and low < 0:
        levels = (low, data.lin_vmax)
    scale = "log" if db else "lin"
    return MapView(
        values=values,
        x=x,
        y=y,
        levels=levels,
        title=f"{which.capitalize()} ({scale})",
        xlabel="Delay (fs)",
        ylabel="Frequency (PHz)",
        unit=unit,
        signed=not db,
        xrange=_delay_window(data),
        yrange=_frequency_window(data),
    )


def residual_map(data: RetrievalPlotData) -> MapView | None:
    """Return measured minus retrieved on a symmetric scale (None without one)."""
    residual = data.processed.residual
    if residual is None:
        return None
    reach = float(np.max(np.abs(residual))) or 1.0
    x, y, values = uniform_image(
        data.result.delays / 1e-15,
        (data.result.omega + data.result.omega0) / _TWOPI_PHZ,
        residual,
    )
    return MapView(
        values=values,
        x=x,
        y=y,
        levels=(-reach, reach),
        title="Residuals",
        xlabel="Delay (fs)",
        ylabel="Frequency (PHz)",
        unit="a.u.",
        colours="bwr",
        xrange=_delay_window(data),
        yrange=_frequency_window(data),
    )


def spectrogram_map(data: RetrievalPlotData) -> MapView:
    """Return the retrieved pulse's Gabor spectrogram, time against wavelength."""
    t_centres, wavelength, s = data.spectrogram
    x, y, values = uniform_image(t_centres / 1e-15, wavelength / 1e-9, s)
    return MapView(
        values=values,
        x=x,
        y=y,
        levels=(0.0, float(np.nanmax(values)) or 1.0),
        title="Spectrogram",
        xlabel="Time (fs)",
        ylabel="Wavelength (nm)",
        unit="a.u.",
        xrange=_delay_window(data),
        yrange=(data.lam_min / 1e-9, data.lam_max / 1e-9),
    )


def temporal_view(data: RetrievalPlotData) -> LineView:
    """Return the retrieved pulse: intensity or power, transform limit, phase."""
    pr = data.processed
    retr, tl, r_label, tl_label, ylabel, power_scale = _temporal_curves(
        pr, data.power_scale
    )
    curves = [
        CurveView(pr.t_over / 1e-15, retr, "retrieved", r_label),
        CurveView(pr.t_tl / 1e-15, tl, "transform-limited", tl_label),
    ]
    truth_curve = None
    truth = data.truth
    if truth is not None:
        truth_curve = _truth_intensity(truth, pr.energy, power_scale)
        curves.append(
            CurveView(
                truth.t / 1e-15,
                truth_curve,
                "truth",
                f"truth ({truth.fwhm / 1e-15:.1f} fs)",
            )
        )
    t = pr.t_over / 1e-15
    curves.append(CurveView(t[pr.mask_t], pr.phi_t[pr.mask_t], "phase", "phase", True))
    truth_phase = None
    if truth is not None and truth.phi_t is not None:
        keep = np.asarray(truth.It) > 0.01
        truth_phase = np.asarray(truth.phi_t)[keep]
        curves.append(
            CurveView(
                np.asarray(truth.t)[keep] / 1e-15,
                truth_phase,
                "truth-phase",
                "truth phase",
                True,
            )
        )
    return LineView(
        curves=tuple(curves),
        title="Retrieved pulse",
        xlabel="Time (fs)",
        ylabel=ylabel,
        right_label="Phase (rad)",
        xrange=_delay_window(data),
        yrange=_intensity_ylim(retr, tl, truth_curve),
        right_range=_phase_ylim(pr.phi_t[pr.mask_t], truth_phase),
    )


def spectral_view(data: RetrievalPlotData) -> LineView:
    """Return the spectrum: retrieved, measured, phase and its fit, the truth.

    Its notes are the GDD/TOD of the phase fit and the energy at the grid's
    edge, a warning above :data:`~croak.plotting.EDGE_ENERGY_WARN`.
    """
    pr = data.processed
    axis = data.spectral_axis
    x, retrieved, measured, xlabel = _spectral_curves(pr, axis)
    curves = [CurveView(x, retrieved, "retrieved", "R")]
    if measured is not None:
        curves.append(CurveView(x, measured, "measured", "M"))
    truth = None if data.truth is None else _truth_spectrum(data.truth, axis)
    if truth is not None:
        curves.append(CurveView(truth[0], truth[1], "truth", "truth"))
    mask = pr.mask_w
    curves.append(CurveView(x[mask], pr.phi_w[mask], "phase", "phase", True))
    curves.append(CurveView(x[mask], pr.phi_w_fit[mask], "fit", "phase fit", True))
    truth_phase = _truth_spectral_phase(pr.wavelength, data.truth)
    if truth_phase is not None:
        curves.append(
            CurveView(x[mask], truth_phase[mask], "truth-phase", "truth phase", True)
        )
    edge = pr.edge_energy
    return LineView(
        curves=tuple(curves),
        title="Spectrum",
        xlabel=xlabel,
        ylabel="Power (a.u.)",
        right_label="Phase (rad)",
        xrange=_spectral_xlim(data.lam_min, data.lam_max, axis),
        yrange=(0.0, 1.05),
        right_range=_phase_ylim(
            pr.phi_w[mask], None if truth_phase is None else truth_phase[mask]
        ),
        notes=(
            (_dispersion_text(pr).replace("\n", "   "), False),
            (f"Edge: {100 * edge:.2f} %", edge > EDGE_ENERGY_WARN),
        ),
    )


def convergence_view(errors: ArrayLike, boundaries: Sequence[int] = ()) -> LineView:
    """Return the FROG error against iteration, with the stage handovers."""
    errors = np.asarray(errors, dtype=float)
    iterations = np.arange(1, errors.size + 1, dtype=float)
    return LineView(
        curves=(CurveView(iterations, errors, "error", "error"),),
        title="Convergence",
        xlabel="Iteration",
        ylabel="Error",
        log_y=True,
        marks=tuple(
            (float(edge) + 0.5, "boundary")
            for edge in boundaries
            if 0 < edge < errors.size
        ),
    )


def _marginal_view(
    x: NDArray[np.float64],
    retrieved: NDArray[np.float64],
    measured: NDArray[np.float64] | None,
    *,
    title: str,
    xlabel: str,
    xrange: tuple[float, float] | None,
) -> LineView:
    curves = [CurveView(x, retrieved, "retrieved", "R")]
    if measured is not None:
        curves.append(CurveView(x, measured, "measured", "M"))
    return LineView(
        curves=tuple(curves),
        title=title,
        xlabel=xlabel,
        ylabel="Marginal (a.u.)",
        xrange=xrange,
        yrange=(0.0, 1.05),
    )


def frequency_marginal_view(data: RetrievalPlotData) -> LineView:
    """Return the frequency marginals of the retrieved and measured traces."""
    pr = data.processed
    x = (data.result.omega + data.result.omega0) / _TWOPI_PHZ
    return _marginal_view(
        x,
        pr.omega_marg_retr,
        pr.omega_marg_meas,
        title="Frequency marginal",
        xlabel="Frequency (PHz)",
        xrange=None,
    )


def delay_marginal_view(data: RetrievalPlotData) -> LineView:
    """Return the delay marginals of the retrieved and measured traces."""
    pr = data.processed
    return _marginal_view(
        data.result.delays / 1e-15,
        pr.tau_marg_retr,
        pr.tau_marg_meas,
        title="Delay marginal",
        xlabel="Delay (fs)",
        xrange=_delay_window(data),
    )


def filter_view(data: RetrievalPlotData) -> LineView:
    """Return the per-frequency scale factor (Rω filter) against wavelength."""
    result = data.result
    # A centred grid can cross zero frequency, a pole of the wavelength axis.
    with np.errstate(divide="ignore", invalid="ignore"):
        lam_nm = wlfreq(result.omega + result.omega0) / 1e-9
    keep = np.isfinite(lam_nm)
    order = np.argsort(lam_nm[keep])
    return LineView(
        curves=(
            CurveView(
                lam_nm[keep][order], _filter_factor(result)[keep][order], "retrieved"
            ),
        ),
        title="Spectral filter",
        xlabel="Wavelength (nm)",
        ylabel="Filter factor",
        xrange=(data.lam_min / 1e-9, data.lam_max / 1e-9),
    )


SIGNAL_FRACTION = 1e-2
"""A wavelength row holds signal if its peak exceeds this fraction of the scale."""


def raw_trace_map(
    delay_fs: ArrayLike,
    wavelength_nm: ArrayLike,
    trace: ArrayLike,
    trust_nm: tuple[float, float],
    *,
    title: str = "Measured trace",
) -> MapView:
    """Return a loaded trace (wavelength × delay) as a signed linear map.

    The colour scale is set by the signal inside ``trust_nm`` alone, so noise far
    outside it (strongly amplified by a ``(λ/µm)^exp`` correction) cannot wash
    the image out, and the view is cropped to the trusted rows that hold signal.

    Parameters
    ----------
    delay_fs, wavelength_nm : array_like
        The trace's axes (any order, any spacing).
    trace : array_like, shape (len(wavelength_nm), len(delay_fs))
        The trace; negative after background subtraction is allowed.
    trust_nm : tuple of float
        The band whose signal sets the scale.
    title : str
        The view's title.
    """
    lam = np.asarray(wavelength_nm, dtype=float)
    trace = np.asarray(trace, dtype=float)
    in_band = (lam >= trust_nm[0]) & (lam <= trust_nm[1])
    band = trace[in_band] if np.any(in_band) else trace
    vmax = float(np.nanmax(np.abs(band))) or 1.0
    rows = lam[in_band & (trace.max(axis=1) > SIGNAL_FRACTION * vmax)]
    x, y, values = uniform_image(delay_fs, lam, trace)
    low = min(float(np.nanmin(values)), 0.0)
    return MapView(
        values=values,
        x=x,
        y=y,
        levels=(low, vmax),
        title=title,
        xlabel="Delay (fs)",
        ylabel="Wavelength (nm)",
        unit="a.u.",
        signed=True,
        yrange=(float(rows.min()), float(rows.max())) if rows.size else None,
    )
