r"""Show a screen view on the spectral axis and logarithmic range the viewer chose.

The views of :mod:`croak.plotting.screen` are built on the axis their data were
computed on: a loaded trace against wavelength, a retrieval against frequency.
:func:`displayed` moves any of them to the axis a :class:`Display` asks for and
sets every logarithmic map's range, so that one setting holds for every plot.

Moving between wavelength λ (nm) and frequency ν (PHz) uses ν = c/λ, with
c = 299.792458 nm·PHz, on positions, ranges, shaded intervals and rules. An
intensity is a density per unit of its axis, so it also takes the Jacobian:
from :math:`I_\lambda\,d\lambda = I_\nu\,d\nu`,

.. math:: I_\nu = I_\lambda \left|\frac{d\lambda}{d\nu}\right|
          = I_\lambda \frac{\lambda^2}{c},
   \qquad I_\lambda = I_\nu \frac{\nu^2}{c}.

Either way the factor is the square of the *old* coordinate, up to the constant,
so a broad spectrum changes shape, not only position. Every map, and every curve
marked as a density (:attr:`~croak.plotting.screen.CurveView.density`), is
converted so; a phase or a filter factor only moves. The views are normalised
displays, so after the Jacobian each map and each density curve is rescaled to
its old peak (in dB: shifted to its old peak), and the colour scale still
reads the same. A map is resampled onto an even grid on the new axis
(:func:`~croak.plotting.screen.uniform_image`); that is for drawing only.

Examples
--------
>>> import numpy as np
>>> from croak.plotting.screen import WAVELENGTH_LABEL, CurveView, LineView
>>> flat = CurveView(np.array([500.0, 1000.0]), np.array([1.0, 1.0]), "measured")
>>> view = LineView(
...     curves=(flat,),
...     title="Spectrum",
...     xlabel=WAVELENGTH_LABEL,
...     ylabel="Intensity",
... )
>>> shown = displayed(view, Display(axis="frequency"))
>>> shown.xlabel, shown.curves[0].x.round(4).tolist(), shown.curves[0].y.tolist()
('Frequency (PHz)', [0.2998, 0.5996], [1.0, 0.25])
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import overload

import numpy as np
from numpy.typing import NDArray

from .screen import (
    DB_DEPTH,
    DB_UNIT,
    FREQUENCY_LABEL,
    WAVELENGTH_LABEL,
    CurveView,
    LineView,
    MapView,
    SpanView,
    uniform_image,
)

__all__ = [
    "DB_RANGE",
    "DB_RANGE_LIMITS",
    "DISPLAY_AXES",
    "SPEED_OF_LIGHT_NM_PHZ",
    "Display",
    "displayed",
]

SPEED_OF_LIGHT_NM_PHZ = 299.792458
"""The speed of light in nm·PHz, so that ν (PHz) = c / λ (nm)."""

DISPLAY_AXES: tuple[str, ...] = ("native", "wavelength", "frequency")
"""The spectral axes a display can ask for; ``"native"`` keeps each view's own."""

DB_RANGE = 30.0
"""The default range of a logarithmic map below its peak, dB."""

DB_RANGE_LIMITS = (5.0, DB_DEPTH)
"""The narrowest and widest logarithmic range, dB (the views keep values down to
:data:`~croak.plotting.screen.DB_DEPTH` below the peak)."""

_LABELS = {"wavelength": WAVELENGTH_LABEL, "frequency": FREQUENCY_LABEL}
_SPECTRAL = frozenset(_LABELS.values())


@dataclass(frozen=True)
class Display:
    """How the views are shown: their spectral axis and logarithmic range.

    Attributes
    ----------
    axis : {"native", "wavelength", "frequency"}
        The spectral axis of every view; ``"native"`` keeps each on its own.
    db_range : float
        Every logarithmic map spans this many dB below its peak, within
        :data:`DB_RANGE_LIMITS`.

    Raises
    ------
    ValueError
        For an unknown axis or a range outside the limits.
    """

    axis: str = "native"
    db_range: float = DB_RANGE

    def __post_init__(self) -> None:
        """Check the axis and the range."""
        if self.axis not in DISPLAY_AXES:
            raise ValueError(f"axis must be one of {DISPLAY_AXES}, got {self.axis!r}")
        low, high = DB_RANGE_LIMITS
        if not low <= self.db_range <= high:
            raise ValueError(
                f"db_range must be within {low:g}–{high:g} dB, got {self.db_range!r}"
            )


@overload
def displayed(view: MapView, display: Display) -> MapView: ...
@overload
def displayed(view: LineView, display: Display) -> LineView: ...
def displayed(view: MapView | LineView, display: Display) -> MapView | LineView:
    """Return ``view`` as ``display`` shows it (the view itself if unchanged).

    Parameters
    ----------
    view : MapView or LineView
        A screen view; its spectral axis is the one labelled
        :data:`~croak.plotting.screen.WAVELENGTH_LABEL` or
        :data:`~croak.plotting.screen.FREQUENCY_LABEL`.
    display : Display
        The axis and the logarithmic range.

    Returns
    -------
    MapView or LineView
        The view on the display's axis, with its logarithmic range.
    """
    if isinstance(view, MapView):
        return _displayed_map(view, display)
    target = _LABELS.get(display.axis)
    if target is None or view.xlabel not in _SPECTRAL or view.xlabel == target:
        return view
    return _moved_line(view, target)


def _displayed_map(view: MapView, display: Display) -> MapView:
    target = _LABELS.get(display.axis)
    if target is not None and view.ylabel in _SPECTRAL and view.ylabel != target:
        view = _moved_map(view, target, along_y=True)
    elif target is not None and view.xlabel in _SPECTRAL and view.xlabel != target:
        view = _moved_map(view, target, along_y=False)
    if view.unit == DB_UNIT:
        view = replace(view, levels=(-display.db_range, 0.0))
    return view


def _moved(axis: NDArray[np.float64]) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """Pick the samples of ``axis`` that can move (positive), and move them.

    Returns their indices, in the new axis's (increasing) order, and their new
    positions.
    """
    keep = np.flatnonzero(np.isfinite(axis) & (axis > 0))
    # ν = c/λ reverses the order: reading the old samples backwards keeps the new
    # positions increasing, as the views need.
    keep = keep[np.argsort(-axis[keep], kind="stable")]
    return keep, SPEED_OF_LIGHT_NM_PHZ / axis[keep]


def _range(window: tuple[float, float] | None) -> tuple[float, float] | None:
    """Move a window to the new axis (``None`` if it reaches zero or below)."""
    if window is None or min(window) <= 0:
        return None
    low, high = sorted(SPEED_OF_LIGHT_NM_PHZ / np.asarray(window, dtype=float))
    return float(low), float(high)


def _moved_map(view: MapView, target: str, *, along_y: bool) -> MapView:
    values = view.values if along_y else view.values.T
    old = view.y if along_y else view.x
    keep, new = _moved(old)
    values = values[keep]
    jacobian = old[keep] ** 2
    if view.unit == DB_UNIT:
        peak = float(np.nanmax(values))
        values = values + 10 * np.log10(jacobian / jacobian.max())[:, None]
        values = np.clip(values - (float(np.nanmax(values)) - peak), -DB_DEPTH, 0.0)
    else:
        peak = float(np.nanmax(np.abs(values)))
        values = values * jacobian[:, None]
        scale = float(np.nanmax(np.abs(values)))
        values = values * (peak / scale if scale > 0 else 1.0)
    if along_y:
        x, y, image = uniform_image(view.x, new, values)
        return replace(
            view, values=image, x=x, y=y, ylabel=target, yrange=_range(view.yrange)
        )
    y, x, image = uniform_image(view.y, new, values)
    return replace(
        view, values=image.T, x=x, y=y, xlabel=target, xrange=_range(view.xrange)
    )


def _moved_curve(curve: CurveView) -> CurveView:
    keep, x = _moved(curve.x)
    y = curve.y[keep]
    if curve.density:
        finite = np.isfinite(curve.y)
        peak = float(np.max(np.abs(curve.y[finite]))) if finite.any() else 0.0
        y = y * curve.x[keep] ** 2
        new_finite = np.isfinite(y)
        scale = float(np.max(np.abs(y[new_finite]))) if new_finite.any() else 0.0
        y = y * (peak / scale if scale > 0 else 1.0)
    return replace(curve, x=x, y=y)


def _moved_span(span: SpanView) -> SpanView:
    if not span.vertical:
        return span
    window = _range((span.low, span.high))
    if window is None:
        return span
    return replace(span, low=window[0], high=window[1])


def _moved_line(view: LineView, target: str) -> LineView:
    bands = []
    for band in view.bands:
        keep, x = _moved(band.x)
        bands.append(replace(band, x=x, low=band.low[keep], high=band.high[keep]))
    return replace(
        view,
        curves=tuple(_moved_curve(curve) for curve in view.curves),
        xlabel=target,
        xrange=_range(view.xrange),
        marks=tuple(
            (SPEED_OF_LIGHT_NM_PHZ / x, role) for x, role in view.marks if x > 0
        ),
        bands=tuple(bands),
        spans=tuple(_moved_span(span) for span in view.spans),
    )
