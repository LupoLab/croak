"""On-screen text density of a figure squeezed below its design size."""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# On-screen density
# ---------------------------------------------------------------------------
# Matplotlib sizes text in points, so when a figure is squeezed into a small window
# the axes shrink and the text does not: on a 1366x768 laptop the twelve-panel
# retrieval summary spent 78 % of its area on titles, labels and padding. The GUI
# canvas (croak.gui.canvas.MplCanvas) therefore scales the rc font sizes by how far
# the canvas has fallen below the size its figure was designed for. The scale is a
# pure function of geometry, kept here so the plotting layer stays Qt-free and the
# thresholds are testable headlessly.

#: Smallest fraction of the design font size that text may shrink to. Below ~70 %
#: a 10 pt label is 7 pt, the limit of comfortable legibility on a laptop panel.
FONT_SCALE_MIN = 0.7
#: Quantisation step of :func:`plot_scale`. A window resize re-plots only when the
#: scale crosses a step, so a few pixels of drag never trigger a redraw.
FONT_SCALE_STEP = 0.05
#: Below this scale :attr:`PlotScale.compact` is set, for plotting code that wants
#: to drop optional decoration (a second legend entry, a colorbar label) when tight.
COMPACT_BELOW = 0.9


@dataclass(frozen=True)
class PlotScale:
    """Text density of a figure relative to the size its layout was designed for.

    Parameters
    ----------
    font : float
        Multiplier applied to every rc font size (and text padding), in
        ``[FONT_SCALE_MIN, 1.0]``. ``1.0`` is matplotlib's default 10 pt base.

    Examples
    --------
    >>> PlotScale(0.85).compact
    True
    >>> FULL_SCALE.compact
    False
    """

    font: float = 1.0

    @property
    def compact(self) -> bool:
        """Whether the figure is tight enough to drop optional decoration."""
        return self.font < COMPACT_BELOW


#: The unscaled density: matplotlib's defaults.
FULL_SCALE = PlotScale()


def plot_scale(
    width_in: float, height_in: float, design_in: tuple[float, float]
) -> PlotScale:
    """Return the text density for a figure of the given size.

    The scale is the fraction of the design size that fits in the *tighter*
    direction, ``min(width / design_width, height / design_height)``, quantised
    to :data:`FONT_SCALE_STEP` and clamped to ``[FONT_SCALE_MIN, 1.0]``. A figure
    at or above its design size is drawn at full scale; one squeezed to 60 % of
    its design height gets the minimum.

    Parameters
    ----------
    width_in, height_in : float
        Current figure size in inches (``Figure.get_size_inches()``). On a Qt
        canvas this is the logical pixel size / 100 on every display, because
        matplotlib folds the device-pixel ratio into ``figure.dpi``.
    design_in : tuple of float
        ``(width, height)`` in inches the figure's layout was designed for — the
        ``figsize`` a stage passes to its canvas.

    Returns
    -------
    PlotScale

    Raises
    ------
    ValueError
        If any size is not positive.

    Examples
    --------
    >>> plot_scale(9.91, 4.21, (12.0, 7.0)).font   # Retrieve page at 1366x768
    0.7
    >>> plot_scale(15.45, 7.33, (12.0, 7.0)).font  # ... at 1920x1080
    1.0
    """
    design_w, design_h = design_in
    if min(width_in, height_in, design_w, design_h) <= 0:
        raise ValueError(
            f"figure and design sizes must be positive, got {width_in}x{height_in} "
            f"in for a {design_w}x{design_h} in design"
        )
    ratio = min(width_in / design_w, height_in / design_h)
    # Quantise to the nearest step; round to two decimals so equal steps compare
    # equal as floats (0.85, not 0.8500000000000001).
    quantised = round(round(ratio / FONT_SCALE_STEP) * FONT_SCALE_STEP, 2)
    return PlotScale(font=min(1.0, max(FONT_SCALE_MIN, quantised)))


# Matplotlib's default sizes spelled out in points. The rc defaults for the title
# and label keys are the *relative* strings 'large'/'medium', which cannot be
# multiplied; writing the resolved points also makes the scaling immune to a user
# matplotlibrc with absolute sizes. The four pads are included because text
# *padding* is a fixed-pixel cost too.
_SCALED_RC_BASE: dict[str, float] = {
    "font.size": 10.0,
    "axes.titlesize": 12.0,
    "axes.labelsize": 10.0,
    "xtick.labelsize": 10.0,
    "ytick.labelsize": 10.0,
    "legend.fontsize": 10.0,
    "figure.titlesize": 12.0,
    "axes.labelpad": 4.0,
    "axes.titlepad": 6.0,
    "xtick.major.pad": 3.5,
    "ytick.major.pad": 3.5,
}


def scaled_rc(scale: PlotScale) -> dict[str, float]:
    """Return rcParams overrides that shrink every font size and pad by ``scale``.

    Use with :external+matplotlib:func:`matplotlib.rc_context` *around the plotting
    call*: matplotlib resolves relative sizes (``fontsize="small"``, the ``'large'``
    title default) against ``font.size`` when an artist is created, not when it is
    drawn, so the context must enclose everything that creates text.

    Parameters
    ----------
    scale : PlotScale
        The density; :data:`FULL_SCALE` reproduces matplotlib's defaults.

    Returns
    -------
    dict
        ``{rc key: value in points}`` for the font-size and padding keys.

    Examples
    --------
    >>> round(scaled_rc(PlotScale(0.7))["axes.titlesize"], 2)
    8.4
    """
    return {key: base * scale.font for key, base in _SCALED_RC_BASE.items()}
