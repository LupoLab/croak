"""Figure lifecycle shared by the composite plotters."""

from __future__ import annotations

import contextlib

from matplotlib.figure import Figure


def _prepare_fig(fig, figsize):
    """Return a cleared figure to draw a composite into (new if None)."""
    if fig is None:
        return Figure(figsize=figsize, layout="constrained")
    fig.clear()
    with contextlib.suppress(Exception):
        fig.set_layout_engine("constrained")
    return fig
