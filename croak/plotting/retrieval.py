"""The retrieval summary: panel registry, pages, the overview and the page view.

The twelve panels of :mod:`croak.plotting.panels` are registered in
:data:`RETRIEVAL_PANELS`; :func:`plot_retrieval` lays them out as the 3×4
overview and :func:`plot_retrieval_page` draws one of the three 2×2 pages
(:data:`RETRIEVAL_PAGES`) the GUI shows as tabs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from ..processing import ProcessedResult, TruthPulse
from ..result import RetrievalResult
from .artists import PanelArtists, ResidualArtists, TraceArtists
from .figure import _prepare_fig
from .panels import (
    RetrievalPlotData,
    draw_convergence,
    draw_delay_marginal,
    draw_freq_marginal,
    draw_measured_lin,
    draw_measured_log,
    draw_residual,
    draw_retrieved_lin,
    draw_retrieved_log,
    draw_spectral,
    draw_spectral_filter,
    draw_spectrogram,
    draw_temporal,
    retrieval_plot_data,
)
from .pulse import SpectralAxis

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh


type PanelDraw = Callable[[Axes, RetrievalPlotData], PanelArtists]


@dataclass(frozen=True)
class PanelSpec:
    """One retrieval panel: registry key, axes title and how to draw it."""

    key: str
    title: str
    draw: PanelDraw


@dataclass(frozen=True)
class PageSpec:
    """A figure's worth of panels laid out as a grid of registry keys.

    Attributes
    ----------
    key, title : str
        Identifier and human-readable name (the GUI's tab label).
    mosaic : tuple of tuple of str
        Rows of panel keys, as
        :external+matplotlib:meth:`~matplotlib.figure.Figure.subplot_mosaic`
        takes them.
    """

    key: str
    title: str
    mosaic: tuple[tuple[str, ...], ...]

    @property
    def keys(self) -> tuple[str, ...]:
        """The page's panel keys in reading order."""
        return tuple(key for row in self.mosaic for key in row)


@dataclass(frozen=True)
class ColorbarSpec:
    """A colorbar shared by the listed panels (drawn only when all are present).

    A positive colorbar is always drawn; a second, ``neg_label``-ed one appears
    when any of the panels holds negative samples (see :func:`signed_pcolormesh`).
    """

    panels: tuple[str, ...]
    shrink: float = 0.6
    neg_label: str = "neg |·|"


#: The twelve retrieval panels, keyed, in the overview's reading order.
RETRIEVAL_PANELS: Mapping[str, PanelSpec] = {
    spec.key: spec
    for spec in (
        PanelSpec("measured_log", "Measured (log)", draw_measured_log),
        PanelSpec("retrieved_log", "Retrieved (log)", draw_retrieved_log),
        PanelSpec("temporal", "Retrieved pulse", draw_temporal),
        PanelSpec("delay_marginal", "Delay marginal", draw_delay_marginal),
        PanelSpec("measured_lin", "Measured (lin)", draw_measured_lin),
        PanelSpec("retrieved_lin", "Retrieved (lin)", draw_retrieved_lin),
        PanelSpec("spectral", "Spectrum", draw_spectral),
        PanelSpec("freq_marginal", "Frequency marginal", draw_freq_marginal),
        PanelSpec("convergence", "Convergence", draw_convergence),
        PanelSpec("residual", "Residuals", draw_residual),
        PanelSpec("spectral_filter", "Spectral filter", draw_spectral_filter),
        PanelSpec("spectrogram", "Spectrogram", draw_spectrogram),
    )
}

#: Colorbars: one per trace pair (they share a scale) and one for the residual.
RETRIEVAL_COLORBARS: tuple[ColorbarSpec, ...] = (
    ColorbarSpec(("measured_log", "retrieved_log")),
    ColorbarSpec(("measured_lin", "retrieved_lin")),
    ColorbarSpec(("residual",)),
)

#: The three 2×2 pages the GUI shows as tabs.
RETRIEVAL_PAGES: Mapping[str, PageSpec] = {
    page.key: page
    for page in (
        PageSpec(
            "traces",
            "Traces",
            (("measured_log", "retrieved_log"), ("measured_lin", "retrieved_lin")),
        ),
        # Each marginal sits under the domain it belongs to: delay under the
        # time-domain pulse, frequency under the spectrum.
        PageSpec(
            "pulse",
            "Pulse",
            (("temporal", "spectral"), ("delay_marginal", "freq_marginal")),
        ),
        PageSpec(
            "diagnostics",
            "Diagnostics",
            (("convergence", "residual"), ("spectral_filter", "spectrogram")),
        ),
    )
}

#: The 3×4 overview :func:`plot_retrieval` draws: traces left, pulse right (the
#: delay marginal beside the time-domain pulse, the frequency marginal beside the
#: spectrum), diagnostics along the bottom.
RETRIEVAL_OVERVIEW = PageSpec(
    "overview",
    "Retrieval overview",
    (
        ("measured_log", "retrieved_log", "temporal", "delay_marginal"),
        ("measured_lin", "retrieved_lin", "spectral", "freq_marginal"),
        ("convergence", "residual", "spectral_filter", "spectrogram"),
    ),
)


def _mappables(artists: PanelArtists) -> tuple[QuadMesh | None, QuadMesh | None]:
    """Return the ``(positive, negative)`` images a colorbar can attach to."""
    if isinstance(artists, TraceArtists):
        return artists.mesh_pos, artists.mesh_neg
    if isinstance(artists, ResidualArtists):
        return artists.mesh, None
    return None, None


def _draw_colorbars(
    fig, axd: Mapping[str, Axes], artists: Mapping[str, PanelArtists]
) -> None:
    """Add every :data:`RETRIEVAL_COLORBARS` entry whose panels are all in ``axd``."""
    for spec in RETRIEVAL_COLORBARS:
        if not all(key in axd for key in spec.panels):
            continue
        axes = [axd[key] for key in spec.panels]
        pairs = [_mappables(artists[key]) for key in spec.panels]
        pos = next((p for p, _ in pairs if p is not None), None)
        if pos is None:
            continue
        # Both panels of a pair share vmin/vmax and cmap, so either can seed the bar.
        fig.colorbar(pos, ax=axes, shrink=spec.shrink)
        neg = next((n for _, n in pairs if n is not None), None)
        if neg is not None:
            fig.colorbar(neg, ax=axes, shrink=spec.shrink, label=spec.neg_label)


def _draw_panels(
    fig, page: PageSpec, data: RetrievalPlotData
) -> tuple[dict[str, Axes], dict[str, PanelArtists]]:
    """Lay ``page``'s mosaic out in ``fig``, draw every panel and its colourbars."""
    axd = fig.subplot_mosaic([list(row) for row in page.mosaic])
    artists = {key: RETRIEVAL_PANELS[key].draw(ax, data) for key, ax in axd.items()}
    _draw_colorbars(fig, axd, artists)
    return axd, artists


def plot_retrieval_page(
    data: RetrievalPlotData,
    page: PageSpec,
    *,
    figsize: tuple[float, float] = (8.5, 6.3),
    fig: Figure | None = None,
) -> Figure:
    """Draw one page of retrieval panels (a :data:`RETRIEVAL_PAGES` entry).

    The default ``figsize`` gives a 2×2 page the same per-panel size as the
    overview's default 17.0×9.45 in.

    Parameters
    ----------
    data : RetrievalPlotData
        From :func:`retrieval_plot_data`.
    page : PageSpec
        Which panels, in which grid; :data:`RETRIEVAL_OVERVIEW` draws all twelve.
    figsize : tuple of float
        Size (in) of a new figure; ignored when ``fig`` is given.
    fig : Figure, optional
        Existing figure to clear and draw into (a GUI canvas).

    Examples
    --------
    >>> data = retrieval_plot_data(result, measured=trace)  # doctest: +SKIP
    >>> fig = plot_retrieval_page(data, RETRIEVAL_PAGES["pulse"])  # doctest: +SKIP
    """
    fig = _prepare_fig(fig, figsize)
    _draw_panels(fig, page, data)
    return fig


def plot_retrieval(
    result: RetrievalResult,
    *,
    measured: ArrayLike | None = None,
    Iomega_meas: ArrayLike | None = None,
    lam_min: float | None = None,
    lam_max: float | None = None,
    flim: tuple[float, float] | None = None,
    tracedb: float = 30.0,
    cmap: str = "viridis",
    truth: TruthPulse | None = None,
    energy: float | None = None,
    figsize=(17.0, 9.45),
    fig: Figure | None = None,
    processed: ProcessedResult | None = None,
    spectral_axis: SpectralAxis = "frequency",
) -> Figure:
    """12-panel retrieval summary: the standard at-a-glance quality check.

    The panels are :data:`RETRIEVAL_PANELS` in the :data:`RETRIEVAL_OVERVIEW`
    layout — measured/retrieved traces (log and linear) on the left, the pulse,
    spectrum and marginals on the right, convergence, residual, spectral filter
    and spectrogram along the bottom. :func:`plot_retrieval_page` draws a subset.

    ``lam_min``/``lam_max`` (m) bound the **wavelength** axes — the spectrum and
    spectrogram panels. ``flim`` bounds the **frequency** axis of the trace and
    residual panels, as ``(f_min, f_max)`` in Hz; the default shows the whole
    retrieval grid. The two are separate because the trace sits at the *signal*
    frequency, which for SHG is twice the pulse's. Reach for ``flim`` when the
    retrieval grid is much wider than the signal — typical when retrieving
    directly on a synthesis grid rather than on one cropped by
    :func:`~croak.preprocess.load_and_clean` — where the trace would otherwise be
    a thin stripe in a mostly empty axis.

    ``energy`` is an optional measured pulse energy (J); when given, the temporal
    panel is shown in absolute power (see :func:`~croak.processing.process_result`).

    ``processed`` is an optional, already-computed
    :class:`~croak.processing.ProcessedResult` **for this same result**: pass it when
    the caller needs the processed quantities too (the GUI shows them in a table
    beside the figure) so they are not derived twice. It must have been built with
    the same ``measured``/``Iomega_meas``/``energy``, which are then used only for
    the trace panels; a mismatched one raises :class:`ValueError`.

    ``spectral_axis`` draws the spectrum panel against ``"frequency"`` (PHz, the
    default) or ``"wavelength"`` (nm); see :func:`plot_spectral`.

    Raises
    ------
    ValueError
        If ``result`` carries no trace, or ``processed`` belongs to another result.
    """
    data = retrieval_plot_data(
        result,
        measured=measured,
        Iomega_meas=Iomega_meas,
        lam_min=lam_min,
        lam_max=lam_max,
        flim=flim,
        tracedb=tracedb,
        cmap=cmap,
        truth=truth,
        energy=energy,
        processed=processed,
        spectral_axis=spectral_axis,
    )
    return plot_retrieval_page(data, RETRIEVAL_OVERVIEW, figsize=figsize, fig=fig)
