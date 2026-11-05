"""Matplotlib plotting for retrieval results.

Provides custom colormaps (white at the low end; a red band for negatives),
reusable **single-axis** plotters (so the GUI embeds exactly the same code), and
**composite** figures: the 12-panel :func:`plot_retrieval`, the 6-panel
:func:`plot_frog_filter`, and :func:`plot_simulated_trace`.

The retrieval summary is assembled from twelve independent **panels**
(``draw_<key>(ax, data)`` over one :class:`RetrievalPlotData` bundle, registered in
:data:`RETRIEVAL_PANELS`). :func:`plot_retrieval` lays them out as the 3×4
overview; :func:`plot_retrieval_page` draws one of the three 2×2 pages
(:data:`RETRIEVAL_PAGES`: Traces, Pulse, Diagnostics) the GUI shows as tabs.

Figures are built with :external+matplotlib:class:`matplotlib.figure.Figure` directly
(no ``pyplot`` global state), so they embed cleanly in a Qt canvas and also save
standalone via ``fig.savefig(...)``.

Every public name is importable from ``croak.plotting`` itself; the submodules
group them by domain:

``scale``
    On-screen text density (:class:`PlotScale`, :func:`plot_scale`,
    :func:`scaled_rc`).
``units``
    Display units and read-out formatting (:func:`sig3`, :func:`si_power`).
``traces``
    FROG-trace images: colormaps, :func:`signed_pcolormesh`, the residual and
    marginal plotters, :func:`plot_frog_filter`, :class:`FrogFilterView` and
    :func:`plot_simulated_trace`.
``pulse``
    The retrieved pulse: :func:`plot_temporal`, :func:`plot_spectral` and
    :func:`plot_spectrogram`.
``artists``
    The artist bundles each panel returns (:class:`TraceArtists`, …).
``panels``
    :class:`RetrievalPlotData` and the twelve ``draw_<key>`` panels.
``updates``
    The ``update_<key>`` in-place twins of the panels, for the live preview.
``retrieval``
    The panel registry, pages and overview, :func:`plot_retrieval_page`,
    :func:`plot_retrieval` and :class:`RetrievalPageView`.
``uncertainty``
    Bootstrap and substrate-thickness uncertainty figures.
``figure``
    The figure set-up the composites share.
"""

from .artists import (
    ConvergenceArtists,
    LineArtists,
    MarginalArtists,
    PanelArtists,
    ResidualArtists,
    SpectralArtists,
    SpectrogramArtists,
    TemporalArtists,
    TraceArtists,
)
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
    plot_convergence,
    retrieval_plot_data,
)
from .pulse import (
    EDGE_ENERGY_WARN,
    SPECTRAL_AXES,
    SpectralAxis,
    plot_spectral,
    plot_spectrogram,
    plot_temporal,
)
from .retrieval import (
    RETRIEVAL_COLORBARS,
    RETRIEVAL_OVERVIEW,
    RETRIEVAL_PAGES,
    RETRIEVAL_PANELS,
    ColorbarSpec,
    PageSpec,
    PanelDraw,
    PanelSpec,
    PanelUpdate,
    RetrievalPageView,
    plot_retrieval,
    plot_retrieval_page,
)
from .scale import (
    COMPACT_BELOW,
    FONT_SCALE_MIN,
    FONT_SCALE_STEP,
    FULL_SCALE,
    PlotScale,
    plot_scale,
    scaled_rc,
)
from .traces import (
    FrogFilterView,
    cmap_negwhite,
    cmap_white,
    filter_view_vmax,
    plot_frog_filter,
    plot_marginal,
    plot_residual,
    plot_simulated_trace,
    signed_pcolormesh,
)
from .uncertainty import (
    plot_error_vs_thickness,
    plot_fwhm_distribution,
    plot_fwhm_vs_thickness,
    plot_intensity_band,
    plot_thickness_sensitivity,
    plot_uncertainty,
)
from .units import (
    si_power,
    sig3,
)
from .updates import (
    update_convergence,
    update_convergence_curve,
    update_delay_marginal,
    update_freq_marginal,
    update_measured_lin,
    update_measured_log,
    update_residual,
    update_retrieved_lin,
    update_retrieved_log,
    update_spectral,
    update_spectral_filter,
    update_spectrogram,
    update_temporal,
)

__all__ = [
    "PlotScale",
    "FULL_SCALE",
    "FONT_SCALE_MIN",
    "FONT_SCALE_STEP",
    "COMPACT_BELOW",
    "plot_scale",
    "scaled_rc",
    "cmap_white",
    "cmap_negwhite",
    "signed_pcolormesh",
    "si_power",
    "sig3",
    "plot_residual",
    "plot_temporal",
    "plot_spectral",
    "SPECTRAL_AXES",
    "EDGE_ENERGY_WARN",
    "plot_convergence",
    "plot_marginal",
    "plot_spectrogram",
    "plot_retrieval",
    "RetrievalPlotData",
    "retrieval_plot_data",
    "TraceArtists",
    "ResidualArtists",
    "TemporalArtists",
    "SpectralArtists",
    "ConvergenceArtists",
    "LineArtists",
    "MarginalArtists",
    "SpectrogramArtists",
    "PanelSpec",
    "PageSpec",
    "ColorbarSpec",
    "RETRIEVAL_PANELS",
    "RETRIEVAL_COLORBARS",
    "RETRIEVAL_PAGES",
    "RETRIEVAL_OVERVIEW",
    "plot_retrieval_page",
    "draw_measured_log",
    "draw_retrieved_log",
    "draw_measured_lin",
    "draw_retrieved_lin",
    "draw_temporal",
    "draw_spectral",
    "draw_convergence",
    "draw_residual",
    "draw_spectral_filter",
    "draw_freq_marginal",
    "draw_delay_marginal",
    "draw_spectrogram",
    "update_measured_log",
    "update_retrieved_log",
    "update_measured_lin",
    "update_retrieved_lin",
    "update_temporal",
    "update_spectral",
    "update_convergence",
    "update_convergence_curve",
    "update_residual",
    "update_spectral_filter",
    "update_freq_marginal",
    "update_delay_marginal",
    "update_spectrogram",
    "RetrievalPageView",
    "plot_frog_filter",
    "filter_view_vmax",
    "FrogFilterView",
    "plot_simulated_trace",
    "plot_fwhm_distribution",
    "plot_intensity_band",
    "plot_uncertainty",
]
