"""In-place updates of the retrieval panels for the GUI's live preview."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from .artists import (
    ConvergenceArtists,
    LineArtists,
    MarginalArtists,
    ResidualArtists,
    SpectralArtists,
    SpectrogramArtists,
    TemporalArtists,
    TraceArtists,
)
from .panels import RetrievalPlotData, _filter_factor, _trace_label
from .pulse import (
    _dispersion_text,
    _edge_text,
    _intensity_ylim,
    _phase_ylim,
    _spectral_curves,
    _temporal_curves,
    _truth_intensity,
    _truth_spectral_phase,
)
from .traces import _signed_magnitude

if TYPE_CHECKING:
    from matplotlib.axes import Axes


# ---------------------------------------------------------------------------
# In-place updates
# ---------------------------------------------------------------------------
# Each update_<key>(artists, data) pushes a new result onto the artists its
# draw_<key> twin created: set_data / set_array / set_text and explicit limits,
# never new artists. The invariant is update(draw(d1), d2) ≡ draw(d2), which the
# tests check panel by panel. Whether an update is possible at all — same grid
# shapes, same optional artists — is decided by _update_signature.


def _axes_of(artist) -> Axes:
    """Return the axes an artist is drawn on (an updated artist always has one)."""
    ax = artist.axes
    if ax is None:
        raise RuntimeError("the artist is not on any axes")
    return ax


def _update_trace(artists: TraceArtists, data: RetrievalPlotData, C, *, db: bool):
    mag, neg, vmin, vhi = _signed_magnitude(
        C, db=db, tracedb=data.tracedb, vmax=None if db else data.lin_vmax
    )
    artists.mesh_pos.set_array(np.ma.masked_where(neg, mag))
    artists.mesh_pos.set_clim(vmin, vhi)  # a shared colorbar follows the norm
    if artists.mesh_neg is not None:
        artists.mesh_neg.set_array(np.ma.masked_where(~neg, mag))
        artists.mesh_neg.set_clim(vmin, vhi)


def update_measured_log(artists: TraceArtists, data: RetrievalPlotData) -> None:
    """Update :func:`draw_measured_log`'s artists in place."""
    _update_trace(artists, data, data.measured, db=True)


def update_retrieved_log(artists: TraceArtists, data: RetrievalPlotData) -> None:
    """Update :func:`draw_retrieved_log`'s artists in place."""
    _update_trace(artists, data, data.retrieved, db=True)


def update_measured_lin(artists: TraceArtists, data: RetrievalPlotData) -> None:
    """Update :func:`draw_measured_lin`'s artists in place."""
    _update_trace(artists, data, data.measured, db=False)


def update_retrieved_lin(artists: TraceArtists, data: RetrievalPlotData) -> None:
    """Update :func:`draw_retrieved_lin`'s artists in place."""
    _update_trace(artists, data, data.retrieved, db=False)
    if artists.label is not None:
        artists.label.set_text(_trace_label(data))


def update_temporal(artists: TemporalArtists, data: RetrievalPlotData) -> None:
    """Update :func:`draw_temporal`'s artists in place."""
    pr = data.processed
    t = pr.t_over / 1e-15
    retr, tl, r_label, tl_label, ylabel, power_scale = _temporal_curves(
        pr, data.power_scale
    )
    artists.line_retr.set_data(t, retr)
    artists.line_tl.set_data(pr.t_tl / 1e-15, tl)
    artists.legend_retr.set_text(r_label)
    artists.legend_tl.set_text(tl_label)
    ax = _axes_of(artists.line_retr)
    ax.set_ylabel(ylabel)
    truth_curve = None
    if artists.line_truth is not None and data.truth is not None:
        truth_curve = _truth_intensity(data.truth, pr.energy, power_scale)
        artists.line_truth.set_ydata(truth_curve)
    ax.set_ylim(*_intensity_ylim(retr, tl, truth_curve))
    artists.line_phase.set_data(t[pr.mask_t], pr.phi_t[pr.mask_t])
    truth_phase = None  # static within a run: the truth does not change
    if artists.line_truth_phase is not None:
        truth_phase = np.asarray(artists.line_truth_phase.get_ydata(), dtype=float)
    limits = _phase_ylim(pr.phi_t[pr.mask_t], truth_phase)
    if limits is not None:
        artists.ax_phase.set_ylim(*limits)


def update_spectral(artists: SpectralArtists, data: RetrievalPlotData) -> None:
    """Update :func:`draw_spectral`'s artists in place."""
    pr = data.processed
    x, retrieved, measured, _xlabel = _spectral_curves(pr, data.spectral_axis)
    artists.line_retr.set_data(x, retrieved)
    if artists.line_meas is not None and measured is not None:
        artists.line_meas.set_data(x, measured)
    artists.line_phase.set_data(x[pr.mask_w], pr.phi_w[pr.mask_w])
    artists.line_phase_fit.set_data(x[pr.mask_w], pr.phi_w_fit[pr.mask_w])
    # The truth phase is static but its mask (where the spectrum is bright) moves.
    tphi = _truth_spectral_phase(pr.wavelength, data.truth)
    if artists.line_truth_phase is not None and tphi is not None:
        artists.line_truth_phase.set_data(x[pr.mask_w], tphi[pr.mask_w])
    limits = _phase_ylim(pr.phi_w[pr.mask_w], None if tphi is None else tphi[pr.mask_w])
    if limits is not None:
        artists.ax_phase.set_ylim(*limits)
    artists.text_dispersion.set_text(_dispersion_text(pr))
    edge, edge_color = _edge_text(pr)
    artists.text_edge.set_text(edge)
    artists.text_edge.set_color(edge_color)


def update_convergence_curve(artists: ConvergenceArtists, errors: ArrayLike) -> None:
    """Grow the error curve of :func:`plot_convergence` in place and rescale."""
    errors = np.asarray(errors, dtype=float)
    artists.line.set_data(np.arange(1, errors.size + 1), errors)
    ax = _axes_of(artists.line)
    ax.relim()  # autoscaling is still on here: plot_convergence sets no limits
    ax.autoscale_view()


def update_convergence(artists: ConvergenceArtists, data: RetrievalPlotData) -> None:
    """Update :func:`draw_convergence`'s artists in place."""
    update_convergence_curve(artists, data.processed.errors)


def update_residual(artists: ResidualArtists, data: RetrievalPlotData) -> None:
    """Update :func:`draw_residual`'s artists in place."""
    resid = data.processed.residual
    if artists.mesh is None or resid is None:
        return
    mr = float(np.max(np.abs(resid))) or 1.0
    artists.mesh.set_array(resid)
    artists.mesh.set_clim(-mr, mr)


def update_spectral_filter(artists: LineArtists, data: RetrievalPlotData) -> None:
    """Update :func:`draw_spectral_filter`'s artists in place (x is the fixed grid)."""
    artists.line.set_ydata(_filter_factor(data.result))


def update_freq_marginal(artists: MarginalArtists, data: RetrievalPlotData) -> None:
    """Update :func:`draw_freq_marginal`'s artists in place."""
    pr = data.processed
    artists.line_retr.set_ydata(pr.omega_marg_retr)
    if artists.line_meas is not None and pr.omega_marg_meas is not None:
        artists.line_meas.set_ydata(pr.omega_marg_meas)


def update_delay_marginal(artists: MarginalArtists, data: RetrievalPlotData) -> None:
    """Update :func:`draw_delay_marginal`'s artists in place."""
    pr = data.processed
    artists.line_retr.set_ydata(pr.tau_marg_retr)
    if artists.line_meas is not None and pr.tau_marg_meas is not None:
        artists.line_meas.set_ydata(pr.tau_marg_meas)


def update_spectrogram(artists: SpectrogramArtists, data: RetrievalPlotData) -> None:
    """Update :func:`draw_spectrogram`'s artists in place.

    Raises
    ------
    ValueError
        If the spectrogram's shape differs from the drawn one (a different grid):
        the panel has to be redrawn, not updated.
    """
    _tc, _lam, S = data.spectrogram
    array = artists.mesh.get_array()
    drawn = () if array is None else tuple(np.shape(array))
    if drawn != S.shape:
        raise ValueError(f"spectrogram shape changed from {drawn} to {S.shape}")
    artists.mesh.set_array(S)
    artists.mesh.autoscale()  # what the first draw did implicitly


def _update_signature(data: RetrievalPlotData) -> tuple:
    """Everything that decides which artists exist and the mesh geometry.

    Two bundles with equal signatures differ only in values, so a panel drawn for
    one can be updated in place for the other. Components: the trace, grid and
    oversampled-axis shapes; whether the traces hold negatives (the second
    colormap); which optional curves exist (residual, measured spectrum and
    marginals, absolute power, the truth and its phases); how many stage-handover
    rules are visible; and the fixed axis bounds.
    """
    pr = data.processed
    result = data.result
    truth = data.truth
    n_rules = sum(1 for edge in result.stage_boundaries if 0 < edge < len(pr.errors))
    return (
        data.retrieved.shape,
        data.measured.shape,
        result.omega.size,
        result.delays.size,
        pr.t_over.size,
        pr.t_tl.size,
        pr.wavelength.size,
        bool((data.measured < 0).any()),
        bool((data.retrieved < 0).any()),
        pr.residual is not None,
        pr.Iw_meas is not None,
        pr.omega_marg_meas is not None,
        pr.tau_marg_meas is not None,
        pr.peak_power is not None,
        truth is not None,
        truth is not None and truth.phi_t is not None,
        truth is not None and truth.lam is not None and truth.Iw is not None,
        truth is not None and truth.lam is not None and truth.phi_w is not None,
        n_rules,
        data.lam_min,
        data.lam_max,
        data.flim,
        data.halfwidth,
        data.tracedb,
        data.spectral_axis,
    )
