"""Uncertainty figures: FWHM distributions, intensity bands, thickness systematic."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.figure import Figure

from .figure import _prepare_fig

if TYPE_CHECKING:
    from ..uncertainty import UncertaintyResult


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------
def plot_fwhm_distribution(
    ax, uresult: UncertaintyResult, *, truth: float | None = None
) -> None:
    """Histogram of bootstrap FWHM samples with the point estimate and intervals.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw into.
    uresult : UncertaintyResult
        The bootstrap result (its ``samples`` are histogrammed).
    truth : float, optional
        A known true FWHM (s) to mark with a dashed line (synthetic tests).
    """
    samples = np.asarray(uresult.samples, dtype=float) / 1e-15
    nbins = int(np.clip(samples.size // 5, 12, 40))
    ax.hist(samples, bins=nbins, color="C0", alpha=0.7, edgecolor="white", lw=0.4)
    lo95, hi95 = uresult.interval_95
    lo68, hi68 = uresult.interval_68
    ax.axvspan(lo95 / 1e-15, hi95 / 1e-15, color="C0", alpha=0.10, label="95 %")
    ax.axvspan(lo68 / 1e-15, hi68 / 1e-15, color="C0", alpha=0.20, label="68 %")
    ax.axvline(uresult.point_estimate / 1e-15, c="C3", lw=1.5, label="estimate")
    if truth is not None:
        ax.axvline(truth / 1e-15, c="0.35", ls="--", lw=1.2, label="truth")
    ax.set_xlabel("FWHM (fs)")
    ax.set_ylabel("Replicates")
    ax.set_title(
        f"{uresult.point_estimate / 1e-15:.2f} ± {uresult.plus_minus / 1e-15:.2f} fs"
        f"  (B={uresult.n_converged})"
    )
    ax.legend(loc="best", handlelength=1.0, frameon=False, fontsize="small")


def plot_intensity_band(ax, uresult: UncertaintyResult) -> None:
    """Plot the median temporal intensity and pointwise 68 %/95 % resampling bands.

    Requires the bootstrap to have been run with ``collect_profiles=True``. The profiles
    are peak-aligned (translation gauge removed by ``shiftnorm``), so the band is the
    spread of the aligned intensity envelope across replicates. The bands are
    pointwise, not simultaneous bands for the whole curve.
    """
    if uresult.profiles is None or uresult.t_profile is None:
        raise ValueError(
            "uresult has no profiles; run the bootstrap with collect_profiles=True"
        )
    t = np.asarray(uresult.t_profile, dtype=float) / 1e-15
    profiles = np.asarray(uresult.profiles, dtype=float)
    med = np.median(profiles, axis=0)
    lo68, hi68 = np.percentile(profiles, [16.0, 84.0], axis=0)
    lo95, hi95 = np.percentile(profiles, [2.5, 97.5], axis=0)
    ax.fill_between(t, lo95, hi95, color="C0", alpha=0.15, label="95 %")
    ax.fill_between(t, lo68, hi68, color="C0", alpha=0.30, label="68 %")
    ax.plot(t, med, c="C0", lw=1.5, label="median")
    ax.set_xlabel("Time (fs)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_ylim(bottom=0.0)
    ax.set_title("Temporal intensity")
    # zoom to where the envelope carries signal
    support = t[med > 0.01 * med.max()]
    if support.size:
        pad = 0.2 * (support.max() - support.min() + 1.0)
        ax.set_xlim(support.min() - pad, support.max() + pad)
    ax.legend(loc="best", handlelength=1.0, frameon=False, fontsize="small")


def plot_uncertainty(
    uresult: UncertaintyResult,
    *,
    truth: float | None = None,
    figsize=(11.0, 4.2),
    fig: Figure | None = None,
) -> Figure:
    """Composite uncertainty figure: FWHM histogram + a confidence band if present.

    Parameters
    ----------
    uresult : UncertaintyResult
        The bootstrap result.
    truth : float, optional
        Known true FWHM (s) for the histogram (synthetic tests).
    figsize : tuple, optional
        Figure size.
    fig : matplotlib.figure.Figure, optional
        Existing figure to draw into (cleared first), e.g. a Qt canvas figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # A thickness-systematic result carries per-draw thicknesses; show the dedicated
    # sensitivity figure so the GUI canvas path needs no special-casing.
    if getattr(uresult, "thicknesses", None) is not None:
        return plot_thickness_sensitivity(uresult, truth=truth, fig=fig)
    fig = _prepare_fig(fig, figsize)
    if uresult.profiles is not None and uresult.t_profile is not None:
        ax_h, ax_b = fig.subplots(1, 2)
        plot_fwhm_distribution(ax_h, uresult, truth=truth)
        plot_intensity_band(ax_b, uresult)
    else:
        ax_h = fig.subplots(1, 1)
        plot_fwhm_distribution(ax_h, uresult, truth=truth)
    fig.suptitle(uresult.summary(), fontsize="small")
    return fig


# ---------------------------------------------------------------------------
# Substrate-thickness systematic
# ---------------------------------------------------------------------------
def _mark_thickness_prior(ax, uresult: UncertaintyResult) -> None:
    """Shade the measured thickness prior ``L₀ ± σ_L`` (µm) on a thickness axis."""
    if uresult.thickness_central is None or uresult.thickness_sigma is None:
        return
    central_um = uresult.thickness_central / 1e-6
    sigma_um = uresult.thickness_sigma / 1e-6
    ax.axvspan(
        central_um - sigma_um,
        central_um + sigma_um,
        color="0.8",
        alpha=0.5,
        label="L₀ ± σ_L",
    )
    ax.axvline(central_um, color="0.4", ls=":", lw=1.0)


def plot_fwhm_vs_thickness(
    ax, uresult: UncertaintyResult, *, truth: float | None = None
) -> None:
    """Scatter the retrieved FWHM against the drawn substrate thickness.

    The horizontal bands give the point estimate and its 68 % interval; the vertical
    band is the measured thickness prior. The slope of the cloud is the
    sensitivity ``dFWHM/dL`` the analysis propagates; solver variation adds to
    the spread.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw into.
    uresult : UncertaintyResult
        A ``method="thickness"`` result (its ``thicknesses`` are the x-values).
    truth : float, optional
        A known true FWHM (s) to mark with a dashed line (synthetic tests).
    """
    if uresult.thicknesses is None:
        raise ValueError("uresult has no thicknesses; run method='thickness'")
    thick_um = np.asarray(uresult.thicknesses, dtype=float) / 1e-6
    fwhm_fs = np.asarray(uresult.samples, dtype=float) / 1e-15
    _mark_thickness_prior(ax, uresult)
    ax.scatter(thick_um, fwhm_fs, s=14, color="C0", alpha=0.6, edgecolors="none")
    ax.axhline(uresult.point_estimate / 1e-15, color="C3", lw=1.5, label="estimate")
    lo68, hi68 = uresult.interval_68
    ax.axhspan(lo68 / 1e-15, hi68 / 1e-15, color="C3", alpha=0.15, label="68 %")
    if truth is not None:
        ax.axhline(truth / 1e-15, color="0.35", ls="--", lw=1.2, label="truth")
    ax.set_xlabel("Substrate thickness (µm)")
    ax.set_ylabel("FWHM (fs)")
    ax.set_title("FWHM vs thickness")
    ax.legend(loc="best", handlelength=1.0, frameon=False, fontsize="small")


def plot_error_vs_thickness(ax, uresult: UncertaintyResult) -> None:
    """Scatter the FROG error against the drawn thickness (a consistency check).

    A flat cloud means the data do not constrain the thickness (the full prior
    propagates); a clear minimum near ``L₀`` means the trace itself prefers a
    thickness under the chosen model. The plot does not reweight the prior.
    """
    if uresult.thicknesses is None:
        raise ValueError("uresult has no thicknesses; run method='thickness'")
    thick_um = np.asarray(uresult.thicknesses, dtype=float) / 1e-6
    errors = np.asarray(uresult.frog_errors, dtype=float)
    _mark_thickness_prior(ax, uresult)
    ax.scatter(thick_um, errors, s=14, color="C1", alpha=0.6, edgecolors="none")
    ax.set_xlabel("Substrate thickness (µm)")
    ax.set_ylabel("FROG error R")
    ax.set_title("Trace misfit vs thickness")
    ax.legend(loc="best", handlelength=1.0, frameon=False, fontsize="small")


def plot_thickness_sensitivity(
    uresult: UncertaintyResult,
    *,
    truth: float | None = None,
    figsize=(13.0, 4.2),
    fig: Figure | None = None,
) -> Figure:
    """Composite figure for the substrate-thickness systematic.

    Panels: FWHM vs thickness, FROG error vs thickness, and — when profiles were
    collected — the temporal intensity confidence band.

    Parameters
    ----------
    uresult : UncertaintyResult
        A ``method="thickness"`` result.
    truth : float, optional
        Known true FWHM (s) for the FWHM panel (synthetic tests).
    figsize : tuple, optional
        Figure size.
    fig : matplotlib.figure.Figure, optional
        Existing figure to draw into (cleared first), e.g. a Qt canvas figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = _prepare_fig(fig, figsize)
    has_band = uresult.profiles is not None and uresult.t_profile is not None
    axes = fig.subplots(1, 3 if has_band else 2)
    plot_fwhm_vs_thickness(axes[0], uresult, truth=truth)
    plot_error_vs_thickness(axes[1], uresult)
    if has_band:
        plot_intensity_band(axes[2], uresult)
    fig.suptitle(uresult.summary(), fontsize="small")
    return fig
