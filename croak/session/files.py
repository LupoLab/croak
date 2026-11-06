"""A session's files on disk: writing a saved session, and checking a saved result.

A saved session is a folder holding ``options.toml`` (the settings, enough to
replay it), ``result.h5`` (the retrieval, with any uncertainty estimates
appended) and ``retrieval.pdf`` (the twelve-panel overview). The GUI adds what
its Analyse pages hold: the dispersed pulse and its figure when a dispersion is
applied (:data:`DISPERSION_FILE`, :data:`DISPERSION_FIGURE`) and the uncertainty
figure on screen (:data:`UNCERTAINTY_FIGURE`). Reopening one
replays the settings and reuses ``result.h5`` only when it was fitted to the
trace the replay produces (:func:`result_mismatch`).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .. import save
from .options import SessionOptions

if TYPE_CHECKING:
    from ..plotting import RetrievalPlotData
    from ..preprocess import TraceData
    from ..processing import ProcessedResult, TruthPulse
    from ..result import RetrievalResult
    from .params import RetrieveParams

__all__ = [
    "DISPERSION_FIGURE",
    "DISPERSION_FILE",
    "OPTIONS_FILE",
    "OVERVIEW_DPI",
    "OVERVIEW_FILE",
    "RESULT_FILE",
    "UNCERTAINTY_FIGURE",
    "result_mismatch",
    "save_session",
]

OPTIONS_FILE = "options.toml"
"""The settings of a saved session."""
RESULT_FILE = "result.h5"
"""The retrieval of a saved session."""
OVERVIEW_FILE = "retrieval.pdf"
"""The overview figure of a saved session."""
OVERVIEW_DPI = 600
"""The overview's resolution: its trace maps are rasterised inside the PDF."""
DISPERSION_FILE = "result_dispersion.h5"
"""The retrieved pulse with the GUI's dispersion applied."""
DISPERSION_FIGURE = "dispersion.pdf"
"""The GUI's dispersion figure."""
UNCERTAINTY_FIGURE = "uncertainty.pdf"
"""The GUI's figure of the uncertainty estimate on screen."""

# Relative tolerance for "the same array": both sides are stored exactly as the
# retrieval saw them, so only floating-point round-off may differ.
_SAME = 1e-9


def save_session(
    folder: str | Path,
    options: SessionOptions,
    *,
    result: RetrievalResult | None = None,
    processed: ProcessedResult | None = None,
    truth: TruthPulse | None = None,
    uncertainty: Mapping[str, Any] | None = None,
    overview: RetrievalPlotData | None = None,
) -> list[str]:
    """Write a session into ``folder`` and return the names of the files written.

    Parameters
    ----------
    folder : str or Path
        An existing folder; files of the same names are overwritten.
    options : SessionOptions
        The settings, always written to :data:`OPTIONS_FILE`.
    result, processed, truth
        The retrieval, its processed form and any known truth, written to
        :data:`RESULT_FILE` when ``result`` is given.
    uncertainty : mapping, optional
        Uncertainty estimates by method, appended to :data:`RESULT_FILE`.
    overview : RetrievalPlotData, optional
        Drawn as the twelve-panel overview into :data:`OVERVIEW_FILE`.

    Raises
    ------
    FileNotFoundError
        If ``folder`` does not exist.
    ValueError
        If ``uncertainty`` is given without a ``result`` to append it to.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"no such folder: {folder}")
    if uncertainty and result is None:
        raise ValueError("uncertainty estimates need the result they belong to")
    written = []
    if result is not None:
        path = folder / RESULT_FILE
        save.save_result(
            result, str(path), processed=processed, truth=truth, force=True
        )
        written.append(RESULT_FILE)
        if uncertainty:
            save.save_uncertainty(dict(uncertainty), str(path), force=True)
    save.save_options(options.to_dict(), str(folder / OPTIONS_FILE))
    written.append(OPTIONS_FILE)
    if overview is not None:
        from .. import plotting  # matplotlib only when a figure is drawn

        figure = plotting.plot_retrieval_page(overview, plotting.RETRIEVAL_OVERVIEW)
        figure.savefig(folder / OVERVIEW_FILE, dpi=OVERVIEW_DPI)
        written.append(OVERVIEW_FILE)
    return written


def result_mismatch(
    saved: Mapping[str, Any], tracedata: TraceData, retrieve: RetrieveParams
) -> str:
    """Why a saved result cannot be paired with ``tracedata``, or ``""`` if it can.

    Reuse is only sound when the saved retrieval was fitted to *this* trace, so
    the saved measured trace and frequency axis are compared with the current
    ones element-wise rather than by shape: any difference (a moved delay grid,
    a changed filter, a new calibration) is grounds to re-run rather than to show
    a fit against data it never saw. A file too old to carry ``trace_meas``
    cannot be verified, so it is refused too: declining costs only a re-run,
    wrongly accepting costs a wrong residual panel.

    Parameters
    ----------
    saved : mapping
        A result file as read by :func:`croak.save.load_result`.
    tracedata : TraceData
        The trace the replayed session produced.
    retrieve : RetrieveParams
        The replayed retrieval settings (their solver must match the file's).
    """
    current = tracedata.trace
    meas = saved.get("trace_meas")
    if meas is None:
        return "the saved result does not store the trace it was fitted to"
    meas = np.asarray(meas, float)
    if meas.shape != np.shape(current):
        return (
            f"the saved result is on a {meas.shape[0]}x{meas.shape[1]} grid but "
            f"this trace regrids to {np.shape(current)[0]}x{np.shape(current)[1]}"
        )
    peak = float(np.max(np.abs(current))) or 1.0
    if float(np.max(np.abs(meas - current))) > _SAME * peak:
        return "preprocessing no longer reproduces the trace the result was fitted to"
    omega_saved = saved.get("omega")
    if omega_saved is not None:
        omega = np.asarray(tracedata.grid.omega, float)
        omega_saved = np.asarray(omega_saved, float)
        span = float(np.max(np.abs(omega))) or 1.0
        if omega_saved.shape != omega.shape or (
            float(np.max(np.abs(omega_saved - omega))) > _SAME * span
        ):
            return "the saved result is on a different frequency grid"
    algorithm = saved.get("algorithm")
    if algorithm is not None and str(algorithm) != str(retrieve.solver):
        return (
            f"the saved result came from {algorithm!s} but the loaded settings "
            f"select {retrieve.solver}"
        )
    return ""
