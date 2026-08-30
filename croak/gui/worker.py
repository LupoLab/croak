"""Background workers: run the session pipeline off the GUI thread.

These ``QThread`` wrappers add only threading, progress signals and cancellation
on top of the Qt-free stage mappings in :mod:`croak.session.pipeline`; the GUI and
the headless engine therefore run identical computations.
"""

from __future__ import annotations

import time

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from ..session import pipeline

__all__ = [
    "RetrievalWorker",
    "PreprocessWorker",
    "UncertaintyWorker",
    "StopRetrieval",
]


class StopRetrieval(Exception):
    """Raised inside the progress callback to abort retrieval cleanly."""


class PreprocessWorker(QThread):
    """Run :func:`croak.session.pipeline.build_tracedata` off the GUI thread.

    Preprocessing is cheap (a few ms) but the *redraw* it triggers is not, and
    both used to block the UI thread, making slider drags laggy. This runs the
    filtering/regridding in a thread and hands the result back via a signal; the
    (main-thread) caller does the plotting. ``params`` is a snapshot the caller
    takes (so later slider changes can't mutate an in-flight run) and ``data``
    holds read-only arrays. ``request_id`` lets the caller ignore superseded
    results. ``fast`` selects the coarse, interactive-preview baseline (the caller
    then refines to the exact baseline once the user settles).

    Signals
    -------
    finished_ok : (object, int, bool)
        The :class:`~croak.preprocess.TraceData`, the originating ``request_id``,
        and whether this was the ``fast`` (preview) computation.
    failed : (str, int)
        An error message and the originating ``request_id``.
    """

    finished_ok = pyqtSignal(object, int, bool)
    failed = pyqtSignal(str, int)

    def __init__(self, params, data, request_id, parent=None, *, fast=False):
        super().__init__(parent)
        self._params = params
        self._data = data
        self._id = int(request_id)
        self._fast = bool(fast)

    def run(self) -> None:  # noqa: D401 - QThread entry point
        try:
            td = pipeline.build_tracedata(
                self._params, self._data, fast_baseline=self._fast
            )
        except Exception as exc:  # pragma: no cover - surfaced in the GUI
            self.failed.emit(str(exc), self._id)
            return
        self.finished_ok.emit(td, self._id, self._fast)


class RetrievalWorker(QThread):
    """Run :func:`croak.session.pipeline.run_retrieval` in a thread, emitting progress.

    For solvers that can cheaply build one, the per-iteration callback also
    carries a *snapshot* — a zero-argument builder of the current in-progress
    :class:`~croak.result.RetrievalResult`. This worker throttles those to at most
    one every :attr:`PREVIEW_MIN_INTERVAL` seconds, building them off the GUI
    thread and emitting them via :attr:`preview` so the stage can show the full
    12-panel view live (not just the convergence curve). A user Stop captures one
    final snapshot at the exact stopping point so the full plot stays on screen.

    Signals
    -------
    progress : (int, float, float)
        ``(iteration, R, best_R)`` after each iteration (cheap; every iteration).
    preview : object
        A throttled in-progress :class:`~croak.result.RetrievalResult` for the
        live full-plot view (only for snapshot-capable solvers).
    finished_ok : (object, float)
        The :class:`~croak.result.RetrievalResult` and the wall-clock time taken
        (seconds) on success.
    stopped : (object, float)
        The partial result captured at the stopping point, and the elapsed time,
        when the user pressed Stop and a snapshot was available.
    failed : str
        An error message on failure (or a bare Stop with no snapshot available).
    """

    #: Minimum wall-clock seconds between live full-plot previews. The 12-panel
    #: render (process + Gabor spectrogram + draw) is far costlier than one solver
    #: iteration, so previews are throttled rather than drawn every iteration.
    PREVIEW_MIN_INTERVAL = 1.0

    progress = pyqtSignal(int, float, float)
    preview = pyqtSignal(object)
    finished_ok = pyqtSignal(object, float)
    stopped = pyqtSignal(object, float)
    failed = pyqtSignal(str)

    def __init__(
        self,
        tracedata,
        params,
        parent=None,
        guess_override=None,
        extras_seed=None,
        truth=None,
        preview=True,
        scan_path=None,
    ):
        super().__init__(parent)
        self._td = tracedata
        self._p = params
        self._guess_override = guess_override
        self._extras_seed = extras_seed
        self._truth = truth
        self._preview = bool(preview)
        # Loaded scan file, forwarded so a focal-mixture retrieval with
        # collection="file" can rebuild the aperture from the file's own record.
        self._scan_path = scan_path
        self._stop = False
        self._t0 = 0.0
        self._last_preview = 0.0
        self._stop_result = None

    def request_stop(self) -> None:
        self._stop = True

    @staticmethod
    def _build(snapshot):
        """Materialise a snapshot, swallowing failures.

        A live preview is best-effort cosmetics: a transient build failure (e.g. a
        degenerate early-iteration field) must never abort the real retrieval, and
        the authoritative final result is built on the normal, must-succeed path.
        """
        try:
            return snapshot()
        except Exception:  # pragma: no cover - defensive; preview is best-effort
            return None

    def _callback(self, iteration: int, R: float, best_R: float, snapshot=None) -> None:
        self.progress.emit(int(iteration), float(R), float(best_R))
        # Throttled live full-plot preview (snapshot-capable solvers only).
        if snapshot is not None and self._preview:
            now = time.perf_counter()
            if (
                self._last_preview == 0.0
                or now - self._last_preview >= self.PREVIEW_MIN_INTERVAL
            ):
                result = self._build(snapshot)
                if result is not None:
                    self._last_preview = now
                    self.preview.emit(result)
        if self._stop:
            # Capture the exact stopping state so the full plot stays on screen,
            # even when periodic previews were disabled.
            if snapshot is not None:
                self._stop_result = self._build(snapshot)
            raise StopRetrieval

    def run(self) -> None:  # noqa: D401 - QThread entry point
        try:
            self._t0 = time.perf_counter()
            result = pipeline.run_retrieval(
                self._p,
                self._td,
                guess_override=self._guess_override,
                extras_seed=self._extras_seed,
                truth=self._truth,
                rng=np.random.default_rng(),
                callback=self._callback,
                scan_path=self._scan_path,
            )
            self.finished_ok.emit(result, time.perf_counter() - self._t0)
        except StopRetrieval:
            elapsed = time.perf_counter() - self._t0
            if self._stop_result is not None:
                self.stopped.emit(self._stop_result, elapsed)
            else:
                self.failed.emit("stopped")
        except Exception as exc:  # pragma: no cover - surfaced in the GUI
            self.failed.emit(str(exc))


class UncertaintyWorker(QThread):
    """Run an FWHM-uncertainty bootstrap in a thread, emitting per-replicate progress.

    Signals
    -------
    progress : (int, int, float)
        ``(done, total, R)`` after each bootstrap replicate.
    finished_ok : object
        The :class:`~croak.uncertainty.UncertaintyResult` on success.
    failed : str
        An error message on failure (excluding a user Stop).
    """

    progress = pyqtSignal(int, int, float)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self, result, tracedata, retrieve_params, uparams, dispersion=None, parent=None
    ):
        super().__init__(parent)
        self._result = result
        self._td = tracedata
        self._rp = retrieve_params
        self._up = uparams
        self._dispersion = dispersion
        self._stop = False

    def request_stop(self) -> None:
        self._stop = True

    def _callback(self, done: int, total: int, error: float) -> None:
        self.progress.emit(int(done), int(total), float(error))
        if self._stop:
            raise StopRetrieval

    def run(self) -> None:  # noqa: D401 - QThread entry point
        try:
            uresult = pipeline.run_uncertainty(
                self._up,
                self._rp,
                self._result,
                self._td,
                dispersion=self._dispersion,
                rng=np.random.default_rng(),
                callback=self._callback,
            )
            self.finished_ok.emit(uresult)
        except StopRetrieval:
            self.failed.emit("stopped")
        except Exception as exc:  # pragma: no cover - surfaced in the GUI
            self.failed.emit(str(exc))
