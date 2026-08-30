"""Terminal / notebook progress reporting for :func:`croak.retrieve`.

When the caller of :func:`~croak.retrieve.retrieve` does not pass its own
``callback``, a :class:`ProgressReporter` is installed automatically. It renders
a live progress bar and prints a one-line summary report when retrieval finishes
(final FROG error, retrieved and transform-limited FWHM, iteration count and
wall-clock time).

The live bar is drawn with `tqdm.auto <https://tqdm.github.io/>`_, which picks
the right widget for the environment: an ``ipywidgets`` bar in a Jupyter
notebook (or a plain text bar if ``ipywidgets`` is not installed) and a text bar
on the terminal. If ``tqdm`` is somehow unavailable, a small built-in
carriage-return bar is used instead so the module stays self-contained.

The GUI passes its own ``callback`` and so is left untouched: it draws its own
convergence curve and status line (see :mod:`croak.gui.stage_retrieve`).

Environments
------------
``detect_environment`` distinguishes three cases the wrapper cares about:

``"notebook"``
    A Jupyter / IPython kernel (``ZMQInteractiveShell``); ``tqdm.auto`` renders
    an inline bar and the report is printed below it.
``"terminal"``
    An interactive TTY. The bar is written to ``stderr`` so it does not pollute
    anything piped from ``stdout``.
``"plain"``
    Non-interactive (output redirected to a file, a CI log, a pytest capture,
    …). Under the default ``progress="auto"`` no live bar *or* report is shown,
    so library calls in scripts and test suites stay silent; pass
    ``progress=True`` to force the report anyway.
"""

from __future__ import annotations

import contextlib
import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .processing import ProcessedResult
    from .result import RetrievalResult

__all__ = ["ProgressReporter", "detect_environment", "make_reporter", "format_report"]

_FS = 1e-15


def detect_environment() -> str:
    """Return ``"notebook"``, ``"terminal"`` or ``"plain"`` (see module docs)."""
    # IPython is not a dependency; it is only importable in notebooks/IPython.
    with contextlib.suppress(Exception):
        # Blanket type-ignore: pyright's bundled stubs flag get_ipython's
        # top-level re-export as private, and in environments without IPython
        # installed (it is not a dependency) the import is unresolvable too.
        from IPython import get_ipython  # type: ignore

        shell = get_ipython()
        if shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell":
            return "notebook"
    stream = sys.stderr
    with contextlib.suppress(Exception):  # exotic stream without isatty
        if stream is not None and stream.isatty():
            return "terminal"
    return "plain"


def _have_ipywidgets() -> bool:
    """Whether ``ipywidgets`` is importable (needed for the notebook tqdm bar)."""
    try:
        # Probes availability only; unresolvable when not installed.
        import ipywidgets  # noqa: F401  # type: ignore
    except Exception:
        return False
    return True


def make_reporter(progress, *, total: int | None) -> ProgressReporter | None:
    """Build a :class:`ProgressReporter` for the ``progress`` setting, or ``None``.

    ``progress`` is the :func:`croak.retrieve.retrieve` argument:

    * ``"auto"`` — report only in a terminal or notebook (``"plain"`` is silent),
    * ``True`` — always report, forcing a live bar even when non-interactive,
    * ``False`` — never report (returns ``None``).
    """
    if progress is False:
        return None
    env = detect_environment()
    if progress is True:
        live = True
    elif env == "plain":
        return None  # auto + non-interactive: stay silent
    else:
        live = True  # terminal / notebook
    return ProgressReporter(total=total, env=env, live=live)


class ProgressReporter:
    """Live progress bar + final summary report for a single retrieval.

    Its :meth:`callback` matches the solver hook signature
    ``callback(iteration, R, best_R, snapshot=None)`` and is what
    :func:`croak.retrieve.retrieve` forwards to the solver (the trailing
    ``snapshot`` is ignored here). :meth:`finish` is called once the solver
    returns.
    """

    def __init__(
        self,
        *,
        total: int | None = None,
        env: str | None = None,
        live: bool = True,
        label: str = "retrieving",
        width: int = 30,
        min_interval: float = 0.1,
        use_tqdm: bool = True,
    ):
        """Configure the bar (``total`` iterations, terminal width, redraw rate).

        ``use_tqdm`` selects the renderer: ``True`` (default) uses
        ``tqdm.auto`` when importable, otherwise the built-in carriage-return
        bar; ``False`` forces the built-in bar (used by the test-suite to get a
        deterministic, capturable rendering).
        """
        self.total = int(total) if total else None
        self.env = env or detect_environment()
        self.live = live
        self.label = label
        self.width = width
        self.min_interval = float(min_interval)
        # stderr keeps stdout clean for piping in a terminal; notebooks render
        # stderr in red, so the built-in fallback bar goes to stdout there.
        self.stream = sys.stdout if self.env == "notebook" else sys.stderr
        self.start = time.perf_counter()
        self.iteration = 0
        self.last_R = float("nan")
        self.best_R = float("inf")
        self._last_draw = 0.0
        self._drawn = False
        self._tqdm = self._make_tqdm() if (self.live and use_tqdm) else None

    def _make_tqdm(self):
        """Create a tqdm bar suited to the environment, or ``None`` if unavailable.

        We dispatch on our own :func:`detect_environment` rather than importing
        ``tqdm.auto``: ``tqdm.auto`` probes for the notebook widget at import
        time and prints a noisy ``TqdmWarning`` ("IProgress not found …") when
        ``ipywidgets`` is missing — even in a plain terminal. Instead we use the
        rich ``tqdm.notebook`` bar only in a real notebook *with* ``ipywidgets``
        installed, and fall back to the text bar (``tqdm.std``) everywhere else.
        The text bar renders fine inside a notebook output cell too.
        """
        import warnings

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if self.env == "notebook" and _have_ipywidgets():
                    from tqdm.notebook import tqdm
                else:
                    from tqdm.std import tqdm
                return tqdm(
                    total=self.total,
                    desc=self.label,
                    mininterval=self.min_interval,
                    leave=True,
                    dynamic_ncols=True,
                )
        except Exception:  # pragma: no cover - tqdm is a declared dependency
            return None

    # -- solver hook --------------------------------------------------------
    def callback(self, iteration: int, R: float, best_R: float, snapshot=None) -> None:
        """Solver progress hook: record the latest iteration and redraw the bar.

        ``snapshot`` (an optional zero-argument in-progress-result builder some
        solvers pass for the GUI live preview) is ignored here — the terminal
        reporter only needs the scalar progress.
        """
        self.iteration = int(iteration)
        self.last_R = float(R)
        self.best_R = float(best_R)
        if not self.live:
            return
        if self._tqdm is not None:
            self._update_tqdm()
            return
        now = time.perf_counter()
        at_end = self.total is not None and self.iteration >= self.total
        # Throttle redraws so a fast solver does not flood the terminal.
        if not at_end and now - self._last_draw < self.min_interval and self._drawn:
            return
        self._last_draw = now
        self._draw()

    def _update_tqdm(self) -> None:
        if self._tqdm is None:  # only meaningful with the tqdm backend active
            return
        delta = self.iteration - self._tqdm.n
        if delta > 0:
            self._tqdm.update(delta)
        self._tqdm.set_postfix_str(f"R={self.last_R:.3%} best={self.best_R:.3%}")
        self._drawn = True

    def _draw(self) -> None:
        elapsed = time.perf_counter() - self.start
        if self.total:
            frac = min(self.iteration / self.total, 1.0)
            filled = int(round(self.width * frac))
            bar = "█" * filled + "·" * (self.width - filled)
            head = f"\r{self.label} |{bar}| {self.iteration}/{self.total}"
        else:
            head = f"\r{self.label}  iter {self.iteration}"
        msg = f"{head}  R={self.last_R:.3%}  best={self.best_R:.3%}  {elapsed:5.1f}s"
        try:
            self.stream.write(msg)
            self.stream.flush()
        except Exception:  # pragma: no cover - closed/odd stream
            self.live = False
            return
        self._drawn = True

    # -- final report -------------------------------------------------------
    def finish(
        self,
        result: RetrievalResult,
        elapsed: float | None = None,
        *,
        processed: ProcessedResult | None = None,
    ) -> None:
        """Close the live bar (if any) and print the summary report."""
        if elapsed is None:
            elapsed = time.perf_counter() - self.start
        if self._tqdm is not None:
            self._tqdm.close()
        elif self.live and self._drawn:
            with contextlib.suppress(Exception):
                self.stream.write("\n")
                self.stream.flush()
        report = format_report(result, elapsed, processed=processed)
        # The report is informational output for a human; print to stdout so it
        # is not mistaken for an error and survives stderr redirection.
        print(report)


def format_report(
    result: RetrievalResult,
    elapsed: float,
    *,
    processed: ProcessedResult | None = None,
) -> str:
    """Format the end-of-retrieval summary (error, FWHMs, iterations, time)."""
    if processed is None:
        import numpy as np

        from .processing import process_result

        # process_result computes a wavelength axis (2πc/ω); with a centred,
        # zero-crossing ω axis that divides by zero at one bin. It is harmless
        # for the FWHM figures we report, so don't surface the RuntimeWarning.
        with np.errstate(divide="ignore", invalid="ignore"):
            processed = process_result(result)
    fwhm = processed.fwhm_retr
    fwhm_tl = processed.fwhm_tl
    fwhm_s = f"{fwhm / _FS:.2f} fs" if fwhm == fwhm else "n/a"  # NaN-safe
    fwhm_tl_s = f"{fwhm_tl / _FS:.2f} fs" if fwhm_tl == fwhm_tl else "n/a"
    return (
        f"retrieval complete — {result.algorithm} / {result.interaction}\n"
        f"  final error R    : {result.error:.4%}\n"
        f"  retrieved FWHM   : {fwhm_s}\n"
        f"  transform limit  : {fwhm_tl_s}\n"
        f"  iterations       : {len(result.errors)}\n"
        f"  time taken       : {elapsed:.2f} s"
    )
