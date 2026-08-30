"""Regenerate the images used by the README and the documentation.

Two images, both derived from a synthetic retrieval so they are reproducible and
carry no measured data:

``docs/_static/hero.png``
    The 12-panel :func:`croak.plot_retrieval` summary — the landing-page figure.

``docs/_static/gui_retrieve.png``
    A screenshot of the wizard's Retrieve stage just after a run, showing the
    plot and the numeric read-out strip.

Run it after any change that alters the plot layout, the read-out columns or the
wizard's controls::

    uv run python tools/make_docs_images.py

The GUI shot is grabbed head*less* (``QT_QPA_PLATFORM=offscreen``), so this works
in a terminal, over SSH and in CI. Two consequences worth knowing: an offscreen
window is never maximised, so the size is whatever :data:`GUI_SIZE` says rather
than the user's screen; and the grab must happen *after* the retrieval finishes,
because the read-out only fills in at end-of-run.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Must precede every Qt import: it selects the windowless platform plugin.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib

matplotlib.use("Agg")

import numpy as np

import croak

STATIC = Path(__file__).resolve().parent.parent / "docs" / "_static"

#: Size of the grabbed wizard window (px). Wide enough that the left-hand control
#: column and the 12-panel canvas both stay legible when the image is scaled down
#: to README width.
GUI_SIZE = (1600, 1000)

#: Pulse energy (J) handed to the GUI run. Without it the read-out's "Peak power"
#: column is all dashes, so the screenshot would show a half-empty table.
GUI_ENERGY_J = 1.5e-6


#: Restart seeds tried for the hero retrieval, best kept. ``retrieve`` draws a
#: *random* Gaussian-spectrum initial guess when no ``guess`` is given, and COPRA
#: stagnates near R ≈ 3e-2 on this trace from a poor start — so a single run is a
#: coin flip. This is the random-restart recipe from
#: docs/howto/solver_selection.md, which is also the honest thing to show.
HERO_SEEDS = range(8)

#: The hero must show the algorithm working. If the best restart cannot beat
#: this, something has regressed and the script fails rather than quietly
#: shipping a poor figure (the previous hero showed a 5.3% error retrieval).
HERO_MAX_ERROR = 1e-4


def make_hero(path: Path) -> Path:
    """Render the 12-panel retrieval summary from a well-converged synthetic run.

    Raises
    ------
    RuntimeError
        If no restart reaches :data:`HERO_MAX_ERROR`.
    """
    # Retrieve directly on the synthesis grid rather than going through
    # load_and_clean. Regridding onto a cropped retrieval grid would fill the
    # trace panels more, but it carries a resampling error that floors R around a
    # few percent (examples/example_workflow.py lands at ~8%). The hero should
    # show the algorithm converging, so accuracy wins over panel framing.
    #
    # The pulse must stay well inside the delay scan: if its wings extend beyond
    # ±tau_max the trace is truncated in delay and *no* solver can fit it, which
    # shows up as a floor in R that more iterations and polishing cannot move.
    grid = croak.Grid(256, dt=0.45e-15)
    # wlfreq is array-in/array-out, so coerce the scalar case for the API.
    omega0 = float(croak.maths.wlfreq(800e-9))
    # Chirped: enough spectral phase to be visibly non-trivial in both domains,
    # while staying compact enough to fit the scan.
    ew = croak.gaussian_pulse(grid, 6e-15, phases=[25e-30, 120e-45])
    delays = np.linspace(-35e-15, 35e-15, 128)
    trace = croak.maketrace(grid.omega, delays, ew, "shg")

    restarts = [
        croak.retrieve(
            trace,
            grid.omega,
            delays,
            "shg",
            algorithm="copra",
            maxiters=400,
            omega0=omega0,
            progress=False,
            rng=np.random.default_rng(seed),
        )
        for seed in HERO_SEEDS
    ]
    best = min(restarts, key=lambda r: r.error)
    if best.error > HERO_MAX_ERROR:
        raise RuntimeError(
            f"best restart reached only R = {best.error:.2e}, above the "
            f"{HERO_MAX_ERROR:.0e} the hero figure requires"
        )

    # The pulse this trace was built from, overlaid on the temporal and spectral
    # panels so the figure shows the retrieval landing on the right answer, not
    # merely converging.
    truth = croak.TruthPulse.from_spectrum(grid, ew, omega0)
    # An SHG trace cannot distinguish a pulse from its mirror image, so the solver
    # returns whichever branch it landed in; pick the one that matches the truth
    # before processing, or the overlay reads as a failed retrieval when it is not.
    best, _ = croak.resolve_time_direction(best, truth)
    processed = croak.process_result(best, measured=trace)
    # Bound both axis families. lam_min/lam_max bound the wavelength panels: the
    # baseband grid runs down to near-zero *absolute* frequency, where
    # lambda = 2*pi*c/(omega+omega0) diverges, so without them the spectrum and
    # spectrogram are squeezed into the left edge of a 200,000 nm axis. flim
    # bounds the trace panels' frequency axis, which is otherwise the full
    # retrieval grid with the signal a thin stripe across the middle. The SHG
    # signal sits at twice the pulse carrier, hence the factor of 2.
    f0 = omega0 / (2.0 * np.pi)
    fig = croak.plot_retrieval(
        best,
        measured=trace,
        processed=processed,
        truth=truth,
        lam_min=600e-9,
        lam_max=1100e-9,
        flim=(2.0 * f0 - 0.30e15, 2.0 * f0 + 0.30e15),
    )
    fig.savefig(path, dpi=110, bbox_inches="tight")
    print(f"wrote {path}  (R = {best.error:.2e}, {trace.shape[0]}×{trace.shape[1]})")
    return path


def _wait_for(signal, timeout_ms: int = 120_000) -> None:
    """Block in a nested Qt event loop until ``signal`` fires (or time runs out).

    The retrieval runs on a worker thread, so the GUI screenshot has to wait for
    it the way the application does rather than by sleeping. This is the
    no-pytest-qt equivalent of ``qtbot.waitSignal``.
    """
    from PyQt6.QtCore import QEventLoop, QTimer

    loop = QEventLoop()
    signal.connect(loop.quit)
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    timer.start(timeout_ms)
    loop.exec()
    if not timer.isActive():
        raise TimeoutError("timed out waiting for the retrieval to finish")
    timer.stop()


def make_gui_screenshot(path: Path) -> Path:
    """Drive the wizard through a synthetic retrieval and grab the Retrieve stage.

    Follows the same route a user takes from the Welcome page: generate a
    synthetic trace, accept the seeded preprocessing, then retrieve. A synthetic
    session is used deliberately — it carries the known ground truth, so the
    read-out's ``truth`` row is populated and the plot gains its truth overlay.
    """
    from PyQt6.QtWidgets import QApplication

    from croak.gui.wizard import Wizard

    app = QApplication.instance() or QApplication(sys.argv)
    wizard = Wizard()
    wizard.resize(*GUI_SIZE)

    # Welcome -> synthetic generator -> build the trace and advance to the
    # marginal-check stage (see croak/gui/stage_synthetic.py:advance).
    wizard.show_synthetic()
    wizard.synthetic.advance()

    # Preprocess runs synchronously from the seeded defaults, so stepping onto it
    # is enough to produce the cleaned TraceData the retrieve stage consumes.
    # Widen the spectral window and lengthen the time range before preprocessing.
    # The seeded defaults regrid onto only 24 frequency points (dt = 10.5 fs),
    # which is too coarse to represent a 10 fs pulse: the retrieval still reaches
    # R < 1% but the recovered FWHM comes out ~18% long. These settings give 98
    # points and bring it within ~3% of the known truth.
    preproc = wizard.state.preproc
    preproc.lam_min_nm, preproc.lam_max_nm = 650.0, 1000.0
    preproc.trange_fs = 600.0

    wizard.state.stage = 3
    retrieve_stage = wizard.stages[3]
    params = retrieve_stage.state.retrieve
    # lbfgs-ad: the autodiff solver, and the most feature-complete one — the
    # right thing to show given how the docs now frame the gradient story.
    params.solver = "lbfgs-ad"
    params.maxiters = 300
    # The defaults stop this trace after ~18 iterations at R ≈ 1%. Tighten the
    # NLopt tolerances so the screenshot shows a converged retrieval whose
    # recovered FWHM matches the known truth.
    params.reltol = 1e-12
    params.abstol = 1e-14
    retrieve_stage.state.load.energy_j = GUI_ENERGY_J
    # _run() reads the state, but the controls are only repainted from it here —
    # without this the screenshot would advertise settings the run did not use.
    retrieve_stage._seed_widgets()

    wizard.state.stage = 4
    retrieve_stage._run()
    _wait_for(wizard.state.result_changed)

    # Let the canvas repaint and the read-out cells update before grabbing.
    app.processEvents()

    pixmap = wizard.grab()
    if pixmap.isNull():
        raise RuntimeError("QWidget.grab() returned a null pixmap")
    if not pixmap.save(str(path), "PNG"):
        raise RuntimeError(f"failed to write {path}")
    print(f"wrote {path}  ({pixmap.width()}×{pixmap.height()})")
    return path


def main(argv: list[str] | None = None) -> int:
    """Regenerate the requested images (both by default)."""
    parser = argparse.ArgumentParser(
        description="Regenerate the README/documentation images."
    )
    parser.add_argument(
        "which",
        nargs="?",
        default="all",
        choices=["all", "hero", "gui"],
        help="which image to regenerate (default: all)",
    )
    args = parser.parse_args(argv)

    STATIC.mkdir(parents=True, exist_ok=True)
    if args.which in ("all", "hero"):
        make_hero(STATIC / "hero.png")
    if args.which in ("all", "gui"):
        make_gui_screenshot(STATIC / "gui_retrieve.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
