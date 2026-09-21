"""The wizard main window.

A welcome page, a synthetic-trace generator and a simulated-scan loader sit in
front of the six linear stages (Load → Marginal check → Preprocess → Retrieve →
Dispersion → Uncertainty). The welcome page routes to the entry points; the
linear stages and the two entry pages share a header and a footer nav bar (only
the welcome menu hides them), so every forward/backward control sits in the same
bottom-right corner. The synthetic and simulated loaders enter at the
marginal-check stage (their "stage 1"); the experimental Load is stage 1.
"""

from __future__ import annotations

import os

import numpy as np
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .. import save
from ..session import retarget
from . import settings
from .base import Stage
from .branding import app_icon
from .stage_dispersion import StageDispersion
from .stage_load import (
    StageLoad,
    assemble_load_data,
    compute_preproc_defaults,
    trace_signature,
)
from .stage_marginal_check import StageMarginalCheck
from .stage_preprocess import StagePreprocess
from .stage_retrieve import StageRetrieve
from .stage_simulated import (
    StageSimulated,
    assemble_simulated_load_data,
    assemble_simulated_tracedata,
    simulated_signature,
)
from .stage_synthetic import StageSynthetic
from .stage_uncertainty import StageUncertainty
from .stage_welcome import StageWelcome
from .state import WizardState

_TITLES = [
    "1. Load",
    "2. Marginal check",
    "3. Preprocess",
    "4. Retrieve",
    "5. Dispersion",
    "6. Uncertainty",
]
_NSTAGES = len(_TITLES)
# the six linear stages live behind the welcome (0), synthetic (1) and
# simulated (2) entry pages
_STAGE_OFFSET = 3


def _result_mismatch(saved, td, retrieve_params) -> str:
    """Why a saved ``result.h5`` cannot be paired with ``td``, or ``""`` if it can.

    Reuse is only sound when the saved retrieval was fitted to *this* trace, so
    the saved measured trace and frequency axis are compared with the current
    ones element-wise rather than by shape. Both are stored exactly as the
    retrieval saw them (``trace_meas`` is the array handed to
    :func:`croak.processing.process_result`), so equality is the honest test and
    any difference — a moved delay grid, a changed filter, a new calibration —
    is grounds to re-run rather than to display a fit against data it never saw.

    A file too old to carry ``trace_meas`` cannot be verified at all, so it is
    also refused: the fast path is an optimisation, and declining it costs only
    a re-run, while wrongly taking it costs a wrong residual panel.
    """
    cur = td.trace
    meas = saved.get("trace_meas")
    if meas is None:
        return "the saved result does not store the trace it was fitted to"
    meas = np.asarray(meas, float)
    if meas.shape != np.shape(cur):
        return (
            f"the saved result is on a {meas.shape[0]}x{meas.shape[1]} grid but "
            f"this trace regrids to {np.shape(cur)[0]}x{np.shape(cur)[1]}"
        )
    peak = float(np.max(np.abs(cur))) or 1.0
    if float(np.max(np.abs(meas - cur))) > 1e-9 * peak:
        return "preprocessing no longer reproduces the trace the result was fitted to"
    w_saved = saved.get("omega")
    if w_saved is not None:
        w_cur = np.asarray(td.grid.omega, float)
        w_saved = np.asarray(w_saved, float)
        span = float(np.max(np.abs(w_cur))) or 1.0
        if w_saved.shape != w_cur.shape or (
            float(np.max(np.abs(w_saved - w_cur))) > 1e-9 * span
        ):
            return "the saved result is on a different frequency grid"
    algo = saved.get("algorithm")
    if algo is not None and str(algo) != str(retrieve_params.solver):
        return (
            f"the saved result came from {algo!s} but the loaded settings select "
            f"{retrieve_params.solver}"
        )
    return ""


class Wizard(QMainWindow):
    """Welcome menu + synthetic generator + four-stage pulse-retrieval wizard."""

    def __init__(self, state: WizardState | None = None):
        super().__init__()
        self.setWindowTitle("croak — pulse-retrieval wizard")
        # run_wizard sets this application-wide, which already covers every window
        # it opens; setting it here too means an embedder who built their own
        # QApplication still gets the icon.
        self.setWindowIcon(app_icon())
        # Un-maximised size: as the user last left it, else a default clamped to
        # the screen (run_wizard maximises on top of this).
        settings.restore_window(self)
        self.state = state or WizardState()

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # header bar: stage title (left) + a context-sensitive help button (right)
        self._header_bar = QWidget()
        self._header_bar.setStyleSheet("background:#22324a;")
        hb = QHBoxLayout(self._header_bar)
        hb.setContentsMargins(8, 4, 8, 4)
        self.header = QLabel()
        self.header.setStyleSheet("color:white; font-weight:bold;")
        self.help_btn = QPushButton("?  Help")
        # The header bar is dark (#22324a); give the button an explicit
        # high-contrast style so it doesn't disappear into the banner.
        self.help_btn.setStyleSheet(
            "QPushButton {"
            " color:white; font-weight:bold; background:#3a557a;"
            " border:1px solid white; border-radius:4px; padding:2px 10px; }"
            "QPushButton:hover { background:#4a6a96; }"
        )
        self.help_btn.setToolTip("Explain this stage's options")
        self.help_btn.clicked.connect(self._show_help)
        hb.addWidget(self.header, stretch=1)
        hb.addWidget(self.help_btn)
        layout.addWidget(self._header_bar)

        # stacked pages: welcome, synthetic, simulated, then the five linear stages
        self.stack = QStackedWidget()
        self.welcome = StageWelcome(self)
        self.synthetic = StageSynthetic(self.state, self)
        self.simulated = StageSimulated(self.state, self)
        self.stages = [
            StageLoad(self.state),
            StageMarginalCheck(self.state, self),
            StagePreprocess(self.state),
            StageRetrieve(self.state),
            StageDispersion(self.state),
            StageUncertainty(self.state),
        ]
        self.stack.addWidget(self.welcome)  # index 0
        self.stack.addWidget(self.synthetic)  # index 1
        self.stack.addWidget(self.simulated)  # index 2
        for s in self.stages:  # indices 3..8
            self.stack.addWidget(s)
        layout.addWidget(self.stack, stretch=1)

        # footer
        self.footer = QWidget()
        fl = QHBoxLayout(self.footer)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.close)
        self.menu_btn = QPushButton("Main menu")
        self.menu_btn.clicked.connect(self.show_welcome)
        self.back_btn = QPushButton("← Back")
        self.back_btn.clicked.connect(self.go_back)
        self.next_btn = QPushButton("Next →")
        self.next_btn.clicked.connect(self.go_next)
        fl.addWidget(self.cancel_btn)
        fl.addWidget(self.menu_btn)
        fl.addStretch(1)
        fl.addWidget(self.back_btn)
        fl.addWidget(self.next_btn)
        layout.addWidget(self.footer)

        self.state.stage_changed.connect(self._refresh)
        for sig in (
            self.state.load_data_changed,
            self.state.tracedata_changed,
            self.state.result_changed,
        ):
            sig.connect(self.refresh_nav)
        self.state.retarget_requested.connect(self._retarget_session)

        # Which entry the current session came from lives on the state
        # (``state.entry``): it selects the live loader params, so it is saved
        # with the session and drives Back from preprocess as well.
        # which page is on screen: "welcome", "synthetic"/"simulated" (the two
        # entry pages) or "stage" (one of the six linear stages). The entry pages
        # sit outside the linear sequence, so the footer's Back/Next cannot be
        # driven from ``state.stage`` alone.
        self._page = "welcome"
        self.show_welcome()

    # -- page chrome --------------------------------------------------------
    def _chrome(self, visible: bool) -> None:
        self._header_bar.setVisible(visible)
        self.footer.setVisible(visible)

    def _entry_page(self) -> Stage | None:
        """The entry page currently on screen, or ``None`` on any other page."""
        if self._page == "synthetic":
            return self.synthetic
        if self._page == "simulated":
            return self.simulated
        return None

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 — Qt override
        """Remember the window geometry for the next launch, then close."""
        settings.save_window(self)
        super().closeEvent(event)

    def _show_help(self) -> None:
        """Open the help dialog for the page currently on screen."""
        page = self._entry_page() or self.stages[self.state.stage - 1]
        page.show_help()

    def show_welcome(self) -> None:
        self._page = "welcome"
        self.stack.setCurrentIndex(0)
        self._chrome(False)

    def show_synthetic(self) -> None:
        self._new_session()
        self.state.entry = "synthetic"
        self._show_synthetic_page()

    def _show_synthetic_page(self) -> None:
        """Display the synthetic generator without resetting in-progress data."""
        self._show_entry_page("synthetic", self.synthetic, "Generate synthetic trace")

    def load_simulated(self) -> None:
        """Menu entry: load a numerically simulated FROG scan, then preprocess."""
        self._new_session()
        self.state.entry = "simulated"
        self._show_simulated_page()

    def _show_simulated_page(self) -> None:
        """Display the simulated-scan loader without resetting in-progress data."""
        self._show_entry_page("simulated", self.simulated, "Load simulated scan")

    def _show_entry_page(self, page: str, entry: Stage, title: str) -> None:
        """Show an entry page with the standard header/footer chrome.

        The entry pages are alternatives to Stage 1 rather than numbered stages,
        so the header carries a plain title instead of "Stage N of M"; the footer
        gives them the same bottom-right Back/Next as every other page, with Back
        returning to the welcome menu and Next running the page's
        :meth:`~croak.gui.base.Stage.advance`.
        """
        self._page = page
        self.stack.setCurrentWidget(entry)
        self._chrome(True)
        self.header.setText(title)
        entry.on_enter()
        self.refresh_nav()

    def enter_marginal_from_simulated(self) -> None:
        """Show the marginal-check stage for the loaded simulated trace.

        The simulated loader is this session's "stage 1"; it hands the trace to
        the marginal-check stage (stage 2), which then leads to preprocessing.
        """
        self.goto_stage(2)

    def enter_retrieve_from_simulated(self) -> None:
        """Jump straight to Retrieve for a raw, directly-loaded simulated trace.

        The "skip filtering & regrid" path on the simulated loader sets
        ``state.tracedata`` itself (the as-simulated trace on its native grid), so
        it bypasses the marginal-check (stage 2) and preprocess (stage 3) and
        lands on Retrieve (stage 4). ``state.entry`` stays ``"simulated"``, so Back from
        Retrieve still reaches Preprocess for a regridded comparison.
        """
        self.goto_stage(4)

    def start_load(self) -> None:
        """Menu entry 1: begin the experimental-trace workflow at Stage 1."""
        self._new_session()
        self.state.entry = "experimental"
        self.goto_stage(1)

    def load_previous(self) -> None:
        """Menu entry 2: load a previous retrieval (saved ``options.toml``).

        The entry is not chosen here: it comes from the file (``state.entry``),
        so a saved simulated session reloads through the simulated loader.
        """
        self.state.reset()  # _load_options repopulates from the file and rebuilds
        self._load_options()

    def _new_session(self) -> None:
        """Clear all data/settings and rebuild the stages for a fresh process."""
        self.state.reset()
        self._recreate_stages()

    # -- navigation ---------------------------------------------------------
    def goto_stage(self, stage: int) -> None:
        """Show a linear stage, restoring the header/footer chrome."""
        self._chrome(True)
        if self.state.stage != stage:
            self.state.stage = stage  # emits stage_changed -> _refresh
        else:
            self._refresh(stage)

    def go_next(self):
        # On an entry page "Next" is that page's own forward action (generate the
        # synthetic trace / load the simulated scan), which then routes onwards.
        entry = self._entry_page()
        if entry is not None:
            if entry.can_advance():
                entry.advance()
            return
        if self.state.stage < _NSTAGES and self.state.ready(self.state.stage):
            self.state.stage += 1

    def go_back(self):
        # Back from an entry page returns to the welcome menu (it is the session's
        # "stage 1", so there is nowhere earlier to go).
        if self._entry_page() is not None:
            self.show_welcome()
            return
        # Back from the marginal-check stage (stage 2) returns to whichever entry
        # page is this session's "stage 1": the synthetic generator or the
        # simulated loader for those entries, else the experimental Load (stage 1).
        if self.state.stage == 2 and self.state.entry == "synthetic":
            self._show_synthetic_page()
        elif self.state.stage == 2 and self.state.entry == "simulated":
            self._show_simulated_page()
        elif self.state.stage > 1:
            self.state.stage -= 1
        else:
            self.show_welcome()

    def enter_marginal_from_synthetic(self) -> None:
        """Show the marginal-check stage for the generated trace.

        The synthetic generator is this session's "stage 1"; it hands the trace
        to the marginal-check stage (stage 2), which then leads to preprocessing.
        """
        self.goto_stage(2)

    def _refresh(self, stage: int):
        self._page = "stage"
        self._chrome(True)
        idx = _STAGE_OFFSET + stage - 1
        self.stack.setCurrentIndex(idx)
        self.header.setText(f"Stage {stage} of {_NSTAGES}    —    {_TITLES[stage - 1]}")
        self.stages[stage - 1].on_enter()
        self.refresh_nav()

    def refresh_nav(self):
        """Re-label and re-enable the footer's Back/Next for the current page.

        Public because the entry pages call it when their readiness or forward
        action changes (a file previews, "skip filtering & regrid" is toggled).
        """
        self.back_btn.setEnabled(True)  # Back at Stage 1 returns to the menu
        entry = self._entry_page()
        if entry is not None:
            self.next_btn.setText(entry.next_label())
            self.next_btn.setEnabled(entry.can_advance())
            return
        stage = self.state.stage
        self.next_btn.setEnabled(stage < _NSTAGES and self.state.ready(stage))
        self.next_btn.setText("Finish" if stage == _NSTAGES else "Next →")

    # -- session persistence ------------------------------------------------
    def _load_options(self):
        fn, _ = QFileDialog.getOpenFileName(
            self, "Load options.toml", filter="TOML (*.toml)"
        )
        if not fn:
            return
        self.state.from_options(save.load_options(fn))
        # The control widgets are bound one-way (widget -> params), so mutating
        # the params alone leaves the UI showing the old values. Rebuild the
        # stage pages from the freshly loaded state so every control reflects it.
        self._rebuild_stages()
        # A sibling result.h5 (written by the Retrieve stage's Save…) lets us
        # reload the retrieved result without re-running the solver.
        result_path = os.path.join(os.path.dirname(fn), "result.h5")
        # Replay the saved settings through the pipeline so the GUI returns to
        # (as near as possible) the saved state: same data, options and plots.
        self._restore_session(result_path if os.path.exists(result_path) else None)

    def _restore_session(self, result_path: str | None = None):
        """Reload the data and re-run preprocess, then retrieve (or reload a result).

        Reproduces the saved session end-to-end: it reloads the trace through the
        loader the session was saved with (``state.entry``), without reseeding the
        preprocess defaults so the saved windowing/filtering survives, and regrids
        with the saved settings. If ``result_path`` points to a saved ``result.h5``
        whose grid matches the regridded trace, the retrieval is **rehydrated from
        it** (no solver re-run); otherwise the retrieval is re-run to reproduce the
        plots. If no trace file is set (or it can't be reloaded) it stops
        gracefully with the controls still populated from the file.
        """
        st = self.state
        if st.entry == "synthetic":
            # The synthetic generator's settings are not part of the saved session
            # (they live only in its controls), so there is no trace to rebuild.
            self._show_synthetic_page()
            self.synthetic.set_status(
                "Options loaded, but the synthetic generator's settings are not "
                "saved with a session — regenerate the trace to replay the rest."
            )
            return
        loaded = (
            self._restore_simulated_load()
            if st.entry == "simulated"
            else self._restore_experimental_load()
        )
        if not loaded:
            return
        # 2) regrid/filter with the saved preprocess settings — unless the loader
        # already built the trace on its native grid (the raw simulated path).
        if st.tracedata is None:
            # stage order: Load(1), Marginal check(2), Preprocess(3), Retrieve(4), …
            preproc_stage = self.stages[2]
            preproc_stage._seed_all()
            preproc_stage.apply(block=True)
        if st.tracedata is None:
            self.goto_stage(3)
            self.set_status(
                "Options loaded and data restored; preprocess did not complete."
            )
            return
        # 3) reproduce the retrieval: rehydrate the saved result if we can (no
        # re-run), else re-run the solver to reproduce the plots.
        if result_path is not None and self._reload_saved_result(result_path):
            self.goto_stage(4)
            self.set_status("Options and saved retrieval loaded — no re-run needed.")
            return
        self.goto_stage(4)
        # Say why the saved result was not reused, so a re-run that returns
        # slightly different numbers from the saved ones is not a mystery.
        note = getattr(self, "_stale_result_note", "")
        if result_path is not None and note:
            self.set_status(f"Options loaded; re-running because {note}.")
        self._stale_result_note = ""
        self.stages[3]._run()

    def _restore_experimental_load(self) -> bool:
        """Reload the measured trace with the saved dataset/unit selection.

        Returns ``True`` when ``state.load_data`` is set (the caller continues
        with preprocess/retrieve); ``False`` after reporting why it stopped.
        """
        st = self.state
        if not st.load.frog_path:
            self.goto_stage(1)
            self.set_status("Options loaded. Choose a trace file and run the stages.")
            return False
        try:
            data = assemble_load_data(st.load)
        except Exception as exc:  # pragma: no cover - surfaced in the GUI
            self.goto_stage(1)
            self.set_status(f"Options loaded, but reloading the trace failed: {exc}")
            return False
        # set_load_data (deliberately, unlike StageLoad._load_preview) does NOT
        # reseed the preprocess defaults, so the saved grid/window/filter values
        # are kept intact. Record the trace signature (and recover the data-seeded
        # defaults for the Reset button) so a later reload of the same trace also
        # preserves the restored settings.
        st.set_load_data(data)
        st.trace_sig = trace_signature(st.load)
        st.preproc_defaults = compute_preproc_defaults(data)
        self.stages[0]._draw(data)
        return True

    def _restore_simulated_load(self) -> bool:
        """Reload the simulated scan with the saved loader options.

        The simulated counterpart of :meth:`_restore_experimental_load`, mirroring
        :meth:`~croak.gui.stage_simulated.StageSimulated.advance` but keeping the
        saved preprocess settings. Its "skip filtering & regrid" option builds the
        retrieval trace here, on the simulation's native grid, exactly as the
        loader page does. Returns ``True`` when the data was restored.
        """
        st = self.state
        if not st.simulated.frog_path:
            self._show_simulated_page()
            self.simulated.set_status(
                "Options loaded. Choose a simulated scan file and run the stages."
            )
            return False
        # The loader page's controls are built from the defaults, so reflect the
        # restored options in them before anything re-reads them.
        self.simulated.seed_widgets(st.simulated)
        try:
            data = assemble_simulated_load_data(st.simulated)
        except Exception as exc:  # pragma: no cover - surfaced in the GUI
            self._show_simulated_page()
            self.simulated.set_status(
                f"Options loaded, but reloading the simulated scan failed: {exc}"
            )
            return False
        st.set_load_data(data)
        st.trace_sig = simulated_signature(st.simulated)
        st.preproc_defaults = compute_preproc_defaults(data)
        self.simulated._draw(data)
        if st.simulated.raw_direct:
            try:
                st.set_tracedata(assemble_simulated_tracedata(st.simulated))
            except Exception as exc:  # pragma: no cover - surfaced in the GUI
                self._show_simulated_page()
                self.simulated.set_status(
                    f"Options loaded, but the raw simulated trace failed: {exc}"
                )
                return False
        return True

    def _reload_saved_result(self, result_path: str) -> bool:
        """Rehydrate a saved ``result.h5`` onto the regridded trace (no re-run).

        Returns ``True`` when the result was loaded and set on the state (the
        Retrieve stage draws it on entry); ``False`` (caller re-runs) when the
        file is unreadable or its grid does not match the regridded trace.

        The check is not a formality, and it is deliberately a check on the
        *data*, not just on the array shapes. Everything downstream pairs the
        saved retrieval with the freshly preprocessed trace — the Retrieve
        stage's residual panel divides one by the other — so the saved result is
        only safe to adopt if that trace is the one it was actually fitted to.
        A shape test alone would not establish this: any preprocessing change
        that alters trace values without changing its size (a filtering default,
        the edge taper, a recalibration) would slip through and be displayed as
        a consistent fit when it is not. See :func:`_result_mismatch`.
        """
        td = self.state.tracedata
        if td is None:
            return False
        try:
            saved = save.load_result(result_path)
            result = save.result_from_saved(saved, grid=td.grid)
        except KeyError, ValueError, OSError:
            return False
        self._stale_result_note = _result_mismatch(saved, td, self.state.retrieve)
        if self._stale_result_note:
            return False
        # processed is recomputed by the Retrieve stage's on_enter/_update_view.
        self.state.set_result(result)
        return True

    def _retarget_session(self, base_dir: str) -> None:
        """Point the current session at a new dataset folder and replay it.

        Triggered by the Retrieve stage's "Retarget…" button. Rewrites the
        dataset-folder load paths to ``base_dir`` (keeping the shared calibration
        files fixed) via :func:`croak.session.retarget`, refreshes the controls,
        then reruns load → preprocess → retrieve with the current settings — so a
        working retrieval can be applied to the next, similarly-laid-out dataset.

        Only experimental sessions have such a folder; for the others
        :func:`croak.session.retarget` raises and the reason is shown rather than
        replaying the session unchanged.
        """
        try:
            moved = retarget(self.state.session_options(), base_dir)
        except ValueError as exc:
            self.set_status(str(exc))
            return
        # Copy the retargeted paths onto the live load params in place (the stage
        # widgets bind to that instance); _recreate_stages then refreshes them.
        for field in self.state.load.dataset_path_fields():
            setattr(self.state.load, field, getattr(moved.load, field))
        self._recreate_stages()
        self._restore_session()

    def _recreate_stages(self):
        """Rebuild the six stage widgets from ``state`` (no navigation).

        The welcome, synthetic and simulated pages (indices 0/1/2) are kept; only
        the six linear stages behind them are recreated so their controls rebind
        to the current parameter dataclasses.
        """
        for s in self.stages:
            self.stack.removeWidget(s)
            s.deleteLater()
        self.stages = [
            StageLoad(self.state),
            StageMarginalCheck(self.state, self),
            StagePreprocess(self.state),
            StageRetrieve(self.state),
            StageDispersion(self.state),
            StageUncertainty(self.state),
        ]
        for s in self.stages:
            self.stack.addWidget(s)

    def _rebuild_stages(self):
        """Recreate the six stage pages and navigate to the current stage."""
        self._recreate_stages()
        self.goto_stage(self.state.stage)

    def set_status(self, text: str):
        self.stages[self.state.stage - 1].set_status(text)
