"""Stage 5 — estimate the uncertainty of the retrieved FWHM.

Runs, off the GUI thread, either a *statistical* resampling bootstrap (parametric
Monte-Carlo, or delay/frequency data resampling) or the *systematic*
substrate-thickness propagation (``thickness``), and shows the FWHM distribution with
its 68 %/95 % intervals plus a temporal confidence band. The bootstraps report the
statistical precision only (calibration/geometry/model systematics excluded); the
thickness method propagates the measured substrate-thickness prior, the one
systematic that dominates short dispersive-retrieval pulses.
"""

from __future__ import annotations

import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QWidget,
)

from .. import plotting, save
from ..processing import TruthPulse
from ..uncertainty import combine_uncertainties
from .base import Stage
from .canvas import MplCanvas, check, combo, dspin, group, spin, with_toolbar

_METHODS = ["parametric", "delay", "frequency", "thickness", "covariance"]
_NOISE_SOURCES = ["residual", "manual"]
_INTERVALS = ["bias-corrected", "percentile"]


def _truth_fwhm(truth: TruthPulse | None) -> float | None:
    """Intensity FWHM (s) of the known ground-truth pulse, or ``None``."""
    return None if truth is None else truth.fwhm


class StageUncertainty(Stage):
    """Estimate and display the statistical uncertainty of the retrieved FWHM."""

    HELP_TITLE = "Uncertainty"
    HELP_INTRO = (
        "Bootstrap a statistical error bar on the retrieved FWHM by resampling. "
        "Parametric: add detector noise to the best-fit trace and re-retrieve. "
        "Delay/frequency: resample the measured trace. The interval is the "
        "statistical precision only — calibration and model systematics are not "
        "included. This re-runs the retrieval many times and can be slow."
    )
    HELP = [
        (
            "Bootstrap",
            [
                (
                    "Method",
                    "Statistical: parametric Monte-Carlo, or delay/frequency "
                    "resampling. Systematic: thickness — propagate the measured "
                    "substrate-thickness uncertainty through the dispersive retrieval.",
                ),
                ("Replicates", "Number of bootstrap/Monte-Carlo retrievals (≈200)."),
                ("Noise from", "Parametric noise: the fit residual, or a manual σ."),
                (
                    "Thickness σ (µm)",
                    "1σ of the measured substrate thickness (the 'thickness' "
                    "method); the central value comes from the retrieval's "
                    "thickness. Yields a systematic FWHM error bar.",
                ),
                (
                    "warm-start replicates",
                    "Seed each replicate from the full-data solution (isolates "
                    "measurement variance from optimiser basin-hopping).",
                ),
            ],
        ),
    ]

    def __init__(self, state):
        super().__init__(state)
        u = self.state.uncertainty
        self._worker = None

        box, form = group("Bootstrap")
        self.method = combo(_METHODS, u.method, self._on_method)
        form.addRow("Method", self.method)
        self.n_resamples = spin(10, 5000, u.n_resamples, self._on_n, step=10)
        form.addRow("Replicates", self.n_resamples)
        self.noise_source = combo(_NOISE_SOURCES, u.noise_source, self._on_noise_source)
        form.addRow("Noise from", self.noise_source)
        self.noise_sigma = dspin(
            0.0, 1.0, u.noise_sigma, self._on_sigma, decimals=4, step=0.005
        )
        form.addRow("Manual σ (frac)", self.noise_sigma)
        self.thickness_sigma = dspin(
            0.0,
            1000.0,
            u.thickness_sigma_um,
            self._on_thickness_sigma,
            decimals=3,
            step=0.01,
        )
        form.addRow("Thickness σ (µm)", self.thickness_sigma)
        self.warm_start = check("warm-start replicates", u.warm_start, self._on_warm)
        form.addRow(self.warm_start)
        self.collect_profiles = check(
            "temporal confidence band", u.collect_profiles, self._on_profiles
        )
        form.addRow(self.collect_profiles)
        self.propagate = check(
            "at dispersion-stage point",
            u.propagate_to_dispersion_point,
            self._on_propagate,
        )
        self.propagate.setToolTip(
            "Report the FWHM/band at the beamline point defined by the Dispersion "
            "stage: propagate every replicate through that dispersion first."
        )
        form.addRow(self.propagate)
        self.interval = combo(_INTERVALS, u.interval, self._on_interval)
        form.addRow("Interval", self.interval)
        self.controls.addWidget(box)

        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        self.run_btn = QPushButton("Estimate")
        self.run_btn.clicked.connect(self._run)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop)
        self.stop_btn.setEnabled(False)
        rl.addWidget(self.run_btn)
        rl.addWidget(self.stop_btn)
        self.controls.addWidget(row)

        # Retained estimates: each completed run is kept so several independent
        # contributions can be combined into a total before saving. Tick the ones to
        # combine (don't mix redundant statistical estimators); click one to view it.
        est_box, est_form = group("Estimates")
        self.est_list = QListWidget()
        self.est_list.setMaximumHeight(110)
        self.est_list.currentItemChanged.connect(self._on_select_estimate)
        self.est_list.itemChanged.connect(lambda _item: self._sync_enabled())
        est_form.addRow(self.est_list)
        est_row = QWidget()
        erl = QHBoxLayout(est_row)
        erl.setContentsMargins(0, 0, 0, 0)
        self.combine_btn = QPushButton("Combine ticked")
        self.combine_btn.setToolTip(
            "Combine the ticked independent estimates in quadrature into a total."
        )
        self.combine_btn.clicked.connect(self._combine)
        self.save_btn = QPushButton("Save…")
        self.save_btn.setToolTip(
            "Append all estimates (intervals + temporal bands) to result.h5."
        )
        self.save_btn.clicked.connect(self._save)
        erl.addWidget(self.combine_btn)
        erl.addWidget(self.save_btn)
        est_form.addRow(est_row)
        self.controls.addWidget(est_box)

        self.controls.addStretch(1)
        self.controls.addWidget(self.status_label)
        self._sync_enabled()

        self.canvas = MplCanvas(figsize=(11, 4.5))
        self.set_plot_area(with_toolbar(self.canvas))
        self.apply_help()

    # -- control bindings ---------------------------------------------------
    def _on_method(self, text):
        self.state.uncertainty.method = text
        self._sync_enabled()

    def _on_n(self, value):
        self.state.uncertainty.n_resamples = int(value)

    def _on_noise_source(self, text):
        self.state.uncertainty.noise_source = text
        self._sync_enabled()

    def _on_sigma(self, value):
        self.state.uncertainty.noise_sigma = float(value)

    def _on_thickness_sigma(self, value):
        self.state.uncertainty.thickness_sigma_um = float(value)

    def _on_warm(self, value):
        self.state.uncertainty.warm_start = bool(value)

    def _on_profiles(self, value):
        self.state.uncertainty.collect_profiles = bool(value)

    def _on_propagate(self, value):
        self.state.uncertainty.propagate_to_dispersion_point = bool(value)

    def _on_interval(self, text):
        self.state.uncertainty.interval = text

    def _sync_enabled(self):
        """Enable the controls relevant to the chosen method and stored estimates."""
        u = self.state.uncertainty
        # The parametric bootstrap and the analytic covariance both use a noise model.
        needs_noise = u.method in ("parametric", "covariance")
        self.noise_source.setEnabled(needs_noise)
        self.noise_sigma.setEnabled(needs_noise and u.noise_source == "manual")
        self.thickness_sigma.setEnabled(u.method == "thickness")
        self.combine_btn.setEnabled(len(self._checked_methods()) >= 2)
        self.save_btn.setEnabled(bool(self.state.uncertainty_results))

    # -- retained estimates -------------------------------------------------
    def _label(self, method: str, u) -> str:
        """One-line ``method: x ± y fs`` label for the estimates list."""
        return (
            f"{method}: {u.point_estimate / 1e-15:.2f} ± {u.plus_minus / 1e-15:.2f} fs"
        )

    def _refresh_estimates(self):
        """Rebuild the estimates list from ``state.uncertainty_results``."""
        self.est_list.blockSignals(True)
        self.est_list.clear()
        for method, u in self.state.uncertainty_results.items():
            item = QListWidgetItem(self._label(method, u))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            # tick the components by default, leave a previous "combined" unticked
            state = (
                Qt.CheckState.Unchecked
                if method == "combined"
                else Qt.CheckState.Checked
            )
            item.setCheckState(state)
            item.setData(Qt.ItemDataRole.UserRole, method)
            self.est_list.addItem(item)
        self.est_list.blockSignals(False)
        self._sync_enabled()

    def _checked_methods(self) -> list[str]:
        """Methods whose list item is ticked (combine inputs)."""
        out = []
        for i in range(self.est_list.count()):
            item = self.est_list.item(i)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                out.append(item.data(Qt.ItemDataRole.UserRole))
        return out

    def _on_select_estimate(self, current, _previous):
        """Display the clicked estimate."""
        if current is None:
            return
        method = current.data(Qt.ItemDataRole.UserRole)
        u = self.state.uncertainty_results.get(method)
        if u is not None:
            self._draw(u)
            self.set_status(u.summary())

    def _combine(self):
        methods = self._checked_methods()
        results = [
            self.state.uncertainty_results[m]
            for m in methods
            if m in self.state.uncertainty_results
        ]
        if len(results) < 2:
            self.set_status("Tick at least two estimates to combine.")
            return
        try:
            combined = combine_uncertainties(results)
        except Exception as exc:  # pragma: no cover - surfaced in the GUI
            self.set_status(f"Combine failed: {exc}")
            return
        self.state.set_uncertainty_result(combined)  # stored under "combined"
        self._refresh_estimates()
        self._draw(combined)
        self.set_status(combined.summary())

    def _save(self):
        if self.state.result is None or not self.state.uncertainty_results:
            self.set_status("Run an estimate first.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Select output folder")
        if not folder:
            return
        path = os.path.join(folder, "result.h5")
        save.save_result(
            self.state.result,
            path,
            processed=self.state.processed,
            truth=self.state.truth,
            force=True,
        )
        save.save_uncertainty(self.state.uncertainty_results, path, force=True)
        save.save_options(self.state.to_options(), os.path.join(folder, "options.toml"))
        self.canvas.figure.savefig(os.path.join(folder, "uncertainty.pdf"), dpi=300)
        self.set_status(
            f"Saved result.h5 (+uncertainty), options.toml, uncertainty.pdf to {folder}"
        )

    # -- lifecycle ----------------------------------------------------------
    def on_enter(self):
        if self.state.result is None:
            self.run_btn.setEnabled(False)
            self.set_status("Retrieve a pulse first.")
            return
        self.run_btn.setEnabled(True)
        self._refresh_estimates()
        if self.state.uncertainty_result is not None:
            self._draw(self.state.uncertainty_result)
            self.set_status(self.state.uncertainty_result.summary())
        else:
            self.set_status("Choose options and click Estimate.")

    # -- bootstrap ----------------------------------------------------------
    def _run(self):
        if self.state.result is None or self.state.tracedata is None:
            self.set_status("Retrieve a pulse first.")
            return
        # Import here so the rest of the GUI imports even if PyQt-less envs build
        # the worker lazily; the worker pulls in the (heavier) solver stack.
        from .worker import UncertaintyWorker

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.set_status("Estimating…")
        self._worker = UncertaintyWorker(
            self.state.result,
            self.state.tracedata,
            self.state.retrieve,
            self.state.uncertainty,
            dispersion=self.state.dispersion,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _stop(self):
        if self._worker is not None:
            self._worker.request_stop()

    def _on_progress(self, done, total, error):
        self.set_status(f"replicate {done}/{total}:  R = {error:.4%}")

    def _on_finished(self, uresult):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.state.set_uncertainty_result(uresult)
        self._refresh_estimates()
        self._draw(uresult)
        self.set_status(uresult.summary())

    def _on_failed(self, message):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if message == "stopped":
            self.set_status("Stopped.")
        else:
            self.set_status(f"Uncertainty estimation failed: {message}")

    def _draw(self, uresult):
        truth = _truth_fwhm(self.state.truth)
        self.canvas.draw_with(
            lambda fig: plotting.plot_uncertainty(uresult, truth=truth, fig=fig)
        )
