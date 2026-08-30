"""Stage 2 — filter and regrid the trace; preview the six-panel filter view.

Three control groups:

* **Grid** — retrieval wavelength range; ``trange`` (time-grid range, always on,
  default 2× the scan delay range); ``τ range`` (the delay range to **use** —
  a hard crop of the delay axis); delay subsampling; time resampling; and the
  resampling pre-filter.
* **Windowing** — measurement spectral window (λm) and delay filtering window
  (τm, a smooth Planck taper); both always on, defaulting to the full
  wavelength/delay extents and narrowed to taste.
* **Filtering** — signal-aware de-fringing (Takeda FTSI) and arPLS baseline
  removal, the legacy fringe-notch / DC / low-pass delay filters, and
  thresholding (the SHG marginal correction lives on the marginal-check stage).

The three delay controls are distinct: ``trange`` sets the time grid, ``τ range``
selects which delays are retrieved, and ``τm`` filters (windows) the delays.

Changes recompute automatically (debounced); there is no Apply button.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TypedDict

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QWidget

from .. import io, plotting
from ..preprocess import suggested_tau_exclude
from ..session import pipeline
from .base import Stage
from .canvas import (
    MplCanvas,
    add_wide_row,
    check,
    combo,
    dspin,
    group,
    spin,
    with_toolbar,
)
from .widgets import RangeControl, SciSpinBox
from .worker import PreprocessWorker


# Range sliders: laid out with ``add_wide_row`` so they span the full controls
# column, at 0.1-unit resolution (slider_scale 10) so they are easy to nudge
# precisely. A TypedDict keeps the per-key types (``decimals`` is int) across the
# ``**_RANGE_KW`` splat.
class _RangeKW(TypedDict):
    decimals: int
    step: float
    slider_scale: float


_RANGE_KW: _RangeKW = {"decimals": 1, "step": 0.1, "slider_scale": 10.0}


class StagePreprocess(Stage):
    HELP_TITLE = "Preprocess"
    HELP_INTRO = (
        "Filter and regrid the measured trace onto the retrieval grid. The three "
        "delay controls are distinct: trange sets the time grid, τ range selects "
        "which delays are retrieved, and τm windows (tapers) the delays. Changes "
        "recompute automatically."
    )
    HELP = [
        (
            "Grid",
            [
                (
                    "λ range (nm)",
                    "Retrieval wavelength-grid extent (the fundamental band). "
                    "Defaults to the trace extents scaled by the interaction's "
                    "carrier factor; widen for spectral headroom.",
                ),
                (
                    "trange (fs)",
                    "Total span of the retrieval time grid. Defaults to 2× the "
                    "scan delay range; larger gives finer frequency resolution.",
                ),
                (
                    "τ range (fs)",
                    "The delay range actually retrieved — a hard crop of the "
                    "delay axis.",
                ),
                ("Subsample τ", "Use every Nth delay to speed up retrieval."),
                ("factor", "Delay subsampling factor N (keep every Nth delay)."),
                (
                    "Resample t",
                    "Resample the trace onto the regular retrieval time grid.",
                ),
                (
                    "Prefilter (anti-alias)",
                    "Low-pass filter before resampling/subsampling to avoid aliasing.",
                ),
            ],
        ),
        (
            "Windowing",
            [
                (
                    "λm (nm)",
                    "Measurement spectral window: wavelengths outside this range "
                    "are tapered out of the trace. Defaults to the full extent.",
                ),
                (
                    "τm (fs)",
                    "Delay filtering window (a smooth Planck taper) applied to the "
                    "trace. Defaults to the full scan range.",
                ),
            ],
        ),
        (
            "Filtering",
            [
                (
                    "De-fringe (Takeda)",
                    "Remove interferometric fringes by a smooth low-pass that "
                    "keeps the DC band below the known optical carrier f_c=c/λ "
                    "(Takeda FTSI) — no ringing. Mutually exclusive with the "
                    "legacy fringe notch.",
                ),
                (
                    "carrier fraction",
                    "De-fringe cutoff as a fraction of the carrier f_c (0–1). "
                    "Lower cuts more fringe but risks the compact signal; ~0.5.",
                ),
                (
                    "on alias",
                    "Behaviour when the delay step is too coarse and the fringe "
                    "carrier aliases: clamp (best-effort, never fails), warn, or "
                    "raise (surface the under-sampling).",
                ),
                (
                    "Baseline (arPLS)",
                    "Remove the broad, slowly-varying leakage background along "
                    "delay per wavelength row by asymmetric least squares — "
                    "handles an asymmetric (before/after τ=0) background. Applied "
                    "after de-fringing.",
                ),
                (
                    "smoothness",
                    "Baseline stiffness. Larger ⇒ flatter; too small bends into "
                    "and removes the signal. ~1e2 suits a broad noisy measured "
                    "background; a clean/simulated or narrow-window/few-sample "
                    "trace needs ~1e4+ (or no baseline at all).",
                ),
                ("ratio", "arPLS convergence tolerance on the baseline change."),
                (
                    "τ exclude (fs)",
                    "Hold out |τ| < this from the baseline fit so the tall τ≈0 "
                    "peak cannot pull the baseline up into it. 0 disables the "
                    "hold-out, and the baseline then eats part of the signal — "
                    "worst on rows that constrain the fit poorly, so it appears "
                    "as a thin stripe at one wavelength, only with baseline "
                    "removal on. 'Auto' sets twice the delay-marginal FWHM.",
                ),
                (
                    "Clip ≥ 0",
                    "Clip the trace non-negative after baseline subtraction "
                    "(off by default keeps small negatives, the MLE-correct choice).",
                ),
                (
                    "Filter fringes (legacy)",
                    "Legacy: remove spectral-interference fringes by a hard "
                    "band-stop notch (superseded by De-fringe above).",
                ),
                (
                    "Filter DC (legacy)",
                    "Legacy: remove the delay-direction DC (zero-frequency) "
                    "component (superseded by Baseline above).",
                ),
                (
                    "Δτmin (fs)",
                    "Smallest delay feature kept by the fringe/low-pass delay "
                    "filter (0 disables it).",
                ),
                (
                    "Threshold",
                    "Zero out trace values below a noise threshold (the only "
                    "step that removes negatives; off by default).",
                ),
                (
                    "threshold",
                    "Noise threshold as a fraction of the peak (e.g. 5e-3); "
                    "stepping moves within the current decade.",
                ),
            ],
        ),
    ]

    def __init__(self, state):
        super().__init__(state)
        p = state.preproc
        self._seeding = False
        # Off-thread preprocessing: the plotting redraw (not the math) is the
        # bottleneck, but both used to block the UI thread on every change.
        self._view = plotting.FrogFilterView()
        self._req_id = 0  # bumped per recompute; stale worker results are ignored
        self._workers: list[PreprocessWorker] = []
        self._pending: tuple | None = None  # (params, data) for the exact refine
        # Debounce: coalesce rapid slider/spin changes into one recompute.
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self.apply)

        # -- Grid --------------------------------------------------------
        box, form = group("Grid")
        self.lam_range = RangeControl(
            p.lam_bound_min_nm,
            p.lam_bound_max_nm,
            p.lam_min_nm,
            p.lam_max_nm,
            **_RANGE_KW,
        )
        self.lam_range.changed.connect(self._on_lam_range)
        add_wide_row(form, "λ range (nm)", self.lam_range)
        self.trange_spin = dspin(
            10, 1e6, p.trange_fs, self._set("trange_fs"), decimals=0
        )
        form.addRow("trange (fs)", self.trange_spin)
        self.tau_range = RangeControl(
            p.tau_bound_min_fs,
            p.tau_bound_max_fs,
            p.tau_min_fs,
            p.tau_max_fs,
            **_RANGE_KW,
        )
        self.tau_range.changed.connect(self._on_tau_range)
        add_wide_row(form, "τ range (fs)", self.tau_range)
        self.subsample_check = check(
            "Subsample τ", p.use_subsample, self._set("use_subsample")
        )
        form.addRow(self.subsample_check)
        self.factor_spin = spin(1, 16, p.subsample_tau, self._set("subsample_tau"))
        form.addRow("factor", self.factor_spin)
        self.resample_check = check("Resample t", p.resample_t, self._set("resample_t"))
        form.addRow(self.resample_check)
        self.prefilter_check = check(
            "Prefilter (anti-alias)", p.prefilter, self._set("prefilter")
        )
        form.addRow(self.prefilter_check)
        self.controls.addWidget(box)

        # -- Windowing ---------------------------------------------------
        box, form = group("Windowing")
        self.lamm_range = RangeControl(
            p.lamm_bound_min_nm,
            p.lamm_bound_max_nm,
            p.lamm_min_nm,
            p.lamm_max_nm,
            **_RANGE_KW,
        )
        self.lamm_range.changed.connect(self._on_lamm_range)
        add_wide_row(form, "λm (nm)", self.lamm_range)
        self.taum_range = RangeControl(
            p.tau_bound_min_fs,
            p.tau_bound_max_fs,
            p.taum_min_fs,
            p.taum_max_fs,
            **_RANGE_KW,
        )
        self.taum_range.changed.connect(self._on_taum_range)
        add_wide_row(form, "τm (fs)", self.taum_range)
        self.controls.addWidget(box)

        # -- Filtering ---------------------------------------------------
        box, form = group("Filtering")

        # Signal-aware cleaning (de-fringe + baseline), above the legacy filters.
        # De-fringe (Takeda FTSI) — a smooth carrier-relative low-pass.
        self.defringe_check = check(
            "De-fringe (Takeda)", p.defringe, self._on_defringe_toggled
        )
        self.defringe_check.setToolTip(
            "Remove interferometric fringes by a smooth low-pass that keeps the "
            "DC band below the optical carrier f_c=c/λ (Takeda FTSI). Mutually "
            "exclusive with the legacy fringe notch."
        )
        form.addRow(self.defringe_check)
        self.defringe_frac_spin = dspin(
            0.0,
            1.0,
            p.defringe_fraction,
            self._set("defringe_fraction"),
            decimals=2,
            step=0.05,
        )
        self.defringe_frac_spin.setToolTip(
            "De-fringe low-pass cutoff as a fraction of the carrier f_c (0–1). "
            "Lower removes more fringe but risks the compact signal; ~0.5 is a "
            "good start."
        )
        self.defringe_frac_spin.setEnabled(p.defringe)
        form.addRow("carrier fraction", self.defringe_frac_spin)
        self.defringe_alias_combo = combo(
            ("raise", "warn", "clamp"),
            p.defringe_on_alias,
            self._set("defringe_on_alias"),
        )
        self.defringe_alias_combo.setToolTip(
            "What to do if the delay step is too coarse and the fringe carrier "
            "aliases: 'clamp' (best-effort, never fails), 'warn', or 'raise' "
            "(surface the under-sampling as an error)."
        )
        self.defringe_alias_combo.setEnabled(p.defringe)
        form.addRow("on alias", self.defringe_alias_combo)

        # Baseline removal (arPLS) — broad, asymmetric leakage background.
        self.baseline_check = check(
            "Baseline (arPLS)", p.baseline, self._on_baseline_toggled
        )
        self.baseline_check.setToolTip(
            "Remove the broad, slowly-varying leakage background along delay per "
            "wavelength row by asymmetric least squares (arPLS). Handles an "
            "asymmetric (before/after τ=0) baseline; applied after de-fringing."
        )
        form.addRow(self.baseline_check)
        self.baseline_smooth_spin = SciSpinBox(
            1.0, 1e6, p.baseline_smoothness, self._set("baseline_smoothness")
        )
        self.baseline_smooth_spin.setToolTip(
            "Baseline stiffness λ. Larger ⇒ flatter baseline; too small bends into "
            "(and removes) the signal itself — the dominant failure. ~1e2 suits a "
            "broad, noisy measured background; a clean/simulated trace, a narrow "
            "delay window, or few delay samples needs a much higher value (~1e4+), "
            "and a clean simulation has no background to remove (leave baseline off)."
        )
        self.baseline_smooth_spin.setEnabled(p.baseline)
        form.addRow("smoothness", self.baseline_smooth_spin)
        self.baseline_ratio_spin = SciSpinBox(
            1e-4, 1.0, p.baseline_ratio, self._set("baseline_ratio")
        )
        self.baseline_ratio_spin.setToolTip(
            "arPLS convergence tolerance on the relative baseline change per "
            "iteration (smaller ⇒ more iterations)."
        )
        self.baseline_ratio_spin.setEnabled(p.baseline)
        form.addRow("ratio", self.baseline_ratio_spin)
        self.baseline_tauex_spin = dspin(
            0.0,
            1000.0,
            p.baseline_tau_exclude_fs,
            self._set("baseline_tau_exclude_fs"),
            decimals=1,
        )
        self.baseline_tauex_spin.setToolTip(
            "Hold out |τ| < this (fs) from the baseline fit so the tall τ≈0 peak "
            "cannot pull the baseline up into it. 0 disables the hold-out, which "
            "lets the baseline eat part of the signal — worst on rows that "
            "constrain the fit poorly, so it shows up as a thin stripe at one "
            "wavelength that appears only with baseline removal on. 'Auto' sets "
            "twice the delay-marginal FWHM, which fixes it."
        )
        self.baseline_tauex_spin.setEnabled(p.baseline)
        # Auto beside the spin, as on the dispersion stage's gas path length.
        self.baseline_tauex_auto = QPushButton("Auto")
        self.baseline_tauex_auto.setToolTip(
            "Set the hold-out to twice the delay-marginal FWHM of the loaded trace."
        )
        self.baseline_tauex_auto.clicked.connect(self._auto_tau_exclude)
        self.baseline_tauex_auto.setEnabled(p.baseline)
        tauex_row = QWidget()
        tauex_layout = QHBoxLayout(tauex_row)
        tauex_layout.setContentsMargins(0, 0, 0, 0)
        tauex_layout.addWidget(self.baseline_tauex_spin, stretch=1)
        tauex_layout.addWidget(self.baseline_tauex_auto)
        form.addRow("τ exclude (fs)", tauex_row)
        self.baseline_clip_check = check(
            "Clip ≥ 0", p.baseline_clip, self._set("baseline_clip")
        )
        self.baseline_clip_check.setToolTip(
            "Clip the trace to be non-negative after baseline subtraction. Off by "
            "default keeps small negatives (the MLE-correct choice)."
        )
        self.baseline_clip_check.setEnabled(p.baseline)
        form.addRow(self.baseline_clip_check)

        # Legacy filters (superseded by de-fringe / baseline above).
        self.fringes_check = check(
            "Filter fringes (legacy)", p.filter_fringes, self._on_fringes_toggled
        )
        form.addRow(self.fringes_check)
        self.dc_check = check("Filter DC (legacy)", p.filter_dc, self._set("filter_dc"))
        form.addRow(self.dc_check)
        self.dtau_spin = dspin(
            0, 1000, p.dtau_min_fs, self._set("dtau_min_fs"), decimals=2
        )
        form.addRow("Δτmin (fs)", self.dtau_spin)
        self.threshold_check = check(
            "Threshold", p.use_threshold, self._set("use_threshold")
        )
        form.addRow(self.threshold_check)
        self.threshold_spin = SciSpinBox(0.0, 1.0, p.threshold, self._set("threshold"))
        form.addRow("threshold", self.threshold_spin)
        self.controls.addWidget(box)

        self.reset_btn = QPushButton("Reset defaults")
        self.reset_btn.setToolTip(
            "Restore the data-seeded preprocess defaults for the loaded trace "
            "(as if this stage were re-initialised)."
        )
        self.reset_btn.clicked.connect(self._reset_defaults)
        self.controls.addWidget(self.reset_btn)
        self.controls.addWidget(self.status_label)
        self.controls.addStretch(1)

        self.canvas = MplCanvas(figsize=(9, 5))
        self.set_plot_area(with_toolbar(self.canvas))

        self.apply_help()

    # -- helpers ------------------------------------------------------------
    def _schedule(self):
        """Queue a debounced recompute (no-op while seeding the widgets)."""
        if not self._seeding:
            self._timer.start()

    def _set(self, attr):
        def setter(value):
            setattr(self.state.preproc, attr, value)
            self._schedule()

        return setter

    def _on_defringe_toggled(self, checked):
        """De-fringe toggle: gate its controls, exclude the legacy fringe notch."""
        self.state.preproc.defringe = checked
        self.defringe_frac_spin.setEnabled(checked)
        self.defringe_alias_combo.setEnabled(checked)
        # filter_frog_trace raises if both fringe removers are on — keep them
        # mutually exclusive in the UI.
        if checked and self.fringes_check.isChecked():
            self.fringes_check.setChecked(False)
        self._schedule()

    def _on_fringes_toggled(self, checked):
        """Legacy fringe notch: mutually exclusive with the Takeda de-fringe."""
        self.state.preproc.filter_fringes = checked
        if checked and self.defringe_check.isChecked():
            self.defringe_check.setChecked(False)
        self._schedule()

    def _on_baseline_toggled(self, checked):
        """Baseline toggle: enable/disable its parameter controls."""
        self.state.preproc.baseline = checked
        for w in (
            self.baseline_smooth_spin,
            self.baseline_ratio_spin,
            self.baseline_tauex_spin,
            self.baseline_tauex_auto,
            self.baseline_clip_check,
        ):
            w.setEnabled(checked)
        self._schedule()

    def _auto_tau_exclude(self):
        """Set the baseline hold-out from the loaded trace's delay marginal.

        Twice the delay-marginal FWHM (see
        :func:`~croak.preprocess.suggested_tau_exclude`) is wide enough to keep the
        arPLS fit off the τ≈0 peak. Writing the spin triggers the usual debounced
        recompute, so the preview follows.
        """
        data = self.state.load_data
        if data is None:
            self.set_status("Load a trace first.")
            return
        scan = data["scanaxis"]
        tau = io.position_to_delay(scan) if data["input_unit"] == "position" else scan
        try:
            value = suggested_tau_exclude(data["trace"], tau)
        except ValueError as exc:
            self.set_status(f"Could not set the hold-out automatically: {exc}")
            return
        self.baseline_tauex_spin.setValue(value / 1e-15)
        self.set_status(f"τ exclude set to {value / 1e-15:.1f} fs (2× marginal FWHM).")

    def _on_lam_range(self, lo, hi):
        self.state.preproc.lam_min_nm, self.state.preproc.lam_max_nm = lo, hi
        self._schedule()

    def _on_tau_range(self, lo, hi):
        self.state.preproc.tau_min_fs, self.state.preproc.tau_max_fs = lo, hi
        self._schedule()

    def _on_lamm_range(self, lo, hi):
        self.state.preproc.lamm_min_nm, self.state.preproc.lamm_max_nm = lo, hi
        self._schedule()

    def _on_taum_range(self, lo, hi):
        self.state.preproc.taum_min_fs, self.state.preproc.taum_max_fs = lo, hi
        self._schedule()

    def _seed_ranges(self):
        """Push the data-seeded bounds/values (from load) into the controls."""
        p = self.state.preproc
        self._seeding = True
        self.lam_range.set_bounds(p.lam_bound_min_nm, p.lam_bound_max_nm)
        self.lam_range.set_values(p.lam_min_nm, p.lam_max_nm)
        self.tau_range.set_bounds(p.tau_bound_min_fs, p.tau_bound_max_fs)
        self.tau_range.set_values(p.tau_min_fs, p.tau_max_fs)
        self.lamm_range.set_bounds(p.lamm_bound_min_nm, p.lamm_bound_max_nm)
        self.lamm_range.set_values(p.lamm_min_nm, p.lamm_max_nm)
        self.taum_range.set_bounds(p.tau_bound_min_fs, p.tau_bound_max_fs)
        self.taum_range.set_values(p.taum_min_fs, p.taum_max_fs)
        self.trange_spin.blockSignals(True)
        self.trange_spin.setValue(p.trange_fs)
        self.trange_spin.blockSignals(False)
        self._seeding = False

    def _seed_all(self):
        """Push *every* preprocess parameter into its widget (signal-safe).

        Extends :meth:`_seed_ranges` (which handles the bound-dependent range
        controls) to the filtering/subsampling toggles, so a full parameter
        reset is reflected in the UI.
        """
        self._seed_ranges()
        p = self.state.preproc
        checks = {
            self.subsample_check: p.use_subsample,
            self.resample_check: p.resample_t,
            self.prefilter_check: p.prefilter,
            self.defringe_check: p.defringe,
            self.baseline_check: p.baseline,
            self.baseline_clip_check: p.baseline_clip,
            self.fringes_check: p.filter_fringes,
            self.dc_check: p.filter_dc,
            self.threshold_check: p.use_threshold,
        }
        for w, val in checks.items():
            w.blockSignals(True)
            w.setChecked(bool(val))
            w.blockSignals(False)
        spins = {
            self.factor_spin: p.subsample_tau,
            self.defringe_frac_spin: p.defringe_fraction,
            self.baseline_smooth_spin: p.baseline_smoothness,
            self.baseline_ratio_spin: p.baseline_ratio,
            self.baseline_tauex_spin: p.baseline_tau_exclude_fs,
            self.dtau_spin: p.dtau_min_fs,
            self.threshold_spin: p.threshold,
        }
        for w, val in spins.items():
            w.blockSignals(True)
            w.setValue(val)
            w.blockSignals(False)
        self.defringe_alias_combo.blockSignals(True)
        self.defringe_alias_combo.setCurrentText(p.defringe_on_alias)
        self.defringe_alias_combo.blockSignals(False)
        # Signals were blocked above, so the toggle handlers did not run — re-apply
        # the enable/disable gating of the dependent controls.
        self.defringe_frac_spin.setEnabled(p.defringe)
        self.defringe_alias_combo.setEnabled(p.defringe)
        for w in (
            self.baseline_smooth_spin,
            self.baseline_ratio_spin,
            self.baseline_tauex_spin,
            self.baseline_tauex_auto,
            self.baseline_clip_check,
        ):
            w.setEnabled(p.baseline)

    def _reset_defaults(self):
        """Restore the data-seeded defaults for the loaded trace, then recompute."""
        defaults = self.state.preproc_defaults
        if defaults is None:
            self.set_status("Load a trace first.")
            return
        for f in type(defaults).__dataclass_fields__:
            setattr(self.state.preproc, f, getattr(defaults, f))
        self._seed_all()
        self.apply(block=True)

    def on_enter(self):
        """Re-seed every control from the parameters, then recompute if needed.

        The controls are bound one-way (widget -> parameter) through edge-triggered
        signals, so a parameter changed elsewhere — notably the data-seeded reset in
        :func:`~croak.gui.stage_load.seed_preproc_defaults`, run whenever a *new*
        trace is loaded — would otherwise leave the widgets showing stale settings
        that are not the ones applied. :meth:`_seed_all` pushes the parameters back
        into every widget (signal-safe, so it schedules no recompute), which keeps
        the page honest about what was actually computed.
        """
        if self.state.load_data is None:
            self.set_status("Load a trace first.")
            return
        self._seed_all()
        if self.state.tracedata is None:
            self.apply(block=True)

    def apply(self, *, block: bool = False):
        """Recompute the cleaned trace and refresh the preview.

        Interactive (debounced) changes run off-thread so slider drags stay
        responsive; they show a fast *coarse* baseline preview, then refine it to
        the exact baseline once the user settles (see :meth:`_on_preprocess_done`),
        so the committed trace is always exact. ``block=True`` runs inline and
        exact for programmatic callers (``on_enter``, reset, session replay) that
        read ``tracedata`` straight after.
        """
        data = self.state.load_data
        if data is None:
            self.set_status("Load a trace first.")
            return
        # Snapshot the parameters now (a shallow copy of the all-scalar dataclass
        # is a full snapshot) so a later slider change can't mutate an in-flight
        # run; the data arrays are read-only so are shared by reference.
        params = replace(self.state.preproc)
        # Bump first: this invalidates any async worker still in flight, so its
        # (now stale) result is ignored whether we run inline or dispatch.
        self._req_id += 1
        if block:
            try:
                td = pipeline.build_tracedata(params, data)  # exact, full-resolution
            except Exception as exc:
                self.set_status(f"Preprocess failed: {exc}")
                return
            self._show_tracedata(td)
            return
        self._dispatch(params, data, self._req_id, fast=True)

    def _dispatch(self, params, data, rid, *, fast):
        """Launch an off-thread preprocess (``fast`` ⇒ coarse baseline preview)."""
        self._pending = (params, data)
        worker = PreprocessWorker(params, data, rid, parent=self, fast=fast)
        worker.finished_ok.connect(self._on_preprocess_done)
        worker.failed.connect(self._on_preprocess_failed)
        worker.finished.connect(lambda w=worker: self._discard_worker(w))
        self._workers.append(worker)
        if fast:
            self.set_status("Preprocessing…")
        worker.start()

    def _discard_worker(self, worker):
        """Drop our reference to a finished worker (kept alive while running)."""
        if worker in self._workers:
            self._workers.remove(worker)
        worker.deleteLater()

    def _show_tracedata(self, td):
        """Store and draw a freshly computed trace (main thread)."""
        self.state.set_tracedata(td)
        if not self._view.attached:
            self._view.attach(self.canvas.figure)
        self._view.update(td)
        self.canvas.draw_idle()
        self.set_status(
            f"Regridded to {td.grid.n}×{td.delays.size}. Proceed to Retrieve."
        )

    def _on_preprocess_done(self, td, req_id, was_fast):
        if req_id != self._req_id:
            return  # a newer request superseded this one
        if was_fast:
            self._show_tracedata(td)  # draw the quick coarse preview
            # Refine to the exact baseline, reusing the same request id so a later
            # parameter change supersedes it; the committed trace ends up exact.
            if self._pending is not None:
                params, data = self._pending
                self._dispatch(params, data, req_id, fast=False)
        else:
            # Exact refinement: replace the stored trace (used for retrieval/save)
            # without a redraw — visually identical to the preview already shown.
            self.state.set_tracedata(td)

    def _on_preprocess_failed(self, message, req_id):
        if req_id != self._req_id:
            return
        self.set_status(f"Preprocess failed: {message}")
