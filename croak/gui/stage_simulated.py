"""Load-simulated-trace panel.

Loads a numerically simulated FROG scan (a Luna ``scansave`` HDF5 file) into the
same ``load_data`` the experimental loader produces, previews the resulting trace
live, and then hands it to the marginal-check stage. Unlike the experimental
Stage 1 the file layout is fixed, so this page exposes only the physics options
(trace window, wavelength band, delay reversal, interaction, known spectrum)
rather than dataset/unit pickers; the (λ/µm)^exp third-order efficiency
correction is tuned on the marginal-check stage.

The transient-grating signal :math:`E(t)\\,|E(t-\\tau)|^2` maps onto croak's PG
interaction; the simulated angular-frequency density is divided by
:math:`\\lambda^2` so the wavelength pipeline returns it undistorted (see
:func:`croak.session.pipeline.assemble_simulated_load_data`).
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QWidget,
)

from ..io import (
    SIMULATED_TRUTH_SOURCES,
    SIMULATED_WINDOW_KEYS,
    read_simulated_truth_keys,
    read_simulated_wavelength_range,
    read_simulated_window_keys,
    read_simulated_z_positions,
)
from ..plotting.screen import simulated_trace_views
from ..session.params import SimulatedLoadParams
from ..session.pipeline import (
    assemble_simulated_load_data,
    assemble_simulated_tracedata,
)
from .base import Stage
from .controls import add_wide_row, check, combo, group
from .retrieval_views import ViewPanel
from .stage_load import seed_preproc_defaults
from .widgets import RangeControl

# TG-FROG keeps the carrier (no doubling), so only the χ³ interactions apply.
_INTERACTIONS = ["pg", "sd"]
# Reference spectrum: the post-mask "beamlet" (the beam that gates) or the
# pre-mask "source" (the ideal input).
_SPECTRUM_SOURCES = ["beamlet", "beamlet_reimaged", "source"]


def simulated_signature(p: SimulatedLoadParams) -> tuple:
    """Identity of the *trace* a simulated load produces, for change detection.

    The simulated-scan counterpart of
    :func:`~croak.gui.stage_load.trace_signature`: two loads share a signature iff
    they yield a trace with the same delay / wavelength extents and carrier
    scaling — i.e. the same data-seeded preprocess defaults. Re-loading with a
    tweaked third-order correction, delay reversal or reference spectrum keeps the
    signature (so the user's windowing/filtering is preserved); changing the file,
    trace window, propagation slice, loaded band or interaction changes it.
    """
    return (
        p.frog_path,
        p.window_key,
        p.z_index,
        p.z_thickness_um,
        p.lam_min_nm,
        p.lam_max_nm,
        p.interaction,
    )


class StageSimulated(Stage):
    """Load + preview a numerically simulated FROG scan (reached from the menu)."""

    HELP_TITLE = "Load simulated trace"
    HELP_INTRO = (
        "Load a numerically simulated TG-FROG scan (a Luna 'scansave' HDF5 file) "
        "and preview it, then hand it to the filtering stage. The simulated signal "
        "is an angular-frequency density and is divided by λ² automatically so the "
        "wavelength pipeline returns it undistorted. The mask vignetting is handled "
        "by the reference-spectrum choice, not by a trace correction."
    )
    HELP = [
        (
            "Simulated scan file",
            [
                (
                    "trace window",
                    "Which signal extraction to load as the trace. 'Iω_win' "
                    "integrates the signal over all k (a detector collecting all "
                    "the light); 'Iω_win_reimaged' takes the on-axis re-imaged "
                    "pixel (a fibre/slit-coupled spectrometer). They differ by a "
                    "factor ≈λ² in their wavelength response, so the on-axis one "
                    "needs a larger third-order exponent. 'Iω_full' (newer files) "
                    "is the full signal beam with no aperture crop — no mask "
                    "vignetting, so it needs little third-order correction. The "
                    "list shows only the windows the file actually stores.",
                ),
                (
                    "thickness",
                    "Which propagation thickness through the medium to retrieve. A "
                    "multi-thickness scan stores the signal at several substrate "
                    "thicknesses in one file (the field at an intermediate depth "
                    "equals a dedicated thinner run); pick one to retrieve the "
                    "pulse there. Defaults to the full thickness (substrate exit). "
                    "Disabled for older single-thickness files.",
                ),
            ],
        ),
        (
            "Trace options",
            [
                (
                    "interaction",
                    "Nonlinear interaction croak models. TG-FROG maps to PG "
                    "(E·|E(t−τ)|²); both PG and SD keep the carrier (no doubling).",
                ),
                (
                    "Reverse delay axis",
                    "Flip the delay-sign convention. Off by default: the "
                    "simulation's τ orientation now matches croak's. Only tick "
                    "this for an older scan written under the opposite "
                    "convention — a wrong setting time-reverses the retrieved "
                    "pulse and flips the sign of every phase order, which is "
                    "invisible on a symmetric transform-limited pulse.",
                ),
                (
                    "Use known input spectrum",
                    "Feed the simulation's known spectrum as an independent "
                    "spectral constraint (enables spectral regularisation). Turn "
                    "off for a fully blind test.",
                ),
                (
                    "reference spectrum",
                    "Which spectrum to feed: 'beamlet' is the post-mask gate "
                    "spectrum (the beam that actually gates — recommended, the "
                    "retrieved spectrum/centroid matches it), 'source' is the "
                    "ideal pre-mask input. The mask blue-shifts the gating "
                    "spectrum, so 'beamlet' is the physically-correct reference.",
                ),
                (
                    "time-domain truth",
                    "Which stored reference pulse to overlay (and save) as the "
                    "known time-domain truth: 'beamlet' is the post-mask gate beam "
                    "(grid/It_beamlet — the beam that actually gates, recommended "
                    "and the default), 'source' is the ideal pre-mask input "
                    "(grid/It). The list shows only what the file stores and is "
                    "disabled when it offers just one; the matching spectrum is "
                    "used for the spectral panel so the overlay is self-consistent.",
                ),
                (
                    "load λ band (nm)",
                    "Wavelength band kept when loading: only bins with λ in this "
                    "range are loaded (the ω≤0 bins are always dropped). It sets "
                    "the extent of the data passed to the marginal-check/filtering "
                    "stages — not the retrieval grid, which is chosen later in the "
                    "filtering stage. Defaults to the file's full data range; "
                    "narrow it to crop noisy or empty wings.",
                ),
                (
                    "Skip filtering & regrid (raw → retrieval)",
                    "Load the as-simulated ω-density trace on its native grid "
                    "straight into the retrieval, skipping the marginal-check and "
                    "preprocess (filtering + regrid) stages. Fully raw: no "
                    "third-order correction; the window/thickness, delay-sign and "
                    "known-spectrum choices still apply. Use it to sanity-check the "
                    "retrieval free of any preprocessing doubt. For a regridded "
                    "comparison, untick it here and Continue again.",
                ),
            ],
        ),
    ]

    def __init__(self, state, adopt_geometry):
        super().__init__(state)
        # Retrieve's StageRetrieve.apply_simulated_geometry: adopt the file's
        # slab thickness and mask on an interactive load (see advance).
        self.adopt_geometry = adopt_geometry
        self._defaults = SimulatedLoadParams()
        # the path the file-driven selectors (trace window + thickness) were last
        # populated for, so re-entering the stage keeps the user's choices
        self._options_path: str | None = None
        # Nothing can be loaded until a file previews successfully; drives the
        # Data page's Continue (see can_advance).
        self._can_load = False

        # debounce rapid control changes into one reload/preview
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self._preview)

        # -- File --------------------------------------------------------
        box, form = group("Simulated scan file")
        self.path_edit = QLineEdit(self._defaults.frog_path)
        self.path_edit.setReadOnly(True)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._pick_file)
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.path_edit, stretch=1)
        rl.addWidget(browse)
        form.addRow(row)
        # Trace window + thickness are both narrowed to what the chosen file
        # actually stores by _populate_file_options(); the window list seeds with
        # every supported key for the pre-file state.
        self.window_key = combo(
            list(SIMULATED_WINDOW_KEYS), self._defaults.window_key, self._changed
        )
        form.addRow("trace window", self.window_key)
        # Thickness selector for the multi-thickness format (one item per saved
        # /grid/zsave slice, carrying its thickness in µm as item data); disabled
        # for the older single-thickness format.
        self.thickness = QComboBox()
        self.thickness.currentIndexChanged.connect(self._changed)
        form.addRow("thickness", self.thickness)
        self.controls.addWidget(box)

        # -- Physics options --------------------------------------------
        box, form = group("Trace options")
        self.interaction = combo(
            _INTERACTIONS, self._defaults.interaction, self._changed
        )
        form.addRow("interaction", self.interaction)
        # reverse_trace=None means "auto from the file marker"; the checkbox
        # cannot express that, so it is seeded explicitly. Default OFF: the
        # delay-axis convention is now fixed at the simulation end, so the
        # old default flip is no longer wanted. A file that does need flipping still
        # carries its own marker, and the box can be ticked by hand.
        self.reverse = check(
            "Reverse delay axis",
            False
            if self._defaults.reverse_trace is None
            else self._defaults.reverse_trace,
            self._changed,
        )
        form.addRow(self.reverse)
        self.uniform_core = check(
            "Uniform delay core only",
            self._defaults.uniform_delay_core,
            self._changed,
        )
        self.uniform_core.setToolTip(
            "Keep only the longest uniformly spaced stretch of the delay "
            "axis. Campaign scans often extend a uniform core with coarser "
            "wing points (e.g. 1 fs-step wings to ±40 fs for the Raman "
            "wake); the smearing kernel's delay-axis convolution needs a "
            "uniform grid, so tick this to retrieve such a scan with the "
            "kernel modelled. Leave off otherwise — the wings are real data."
        )
        form.addRow(self.uniform_core)
        # The (λ/µm)^exp third-order correction is tuned on the marginal-check
        # stage (with the predicted marginal in view), not here.
        self.use_spectrum = check(
            "Use known input spectrum", self._defaults.use_spectrum, self._changed
        )
        form.addRow(self.use_spectrum)
        self.spectrum_source = combo(
            _SPECTRUM_SOURCES, self._defaults.spectrum_source, self._changed
        )
        form.addRow("reference spectrum", self.spectrum_source)
        # Time-domain truth overlay source (beamlet/source); populated from the
        # file in _populate_file_options (defaults to the beamlet when present,
        # and is disabled when the file offers only one).
        self.truth_source = QComboBox()
        self.truth_source.currentIndexChanged.connect(self._changed)
        form.addRow("time-domain truth", self.truth_source)
        self._band = (self._defaults.lam_min_nm, self._defaults.lam_max_nm)
        self.band = RangeControl(50.0, 3000.0, *self._band, decimals=0, step=10.0)
        self.band.changed.connect(self._on_band)
        add_wide_row(form, "load λ band (nm)", self.band)
        # Skip the marginal-check + preprocess stages and load the as-simulated
        # ω-density trace on its native grid straight into the retrieval. "&&" is
        # Qt's escape for a literal ampersand in a widget label (see sheet.plain).
        self.raw_direct = check(
            "Skip filtering && regrid (raw → retrieval)",
            self._defaults.raw_direct,
            self._on_raw_direct,
        )
        form.addRow(self.raw_direct)
        self.controls.addWidget(box)

        # Continue (which calls advance) and Help live in the window's top bar.
        self.controls.addStretch(1)

        self.views = {
            "lin": ViewPanel("map", signed=True),
            "db": ViewPanel("map"),
            "delay": ViewPanel("line"),
            "spectral": ViewPanel("line"),
        }
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(10)
        for index, view in enumerate(self.views.values()):
            grid.addWidget(view, index // 2, index % 2)
        plots = QWidget()
        plots.setLayout(grid)
        self.set_plot_area(plots)

        self.apply_help()

    # -- lifecycle ----------------------------------------------------------
    def on_enter(self):
        if self.path_edit.text():
            self._populate_file_options(self.path_edit.text())
            self._preview()
        else:
            self.set_status("Choose a simulated scan file (.h5) to begin.")

    def _populate_file_options(self, path: str) -> None:
        """Populate the trace-window and thickness selectors from the file.

        The trace-window list is narrowed to the windows actually stored (older
        files lack ``Iω_full``); the thickness list comes from ``/grid/zsave``
        (one item per saved depth, largest/exit first so the default is the full
        thickness, each carrying its µm value as item data — disabled when the
        file has no ``zsave``). Both keep the current choice when it is still
        valid, and re-populating is skipped when the path is unchanged, preserving
        the user's selections on re-entry.
        """
        if path == self._options_path and self.window_key.count():
            return
        try:
            windows = read_simulated_window_keys(path)
        except Exception:
            # A malformed/non-scansave file: fall back to the full list; the live
            # preview below surfaces the actual load error.
            windows = []
        try:
            zsave = read_simulated_z_positions(path)
        except Exception:
            zsave = None

        # trace windows: keep the current choice if the new file still has it
        items = windows or list(SIMULATED_WINDOW_KEYS)
        current = self.window_key.currentText()
        self.window_key.blockSignals(True)
        self.window_key.clear()
        self.window_key.addItems(items)
        self.window_key.setCurrentText(current if current in items else items[0])
        self.window_key.blockSignals(False)

        # thickness slices (largest/exit first; disabled for older files)
        self.thickness.blockSignals(True)
        self.thickness.clear()
        if zsave is None or zsave.size == 0:
            self.thickness.addItem("full thickness")
            self.thickness.setEnabled(False)
        else:
            zmax = float(zsave.max())
            for z in sorted((float(v) for v in zsave), reverse=True):
                um = z * 1e6
                if z == zmax:
                    label = f"{um:.4g} µm (full)"
                elif z == 0.0:
                    label = f"{um:.4g} µm (entrance)"
                else:
                    label = f"{um:.4g} µm"
                self.thickness.addItem(label, um)
            self.thickness.setEnabled(True)
            self.thickness.setCurrentIndex(0)
        self.thickness.blockSignals(False)

        # time-domain truth source: list what the file stores (beamlet first), so
        # It_beamlet is the default; disable the picker when only one is offered.
        try:
            truth_keys = read_simulated_truth_keys(path)
        except Exception:
            truth_keys = ()
        self.truth_source.blockSignals(True)
        self.truth_source.clear()
        if truth_keys:
            # read_simulated_truth_keys lists the beamlet first, so default to it
            # (the "It_beamlet by default" rule); enable the picker only when the
            # file offers a genuine choice.
            self.truth_source.addItems(list(truth_keys))
            self.truth_source.setCurrentText(truth_keys[0])
            self.truth_source.setEnabled(len(truth_keys) > 1)
        else:
            # malformed/non-scansave file: a disabled placeholder; the live
            # preview below surfaces the actual load error.
            self.truth_source.addItem(self._defaults.truth_source)
            self.truth_source.setEnabled(False)
        self.truth_source.blockSignals(False)

        # default the load band to the file's full positive-frequency λ range
        # (set_bounds/set_values are signal-safe, so update self._band by hand)
        try:
            lam_lo, lam_hi = read_simulated_wavelength_range(path)
        except Exception:
            lam_lo = lam_hi = None
        if lam_lo is not None and lam_hi is not None:
            lo, hi = float(np.floor(lam_lo)), float(np.ceil(lam_hi))
            self.band.set_bounds(min(50.0, lo), max(3000.0, hi))
            self.band.set_values(lo, hi)
            self._band = (lo, hi)
        self._options_path = path

    def seed_widgets(self, p: SimulatedLoadParams) -> None:
        """Reflect a restored :class:`SimulatedLoadParams` in the controls.

        Used when a saved session is reloaded (the controls are otherwise built
        from the defaults and only ever written *to* the params). The file-driven
        selectors are populated from the file first — so the trace-window,
        thickness and truth lists exist — and the saved choices are then applied
        on top, with signals blocked so seeding never triggers a reload.
        """
        self.path_edit.setText(p.frog_path)
        if p.frog_path:
            self._populate_file_options(p.frog_path)
        for widget, setter in (
            (self.window_key, lambda: self.window_key.setCurrentText(p.window_key)),
            (self.interaction, lambda: self.interaction.setCurrentText(p.interaction)),
            (
                self.reverse,
                # Same default as the construction-time seed above: unset means
                # off, because the delay convention is now fixed at the source.
                lambda: self.reverse.setChecked(
                    False if p.reverse_trace is None else p.reverse_trace
                ),
            ),
            (self.use_spectrum, lambda: self.use_spectrum.setChecked(p.use_spectrum)),
            (
                self.uniform_core,
                lambda: self.uniform_core.setChecked(p.uniform_delay_core),
            ),
            (
                self.spectrum_source,
                lambda: self.spectrum_source.setCurrentText(p.spectrum_source),
            ),
            (
                self.truth_source,
                lambda: self.truth_source.setCurrentText(p.truth_source),
            ),
            (self.raw_direct, lambda: self.raw_direct.setChecked(p.raw_direct)),
        ):
            widget.blockSignals(True)
            setter()
            widget.blockSignals(False)
        # The thickness items carry their µm value as item data; match the saved
        # slice exactly (it was written from the same list) and otherwise leave the
        # default (full thickness), which is also the no-zsave behaviour of older files.
        if p.z_thickness_um is not None:
            index = self.thickness.findData(p.z_thickness_um)
            if index >= 0:
                self.thickness.blockSignals(True)
                self.thickness.setCurrentIndex(index)
                self.thickness.blockSignals(False)
        # set_values is signal-safe, so track the band by hand (as _on_band does)
        self.band.set_values(p.lam_min_nm, p.lam_max_nm)
        self._band = (p.lam_min_nm, p.lam_max_nm)

    def _changed(self, *_):
        self._timer.start()

    def _on_band(self, lo, hi):
        self._band = (lo, hi)
        self._changed()

    def _on_raw_direct(self, _checked: bool) -> None:
        # Only changes what Load does (not the preview), so just relabel the
        # forward action.
        self.changed.emit()

    def _set_can_load(self, can: bool) -> None:
        """Record whether a trace previewed, and refresh the wizard's Next button."""
        self._can_load = can
        self.changed.emit()

    def _pick_file(self):
        fn, _ = QFileDialog.getOpenFileName(
            self, "Select simulated scan file", filter="Scan files (*.h5 *.hdf5)"
        )
        if fn:
            self.path_edit.setText(fn)
            self._populate_file_options(fn)
            self._preview()

    # -- params -------------------------------------------------------------
    def _params(self) -> SimulatedLoadParams:
        """Build a :class:`SimulatedLoadParams` from the current controls."""
        # The thickness item data is the chosen slice in µm; None (the disabled
        # placeholder of an older file) falls back to the exit slice.
        data = self.thickness.currentData()
        z_thickness_um = float(data) if data is not None else None
        truth = self.truth_source.currentText()
        truth_source = (
            truth if truth in SIMULATED_TRUTH_SOURCES else self._defaults.truth_source
        )
        # The third-order correction has no control here — it is tuned on the
        # marginal-check stage, which writes it onto state.simulated — and neither
        # has the low-level z_index. Carry all three across so re-previewing (or a
        # restored session) does not silently reset them to the defaults.
        live = self.state.simulated
        return SimulatedLoadParams(
            frog_path=self.path_edit.text(),
            window_key=self.window_key.currentText(),
            z_index=live.z_index,
            z_thickness_um=z_thickness_um,
            third_order=live.third_order,
            third_order_exp=live.third_order_exp,
            lam_min_nm=self._band[0],
            lam_max_nm=self._band[1],
            reverse_trace=self.reverse.isChecked(),
            uniform_delay_core=self.uniform_core.isChecked(),
            interaction=self.interaction.currentText(),
            use_spectrum=self.use_spectrum.isChecked(),
            spectrum_source=self.spectrum_source.currentText(),
            truth_source=truth_source,
            raw_direct=self.raw_direct.isChecked(),
        )

    # -- preview ------------------------------------------------------------
    def _preview(self):
        if not self.path_edit.text():
            return
        try:
            data = assemble_simulated_load_data(self._params())
        except Exception as exc:
            self._set_can_load(False)
            self.set_status(f"Load failed: {exc}")
            return
        self._draw(data)
        self._set_can_load(True)
        if self.thickness.isEnabled():
            self.set_status(
                f"Loaded the {self.thickness.currentText()} slice — review the "
                "trace, then Continue to check and filter it."
            )
        else:
            self.set_status("Review the trace, then Continue to check and filter it.")

    def _draw(self, data):
        """Preview the scan: linear and dB maps, delay and spectral marginals."""
        for view, shown in zip(
            self.views.values(), simulated_trace_views(data), strict=True
        ):
            view.show_view(shown)

    # -- load (the entry-page forward action) -------------------------------

    def can_advance(self) -> bool:
        return self._can_load

    def advance(self):
        """Load the chosen simulated scan and hand it to the next stage.

        The Data page's Continue calls it. Normally the trace goes into
        the marginal-check stage; with "Skip filtering & regrid" ticked the
        as-simulated trace is built on its native grid and the session goes
        straight to Retrieve (:attr:`advanced` says which). On failure the reason
        is shown in the status label and the page stays put.
        """
        params = self._params()
        try:
            data = assemble_simulated_load_data(params)
        except Exception as exc:
            self.set_status(f"Load failed: {exc}")
            return
        # record the chosen options on the shared state, then hand the trace into
        # the normal preprocess pipeline (mirrors the synthetic generator).
        self.state.simulated = params
        # A new/different trace resets the preprocess windows and filters; loading
        # the same trace again (e.g. after tweaking the third-order correction or
        # coming Back from a later stage) keeps them — matching StageLoad.
        sig = simulated_signature(params)
        new_trace = sig != self.state.trace_sig
        self.state.trace_sig = sig
        self.state.set_load_data(data)
        seed_preproc_defaults(self.state, reset_values=new_trace)
        # Adopt the instrument geometry the file records (slab thickness of the
        # chosen slice, mask hole diameter and spacing). Only on this
        # interactive path: a restored session keeps its own saved retrieval
        # settings, which the user may have chosen deliberately.
        note = self.adopt_geometry(self.state.geometry)
        if note:
            self.set_status(note)
        if params.raw_direct:
            # Build the as-simulated trace on its native grid and go straight to
            # retrieval, skipping the marginal-check + preprocess stages. The
            # normal load_data above stays set (it carries the truth overlay, and
            # unticking the option re-runs the regridded path for a comparison).
            try:
                td = assemble_simulated_tracedata(params)
            except Exception as exc:
                self.set_status(f"Raw load failed: {exc}")
                return
            self.state.set_tracedata(td)
            self.advanced.emit("retrieve")
        else:
            self.advanced.emit("check")
