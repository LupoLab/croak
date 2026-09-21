"""Stage 4 — tune residual dispersion (GDD/TOD/FOD, material) and save.

GDD/TOD/FOD and material thicknesses use slider + spin-box controls (default 0).
The auto-tuners maximise the absolute temporal
peak power over the slider bounds, using a selectable optimiser (global by
default).
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import replace

import numpy as np
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from .. import dispersion, gases, materials, plotting, refractive_db, save
from ..processing import process_result, spectrogram
from .base import Stage
from .canvas import MplCanvas, combo, group, with_toolbar
from .mirror_stack import MirrorStack
from .widgets import SciSpinBox, ValueControl

_GAS_NONE = "(none)"
# Gas controls: pressure in bar at millibar resolution (see _GAS_PRESSURE_KW) and
# path length in cm (gas paths run cm–m, far longer than the ±20 mm solid range);
# a negative path pre-compensates, mirroring the solid-thickness convention.
# The 10 bar ceiling covers hollow-fibre compression and is also where the
# ideal-gas scaling stops being quantitative (see croak.gases).
_GAS_PRESSURE_RANGE_BAR = (0.0, 10.0)
_GAS_PATH_RANGE_CM = (-999.0, 999.0)

# Material thickness controls follow the single source of truth in
# ``materials.MATERIALS`` (built-in Sellmeier + tabulated), so new materials
# surface here automatically and stay in sync with the other GUI stages.
_MATERIALS = list(materials.MATERIALS)
# Slider bounds (fs² / fs³ / fs⁴, mm). Chosen to span the residual dispersion a
# typical compressor leaves behind while keeping the slider resolution usable.
_GDD_RANGE = (-10000.0, 10000.0)
_TOD_RANGE = (-100000.0, 100000.0)
_FOD_RANGE = (-1000000.0, 1000000.0)
_MAT_RANGE_MM = (-20.0, 20.0)


class StageDispersion(Stage):
    HELP_TITLE = "Dispersion"
    HELP_INTRO = (
        "Tune residual dispersion on the retrieved pulse: add Taylor phase "
        "(GDD/TOD/FOD) and/or material thickness and watch the temporal pulse "
        "update. The Auto buttons maximise the peak temporal power. This does "
        "not re-run the retrieval."
    )
    HELP = [
        (
            "Optimiser",
            [
                (
                    "algorithm",
                    "Optimiser used by the Auto buttons. differential_evolution, "
                    "dual_annealing and direct are global searches; nelder-mead is "
                    "a local refinement.",
                ),
            ],
        ),
        (
            "Taylor coefficients",
            [
                (
                    "GDD (fs²)",
                    "Group-delay dispersion added to the retrieved spectral phase "
                    "(second order, fs²).",
                ),
                (
                    "TOD (fs³)",
                    "Third-order dispersion added to the spectral phase (fs³).",
                ),
                (
                    "FOD (fs⁴)",
                    "Fourth-order dispersion added to the spectral phase (fs⁴).",
                ),
            ],
        ),
        (
            "Materials (mm)",
            [
                (
                    "SiO2",
                    "Thickness (mm) of fused silica to add; negative "
                    "pre-compensates. 'Auto' optimises it.",
                ),
                ("BK7", "Thickness (mm) of N-BK7 to add (negative pre-compensates)."),
                ("CaF2", "Thickness (mm) of CaF₂ to add (negative pre-compensates)."),
                ("BaF2", "Thickness (mm) of BaF₂ to add (negative pre-compensates)."),
                ("MgF2", "Thickness (mm) of MgF₂ to add (negative pre-compensates)."),
                (
                    "SiO2-Franta",
                    "Thickness (mm) of fused silica using Franta et al. broadband "
                    "tabulated optical constants (24.8 nm – 125 µm), valid far "
                    "beyond the Sellmeier range.",
                ),
            ],
        ),
        (
            "refractiveindex.info material",
            [
                (
                    "shelf",
                    "Top-level refractiveindex.info category (click 'Load "
                    "database' first; the catalogue downloads on first use).",
                ),
                ("book", "Material within the selected shelf."),
                (
                    "page",
                    "Data source/reference for the material; the valid wavelength "
                    "range is shown below.",
                ),
                (
                    "thickness (mm)",
                    "Thickness of the selected refractiveindex.info material to "
                    "add (negative pre-compensates). 'Auto' optimises it.",
                ),
            ],
        ),
        (
            "Gases",
            [
                (
                    "gas",
                    "Gas filling the beam path (Helium or Air), or '(none)'. "
                    "Sellmeier data ported from Luna.jl; the index scales with "
                    "pressure via the ideal-gas law.",
                ),
                (
                    "pressure (bar)",
                    "Gas pressure in bar (n−1 scales linearly with it; 0 is "
                    "vacuum). Coefficients are referenced to 1 bar.",
                ),
                (
                    "path length (cm)",
                    "Optical path length through the gas in cm; negative "
                    "pre-compensates. 'Auto' optimises it at the set pressure.",
                ),
            ],
        ),
        (
            "Mirrors",
            [
                (
                    "mirrors",
                    "An editable stack of beam-path mirrors; '+ Add mirror' appends "
                    "a row. Each row picks a type — a coating (bare Si, MgF₂/Al at "
                    "several angles), a built-in chirped compressor (ported from "
                    "Luna.jl), or 'Custom file…' — a polarisation (s or p, for "
                    "coatings at non-normal incidence; identical at 0°), a bounce "
                    "count and a direction. 'remove' (the default) back-propagates "
                    "the mirror out of the measured pulse — the usual case — and "
                    "'add' applies it. Removing a coating undoes its spectral phase "
                    "AND divides out its reflectivity (clamped to avoid blow-up "
                    "where R is small); chirped/custom mirrors are phase-only. "
                    "'Auto' scans the bounce count that maximises the peak power.",
                ),
            ],
        ),
        (
            "Pulse info",
            [
                (
                    "Pulse energy (J)",
                    "Optional measured pulse energy in joules (e.g. from a power "
                    "meter). When > 0 the temporal plot and the peak-power "
                    "readouts switch from normalised units to absolute power; 0 "
                    "keeps normalised 'a.u.'. Shared with the load stage.",
                ),
                (
                    "Current FWHM",
                    "Intensity FWHM of the current (dispersion-adjusted) temporal "
                    "pulse.",
                ),
                (
                    "TL FWHM",
                    "Transform-limited FWHM — the shortest pulse the spectrum "
                    "supports.",
                ),
                (
                    "Peak power",
                    "Peak instantaneous power of the current pulse, = pulse "
                    "energy / effective duration. Shown once a pulse energy is set.",
                ),
                (
                    "TL peak power",
                    "Peak power of the transform-limited pulse (same energy, "
                    "shortest duration) — the maximum achievable by compression.",
                ),
            ],
        ),
    ]

    def __init__(self, state):
        super().__init__(state)
        self._base = None  # original retrieved spectrum
        self._ri_key = None  # registered refractiveindex.info material name
        self._ri_range = None  # (lo, hi) m valid range of the selected RI material
        self._gas_key = None  # registered active-gas material name, or None
        self._gas_name = ""  # name of the active gas (for re-registration)
        self._ri_shelf_ids: list[str] = []
        self._ri_book_ids: list[str] = []
        self._ri_page_ids: list[str] = []

        box, form = group("Optimiser")
        self.method = combo(
            [
                "differential_evolution",
                "dual_annealing",
                "direct",
                "nelder-mead",
            ],
            "differential_evolution",
        )
        form.addRow("algorithm", self.method)
        self.controls.addWidget(box)

        box, form = group("Taylor coefficients")
        self.gdd = ValueControl(
            *_GDD_RANGE, 0.0, decimals=2, step=0.1, slider_scale=0.02
        )
        self.tod = ValueControl(
            *_TOD_RANGE, 0.0, decimals=2, step=1.0, slider_scale=0.002
        )
        self.fod = ValueControl(
            *_FOD_RANGE, 0.0, decimals=2, step=10.0, slider_scale=0.0002
        )
        for c in (self.gdd, self.tod, self.fod):
            c.changed.connect(self._changed)
        form.addRow("GDD (fs²)", self.gdd)
        form.addRow("TOD (fs³)", self.tod)
        form.addRow("FOD (fs⁴)", self.fod)
        auto = QPushButton("Auto GDD+TOD+FOD")
        auto.clicked.connect(self._auto_taylor)
        form.addRow(auto)
        self.controls.addWidget(box)

        box, form = group("Materials (mm)")
        self._mat_ctrls = {}
        for mat in _MATERIALS:
            ctrl = ValueControl(*_MAT_RANGE_MM, 0.0, decimals=3, slider_scale=100.0)
            ctrl.changed.connect(self._changed)
            self._mat_ctrls[mat] = ctrl
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.addWidget(ctrl, stretch=1)
            btn = QPushButton("Auto")
            btn.clicked.connect(lambda _, m=mat: self._auto_material(m))
            rl.addWidget(btn)
            form.addRow(mat, row)
        self.controls.addWidget(box)

        self._build_ri_group()
        self._build_gas_group()
        self._build_mirror_group()

        box, form = group("Pulse info")
        # Optional measured pulse energy (J): when > 0 the temporal plot and the
        # peak-power readouts switch from normalised units to absolute power. The
        # value is shared with the load stage (state.load.energy_j).
        self.energy = SciSpinBox(0.0, 1e6, 0.0, self._on_energy_changed, sigfigs=3)
        form.addRow("Pulse energy (J)", self.energy)
        self.fwhm_label = QLabel("–")
        self.tl_label = QLabel("–")
        self.peak_label = QLabel("–")
        self.peak_tl_label = QLabel("–")
        form.addRow("Current FWHM", self.fwhm_label)
        form.addRow("TL FWHM", self.tl_label)
        form.addRow("Peak power", self.peak_label)
        form.addRow("TL peak power", self.peak_tl_label)
        self.controls.addWidget(box)

        btns = QWidget()
        bl = QHBoxLayout(btns)
        bl.setContentsMargins(0, 0, 0, 0)
        reset = QPushButton("Reset")
        reset.clicked.connect(self._reset)
        save_btn = QPushButton("Save…")
        save_btn.clicked.connect(self._save)
        bl.addWidget(reset)
        bl.addWidget(save_btn)
        self.controls.addWidget(btns)
        self.controls.addWidget(self.status_label)
        self.controls.addStretch(1)

        self.canvas = MplCanvas(figsize=(9, 7))
        self.set_plot_area(with_toolbar(self.canvas))

        self.apply_help()

    # -- group builders -----------------------------------------------------
    def _build_ri_group(self):
        box, form = group("refractiveindex.info material")
        self.ri_load_btn = QPushButton("Load database")
        self.ri_load_btn.clicked.connect(self._load_ri_db)
        form.addRow(self.ri_load_btn)
        self.ri_shelf = QComboBox()
        self.ri_book = QComboBox()
        self.ri_page = QComboBox()
        self.ri_shelf.currentIndexChanged.connect(self._on_ri_shelf)
        self.ri_book.currentIndexChanged.connect(self._on_ri_book)
        self.ri_page.currentIndexChanged.connect(self._on_ri_page)
        for c in (self.ri_shelf, self.ri_book, self.ri_page):
            c.setEnabled(False)
        form.addRow("shelf", self.ri_shelf)
        form.addRow("book", self.ri_book)
        form.addRow("page", self.ri_page)
        self.ri_range_label = QLabel("–")
        form.addRow("valid λ", self.ri_range_label)
        self.ri_thickness = ValueControl(
            *_MAT_RANGE_MM, 0.0, decimals=3, slider_scale=100.0
        )
        self.ri_thickness.changed.connect(self._changed)
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.ri_thickness, stretch=1)
        btn = QPushButton("Auto")
        btn.clicked.connect(self._auto_ri)
        rl.addWidget(btn)
        form.addRow("thickness (mm)", row)
        if not refractive_db.available():
            self.ri_load_btn.setEnabled(False)
            self.ri_load_btn.setText("Install 'ridb' extra")
            self.ri_load_btn.setToolTip(
                "The optional 'refractiveindex' package is not installed. Install "
                "it (e.g. `uv pip install 'croak[ridb]'` or `pip install "
                "'croak[ridb]'`) and restart; this button then downloads the "
                "refractiveindex.info database on first use."
            )
        self.controls.addWidget(box)

    def _build_mirror_group(self):
        box, form = group("Mirrors")
        self.mirror_stack = MirrorStack()
        self.mirror_stack.changed.connect(self._changed)
        self.mirror_stack.status.connect(self.set_status)
        self.mirror_stack.auto_requested.connect(self._auto_row)
        # a labelled row so the help tooltip attaches (see help.apply_tooltips)
        form.addRow("mirrors", self.mirror_stack)
        self.controls.addWidget(box)

    def _build_gas_group(self):
        box, form = group("Gases")
        self.gas_combo = combo(
            [_GAS_NONE, *gases.GASES], _GAS_NONE, self._on_gas_changed
        )
        form.addRow("gas", self.gas_combo)
        # 3 decimals, a 1e-3 bar step and a slider scaled by 1000 make the spin
        # box, its arrows and the slider all resolve one millibar: low-pressure
        # work (a few mbar over a long purge path) has to dial the bottom of the
        # range, not just type into the spin box.
        self.gas_pressure = ValueControl(
            *_GAS_PRESSURE_RANGE_BAR, 1.0, decimals=3, step=0.001, slider_scale=1000.0
        )
        self.gas_pressure.changed.connect(self._on_gas_pressure)
        form.addRow("pressure (bar)", self.gas_pressure)
        self.gas_path = ValueControl(
            *_GAS_PATH_RANGE_CM, 0.0, decimals=2, slider_scale=10.0
        )
        self.gas_path.changed.connect(self._changed)
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.gas_path, stretch=1)
        auto = QPushButton("Auto")
        auto.clicked.connect(self._auto_gas)
        rl.addWidget(auto)
        form.addRow("path length (cm)", row)
        self.controls.addWidget(box)

    @staticmethod
    def _range_text(rng):
        """Format a ``(lo, hi)`` wavelength range (m) as ``"lo–hi nm"``."""
        return f"{rng[0] / 1e-9:.0f}–{rng[1] / 1e-9:.0f} nm" if rng else "(unknown)"

    def _fill_combo(self, box, items, index):
        box.blockSignals(True)
        box.clear()
        box.addItems([str(i) for i in items])
        if index is not None and 0 <= index < len(items):
            box.setCurrentIndex(index)
        box.blockSignals(False)

    # -- refractiveindex.info handlers --------------------------------------
    def _load_ri_db(self):
        if not refractive_db.available():
            self.set_status("pip install croak[ridb] to enable refractiveindex.info")
            return
        self.set_status(
            "Loading refractiveindex.info catalogue (first use may "
            "download the database)…"
        )
        try:
            shelves = refractive_db.shelves()
        except Exception as exc:
            self.set_status(f"refractiveindex.info unavailable: {exc}")
            return
        self._ri_shelf_ids = [sid for sid, _ in shelves]
        self._fill_combo(self.ri_shelf, [label for _, label in shelves], 0)
        for c in (self.ri_shelf, self.ri_book, self.ri_page):
            c.setEnabled(True)
        self._on_ri_shelf()  # cascade-fill books, pages and register material
        self.set_status(f"{len(shelves)} shelves loaded.")

    def _on_ri_shelf(self, *_):
        if not self._ri_shelf_ids:
            return
        sid = self._ri_shelf_ids[self.ri_shelf.currentIndex()]
        self.state.dispersion.ri_shelf = sid
        bks = refractive_db.books(sid)
        self._ri_book_ids = [bid for bid, _ in bks]
        self._fill_combo(self.ri_book, [label for _, label in bks], 0)
        self._on_ri_book()

    def _on_ri_book(self, *_):
        if not self._ri_book_ids:
            return
        sid = self._ri_shelf_ids[self.ri_shelf.currentIndex()]
        bid = self._ri_book_ids[self.ri_book.currentIndex()]
        self.state.dispersion.ri_book = bid
        pgs = refractive_db.pages(sid, bid)
        self._ri_page_ids = [pid for pid, _ in pgs]
        self._fill_combo(self.ri_page, [label for _, label in pgs], 0)
        self._on_ri_page()

    def _on_ri_page(self, *_):
        if not self._ri_page_ids:
            return
        sid = self._ri_shelf_ids[self.ri_shelf.currentIndex()]
        bid = self._ri_book_ids[self.ri_book.currentIndex()]
        pid = self._ri_page_ids[self.ri_page.currentIndex()]
        self.state.dispersion.ri_page = pid
        try:
            n_func, rng = refractive_db.make_material(sid, bid, pid)
        except Exception as exc:
            self.set_status(f"Could not load material: {exc}")
            return
        if self._ri_key:
            materials.unregister_material(self._ri_key)
        self._ri_key = f"RI:{bid}/{pid}"
        materials.register_material(self._ri_key, n_func)
        self._ri_range = rng
        self.ri_range_label.setText(self._range_text(rng))
        self._changed()

    def _auto_ri(self):
        if self._base is None or not self._ri_key:
            return
        r = self.state.result
        if r is None:
            return
        base = self._mirror_base(r)
        pr = process_result(replace(r, spectrum=base))
        gdd_per_mm = dispersion.material_gdd(self._ri_key, r.omega0, 1e-3)
        if gdd_per_mm and np.isfinite(gdd_per_mm):
            t_need = -pr.gdd_fs2 / gdd_per_mm
            span = max(10.0 * abs(t_need), 1.0)  # mm
        else:
            span = 20.0
        thickness = dispersion.auto_material(
            r.grid,
            base,
            r.omega0,
            self._ri_key,
            bounds=(-span * 1e-3, span * 1e-3),
            method=self.method.currentText(),
        )
        self.ri_thickness.blockSignals(True)
        self.ri_thickness.set_value(thickness * 1e3)
        self.ri_thickness.blockSignals(False)
        self._changed()

    # -- mirror-stack handlers ----------------------------------------------
    def _auto_row(self, row):
        """Scan ``row``'s bounce count to maximise the peak power.

        Holds the other rows and every other dispersion control at their current
        values, so it finds the best count for this mirror *in context* (whether
        the row removes or adds). Mirrors the old single-mirror auto-bounces.
        """
        if self._base is None:
            return
        best_n, best_p = 0, -np.inf
        for n in range(row.bounces.maximum() + 1):
            row.set_bounces(n, silent=True)
            power = float(np.max(self._modified().intensity_t))
            if power > best_p:
                best_p, best_n = power, n
        row.set_bounces(best_n)  # emits changed → redraw

    # -- gas handlers -------------------------------------------------------
    def _register_gas(self):
        """(Re)register the active gas's ``n(λ)`` at the current pressure.

        Pressure is baked into the registered callable (the dispersion pipeline
        keys materials by name and looks up a single-argument ``n(λ_m)``), so a
        pressure change rebuilds and re-registers it under the same key.
        """
        self._gas_key = "_active_gas"
        n_func = gases.gas_refractive_index(self._gas_name, self.gas_pressure.value())
        materials.register_material(self._gas_key, n_func)

    def _unregister_gas(self):
        if self._gas_key:
            materials.unregister_material(self._gas_key)
            self._gas_key = None

    def _on_gas_changed(self, text):
        if text == _GAS_NONE:
            self._gas_name = ""
            self._unregister_gas()
        else:
            self._gas_name = text
            self._register_gas()
        self._changed()

    def _on_gas_pressure(self, _v):
        if self._gas_name:
            self._register_gas()
        self._changed()

    def _auto_gas(self):
        r = self.state.result
        if self._base is None or not self._gas_key or r is None:
            return
        # Gas dispersion is weak, so the solid GDD-need heuristic would ask for an
        # unboundedly long path; search the path control's own range instead.
        lo, hi = self.gas_path.bounds()  # cm
        length = dispersion.auto_material(
            r.grid,
            self._mirror_base(r),
            r.omega0,
            self._gas_key,
            bounds=(lo * 1e-2, hi * 1e-2),
            method=self.method.currentText(),
        )
        self.gas_path.blockSignals(True)
        self.gas_path.set_value(length * 1e2)  # m -> cm
        self.gas_path.blockSignals(False)
        self._changed()

    def on_enter(self):
        if self.state.result is None:
            self.set_status("Retrieve a pulse first.")
            return
        self._base = self.state.result.spectrum.copy()
        # Restore any saved/loaded dispersion settings into the controls so a
        # reloaded session re-applies its compression, then recompute once.
        self._restore_from_params()
        # Reflect the (load-stage) pulse energy without retriggering a recompute.
        self.energy.blockSignals(True)
        self.energy.setValue(self.state.load.energy_j)
        self.energy.blockSignals(False)
        self._changed()

    def _on_energy_changed(self, _v=None):
        """Persist the entered pulse energy (shared with the load stage) and redraw."""
        self.state.load.energy_j = self.energy.value()
        self._changed()

    # -- persistence: controls <-> DispersionParams -------------------------
    def _sync_to_params(self) -> None:
        """Write the live control values into ``state.dispersion`` for saving.

        The GUI composes dispersion interactively from the live controls/registry
        (:meth:`_modified`); this mirrors those values into the persisted
        :class:`~croak.session.params.DispersionParams` so a save captures them and
        a reload or headless replay (:func:`croak.session.compose_dispersion`)
        reproduces the same compression.
        """
        d = self.state.dispersion
        d.gdd_fs2 = self.gdd.value()
        d.tod_fs3 = self.tod.value()
        d.fod_fs4 = self.fod.value()
        d.material_thickness_mm = {
            mat: c.value() for mat, c in self._mat_ctrls.items() if c.value()
        }
        d.gas_name = self._gas_name
        d.gas_pressure_bar = self.gas_pressure.value()
        d.gas_path_cm = self.gas_path.value()
        d.ri_thickness_mm = self.ri_thickness.value()
        d.mirrors = self.mirror_stack.to_specs()
        # ri_shelf/ri_book/ri_page are kept current by the RI selection handlers.

    def _restore_from_params(self) -> None:
        """Populate the controls from ``state.dispersion`` (signals blocked)."""
        d = self.state.dispersion
        for ctrl, val in (
            (self.gdd, d.gdd_fs2),
            (self.tod, d.tod_fs3),
            (self.fod, d.fod_fs4),
            (self.ri_thickness, d.ri_thickness_mm),
            (self.gas_pressure, d.gas_pressure_bar),
            (self.gas_path, d.gas_path_cm),
        ):
            ctrl.blockSignals(True)
            ctrl.set_value(val)
            ctrl.blockSignals(False)
        for mat, c in self._mat_ctrls.items():
            c.blockSignals(True)
            c.set_value(d.material_thickness_mm.get(mat, 0.0))
            c.blockSignals(False)
        # active gas: reflect the combo and (re)register n(λ) at the saved pressure
        self.gas_combo.blockSignals(True)
        self.gas_combo.setCurrentText(d.gas_name or _GAS_NONE)
        self.gas_combo.blockSignals(False)
        self._gas_name = d.gas_name
        self._unregister_gas()
        if d.gas_name:
            self._register_gas()
        # mirror stack (suppress its bulk-changed; on_enter recomputes once)
        self.mirror_stack.blockSignals(True)
        self.mirror_stack.load_specs(d.mirrors)
        self.mirror_stack.blockSignals(False)
        self._restore_ri(d)

    def _restore_ri(self, d) -> None:
        """Re-register a saved refractiveindex.info material if the db is available.

        Best-effort: the cascading shelf/book/page combos are only populated once
        the user loads the database, but the saved page is enough to re-register
        the material (and thus re-apply it) so the reloaded pulse matches.
        """
        if not (d.ri_page and refractive_db.available()):
            return
        with contextlib.suppress(Exception):
            n_func, rng = refractive_db.make_material(d.ri_shelf, d.ri_book, d.ri_page)
            if self._ri_key:
                materials.unregister_material(self._ri_key)
            self._ri_key = f"RI:{d.ri_book}/{d.ri_page}"
            materials.register_material(self._ri_key, n_func)
            self._ri_range = rng

    # -- tuning -------------------------------------------------------------
    def _extra_dispersion(self):
        """Material thicknesses (m) and mirror bounces from all controls.

        Combines the built-in Sellmeier rows, the optional refractiveindex.info
        material, the optional active gas (path length as thickness, pressure
        baked into its registered ``n(λ)``), and the mirror stack (signed bounce
        counts; negative back-propagates a beam-path mirror out of the pulse).
        """
        mats = {
            m: c.value() * 1e-3 for m, c in self._mat_ctrls.items() if c.value() != 0
        }
        if self._ri_key and self.ri_thickness.value() != 0:
            mats[self._ri_key] = self.ri_thickness.value() * 1e-3
        if self._gas_key and self.gas_path.value() != 0:
            mats[self._gas_key] = self.gas_path.value() * 1e-2  # cm -> m
        bounces = self.mirror_stack.bounces_dict() or None
        return mats, bounces

    def _mirror_base(self, r):
        """The retrieved spectrum (``r``) with only the mirror stack applied.

        Auto-tuners optimise a knob (Taylor, material, gas) on *this* spectrum,
        not the raw retrieval, so "remove the beam-path mirrors, then Auto GDD"
        compresses the residual of the back-propagated pulse — the intended
        workflow. Mirror bounces are physical (fixed), so they are baked in here
        rather than searched.
        """
        # Callers gate on ``self._base``; the guard restates that for the type
        # checker (and fails loudly if the contract is ever broken).
        if self._base is None:
            raise RuntimeError("dispersion stage has no base spectrum yet")
        bounces = self.mirror_stack.bounces_dict() or None
        if bounces is None:
            return self._base
        return dispersion.apply_dispersion(
            r.grid.omega, r.omega0, self._base, mirror_bounces=bounces
        )

    def _modified(self):
        r = self.state.result
        if r is None or self._base is None:
            raise RuntimeError("no retrieval result to apply dispersion to")
        mats, bounces = self._extra_dispersion()
        ew = dispersion.apply_dispersion(
            r.grid.omega,
            r.omega0,
            self._base,
            gdd=self.gdd.value() * 1e-30,
            tod=self.tod.value() * 1e-45,
            fod=self.fod.value() * 1e-60,
            material_thicknesses=mats,
            mirror_bounces=bounces,
        )
        return replace(r, spectrum=ew)

    def _changed(self, *_):
        if self._base is None:
            return
        # Keep the persisted parameters in step with the live controls so saving
        # (here or from a later stage) captures the dispersion settings.
        self._sync_to_params()
        mod = self._modified()
        pr = process_result(mod, energy=self.state.load.energy_j or None)
        self.fwhm_label.setText(f"{pr.fwhm_retr / 1e-15:.2f} fs")
        self.tl_label.setText(f"{pr.fwhm_tl / 1e-15:.2f} fs")
        self._set_power_labels(pr)
        self._draw(mod, pr)

    def _set_power_labels(self, pr) -> None:
        """Show the peak powers (SI-prefixed) or '–' when no pulse energy is set."""
        if pr.peak_power is None:
            self.peak_label.setText("–")
            self.peak_tl_label.setText("–")
            return
        scale, unit = plotting.si_power(max(pr.peak_power, pr.peak_power_tl or 0.0))
        self.peak_label.setText(f"{pr.peak_power / scale:.3g} {unit}")
        self.peak_tl_label.setText(
            f"{pr.peak_power_tl / scale:.3g} {unit}"
            if pr.peak_power_tl is not None
            else "–"
        )

    def _applied_phase(self):
        r = self.state.result
        if r is None:
            raise RuntimeError("no retrieval result to apply dispersion to")
        mats, bounces = self._extra_dispersion()
        return dispersion.dispersion_phase(
            r.grid.omega,
            r.omega0,
            gdd=self.gdd.value() * 1e-30,
            tod=self.tod.value() * 1e-45,
            fod=self.fod.value() * 1e-60,
            material_thicknesses=mats,
            mirror_bounces=bounces,
        )

    def _draw(self, mod, pr):
        self.canvas.render(lambda fig: self._plot(fig, mod, pr))

    def _plot(self, fig, mod, pr) -> None:
        """Draw the four dispersion panels into ``fig``."""
        axd = fig.subplot_mosaic("ab\ncd")
        # Wavelength, to match the applied-phase panel beneath it.
        plotting.plot_spectral(axd["a"], pr, axis="wavelength")  # phase + fit
        plotting.plot_temporal(axd["b"], pr)
        # bottom-left: only the *applied* dispersion phase
        phi_applied = self._applied_phase()
        phi_applied = (
            phi_applied - phi_applied[pr.mask_w][int(np.argmax(pr.Iw[pr.mask_w]))]
        )
        axd["c"].plot(pr.wavelength[pr.mask_w] / 1e-9, phi_applied[pr.mask_w], c="C0")
        axd["c"].set_xlabel("Wavelength (nm)")
        axd["c"].set_ylabel("Phase (rad)")
        axd["c"].set_title("Applied phase")
        tc, lam, S = spectrogram(mod)
        plotting.plot_spectrogram(axd["d"], tc, lam, S)

    def _auto_taylor(self):
        if self._base is None:
            return
        r = self.state.result
        if r is None:
            return
        base = self._mirror_base(r)
        # Search bounds from ~10× the residual dispersion fitted on the pulse, so
        # the optimiser resolves a small optimum (a short pulse needs little GDD)
        # instead of searching the full ±10⁴ fs² slider range.
        pr = process_result(replace(r, spectrum=base))

        def _b(fit, floor):
            span = max(10.0 * abs(fit), floor)
            return (-span, span)

        bounds = (_b(pr.gdd_fs2, 50.0), _b(pr.tod_fs3, 500.0), _b(pr.fod_fs4, 5000.0))
        self.set_status("Optimising…")
        g, t, f = dispersion.auto_taylor(
            r.grid,
            base,
            r.omega0,
            bounds=bounds,
            method=self.method.currentText(),
        )
        for ctrl, val in (
            (self.gdd, g / 1e-30),
            (self.tod, t / 1e-45),
            (self.fod, f / 1e-60),
        ):
            ctrl.blockSignals(True)
            ctrl.set_value(val)
            ctrl.blockSignals(False)
        self.set_status(
            f"Auto: GDD {g / 1e-30:.0f} fs², TOD {t / 1e-45:.0f} fs³, "
            f"FOD {f / 1e-60:.0f} fs⁴"
        )
        self._changed()

    def _auto_material(self, mat):
        if self._base is None:
            return
        r = self.state.result
        if r is None:
            return
        base = self._mirror_base(r)
        pr = process_result(replace(r, spectrum=base))
        # thickness (mm) that would cancel the fitted GDD, then search ±10× that
        gdd_per_mm = dispersion.material_gdd(mat, r.omega0, 1e-3)
        t_need = -pr.gdd_fs2 / gdd_per_mm if gdd_per_mm else 0.0
        span = max(10.0 * abs(t_need), 1.0)  # mm
        thickness = dispersion.auto_material(
            r.grid,
            base,
            r.omega0,
            mat,
            bounds=(-span * 1e-3, span * 1e-3),
            method=self.method.currentText(),
        )
        ctrl = self._mat_ctrls[mat]
        ctrl.blockSignals(True)
        ctrl.set_value(thickness * 1e3)
        ctrl.blockSignals(False)
        self._changed()

    def _reset(self):
        for ctrl in (
            self.gdd,
            self.tod,
            self.fod,
            self.ri_thickness,
            self.gas_path,
            *self._mat_ctrls.values(),
        ):
            ctrl.blockSignals(True)
            ctrl.set_value(0.0)
            ctrl.blockSignals(False)
        self.mirror_stack.blockSignals(True)
        self.mirror_stack.clear()
        self.mirror_stack.blockSignals(False)
        # clear the active gas (combo back to none, pressure to 1 bar, unregister)
        self.gas_combo.blockSignals(True)
        self.gas_combo.setCurrentText(_GAS_NONE)
        self.gas_combo.blockSignals(False)
        self.gas_pressure.blockSignals(True)
        self.gas_pressure.set_value(1.0)
        self.gas_pressure.blockSignals(False)
        self._gas_name = ""
        self._unregister_gas()
        self._changed()

    def _save(self):
        if self.state.result is None:
            return
        folder = QFileDialog.getExistingDirectory(self, "Select output folder")
        if not folder:
            return
        mod = self._modified()
        self._sync_to_params()  # ensure the saved options.toml captures dispersion
        measured = self.state.tracedata.trace if self.state.tracedata else None
        # Process with the pulse energy so the saved HDF5 carries peak power.
        pr = process_result(
            mod, measured=measured, energy=self.state.load.energy_j or None
        )
        save.save_result(
            mod,
            os.path.join(folder, "result_dispersion.h5"),
            processed=pr,
            measured=measured,
            # a synthetic/simulated session's ground truth belongs in every saved
            # file, as it does in the Retrieve stage's result.h5
            truth=self.state.truth,
            force=True,
        )
        save.save_options(self.state.to_options(), os.path.join(folder, "options.toml"))
        self.canvas.save_figure(os.path.join(folder, "dispersion.pdf"), dpi=600)
        self.set_status(f"Saved to {folder}")
