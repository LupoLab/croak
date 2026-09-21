"""Stage 1 — load a measured trace (+ optional spectrum/backgrounds), preview it.

2-D datasets feed the trace picker
and 1-D datasets the wavelength/scan pickers; units and scan-type are
auto-detected; the spectrum may be CSV or HDF5/NPZ (with its own dataset
picker) and supports background subtraction and calibration. On a successful
load the stage seeds the preprocess windows from the data extents.
"""

from __future__ import annotations

import contextlib
import os

import numpy as np
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QWidget,
)

from .. import io
from ..interactions import get_interaction
from ..plotting import cmap_white, signed_pcolormesh
from ..session.pipeline import assemble_load_data
from .base import Stage
from .canvas import MplCanvas, check, combo, group, with_toolbar
from .state import PreprocParams
from .widgets import SciSpinBox

_LAM_UNITS = ["nm", "µm", "m"]
_SCAN_UNITS = ["fs", "ps", "s", "µm", "mm", "m"]

# Trust window (nm) for scaling the trace preview: the display ``vmax`` is taken
# from the in-window signal so out-of-band detector noise — strongly amplified by
# any (λ/µm)^exp tilt set on the marginal-check stage — cannot wash the image out.
# The marginal-vs-prediction comparison itself lives on the marginal-check stage.
_PREVIEW_TRUST_NM = (190.0, 1000.0)


# Bounds depend only on the data extents (not on user choices), so they are
# refreshed on every load even when the user's window *values* are preserved.
_PREPROC_BOUND_FIELDS = (
    "tau_bound_min_fs",
    "tau_bound_max_fs",
    "lamm_bound_min_nm",
    "lamm_bound_max_nm",
    "lam_bound_min_nm",
    "lam_bound_max_nm",
)


def trace_signature(p) -> tuple:
    """Identity of the *trace* a load produces, for change detection.

    Two loads share a signature iff they yield a trace with the same delay /
    wavelength extents and carrier scaling — i.e. the same data-seeded
    preprocess defaults. Reloading the same trace with tweaked corrections
    (background, calibration, an independent spectrum) keeps the signature, so
    the user's windowing/filtering is preserved; loading a different file,
    dataset, orientation, unit, scan type or interaction changes it.
    """
    return (
        p.frog_path,
        p.ifrog_name,
        p.lam_name,
        p.scan_name,
        p.transpose,
        p.lam_unit,
        p.scan_unit,
        p.scan_type,
        p.interaction,
    )


def compute_preproc_defaults(data) -> PreprocParams:
    """A fresh, data-seeded :class:`PreprocParams`.

    Returns the parameters the preprocess stage would start from for this trace
    if re-initialised: a default :class:`PreprocParams` with the grid/window
    extents and bounds seeded from the loaded data. Used both to seed a new
    trace and to back the preprocess "Reset defaults" button.
    """
    pp = PreprocParams()
    scan = data["scanaxis"]
    tau = io.position_to_delay(scan) if data["input_unit"] == "position" else scan
    tmin = float(np.floor(tau.min() / 1e-15))
    tmax = float(np.ceil(tau.max() / 1e-15))
    # τ range (delay to use) and τm window (delay filtering) default to the full
    # scan delay range; the time grid spans 2× it
    pp.tau_min_fs, pp.tau_max_fs = tmin, tmax
    pp.taum_min_fs, pp.taum_max_fs = tmin, tmax
    pp.tau_bound_min_fs, pp.tau_bound_max_fs = tmin, tmax
    pp.trange_fs = 2.0 * (tmax - tmin)

    lam_nm = data["lam"] / 1e-9
    lmin = float(np.floor(lam_nm.min()))
    lmax = float(np.ceil(lam_nm.max()))
    # measurement spectral window (λm): bounds are the trace extents, can narrow
    pp.lamm_min_nm, pp.lamm_max_nm = lmin, lmax
    pp.lamm_bound_min_nm, pp.lamm_bound_max_nm = lmin, lmax

    # retrieval grid λ: default to the trace extents × ω0 scaling (= fundamental
    # range). Bounds stay generous so the user can widen the grid if needed.
    scale = get_interaction(data["interaction"]).omega0_scale
    pp.lam_min_nm = float(np.floor(lmin * scale))
    pp.lam_max_nm = float(np.ceil(lmax * scale))
    pp.lam_bound_min_nm = max(50.0, 0.5 * pp.lam_min_nm)
    pp.lam_bound_max_nm = min(6000.0, 1.5 * pp.lam_max_nm)
    return pp


def _clamp_preproc_to_bounds(pp: PreprocParams) -> None:
    """Clamp the window *values* into the (possibly refreshed) slider bounds."""
    pp.lam_min_nm = min(max(pp.lam_min_nm, pp.lam_bound_min_nm), pp.lam_bound_max_nm)
    pp.lam_max_nm = min(max(pp.lam_max_nm, pp.lam_bound_min_nm), pp.lam_bound_max_nm)
    pp.lamm_min_nm = min(
        max(pp.lamm_min_nm, pp.lamm_bound_min_nm), pp.lamm_bound_max_nm
    )
    pp.lamm_max_nm = min(
        max(pp.lamm_max_nm, pp.lamm_bound_min_nm), pp.lamm_bound_max_nm
    )
    for attr in ("tau_min_fs", "tau_max_fs", "taum_min_fs", "taum_max_fs"):
        v = getattr(pp, attr)
        setattr(pp, attr, min(max(v, pp.tau_bound_min_fs), pp.tau_bound_max_fs))


def seed_preproc_defaults(state, *, reset_values: bool = True) -> None:
    """Seed the preprocess parameters from the loaded data extents.

    Always records the fresh data-seeded defaults on ``state.preproc_defaults``
    (for the "Reset defaults" button) and refreshes the slider *bounds*. When
    ``reset_values`` is true (a genuinely new trace) the window/grid *values*
    are reset to those defaults; otherwise the user's current values are kept
    (only clamped into the refreshed bounds), so moving back to Load and
    reloading the same trace does not discard their windowing.
    """
    defaults = compute_preproc_defaults(state.load_data)
    state.preproc_defaults = defaults
    pp = state.preproc
    if reset_values:
        for f in PreprocParams.__dataclass_fields__:
            setattr(pp, f, getattr(defaults, f))
    else:
        for f in _PREPROC_BOUND_FIELDS:
            setattr(pp, f, getattr(defaults, f))
        _clamp_preproc_to_bounds(pp)


# ---------------------------------------------------------------------------
# Stage widget
# ---------------------------------------------------------------------------
class StageLoad(Stage):
    HELP_TITLE = "Load"
    HELP_INTRO = (
        "Load a measured FROG trace (and an optional independent spectrum and "
        "backgrounds), pick the datasets and units, then preview. 2-D datasets "
        "feed the trace picker and 1-D datasets the wavelength/scan pickers; "
        "units and scan-type are auto-detected but freely overridable."
    )
    HELP = [
        (
            "Trace file",
            [
                (
                    "Ifrog",
                    "The 2-D measured FROG trace dataset (intensity vs "
                    "wavelength × scan). Auto-selected by dimensionality and name.",
                ),
                (
                    "λ vector",
                    "The 1-D wavelength-axis dataset for the trace's spectral "
                    "dimension.",
                ),
                (
                    "scan vector",
                    "The 1-D scan-axis dataset (delay or stage position) for the "
                    "trace's scan dimension.",
                ),
                (
                    "Transpose",
                    "Swap the trace's axes if it loaded as (delay × wavelength) "
                    "instead of (wavelength × delay).",
                ),
            ],
        ),
        (
            "Units & interaction",
            [
                (
                    "λ unit",
                    "Physical unit of the wavelength axis (auto-guessed from the "
                    "values).",
                ),
                (
                    "scan unit",
                    "Physical unit of the scan axis (time fs/ps/s or length µm/mm/m).",
                ),
                (
                    "scan type",
                    "Whether the scan axis is a delay (time) or a stage position "
                    "(length, converted to delay as τ = 2z/c).",
                ),
                (
                    "interaction",
                    "Nonlinear FROG geometry: SHG (second-harmonic), SD "
                    "(self-diffraction) or PG (polarization gating).",
                ),
                (
                    "pulse energy (J)",
                    "Optional measured pulse energy in joules (e.g. from a power "
                    "meter). When > 0 the temporal envelope plots are rescaled "
                    "from normalised units to absolute power and a peak power is "
                    "reported; 0 keeps normalised 'a.u.'. Also editable on the "
                    "dispersion stage.",
                ),
            ],
        ),
        (
            "Trace corrections",
            [
                (
                    "Subtract trace background",
                    "Subtract a separately-measured background spectrum from the "
                    "trace, matched over the overlapping λ range.",
                ),
                ("BG λ", "Wavelength-axis dataset of the trace-background file."),
                ("BG spectrum", "Intensity dataset of the trace-background file."),
                (
                    "Trace calibration curve",
                    "Multiply the trace by a measured spectral calibration curve "
                    "(λ, scale) from a CSV file.",
                ),
                (
                    "calib λ unit",
                    "Wavelength unit of the trace calibration CSV's first column.",
                ),
            ],
        ),
        (
            "Independent spectrum (optional)",
            [
                (
                    "Use independent spectrum",
                    "Load a separately-measured fundamental spectrum, used for the "
                    "spectral-match regularisation and overlay plots.",
                ),
                (
                    "λ data/col",
                    "Wavelength dataset (HDF5/NPZ) or CSV column for the "
                    "independent spectrum.",
                ),
                (
                    "I data/col",
                    "Intensity dataset or CSV column for the independent spectrum.",
                ),
                ("λ unit", "Wavelength unit of the independent spectrum."),
                (
                    "Subtract spectrum background",
                    "Subtract a background spectrum from the independent spectrum.",
                ),
                (
                    "Spectrum calibration curve",
                    "Apply a spectral calibration curve to the independent spectrum.",
                ),
                (
                    "calib λ unit",
                    "Wavelength unit of the spectrum calibration CSV's first column.",
                ),
            ],
        ),
    ]

    def __init__(self, state):
        super().__init__(state)
        p = state.load
        self._frog_fd = None
        self._spec_fd = None
        self._bg_fd = None

        # FROG file
        box, form = group("Trace file")
        self.path_edit = QLineEdit(p.frog_path)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._pick_frog)
        form.addRow(self._row(self.path_edit, browse))
        # all three list every dataset in the file (name + shape); auto-selected
        # by dimensionality + name pattern, but freely overridable.
        self.ifrog_combo = QComboBox()
        self.lam_combo = QComboBox()
        self.scan_combo = QComboBox()
        self.ifrog_combo.currentIndexChanged.connect(self._on_ifrog_changed)
        self.lam_combo.currentIndexChanged.connect(self._on_lam_changed)
        self.scan_combo.currentIndexChanged.connect(self._on_scan_changed)
        form.addRow("Ifrog", self.ifrog_combo)
        form.addRow("λ vector", self.lam_combo)
        form.addRow("scan vector", self.scan_combo)
        form.addRow(
            check("Transpose", p.transpose, lambda v: setattr(p, "transpose", v))
        )
        self.controls.addWidget(box)

        # Units & interaction ("&&" escapes the literal ampersand; see help._plain)
        box, form = group("Units && interaction")
        self.lam_unit_combo = combo(
            _LAM_UNITS, p.lam_unit, lambda v: setattr(p, "lam_unit", v)
        )
        self.scan_unit_combo = combo(
            _SCAN_UNITS, p.scan_unit, lambda v: setattr(p, "scan_unit", v)
        )
        self.scan_type_combo = combo(
            ["delay", "position"], p.scan_type, lambda v: setattr(p, "scan_type", v)
        )
        form.addRow("λ unit", self.lam_unit_combo)
        form.addRow("scan unit", self.scan_unit_combo)
        form.addRow("scan type", self.scan_type_combo)
        self.energy_spin = SciSpinBox(
            0.0, 1e6, p.energy_j, lambda v: setattr(p, "energy_j", v), sigfigs=3
        )
        form.addRow("pulse energy (J)", self.energy_spin)
        form.addRow(
            "interaction",
            combo(
                ["shg", "sd", "pg"],
                p.interaction,
                lambda v: setattr(p, "interaction", v),
            ),
        )
        self.controls.addWidget(box)

        # FROG bg/calib (the (λ/µm)^exp third-order correction is tuned on the
        # marginal-check stage, where its effect on the marginal is visible).
        box, form = group("Trace corrections")
        form.addRow(
            check(
                "Subtract trace background",
                p.frog_bg_use,
                lambda v: setattr(p, "frog_bg_use", v),
            )
        )
        form.addRow(self._file_row("frog_bg_path", self._on_bg_chosen))
        self.bg_lam_combo = combo(
            [], on_change=lambda v: setattr(p, "frog_bg_lam_name", v)
        )
        self.bg_int_combo = combo(
            [], on_change=lambda v: setattr(p, "frog_bg_int_name", v)
        )
        form.addRow("BG λ", self.bg_lam_combo)
        form.addRow("BG spectrum", self.bg_int_combo)
        form.addRow(
            check(
                "Trace calibration curve",
                p.frog_calib_use,
                lambda v: setattr(p, "frog_calib_use", v),
            )
        )
        form.addRow(self._file_row("frog_calib_path", self._on_frog_calib_chosen))
        self.frog_calib_unit_combo = combo(
            _LAM_UNITS,
            p.frog_calib_lam_unit,
            lambda v: setattr(p, "frog_calib_lam_unit", v),
        )
        form.addRow("calib λ unit", self.frog_calib_unit_combo)
        self.controls.addWidget(box)

        # Independent spectrum (+ bg + calib)
        box, form = group("Independent spectrum (optional)")
        form.addRow(
            check(
                "Use independent spectrum",
                p.use_spec,
                lambda v: setattr(p, "use_spec", v),
            )
        )
        form.addRow(self._file_row("spec_path", self._on_spec_chosen))
        self.spec_lam_combo = combo([], on_change=self._on_spec_lam_changed)
        self.spec_int_combo = combo([], on_change=self._on_spec_int_changed)
        form.addRow("λ data/col", self.spec_lam_combo)
        form.addRow("I data/col", self.spec_int_combo)
        self.spec_unit_combo = combo(
            _LAM_UNITS, p.spec_lam_unit, lambda v: setattr(p, "spec_lam_unit", v)
        )
        form.addRow("λ unit", self.spec_unit_combo)
        form.addRow(
            check(
                "Subtract spectrum background",
                p.spec_bg_use,
                lambda v: setattr(p, "spec_bg_use", v),
            )
        )
        form.addRow(self._file_row("spec_bg_path"))
        form.addRow(
            check(
                "Spectrum calibration curve",
                p.spec_calib_use,
                lambda v: setattr(p, "spec_calib_use", v),
            )
        )
        form.addRow(self._file_row("spec_calib_path", self._on_spec_calib_chosen))
        self.spec_calib_unit_combo = combo(
            _LAM_UNITS,
            p.spec_calib_lam_unit,
            lambda v: setattr(p, "spec_calib_lam_unit", v),
        )
        form.addRow("calib λ unit", self.spec_calib_unit_combo)
        self.controls.addWidget(box)

        load_btn = QPushButton("Load && Preview")
        load_btn.clicked.connect(self._load_preview)
        self.controls.addWidget(load_btn)
        self.controls.addWidget(self.status_label)
        self.controls.addStretch(1)

        self.trace_canvas = MplCanvas()
        self.set_plot_area(with_toolbar(self.trace_canvas))

        self.apply_help()

    def on_enter(self) -> None:
        """Repopulate the dataset combos from the saved files when shown.

        The dataset pickers are filled only when a file is browsed, so after a
        session restore (which rebuilds the page from the loaded options but
        does not re-pick files) they start empty. When entering the stage with a
        file already set but its combos unpopulated, read the file and select
        the saved dataset names so the selection is visible and editable.
        """
        p = self.state.load
        if p.frog_path:
            self.path_edit.setText(p.frog_path)
            if self._frog_fd is None:
                self._seed_frog_combos_from_params()
        if self._spec_fd is None:
            self._seed_spec_combos_from_params()
        if self._bg_fd is None:
            self._seed_bg_combos_from_params()

    # -- small layout helpers ----------------------------------------------
    def _row(self, *widgets):
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        for w in widgets:
            rl.addWidget(w)
        return row

    def _start_dir(self, *candidates: str) -> str:
        """Folder to open a file dialog in.

        The first candidate path with an existing parent directory, so browsing
        starts where the data lives (e.g. the trace folder) rather than in the
        process's working directory.
        """
        for c in candidates:
            if c:
                d = os.path.dirname(c)
                if d and os.path.isdir(d):
                    return d
        return ""

    def _file_row(self, attr, on_chosen=None):
        edit = QLineEdit(getattr(self.state.load, attr))
        edit.textChanged.connect(lambda t: setattr(self.state.load, attr, t))
        btn = QPushButton("…")

        def pick():
            start = self._start_dir(
                getattr(self.state.load, attr), self.state.load.frog_path
            )
            fn, _ = QFileDialog.getOpenFileName(self, "Select file", start)
            if fn:
                edit.setText(fn)
                setattr(self.state.load, attr, fn)
                if on_chosen:
                    on_chosen()

        btn.clicked.connect(pick)
        return self._row(edit, btn)

    # -- FROG file handling -------------------------------------------------
    def _pick_frog(self):
        fn, _ = QFileDialog.getOpenFileName(
            self,
            "Select trace file",
            self._start_dir(self.state.load.frog_path),
            filter="Trace files (*.h5 *.hdf5 *.npz)",
        )
        if fn:
            self.path_edit.setText(fn)
            self.state.load.frog_path = fn
            self._populate_frog()

    def _populate_frog(self):
        p = self.state.load
        try:
            fd = io.list_datasets(p.frog_path)
        except Exception as exc:
            self.set_status(f"Could not read file: {exc}")
            return
        self._frog_fd = fd
        names, labels = fd.names, fd.option_labels

        def auto(kind, ndim):
            cands = [i for i, d in enumerate(fd.datasets) if d.ndim == ndim]
            if not cands:
                cands = list(range(len(names)))
            j = io.match_dataset([names[i] for i in cands], io.PATTERNS[kind])
            return cands[j] if j is not None else cands[0]

        ii, li, si = auto("ifrog", 2), auto("lambda", 1), auto("scan", 1)
        self._fill(self.ifrog_combo, labels, ii)
        self._fill(self.lam_combo, labels, li)
        self._fill(self.scan_combo, labels, si)
        p.ifrog_name, p.lam_name, p.scan_name = names[ii], names[li], names[si]
        self._autodetect_units()
        self.set_status(f"{len(names)} datasets found. Click 'Load & Preview'.")

    def _seed_frog_combos_from_params(self):
        """Fill the trace dataset combos from the saved file, keeping saved names.

        Unlike :meth:`_populate_frog` (which auto-detects), this selects the
        dataset names already in ``state.load`` — used by :meth:`on_enter` after
        a session restore, where the combos start empty.
        """
        p = self.state.load
        try:
            fd = io.list_datasets(p.frog_path)
        except Exception:
            return
        self._frog_fd = fd
        labels, names = fd.option_labels, fd.names

        def index_of(name):
            return names.index(name) if name in names else 0

        self._fill(self.ifrog_combo, labels, index_of(p.ifrog_name))
        self._fill(self.lam_combo, labels, index_of(p.lam_name))
        self._fill(self.scan_combo, labels, index_of(p.scan_name))

    def _seed_spec_combos_from_params(self):
        """Fill the spectrum dataset/column combos from the saved selection."""
        p = self.state.load
        if not (p.use_spec and p.spec_path):
            return
        if p.spec_is_csv:
            self._fill(self.spec_lam_combo, ["column 1", "column 2"], p.spec_lam_col)
            self._fill(self.spec_int_combo, ["column 1", "column 2"], p.spec_int_col)
            return
        try:
            fd = io.list_datasets(p.spec_path)
        except Exception:
            return
        self._spec_fd = fd
        labels, names = fd.option_labels, fd.names

        def index_of(name):
            return names.index(name) if name in names else 0

        self._fill(self.spec_lam_combo, labels, index_of(p.spec_lam_name))
        self._fill(self.spec_int_combo, labels, index_of(p.spec_int_name))

    def _seed_bg_combos_from_params(self):
        """Fill the trace-background dataset combos from the saved selection."""
        p = self.state.load
        if not (p.frog_bg_use and p.frog_bg_path):
            return
        try:
            fd = io.list_datasets(p.frog_bg_path)
        except Exception:
            return
        self._bg_fd = fd
        labels, names = fd.option_labels, fd.names

        def index_of(name):
            return names.index(name) if name in names else 0

        self._fill(self.bg_lam_combo, labels, index_of(p.frog_bg_lam_name))
        self._fill(self.bg_int_combo, labels, index_of(p.frog_bg_int_name))

    def _autodetect_units(self):
        p = self.state.load
        fd = self._frog_fd
        if fd is None:
            return
        with contextlib.suppress(Exception):
            lamv = fd.load(p.lam_name)
            p.lam_unit = io.guess_wavelength_unit(lamv)
            self._set_combo(self.lam_unit_combo, p.lam_unit)
        with contextlib.suppress(Exception):
            scanv = fd.load(p.scan_name)
            stype = io.guess_scanaxis_type(p.scan_name) or "delay"
            p.scan_type = stype
            self._set_combo(self.scan_type_combo, stype)
            p.scan_unit = io.guess_scanaxis_unit(scanv, stype)
            self._set_combo(self.scan_unit_combo, p.scan_unit)

    def _on_ifrog_changed(self, idx):
        fd = self._frog_fd
        if fd is not None and 0 <= idx < len(fd.names):
            self.state.load.ifrog_name = fd.names[idx]

    def _on_lam_changed(self, idx):
        fd = self._frog_fd
        if fd is None or not (0 <= idx < len(fd.names)):
            return
        name = fd.names[idx]
        self.state.load.lam_name = name
        with contextlib.suppress(Exception):
            self.state.load.lam_unit = io.guess_wavelength_unit(fd.load(name))
            self._set_combo(self.lam_unit_combo, self.state.load.lam_unit)

    def _on_scan_changed(self, idx):
        fd = self._frog_fd
        if fd is None or not (0 <= idx < len(fd.names)):
            return
        name = fd.names[idx]
        self.state.load.scan_name = name
        stype = io.guess_scanaxis_type(name) or self.state.load.scan_type
        self.state.load.scan_type = stype
        self._set_combo(self.scan_type_combo, stype)
        with contextlib.suppress(Exception):
            self.state.load.scan_unit = io.guess_scanaxis_unit(fd.load(name), stype)
            self._set_combo(self.scan_unit_combo, self.state.load.scan_unit)

    # -- spectrum / background handling -------------------------------------
    def _on_spec_chosen(self):
        p = self.state.load
        ext = os.path.splitext(p.spec_path)[1].lower()
        p.spec_is_csv = ext in (".csv", ".txt")
        if p.spec_is_csv:
            self._fill(self.spec_lam_combo, ["column 1", "column 2"], 0)
            self._fill(self.spec_int_combo, ["column 1", "column 2"], 1)
            p.spec_lam_col, p.spec_int_col = 0, 1
            p.spec_lam_unit = io.csv_wavelength_unit(p.spec_path)
            self._set_combo(self.spec_unit_combo, p.spec_lam_unit)
        else:
            fd = io.list_datasets(p.spec_path)
            self._spec_fd = fd
            labels = fd.option_labels
            names = fd.names
            li = io.match_dataset(names, io.PATTERNS["lambda"]) or 0
            ii = io.match_dataset(names, io.PATTERNS["ifrog"]) or min(1, len(names) - 1)
            self._fill(self.spec_lam_combo, labels, li)
            self._fill(self.spec_int_combo, labels, ii)
            p.spec_lam_name, p.spec_int_name = names[li], names[ii]
            with contextlib.suppress(Exception):
                p.spec_lam_unit = io.guess_wavelength_unit(
                    io.to_1d(fd.load(names[li]), names[li])
                )
                self._set_combo(self.spec_unit_combo, p.spec_lam_unit)

    def _on_spec_lam_changed(self, label):
        p = self.state.load
        if p.spec_is_csv:
            p.spec_lam_col = self.spec_lam_combo.currentIndex()
        elif self._spec_fd is not None:
            p.spec_lam_name = self._spec_fd.names[self.spec_lam_combo.currentIndex()]

    def _on_spec_int_changed(self, label):
        p = self.state.load
        if p.spec_is_csv:
            p.spec_int_col = self.spec_int_combo.currentIndex()
        elif self._spec_fd is not None:
            p.spec_int_name = self._spec_fd.names[self.spec_int_combo.currentIndex()]

    def _on_bg_chosen(self):
        p = self.state.load
        try:
            fd = io.list_datasets(p.frog_bg_path)
        except Exception as exc:
            self.set_status(f"BG file unreadable: {exc}")
            return
        self._bg_fd = fd
        names = fd.names
        li = io.match_dataset(names, io.PATTERNS["lambda"]) or 0
        ii = io.match_dataset(names, io.PATTERNS["ifrog"]) or min(1, len(names) - 1)
        self._fill(self.bg_lam_combo, fd.option_labels, li)
        self._fill(self.bg_int_combo, fd.option_labels, ii)
        p.frog_bg_lam_name, p.frog_bg_int_name = names[li], names[ii]

    def _on_frog_calib_chosen(self):
        p = self.state.load
        p.frog_calib_lam_unit = io.csv_wavelength_unit(p.frog_calib_path)
        self._set_combo(self.frog_calib_unit_combo, p.frog_calib_lam_unit)

    def _on_spec_calib_chosen(self):
        p = self.state.load
        p.spec_calib_lam_unit = io.csv_wavelength_unit(p.spec_calib_path)
        self._set_combo(self.spec_calib_unit_combo, p.spec_calib_lam_unit)

    # -- combo utilities (the bg/spec combos hold labels; we map to names) --
    def _fill(self, box, items, index):
        box.blockSignals(True)
        box.clear()
        box.addItems([str(i) for i in items])
        if index is not None and 0 <= index < len(items):
            box.setCurrentIndex(index)
        box.blockSignals(False)

    def _set_combo(self, box, value):
        box.blockSignals(True)
        box.setCurrentText(str(value))
        box.blockSignals(False)

    # need bg/spec combos to map their selected index back to dataset names:
    def _sync_named_combos(self):
        p = self.state.load
        if self._spec_fd is not None and not p.spec_is_csv:
            i = self.spec_lam_combo.currentIndex()
            j = self.spec_int_combo.currentIndex()
            if 0 <= i < len(self._spec_fd.names):
                p.spec_lam_name = self._spec_fd.names[i]
            if 0 <= j < len(self._spec_fd.names):
                p.spec_int_name = self._spec_fd.names[j]
        if self._bg_fd is not None:
            i = self.bg_lam_combo.currentIndex()
            j = self.bg_int_combo.currentIndex()
            if 0 <= i < len(self._bg_fd.names):
                p.frog_bg_lam_name = self._bg_fd.names[i]
            if 0 <= j < len(self._bg_fd.names):
                p.frog_bg_int_name = self._bg_fd.names[j]

    # -- load & preview -----------------------------------------------------
    def _load_preview(self):
        self._sync_named_combos()
        try:
            data = assemble_load_data(self.state.load)
        except Exception as exc:
            self.set_status(f"Load failed: {exc}")
            return
        # A new/different trace resets the preprocess windows; reloading the same
        # trace (e.g. after tweaking a background or calibration) keeps them.
        sig = trace_signature(self.state.load)
        new_trace = sig != self.state.trace_sig
        self.state.trace_sig = sig
        self.state.set_load_data(data)
        seed_preproc_defaults(self.state, reset_values=new_trace)
        self._draw(data)
        msg = "Loaded. Proceed to the marginal check."
        if not new_trace:
            msg += " (kept your preprocess settings; use Reset on the preprocess "
            msg += "page to restore defaults.)"
        self.set_status(msg)

    def _draw(self, data):
        """Preview the raw trace with the preprocess-page colour scheme.

        Positive samples use white→viridis, negatives white→red (matching the
        filtering stage), and the linear ``vmax`` is taken from the in-trust-band
        signal so an aggressive (λ/µm)^exp tilt (set on the marginal-check stage)
        cannot let amplified out-of-band noise wash the image out.
        """
        lam_nm = data["lam"] / 1e-9
        scan = data["scanaxis"]
        tau_fs = (
            io.position_to_delay(scan) if data["input_unit"] == "position" else scan
        ) / 1e-15
        trace = data["trace"]
        lo, hi = _PREVIEW_TRUST_NM
        in_band = (lam_nm >= lo) & (lam_nm <= hi)
        band = trace[in_band] if np.any(in_band) else trace
        vmax = float(np.nanmax(np.abs(band))) if band.size else 0.0
        self.trace_canvas.render(
            lambda fig: self._plot_trace(
                fig.add_subplot(111), tau_fs, lam_nm, trace, in_band, vmax
            )
        )

    @staticmethod
    def _plot_trace(ax, tau_fs, lam_nm, trace, in_band, vmax) -> None:
        """Draw the raw-trace preview into ``ax`` (the body of :meth:`_draw`)."""
        signed_pcolormesh(
            ax,
            tau_fs,
            lam_nm,
            trace,
            db=False,
            cmap_pos=cmap_white("viridis"),
            vmax=vmax if vmax > 0 else None,
        )
        sig = lam_nm[in_band & (trace.max(axis=1) > 1e-2 * (vmax or 1.0))]
        if sig.size:
            ax.set_ylim(sig.min(), sig.max())
        ax.set_xlabel("Delay (fs)")
        ax.set_ylabel("Wavelength (nm)")
        ax.set_title("Measured trace")
