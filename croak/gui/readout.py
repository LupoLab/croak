"""Numeric read-out strip shown beneath the Retrieve stage's figure.

The 12-panel figure is a *qualitative* view: it annotates only the FROG error, the
fitted GDD/TOD and the three temporal FWHMs. This strip carries the numbers you
actually compare between runs, and it updates live as the retrieval converges.

Two blocks, stacked:

* a **parameter table** with one row per curve drawn in the figure — ``R``
  (retrieved), ``M`` (measured spectrum), ``TL`` (transform-limited) and ``truth``
  — and one column per quantity: peak wavelength, spectral FWHM, pulse FWHM and
  peak power. The row labels deliberately match the figure's legend entries so the
  table reads as a key to the panels above it. Not every cell is defined: the
  measured spectrum has no phase, hence no temporal envelope, and the
  transform-limited pulse is the retrieved ``|E(ω)|`` with the phase removed, so
  its spectrum *is* the retrieved one and repeating it would imply independent
  information. Undefined cells show an en dash.
* a **scalar block**: iteration, elapsed time and FROG error on one line; then the
  three forward-model extras (delay-zero offset, slab thickness, geometric
  smearing), each marked ``(fit)`` or ``(fixed)`` so it is obvious at a glance
  whether the retriever was free to move it; then, when the pulse is known, the
  three truth errors (:mod:`croak.truth_metrics`). The last line answers a
  different question from the FROG error above it: ``R`` says how well the model
  fits the *trace*, the ε's say how close the answer is to the true *pulse*, and
  on a non-unique inverse problem those can disagree.

The widget is pure display: it holds no state beyond the label text and is fed by
:class:`~croak.gui.stage_retrieve.StageRetrieve`.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFontDatabase
from PyQt6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

__all__ = ["RetrievalReadout", "DASH"]

#: Shown in every cell whose quantity is undefined for that row (the house
#: convention, matching the dispersion stage's power read-outs).
DASH = "–"

# Curve rows, keyed to the figure's legend labels.
_ROWS: tuple[tuple[str, str], ...] = (
    ("R", "Retrieved pulse."),
    ("M", "Independent measured spectrum (intensity only — no temporal envelope)."),
    ("TL", "Transform limit of the retrieved spectrum (its spectrum is R's)."),
    ("truth", "Known ground truth, for synthetic and simulated traces."),
)

# Quantity columns. The peak-power header gains an SI unit when a pulse energy is
# known; the others carry their unit permanently.
_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("lam_peak", "λ peak (nm)", "Wavelength of the spectral intensity maximum."),
    ("lam_fwhm", "λ FWHM (nm)", "Full width at half maximum of the spectrum."),
    ("fwhm", "FWHM (fs)", "Full width at half maximum of the temporal intensity."),
    ("peak_power", "Peak power", "Needs a pulse energy — set one on the Load stage."),
)

_SCALARS: tuple[tuple[str, str, str], ...] = (
    ("iteration", "Iteration", "Solver iterations completed."),
    ("elapsed", "Elapsed", "Wall-clock time since Run was pressed."),
    ("error", "Error R", "Normalised FROG error of the current iterate."),
    (
        "tau0",
        "τ₀",
        "Delay-zero offset of the measured trace: fitted with 'fit τ₀', else 0.",
    ),
    (
        "thickness",
        "Thickness",
        "Dispersive-slab thickness the forward model propagated through.",
    ),
    (
        "smearing",
        "Smearing",
        "Geometric time-smearing width (delay σ) the forward model convolved with.",
    ),
    # Known-truth errors: only the synthetic and simulated workflows can fill
    # these, and the two spectral ones need the truth's complex field.
    (
        "eps_It",
        "ε(Iₜ)",
        "Temporal-intensity error against the known truth: the energy-normalised "
        "rms distance between |E(t)|², blind to energy scale and time origin.",
    ),
    (
        "eps_Iw",
        "ε(I𝜔)",
        "Spectral-intensity error against the known truth: |Ẽ(ω)|² over the "
        "truth's band, blind to scale and to ALL phase. Needs a complex truth. "
        "Scored in the measurement frame: any spectrum_frame_p reweighting is "
        "undone first, so a phase-only retrieval reads 0.",
    ),
    (
        "eps_Ew",
        "ε(E𝜔)",
        "Complex-field error against the known truth (Geib's ε): minimised over "
        "amplitude scale, absolute phase and delay — the three a PNPS "
        "measurement cannot determine. Small ε(I𝜔) with large ε(E𝜔) means the "
        "amplitude is right and the phase is not. Needs a complex truth. "
        "Scored in the measurement frame (spectrum_frame_p undone).",
    ),
)


def _cell(*, bold: bool = False, fixed_width: bool = False) -> QLabel:
    """A right-aligned read-out label (fixed-width digits keep columns lined up)."""
    label = QLabel(DASH if not bold else "")
    label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    if bold:
        font = label.font()
        font.setBold(True)
        label.setFont(font)
    elif fixed_width:
        label.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
    return label


class RetrievalReadout(QWidget):
    """The parameter table and scalar read-outs beneath the retrieval figure.

    Built from plain :class:`~PyQt6.QtWidgets.QLabel` grids rather than a table
    widget: the labels size themselves to their content, so the strip stays the
    few-line height the figure can spare without any per-style height arithmetic,
    and a refresh is a handful of ``setText`` calls with no item allocation.

    All three update methods are safe to call in any order and at any time; the
    widget starts, and :meth:`clear` returns it to, an all-:data:`DASH` state.
    """

    def __init__(self):
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 2, 0, 2)
        outer.setSpacing(2)

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(self._build_table(), stretch=1)
        outer.addWidget(row)
        outer.addWidget(self._build_scalars())

    # -- construction -------------------------------------------------------
    def _build_table(self) -> QGroupBox:
        box = QGroupBox("Pulse parameters")
        grid = QGridLayout(box)
        grid.setContentsMargins(8, 2, 8, 4)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(1)

        self._header: dict[str, QLabel] = {}
        for col, (key, title, tip) in enumerate(_COLUMNS, start=1):
            head = _cell(bold=True)
            head.setText(title)
            head.setToolTip(tip)
            grid.addWidget(head, 0, col)
            self._header[key] = head

        self._cells: dict[tuple[str, str], QLabel] = {}
        for r, (name, tip) in enumerate(_ROWS, start=1):
            label = QLabel(name)
            label.setToolTip(tip)
            font = label.font()
            font.setBold(True)
            label.setFont(font)
            grid.addWidget(label, r, 0)
            for col, (key, _title, _tip) in enumerate(_COLUMNS, start=1):
                cell = _cell(fixed_width=True)
                cell.setToolTip(tip)
                grid.addWidget(cell, r, col)
                self._cells[name, key] = cell
        grid.setColumnStretch(len(_COLUMNS) + 1, 1)
        return box

    def _build_scalars(self) -> QWidget:
        # A bare grid rather than a second group box: the "Name:" labels are
        # self-describing, and dropping the title and frame keeps the whole strip
        # to the few lines the figure can spare.
        box = QWidget()
        grid = QGridLayout(box)
        grid.setContentsMargins(10, 0, 8, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(2)
        self._scalars: dict[str, QLabel] = {}
        # Two rows of three (name, value) pairs: the solver's progress on top, the
        # forward model's fitted-or-fixed extras beneath.
        for i, (key, title, tip) in enumerate(_SCALARS):
            r, c = divmod(i, 3)
            name = QLabel(f"{title}:")
            name.setToolTip(tip)
            value = _cell()
            value.setAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
            value.setToolTip(tip)
            grid.addWidget(name, r, 2 * c)
            grid.addWidget(value, r, 2 * c + 1)
            self._scalars[key] = value
        grid.setColumnStretch(6, 1)
        return box

    # -- updates ------------------------------------------------------------
    def clear(self) -> None:
        """Blank every cell and scalar back to :data:`DASH`."""
        for cell in self._cells.values():
            cell.setText(DASH)
        for cell in self._scalars.values():
            cell.setText(DASH)
        self.set_power_unit(None)

    def set_power_unit(self, unit: str | None) -> None:
        """Put the shared SI power prefix in the peak-power column header.

        One prefix for the whole column (rather than a unit per cell) keeps the
        numbers comparable at a glance and the digits aligned.
        """
        title = "Peak power" if unit is None else f"Peak power ({unit})"
        self._header["peak_power"].setText(title)

    def set_cells(self, row: str, values: dict[str, str]) -> None:
        """Set one curve row from ``{column key: formatted text}``.

        Columns absent from ``values`` are reset to :data:`DASH`, so a caller need
        only supply what that row defines.
        """
        for key, _title, _tip in _COLUMNS:
            self._cells[row, key].setText(values.get(key, DASH))

    def set_scalars(self, values: dict[str, str]) -> None:
        """Set the named scalar read-outs; keys absent are left untouched."""
        for key, text in values.items():
            self._scalars[key].setText(text)
