"""The numbers beneath the Retrieve stage's plots, as large derived values.

The plots are a *qualitative* view; these are the numbers you compare between
runs, and they move live as the retrieval converges. Each is a small grey name
over a large value (:class:`~croak.gui.sheet.DerivedGrid`), in two rows:

* **the run and its pulse:** iteration, FROG error R and elapsed time, then the
  retrieved spectrum's peak wavelength and the pulse's FWHM;
* **the forward model and the truth:** the three forward-model extras
  (delay-zero offset, slab thickness, geometric smearing), each named with
  ``fit`` or ``fixed`` so it is obvious whether the retriever was free to move
  it; then, when the pulse is known, the three truth errors
  (:mod:`croak.truth_metrics`). ``R`` says how well the model fits the *trace*,
  the ε's say how close the answer is to the true *pulse*, and on a non-unique
  inverse problem those can disagree.

The widget is pure display, fed by :class:`~croak.gui.stage_retrieve.StageRetrieve`.
"""

from __future__ import annotations

from collections.abc import Mapping

from PySide6.QtWidgets import QVBoxLayout, QWidget

from .sheet import DerivedGrid

__all__ = ["DASH", "NUMBERS", "RetrievalReadout"]

DASH = "–"
"""Shown for a number that is undefined (the house convention)."""

NUMBERS: tuple[tuple[tuple[str, str, str], ...], ...] = (
    (
        ("iteration", "Iteration", "Solver iterations completed."),
        ("error", "Error R", "Normalised FROG error of the current iterate."),
        ("elapsed", "Elapsed", "Wall-clock time since the run started."),
        (
            "lam_peak",
            "λ peak",
            "Wavelength of the retrieved spectrum's intensity maximum.",
        ),
        ("fwhm", "FWHM", "Full width at half maximum of the retrieved pulse."),
    ),
    (
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
            "Geometric time-smearing width (delay σ) the forward model convolved "
            "with, and its factor on the nominal kernel.",
        ),
        # Known-truth errors: only the synthetic and simulated workflows can fill
        # these, and the two spectral ones need the truth's complex field.
        (
            "eps_It",
            "ε(Iₜ)",
            "Temporal-intensity error against the known truth: the energy-"
            "normalised rms distance between |E(t)|², blind to energy scale and "
            "time origin.",
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
            "Complex-field error against the known truth (Geib's ε): minimised "
            "over amplitude scale, absolute phase and delay — the three a PNPS "
            "measurement cannot determine. Small ε(I𝜔) with large ε(E𝜔) means "
            "the amplitude is right and the phase is not. Needs a complex truth. "
            "Scored in the measurement frame (spectrum_frame_p undone).",
        ),
    ),
)
"""The rows of numbers: ``(key, name, tooltip)`` for each, in reading order."""

_NAMES = {key: name for row in NUMBERS for key, name, _tip in row}


class RetrievalReadout(QWidget):
    """The retrieval's numbers; every update is safe to make at any time."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.grid = DerivedGrid(columns=max(len(row) for row in NUMBERS))
        for row in NUMBERS:
            for _key, name, tip in row:
                self.grid.add(name, tip)
            self.grid.end_row()
        layout.addWidget(self.grid)

    def text(self, key: str) -> str:
        """The value shown for ``key``."""
        return self.grid.values[_NAMES[key]].text()

    def name(self, key: str) -> str:
        """The name shown over ``key``, with any note."""
        return self.grid.names[_NAMES[key]].text()

    def clear(self) -> None:
        """Show every number as :data:`DASH`, with no notes."""
        self.grid.clear()
        for name in _NAMES.values():
            self.grid.note(name, "")

    def set_scalars(self, values: Mapping[str, str]) -> None:
        """Show ``{key: text}``; keys absent are left as they are."""
        for key, text in values.items():
            self.grid.set(_NAMES[key], text)

    def set_notes(self, notes: Mapping[str, str]) -> None:
        """Add ``{key: note}`` to the numbers' names, e.g. ``fit`` or ``fixed``."""
        for key, note in notes.items():
            self.grid.note(_NAMES[key], note)
