"""A stack of beam-path mirrors for the dispersion stage.

The retrieved pulse is usually measured *after* bouncing off several optics, and
we want to back-propagate (remove) their dispersion to recover it upstream. A
real measurement mixes mirrors — e.g. 1× Si @ 0°, 6× MgF₂/Al @ 0°, 2× MgF₂/Al
@ 45° (s-pol) — so this module provides an editable list of mirror *rows*, each
``(mirror, polarisation, bounces, add/remove)``.

Each :class:`MirrorRow` registers its per-bounce phase (and, for coatings, its
reflectivity) under a unique key in :mod:`croak.dispersion`, and contributes a
*signed* bounce count (negative = remove) to the ``mirror_bounces`` dictionary
that :func:`croak.dispersion.apply_dispersion` consumes. :class:`MirrorStack`
aggregates the rows.
"""

from __future__ import annotations

import contextlib

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import dispersion, io, mirrors
from ..session.params import BeamPathMirror
from .canvas import combo, spin

__all__ = ["MirrorRow", "MirrorStack"]

_CUSTOM = "Custom file…"
_LAM_UNITS = ["nm", "µm", "m"]
_MAX_BOUNCES = 200
#: directions: a remove row back-propagates (negative bounce count).
_REMOVE, _ADD = "remove", "add"


def _range_text(rng: tuple[float, float] | None) -> str:
    """Format a ``(lo, hi)`` wavelength range (m) as ``"lo–hi nm"``."""
    return f"{rng[0] / 1e-9:.0f}–{rng[1] / 1e-9:.0f} nm" if rng else "–"


class MirrorRow(QWidget):
    """One mirror in the stack: type, polarisation, bounces and direction.

    Loads the selected mirror on any change and (re)registers it in
    :mod:`croak.dispersion` under :attr:`key`; emits :attr:`changed` so the stage
    recomputes. Built-in coatings register reflectivity too (applied on
    removal); chirped compressors and custom files are phase-only.
    """

    changed = pyqtSignal()
    status = pyqtSignal(str)
    remove_requested = pyqtSignal(object)
    auto_requested = pyqtSignal(object)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.key = f"_mirror_row_{id(self)}"
        # unregister even if the widget is torn down without an explicit remove
        # (e.g. the whole stage is rebuilt), so module-global keys never leak.
        key = self.key
        self.destroyed.connect(lambda *_: dispersion.unregister_mirror(key))
        self._range: tuple[float, float] | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)

        line = QWidget()
        row = QHBoxLayout(line)
        row.setContentsMargins(0, 0, 0, 0)
        self.type_combo = combo(
            [*mirrors.BUILTIN_COATINGS, *mirrors.BUILTIN_MIRRORS, _CUSTOM],
            next(iter(mirrors.BUILTIN_COATINGS)),
            self._on_type,
        )
        self.pol_combo = combo(["s", "p"], "s", self._on_change)
        self.bounces = spin(0, _MAX_BOUNCES, 1, self._on_change)
        self.dir_combo = combo([_REMOVE, _ADD], _REMOVE, self._on_change)
        auto = QPushButton("Auto")
        auto.setToolTip("Scan the bounce count that maximises the peak power.")
        auto.clicked.connect(lambda: self.auto_requested.emit(self))
        remove = QPushButton("✕")
        remove.setFixedWidth(28)
        remove.setToolTip("Remove this mirror.")
        remove.clicked.connect(lambda: self.remove_requested.emit(self))
        row.addWidget(self.type_combo, stretch=2)
        row.addWidget(self.pol_combo)
        row.addWidget(self.bounces)
        row.addWidget(self.dir_combo)
        row.addWidget(auto)
        row.addWidget(remove)
        outer.addWidget(line)

        self.range_label = QLabel("–")
        self.range_label.setStyleSheet("color: gray;")
        outer.addWidget(self.range_label)

        # custom-file sub-controls, shown only when 'Custom file…' is selected
        self.custom = QWidget()
        self._build_custom(self.custom)
        self.custom.setVisible(False)
        outer.addWidget(self.custom)

        self._reload()

    # -- custom-file controls ----------------------------------------------
    def _build_custom(self, container: QWidget) -> None:
        form = QFormLayout(container)
        form.setContentsMargins(12, 0, 0, 0)
        self.cm_path = QLineEdit()
        pick = QPushButton("…")
        pick.clicked.connect(self._pick_custom)
        prow = QWidget()
        pl = QHBoxLayout(prow)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.addWidget(self.cm_path)
        pl.addWidget(pick)
        form.addRow("file", prow)
        self.cm_lam_col = QComboBox()
        self.cm_val_col = QComboBox()
        form.addRow("λ column", self.cm_lam_col)
        form.addRow("data column", self.cm_val_col)
        self.cm_unit = combo(_LAM_UNITS, "nm")
        form.addRow("λ unit", self.cm_unit)
        self.cm_datatype = combo(["phase (rad)", "GDD (fs²)"], "phase (rad)")
        form.addRow("data type", self.cm_datatype)
        load = QPushButton("Load custom mirror")
        load.clicked.connect(self._load_custom)
        form.addRow(load)

    def _pick_custom(self) -> None:
        fn, _ = QFileDialog.getOpenFileName(self, "Select mirror file")
        if not fn:
            return
        self.cm_path.setText(fn)
        with contextlib.suppress(Exception):
            ncol = io.load_csv(fn).shape[1]
            labels = [f"column {i + 1}" for i in range(ncol)]
            self._fill_combo(self.cm_lam_col, labels, 0)
            self._fill_combo(self.cm_val_col, labels, min(1, ncol - 1))
            self.cm_unit.setCurrentText(io.csv_wavelength_unit(fn))

    def _load_custom(self) -> None:
        path = self.cm_path.text()
        if not path:
            self.status.emit("Choose a mirror file first.")
            return
        try:
            mat = io.load_csv(path)
            lam = mat[:, self.cm_lam_col.currentIndex()] * io.unit_to_si(
                self.cm_unit.currentText()
            )
            vals = mat[:, self.cm_val_col.currentIndex()]
            if self.cm_datatype.currentText().startswith("GDD"):
                mode, vals = "gdd", vals * 1e-30  # fs² → s²
            else:
                mode = "phase"
            phase_fn, rng = mirrors.mirror_from_arrays(lam, vals, mode=mode)
        except Exception as exc:
            self.status.emit(f"Custom mirror failed: {exc}")
            return
        dispersion.register_mirror(self.key, phase_fn)  # phase-only
        self._set_range(rng)
        self.status.emit("Custom mirror loaded.")
        self.changed.emit()

    @staticmethod
    def _fill_combo(box: QComboBox, items: list[str], index: int) -> None:
        box.blockSignals(True)
        box.clear()
        box.addItems([str(i) for i in items])
        if 0 <= index < len(items):
            box.setCurrentIndex(index)
        box.blockSignals(False)

    # -- loading / registration --------------------------------------------
    def _on_type(self, _text: str = "") -> None:
        name = self.type_combo.currentText()
        is_coating = name in mirrors.BUILTIN_COATINGS
        self.pol_combo.setEnabled(is_coating)
        self.custom.setVisible(name == _CUSTOM)
        self._reload()

    def _on_change(self, *_) -> None:
        self.changed.emit()

    def _reload(self) -> None:
        """(Re)load and register the selected built-in mirror, then signal."""
        name = self.type_combo.currentText()
        if name == _CUSTOM:
            dispersion.unregister_mirror(self.key)  # registered on explicit Load
            self._set_range(None, text="load a file…")
            self.changed.emit()
            return
        try:
            if name in mirrors.BUILTIN_COATINGS:
                phase_fn, amp_fn, rng = mirrors.load_coating(
                    name, self.pol_combo.currentText()
                )
            else:
                phase_fn, amp_fn, rng = mirrors.load_builtin(name)
            dispersion.register_mirror(self.key, phase_fn, amp_fn)
        except Exception as exc:
            dispersion.unregister_mirror(self.key)
            self.status.emit(f"Mirror load failed: {exc}")
            return
        self._set_range(rng)
        self.changed.emit()

    def _set_range(self, rng: tuple[float, float] | None, text: str = "") -> None:
        self._range = rng
        self.range_label.setText(text or f"valid λ: {_range_text(rng)}")

    # -- public API used by the stack / stage ------------------------------
    def contribution(self) -> tuple[str, int] | None:
        """``(key, signed_bounces)`` for the active mirror, else ``None``.

        The sign encodes the direction: ``remove`` back-propagates (negative).
        Returns ``None`` when the row is inactive (zero bounces or no registered
        phase, e.g. a custom file not yet loaded).
        """
        n = int(self.bounces.value())
        if n == 0 or self.key not in dispersion.MIRRORS:
            return None
        sign = -1 if self.dir_combo.currentText() == _REMOVE else 1
        return self.key, sign * n

    def set_bounces(self, n: int, *, silent: bool = False) -> None:
        """Set the bounce count; ``silent`` suppresses the ``changed`` signal."""
        self.bounces.blockSignals(True)
        self.bounces.setValue(int(n))
        self.bounces.blockSignals(False)
        if not silent:
            self.changed.emit()

    def to_spec(self) -> BeamPathMirror:
        """Capture this row as a serialisable :class:`BeamPathMirror`."""
        name = self.type_combo.currentText()
        pol = self.pol_combo.currentText()
        bounces = int(self.bounces.value())
        direction = self.dir_combo.currentText()
        if name != _CUSTOM:
            return BeamPathMirror(
                name=name, polarization=pol, bounces=bounces, direction=direction
            )
        return BeamPathMirror(
            name="",
            polarization=pol,
            bounces=bounces,
            direction=direction,
            custom_file=self.cm_path.text(),
            custom_lam_col=self.cm_lam_col.currentIndex(),
            custom_val_col=self.cm_val_col.currentIndex(),
            custom_unit=self.cm_unit.currentText(),
            custom_datatype=self.cm_datatype.currentText().split()[0].lower(),
        )

    def load_spec(self, spec: BeamPathMirror) -> None:
        """Populate the row from a :class:`BeamPathMirror` and (re)register it.

        Sets the controls with their signals blocked, then triggers a single
        (re)load so the mirror is registered exactly once and the stage redraws.
        """
        name = spec.name or _CUSTOM
        widgets = (self.type_combo, self.pol_combo, self.bounces, self.dir_combo)
        for w in widgets:
            w.blockSignals(True)
        self.type_combo.setCurrentText(name)
        self.pol_combo.setCurrentText(spec.polarization)
        self.bounces.setValue(int(spec.bounces))
        self.dir_combo.setCurrentText(spec.direction)
        for w in widgets:
            w.blockSignals(False)
        # reflect the type's polarisation/custom visibility (as _on_type would)
        self.pol_combo.setEnabled(name in mirrors.BUILTIN_COATINGS)
        self.custom.setVisible(name == _CUSTOM)
        if name == _CUSTOM:
            self._restore_custom(spec)
        else:
            self._reload()  # registers the built-in mirror + emits changed

    def _restore_custom(self, spec: BeamPathMirror) -> None:
        """Restore the custom-file sub-controls and load the mirror when set."""
        self.cm_path.setText(spec.custom_file)
        if spec.custom_file:
            with contextlib.suppress(Exception):
                ncol = io.load_csv(spec.custom_file).shape[1]
                labels = [f"column {i + 1}" for i in range(ncol)]
                self._fill_combo(self.cm_lam_col, labels, spec.custom_lam_col)
                self._fill_combo(self.cm_val_col, labels, spec.custom_val_col)
        self.cm_unit.setCurrentText(spec.custom_unit)
        self.cm_datatype.setCurrentText(
            "GDD (fs²)" if spec.custom_datatype == "gdd" else "phase (rad)"
        )
        if spec.custom_file:
            self._load_custom()  # registers + emits changed
        else:
            self.changed.emit()

    def cleanup(self) -> None:
        """Unregister this row's mirror from the dispersion registries."""
        dispersion.unregister_mirror(self.key)


class MirrorStack(QWidget):
    """An editable list of :class:`MirrorRow` widgets plus an 'Add' button."""

    changed = pyqtSignal()
    status = pyqtSignal(str)
    auto_requested = pyqtSignal(object)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._rows: list[MirrorRow] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._rows_box = QVBoxLayout()
        self._rows_box.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self._rows_box)
        add = QPushButton("+ Add mirror")
        add.clicked.connect(lambda: self.add_row())
        layout.addWidget(add)

    def add_row(self) -> MirrorRow:
        """Append a new mirror row and wire its signals."""
        row = MirrorRow()
        row.changed.connect(self.changed)
        row.status.connect(self.status)
        row.auto_requested.connect(self.auto_requested)
        row.remove_requested.connect(self._remove_row)
        self._rows.append(row)
        self._rows_box.addWidget(row)
        self.changed.emit()
        return row

    def _remove_row(self, row: MirrorRow) -> None:
        if row not in self._rows:
            return
        self._rows.remove(row)
        row.cleanup()
        self._rows_box.removeWidget(row)
        row.deleteLater()
        self.changed.emit()

    def clear(self) -> None:
        """Remove every row (unregistering each mirror)."""
        for row in list(self._rows):
            self._rows.remove(row)
            row.cleanup()
            self._rows_box.removeWidget(row)
            row.deleteLater()
        self.changed.emit()

    def bounces_dict(self) -> dict[str, int]:
        """Aggregate ``{key: signed_bounces}`` over the active rows."""
        out: dict[str, int] = {}
        for row in self._rows:
            contrib = row.contribution()
            if contrib is not None:
                key, n = contrib
                out[key] = out.get(key, 0) + n
        return out

    def to_specs(self) -> list[BeamPathMirror]:
        """Capture the stack as a serialisable list of :class:`BeamPathMirror`."""
        return [row.to_spec() for row in self._rows]

    def load_specs(self, specs: list[BeamPathMirror]) -> None:
        """Rebuild the stack from saved specs, emitting ``changed`` once at the end."""
        self.blockSignals(True)
        try:
            self.clear()
            for spec in specs:
                self.add_row().load_spec(spec)
        finally:
            self.blockSignals(False)
        self.changed.emit()
