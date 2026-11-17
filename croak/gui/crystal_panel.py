"""The SHG crystal and the two beams, as one panel for the Data stage.

The crystal an SHG trace was measured through and the beams that cross in it
describe the apparatus, so they are set with the data: the marginal check
predicts the marginal through them, the marginal correction corrects onto that
prediction, a synthetic trace is generated through them and the retrieval models
them. :class:`CrystalPanel` is that one set of controls, bound to the session's
:class:`~croak.session.params.RetrieveParams` (where the settings are stored, for
session-file compatibility), and each Data page shows it for an SHG trace. The
Retrieve stage shows :func:`retrieve_summary` of the same settings.
"""

from __future__ import annotations

import math

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from .. import crystals, smearing
from ..constants import C
from ..session.params import RetrieveParams
from ..session.pipeline import (
    beam_geometry,
    crystal_setup,
    pulse_carrier,
    pulse_wavelength,
    retrieval_npoints,
)
from .controls import check, combo, dspin, group
from .help import HelpSections, apply_tooltips, flatten
from .state import WizardState

__all__ = [
    "CRYSTAL_HELP",
    "CrystalPanel",
    "crystal_summary",
    "geometry_summary",
    "readout_carrier",
    "retrieve_summary",
]

#: The crystal settings' labels in the GUI and their values in RetrieveParams.
_ANGLE_MODES = {
    "Phase-matched wavelength": "phase-match",
    "External tilt": "tilt",
}
_POLARIZATIONS = {
    "In the crossing plane": "parallel",
    "Perpendicular to the crossing plane": "perpendicular",
}
_DELAYED_BEAMS = {
    "Toward the optic axis": "toward_axis",
    "Away from the optic axis": "away_from_axis",
}

#: The panel's help, for the Data pages' help dialogs and the panel's tooltips.
CRYSTAL_HELP: HelpSections = [
    (
        "Nonlinear crystal",
        [
            (
                "Model the crystal",
                "Model the SHG crystal: phase matching, group-velocity mismatch "
                "and the generation prefactor, for crystals thick enough that the "
                "ideal-SHG model fails. The marginal check then predicts the "
                "marginal through it (and the marginal correction corrects onto "
                "that prediction), a synthetic trace is generated through it, and "
                "the retrieval models it; the retrieved field is the field at the "
                "crystal's entrance face.",
            ),
            ("Crystal", "Uniaxial crystal (Type I, oo→e)."),
            (
                "Thickness (µm)",
                "Crystal thickness along its face normal. The depth nodes are the "
                "number the phase mismatch needs (croak.crystals.depth_points; "
                "read-out below, once the trace is prepared).",
            ),
            (
                "Cut angle θc (°)",
                "The manufacturer's cut: the angle between the optic axis and the "
                "face normal.",
            ),
            (
                "Angle set by",
                "Phase-matched wavelength: the crystal is tilted to phase-match "
                "the wavelength below (the tilt is solved and shown). External "
                "tilt: the tilt you set, measured outside the crystal.",
            ),
            (
                "Phase-matched λ (nm)",
                "Fundamental wavelength the crystal is tuned to phase-match.",
            ),
            ("External tilt γ (°)", "Tilt of the crystal from normal incidence."),
            (
                "Polarisation",
                "The fundamental's polarisation relative to the plane in which the "
                "beams cross. In the crossing plane, the beams cross perpendicular "
                "to the crystal's principal plane and the trace stays symmetric in "
                "delay. Perpendicular to it, they cross in the principal plane: "
                "the signal's tilt then makes the trace asymmetric, and the "
                "delayed beam's side matters.",
            ),
            (
                "Delayed beam",
                "With the beams crossing in the principal plane: whether the "
                "delayed beam is on the optic axis's side of the bisector. A "
                "crystal's markings do not show it; the two choices mirror the "
                "model trace in delay, so retrieve with both and keep the better "
                "fit.",
            ),
        ],
    ),
    (
        "Beam geometry",
        [
            (
                "Beam separation (mm)",
                "Distance between the two beams at the focusing optic. With the "
                "focal length it sets the crossing half-angle atan(d / 2f), which "
                "enters the phase matching. 0 models collinear beams.",
            ),
            (
                "Focal length (mm)",
                "Focal length of the optic that crosses the beams (the same value "
                "as the Retrieve stage's focal mixture).",
            ),
            (
                "Beam diameter (mm)",
                "1/e² diameter of each beam at the focusing optic. It sets the "
                "focal spot, across which the two beams' relative delay varies: "
                "the delay smearing (read-out below), which the Retrieve stage "
                "can model.",
            ),
        ],
    ),
]


def _label(labels: dict[str, str], value: str) -> str:
    """Return the GUI label of a stored setting ``value``, or ``value`` if unknown.

    An unknown value (from an edited session file) is then shown as it is, and the
    crystal read-out reports why the settings give no crystal.
    """
    return next((k for k, v in labels.items() if v == value), value)


def readout_carrier(state: WizardState) -> tuple[float, str]:
    """Return the carrier (rad/s) the read-outs are evaluated at, and its source.

    The prepared trace's carrier once there is one; before that, the carrier the
    retrieval grid will have (:func:`~croak.session.pipeline.pulse_carrier`) once
    a trace is loaded; and before any data, the phase-matched wavelength.
    """
    if state.tracedata is not None:
        return state.tracedata.omega0_pulse, "the prepared trace's carrier"
    if state.load_data is not None:
        return pulse_carrier(state.preproc), "the centre of the Prepare band"
    wavelength = state.retrieve.phase_match_nm * 1e-9
    return 2.0 * math.pi * C / wavelength, "the phase-matched wavelength"


def crystal_summary(p: RetrieveParams, td, omega0: float) -> str:
    """Describe the crystal the settings give at ``omega0``, or why they give none.

    ``td`` is the prepared trace (:class:`~croak.preprocess.TraceData`), whose grid
    sets the depth nodes; ``None`` before Prepare.
    """
    try:
        setup = crystal_setup(p)
        thickness = p.thickness_um * 1e-6
        lines = [
            f"θ inside = {math.degrees(setup.internal_angle(omega0)):.3f}°, "
            f"external tilt γ = {math.degrees(setup.external_tilt(omega0)):.3f}°",
            f"path = {setup.path_length(thickness, omega0) * 1e6:.2f} µm",
        ]
        if td is None:
            lines.append("depth nodes: after Prepare")
        elif thickness > 0:
            lines.append(f"depth nodes = {retrieval_npoints(p, td, setup, thickness)}")
    except ValueError as err:
        return f"No valid crystal: {err}"
    return "\n".join(lines)


def geometry_summary(p: RetrieveParams, wavelength: float) -> str:
    """Describe the beams' crossing and their delay smearing at ``wavelength`` (m)."""
    geometry = beam_geometry(p)
    if geometry is None:
        return "Collinear beams: no delay smearing"
    alpha = math.degrees(geometry.crossing_half_angle)
    sigma = smearing.shg_two_beam_kernel(geometry, wavelength).sigma_delta
    return (
        f"Crossing half-angle α = {alpha:.3f}°\n"
        f"Delay smearing σ = {sigma * 1e15:.2f} fs rms "
        f"({2.3548 * sigma * 1e15:.2f} fs FWHM)"
    )


def _wavelength(state: WizardState, omega0: float) -> float:
    """The pulse wavelength (m): the prepared spectrum's centroid, else the carrier."""
    if state.tracedata is not None:
        return pulse_wavelength(state.tracedata)
    return 2.0 * math.pi * C / omega0


def retrieve_summary(state: WizardState) -> str:
    """Summarise the crystal and the beams for the Retrieve stage, read-only."""
    p = state.retrieve
    omega0, _ = readout_carrier(state)
    if p.dispersive:
        crystal = (
            f"{p.crystal}, {p.thickness_um:g} µm, cut {p.cut_angle_deg:g}°\n"
            + crystal_summary(p, state.tracedata, omega0)
        )
    else:
        crystal = "Crystal not modelled: thin-medium SHG."
    if p.beam_separation_mm > 0:
        beams = (
            f"Beams {p.beam_separation_mm:g} mm apart, f = {p.focal_f_mm:g} mm, "
            f"⌀ {p.beam_diameter_mm:g} mm\n"
        )
    else:
        beams = ""
    return f"{crystal}\n{beams}{geometry_summary(p, _wavelength(state, omega0))}"


class CrystalPanel(QWidget):
    """The "Nonlinear crystal" and "Beam geometry" sections, on ``state.retrieve``.

    Every edit is stored on the session's retrieval settings at once, so the
    marginal check, the preparation and the retrieval read the same crystal.
    A Data page adds the panel to its form and shows it for an SHG trace
    (:meth:`set_shg`).

    Parameters
    ----------
    state : WizardState
        The session; the panel reads and writes ``state.retrieve`` (looked up on
        each use, so a replaced settings object is followed).

    Attributes
    ----------
    edited : Signal
        Emitted after an edit has been stored.
    """

    edited = Signal()

    def __init__(self, state: WizardState) -> None:
        super().__init__()
        self.state = state
        p = state.retrieve
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.crystal_box, form = group("Nonlinear crystal")
        self.crystal_check = check(
            "Model the crystal", p.dispersive, self._setter("dispersive", bool)
        )
        form.addRow(self.crystal_check)
        self.crystal_combo = combo(
            list(crystals.CRYSTALS), p.crystal, self._setter("crystal", str)
        )
        form.addRow("Crystal", self.crystal_combo)
        # the same range as the Retrieve stage's slab thickness, which is the
        # same setting
        self.thickness_spin = dspin(
            0, 1000, p.thickness_um, self._setter("thickness_um", float), decimals=1
        )
        form.addRow("Thickness (µm)", self.thickness_spin)
        self.cut_angle_spin = dspin(
            0, 90, p.cut_angle_deg, self._setter("cut_angle_deg", float), decimals=2
        )
        form.addRow("Cut angle θc (°)", self.cut_angle_spin)
        self.angle_mode_combo = combo(
            list(_ANGLE_MODES),
            _label(_ANGLE_MODES, p.angle_mode),
            lambda v: self._store("angle_mode", _ANGLE_MODES[v]),
        )
        form.addRow("Angle set by", self.angle_mode_combo)
        self.phase_match_spin = dspin(
            200,
            5000,
            p.phase_match_nm,
            self._setter("phase_match_nm", float),
            decimals=1,
        )
        form.addRow("Phase-matched λ (nm)", self.phase_match_spin)
        self.tilt_spin = dspin(
            -60, 60, p.tilt_deg, self._setter("tilt_deg", float), decimals=2, step=0.1
        )
        form.addRow("External tilt γ (°)", self.tilt_spin)
        self.polarization_combo = combo(
            list(_POLARIZATIONS),
            _label(_POLARIZATIONS, p.polarization),
            lambda v: self._store("polarization", _POLARIZATIONS[v]),
        )
        form.addRow("Polarisation", self.polarization_combo)
        self.delayed_beam_combo = combo(
            list(_DELAYED_BEAMS),
            _label(_DELAYED_BEAMS, p.delayed_beam),
            lambda v: self._store("delayed_beam", _DELAYED_BEAMS[v]),
        )
        form.addRow("Delayed beam", self.delayed_beam_combo)
        self.crystal_readout = QLabel()
        self.crystal_readout.setWordWrap(True)
        form.addRow(self.crystal_readout)
        layout.addWidget(self.crystal_box)

        self.geometry_box, form = group("Beam geometry")
        self.separation_spin = dspin(
            0,
            1000,
            p.beam_separation_mm,
            self._setter("beam_separation_mm", float),
            decimals=2,
            step=0.5,
        )
        form.addRow("Beam separation (mm)", self.separation_spin)
        # the same range as the focal mixture's focal length, the same setting
        self.focal_f_spin = dspin(
            1, 10000, p.focal_f_mm, self._setter("focal_f_mm", float), decimals=1
        )
        form.addRow("Focal length (mm)", self.focal_f_spin)
        self.beam_diameter_spin = dspin(
            0.01,
            100,
            p.beam_diameter_mm,
            self._setter("beam_diameter_mm", float),
            decimals=2,
            step=0.1,
        )
        form.addRow("Beam diameter (mm)", self.beam_diameter_spin)
        self.geometry_readout = QLabel()
        self.geometry_readout.setWordWrap(True)
        form.addRow(self.geometry_readout)
        layout.addWidget(self.geometry_box)

        apply_tooltips(self, flatten(CRYSTAL_HELP))
        self._update()

    def _setter(self, name: str, kind):
        """Return a handler that stores ``kind(value)`` as ``name``."""
        return lambda value: self._store(name, kind(value))

    def _store(self, name: str, value) -> None:
        setattr(self.state.retrieve, name, value)
        self._update()
        self.edited.emit()

    def set_shg(self, shg: bool) -> None:
        """Show the panel for an SHG trace (refreshed from the settings), else hide."""
        self.setHidden(not shg)
        if shg:
            self.refresh()

    def refresh(self) -> None:
        """Show the stored settings, which another stage may have changed."""
        p = self.state.retrieve
        values = {
            self.crystal_check: p.dispersive,
            self.crystal_combo: p.crystal,
            self.thickness_spin: p.thickness_um,
            self.cut_angle_spin: p.cut_angle_deg,
            self.angle_mode_combo: _label(_ANGLE_MODES, p.angle_mode),
            self.phase_match_spin: p.phase_match_nm,
            self.tilt_spin: p.tilt_deg,
            self.polarization_combo: _label(_POLARIZATIONS, p.polarization),
            self.delayed_beam_combo: _label(_DELAYED_BEAMS, p.delayed_beam),
            self.separation_spin: p.beam_separation_mm,
            self.focal_f_spin: p.focal_f_mm,
            self.beam_diameter_spin: p.beam_diameter_mm,
        }
        for widget, value in values.items():
            widget.blockSignals(True)
            if isinstance(value, bool):
                widget.setChecked(value)
            elif isinstance(value, str):
                widget.setCurrentText(value)
            else:
                widget.setValue(value)
            widget.blockSignals(False)
        self._update()

    def _update(self) -> None:
        """Enable the controls the settings use and describe what they give."""
        p = self.state.retrieve
        tilt_mode = p.angle_mode == "tilt"
        self.tilt_spin.setEnabled(tilt_mode)
        self.phase_match_spin.setEnabled(not tilt_mode)
        crossing = p.beam_separation_mm > 0
        self.polarization_combo.setEnabled(crossing)
        self.delayed_beam_combo.setEnabled(
            crossing and p.polarization == "perpendicular"
        )
        omega0, source = readout_carrier(self.state)
        at = f"at {2.0 * math.pi * C / omega0 * 1e9:.0f} nm ({source})"
        self.crystal_readout.setText(
            f"{crystal_summary(p, self.state.tracedata, omega0)}\n{at}"
        )
        self.geometry_readout.setText(
            geometry_summary(p, _wavelength(self.state, omega0))
        )
