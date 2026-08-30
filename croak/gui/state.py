"""Wizard state: per-stage parameter dataclasses and the shared state object.

``WizardState`` (a ``QObject``) holds the four stages' parameters and the
pipeline outputs (``load_data`` → ``tracedata`` → ``result`` → ``processed``),
emits change signals, and implements cascading invalidation so editing an
upstream stage clears the downstream results — the wizard can then never show a
retrieved pulse that belongs to a superseded set of load/preprocess parameters.
"""

from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING

from PyQt6.QtCore import QObject, pyqtSignal

from ..session.options import SessionOptions
from ..session.params import (
    DispersionParams,
    LoadParams,
    PreprocParams,
    RetrieveParams,
    SimulatedLoadParams,
    UncertaintyParams,
)

if TYPE_CHECKING:
    from ..io import FileDatasets
    from ..preprocess import TraceData
    from ..processing import ProcessedResult, TruthPulse
    from ..result import RetrievalResult
    from ..uncertainty import UncertaintyResult

# The per-stage parameter dataclasses now live in croak.session.params (Qt-free,
# shared with the headless engine); they are re-exported here so existing
# ``from .state import LoadParams`` imports keep working.
__all__ = [
    "LoadParams",
    "PreprocParams",
    "RetrieveParams",
    "DispersionParams",
    "UncertaintyParams",
    "WizardState",
]


class WizardState(QObject):
    """Holds wizard parameters and pipeline outputs, with change signals."""

    stage_changed = pyqtSignal(int)
    load_data_changed = pyqtSignal()
    tracedata_changed = pyqtSignal()
    result_changed = pyqtSignal()
    uncertainty_changed = pyqtSignal()
    status = pyqtSignal(str)
    # Request the wizard to point the current session at a new dataset folder
    # (its argument) and replay load → preprocess → retrieve. Emitted by the
    # Retrieve stage's "Retarget…" button; handled in croak.gui.wizard.Wizard.
    retarget_requested = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        # Which stage-1 entry produced the trace ("experimental" / "simulated" /
        # "synthetic"; see croak.session.options.ENTRIES). It selects which loader
        # params are live, so it is session state rather than view state: the
        # marginal-check stage reads it, and it is saved with the options so a
        # reload knows which workflow to replay.
        self.entry = "experimental"
        self.load = LoadParams()
        # the simulated-scan loader's options (its own entry page), saved with the
        # session; not part of the experimental LoadParams schema
        self.simulated = SimulatedLoadParams()
        self.preproc = PreprocParams()
        self.retrieve = RetrieveParams()
        self.dispersion = DispersionParams()
        self.uncertainty = UncertaintyParams()

        self._stage = 1
        self.file_datasets: FileDatasets | None = None  # FROG file datasets
        self.load_data: dict | None = None  # lam, tau, trace, lam_spec, Ilam_spec
        self.tracedata: TraceData | None = None
        self.result: RetrievalResult | None = None
        self.processed: ProcessedResult | None = None
        self.uncertainty_result: UncertaintyResult | None = None  # most recent
        # All retained estimates, keyed by method ("parametric"/"thickness"/"combined"
        # …), so several can be combined into a total before saving.
        self.uncertainty_results: dict = {}
        # Data-seeded preprocess defaults from the most recent load (a fresh,
        # re-initialised PreprocParams), kept so the preprocess stage's "Reset
        # defaults" button can restore them. The trace signature lets the load
        # stage tell a genuinely new trace (→ reseed the windows) from a reload
        # of the same trace with tweaked options (→ keep the user's windows).
        self.preproc_defaults: PreprocParams | None = None
        self.trace_sig: tuple | None = None

    @property
    def truth(self) -> TruthPulse | None:
        """Known ground-truth pulse for the current load, or ``None``.

        Carried inside ``load_data`` (under ``"truth"``) by the synthetic and
        simulated loaders, so it survives :meth:`set_load_data` and the
        marginal-check stage's re-assembly without a separate, easily-stale field.
        Experimental loads have no truth.
        """
        if isinstance(self.load_data, dict):
            return self.load_data.get("truth")
        return None

    @property
    def geometry(self):
        """Instrument geometry recorded by the current load, or ``None``.

        A :class:`~croak.session.pipeline.SimulatedGeometry` for simulated
        loads (slab thickness of the selected slice, mask hole diameter and
        spacing), carried inside ``load_data`` like :attr:`truth`. Experimental
        and synthetic loads have none: only the simulator knows these numbers.
        """
        if isinstance(self.load_data, dict):
            return self.load_data.get("geometry")
        return None

    # -- navigation ---------------------------------------------------------
    @property
    def stage(self) -> int:
        return self._stage

    @stage.setter
    def stage(self, value: int) -> None:
        value = max(1, min(6, value))
        if value != self._stage:
            self._stage = value
            self.stage_changed.emit(value)

    def ready(self, stage: int) -> bool:
        """Whether the Next button should be enabled for ``stage``.

        Stage order: Load(1) → Marginal check(2) → Preprocess(3) → Retrieve(4) →
        Dispersion(5) → Uncertainty(6). The marginal-check and preprocess stages
        only need a loaded trace; the later stages need a completed retrieval.
        """
        return {
            1: self.load_data is not None,
            2: self.load_data is not None,
            3: self.load_data is not None,
            4: self.result is not None,
            5: self.result is not None,
            6: True,
        }.get(stage, False)

    # -- outputs with cascading invalidation --------------------------------
    def set_load_data(self, data) -> None:
        self.load_data = data
        self.tracedata = None
        self.result = None
        self.processed = None
        self.load_data_changed.emit()

    def set_tracedata(self, td) -> None:
        self.tracedata = td
        self.result = None
        self.processed = None
        self.tracedata_changed.emit()

    def set_result(self, result, processed=None) -> None:
        self.result = result
        self.processed = processed
        # a new retrieval invalidates every old error bar
        self.uncertainty_result = None
        self.uncertainty_results = {}
        self.result_changed.emit()
        self.uncertainty_changed.emit()

    def set_uncertainty_result(self, uresult) -> None:
        """Retain an estimate (keyed by its method) and mark it the current one."""
        self.uncertainty_results[uresult.method] = uresult
        self.uncertainty_result = uresult
        self.uncertainty_changed.emit()

    def reset(self) -> None:
        """Clear all data and per-stage settings for a fresh session.

        Used when the user starts a new process from the welcome menu, so a
        synthetic run never leaks into an experimental one (or vice versa).
        Recreates the parameter dataclasses (so callers must rebuild the stage
        widgets to rebind them) and clears every pipeline output.
        """
        self.entry = "experimental"
        self.load = LoadParams()
        self.simulated = SimulatedLoadParams()
        self.preproc = PreprocParams()
        self.retrieve = RetrieveParams()
        self.dispersion = DispersionParams()
        self.uncertainty = UncertaintyParams()
        self.file_datasets = None
        self.load_data = None
        self.tracedata = None
        self.result = None
        self.processed = None
        self.uncertainty_result = None
        self.uncertainty_results = {}
        self.preproc_defaults = None
        self.trace_sig = None
        self._stage = 1
        self.load_data_changed.emit()
        self.tracedata_changed.emit()
        self.result_changed.emit()

    # -- session persistence ------------------------------------------------
    def session_options(self) -> SessionOptions:
        """Snapshot the current parameters as a :class:`SessionOptions`."""
        return SessionOptions(
            entry=self.entry,
            load=self.load,
            simulated=self.simulated,
            preproc=self.preproc,
            retrieve=self.retrieve,
            dispersion=self.dispersion,
            uncertainty=self.uncertainty,
        )

    def to_options(self) -> dict:
        """Serialise the session to the nested options-dict layout."""
        return self.session_options().to_dict()

    def from_options(self, options: dict) -> None:
        """Apply a loaded options dict onto the live parameter objects.

        The stage widgets bind to these dataclass *instances* by reference, so we
        copy the parsed field values into the existing objects rather than
        replacing them (the wizard then rebuilds the controls to reflect them).
        """
        parsed = SessionOptions.from_dict(options)
        self.entry = parsed.entry
        for name in (
            "load",
            "simulated",
            "preproc",
            "retrieve",
            "dispersion",
            "uncertainty",
        ):
            dst = getattr(self, name)
            src = getattr(parsed, name)
            for f in fields(src):
                setattr(dst, f.name, getattr(src, f.name))
