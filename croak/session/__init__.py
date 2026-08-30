"""Headless retrieval sessions: parameters, serialisation, and replay.

This subpackage is the Qt-free heart shared by the GUI, the ``croak`` CLI, and
script generation. It defines the per-stage parameter dataclasses
(:mod:`~croak.session.params`), bundles them into a serialisable
:class:`~croak.session.options.SessionOptions`, maps each stage onto croak's public
API (:mod:`~croak.session.pipeline`), and replays a whole session with
:func:`~croak.session.engine.run_session`. :func:`~croak.session.retarget.retarget`
points an existing session at a new dataset folder.
"""

from __future__ import annotations

from .dispersion import compose_dispersion
from .engine import DEFAULT_STAGES, SessionResult, run_session
from .options import SessionOptions
from .params import (
    BeamPathMirror,
    DispersionParams,
    LoadParams,
    PreprocParams,
    RetrieveParams,
    SimulatedLoadParams,
    UncertaintyParams,
)
from .pipeline import (
    assemble_load_data,
    assemble_simulated_load_data,
    assemble_simulated_tracedata,
    build_tracedata,
    run_retrieval,
    run_uncertainty,
)
from .retarget import retarget

__all__ = [
    # parameters
    "LoadParams",
    "SimulatedLoadParams",
    "PreprocParams",
    "RetrieveParams",
    "BeamPathMirror",
    "DispersionParams",
    "UncertaintyParams",
    # serialisable bundle
    "SessionOptions",
    # stage mappings
    "assemble_load_data",
    "assemble_simulated_load_data",
    "assemble_simulated_tracedata",
    "build_tracedata",
    "run_retrieval",
    "compose_dispersion",
    "run_uncertainty",
    # replay
    "run_session",
    "SessionResult",
    "DEFAULT_STAGES",
    # retargeting to new datasets
    "retarget",
]
