"""Headless session engine: replay a :class:`SessionOptions` end-to-end.

``run_session`` drives the same per-stage mappings the GUI uses
(:mod:`croak.session.pipeline`) without any Qt dependency, so a saved
``options.toml`` can be replayed from a script or the ``croak`` CLI. It returns a
:class:`SessionResult` holding every intermediate (loaded arrays, cleaned trace,
retrieved + dispersion-compensated pulse, processed summary, uncertainty).

Reads happen at the load stage; the engine itself writes nothing — saving is the
caller's job (see :mod:`croak.cli`), keeping the core pure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..preprocess import TraceData
from ..processing import ProcessedResult
from ..result import RetrievalResult
from ..uncertainty import UncertaintyResult
from . import pipeline
from .dispersion import compose_dispersion
from .options import SessionOptions

__all__ = ["SessionResult", "run_session", "DEFAULT_STAGES"]

#: Stages run by default. Dispersion compensation is included (it is deterministic
#: and cheap). Uncertainty is opt-in: it is a slow, stochastic bootstrap (many
#: retrievals) whose results are saved separately, so a plain replay reproduces
#: the retrieved + compressed pulse and leaves error bars to an explicit request
#: (``stages=(..., "uncertainty")`` or ``croak replay --uncertainty``).
DEFAULT_STAGES: tuple[str, ...] = (
    "load",
    "preproc",
    "retrieve",
    "dispersion",
    "process",
)

_STAGE_ORDER: tuple[str, ...] = (
    "load",
    "preproc",
    "retrieve",
    "dispersion",
    "process",
    "uncertainty",
)


@dataclass
class SessionResult:
    """Every intermediate produced by :func:`run_session`.

    Fields are populated up to the last stage that ran; later ones stay ``None``.
    ``result`` is the (post-filtered) retrieval; ``dispersed`` is it after the
    stage-4 dispersion compensation; ``processed`` summarises the final pulse
    (dispersed when dispersion ran, else the retrieval).
    """

    options: SessionOptions
    data: dict | None = None
    tracedata: TraceData | None = None
    result: RetrievalResult | None = None
    dispersed: RetrievalResult | None = None
    processed: ProcessedResult | None = None
    uncertainty: UncertaintyResult | None = None


def _load_stage(options: SessionOptions) -> dict:
    """Load the trace arrays through the loader the session was saved with.

    ``options.entry`` selects between the mutually exclusive stage-1 loaders (see
    :data:`croak.session.options.ENTRIES`). The synthetic generator's settings are
    not serialisable yet, so that entry raises instead of silently loading the
    empty experimental table.
    """
    if options.entry == "simulated":
        return pipeline.assemble_simulated_load_data(options.simulated)
    if options.entry == "synthetic":
        raise ValueError(
            "cannot replay a synthetic session: the generator's settings are not "
            "saved with the options (regenerate the trace in the GUI instead)"
        )
    return pipeline.assemble_load_data(options.load)


def _preproc_stage(options: SessionOptions, data: dict) -> TraceData:
    """Clean and regrid the loaded trace, honouring the raw simulated shortcut.

    The simulated loader's ``raw_direct`` option deliberately bypasses the
    filtering and regrid, building the retrieval trace on the simulation's native
    grid instead (:func:`croak.session.pipeline.assemble_simulated_tracedata`).
    """
    if options.entry == "simulated" and options.simulated.raw_direct:
        return pipeline.assemble_simulated_tracedata(options.simulated)
    return pipeline.build_tracedata(options.preproc, data)


def _check_stages(stages: tuple[str, ...]) -> list[str]:
    """Return the requested stages in pipeline order, validating names.

    Stages are run in their fixed dependency order regardless of the order they
    are requested in, so ``("retrieve", "load")`` still loads before retrieving.
    """
    unknown = set(stages) - set(_STAGE_ORDER)
    if unknown:
        raise ValueError(
            f"unknown session stage(s) {sorted(unknown)}; valid: {list(_STAGE_ORDER)}"
        )
    return [s for s in _STAGE_ORDER if s in stages]


def run_session(
    options: SessionOptions,
    *,
    stages: tuple[str, ...] = DEFAULT_STAGES,
    rng: np.random.Generator | None = None,
    retrieve_callback=None,
    uncertainty_callback=None,
) -> SessionResult:
    """Replay a session from its parameters, returning all intermediates.

    Parameters
    ----------
    options : SessionOptions
        The session to replay (typically ``SessionOptions.from_toml(path)``).
    stages : tuple of str, optional
        Which stages to run; see :data:`DEFAULT_STAGES`. Always executed in
        dependency order. Include ``"uncertainty"`` to also bootstrap error bars.
    rng : numpy.random.Generator, optional
        Generator threaded through retrieval and the uncertainty bootstrap for
        reproducibility; a fresh default is used if None.
    retrieve_callback, uncertainty_callback : callable, optional
        Progress hooks forwarded to the retrieval and bootstrap respectively.

    Returns
    -------
    SessionResult
    """
    rng = np.random.default_rng() if rng is None else rng
    todo = _check_stages(stages)
    out = SessionResult(options=options)

    if "load" in todo:
        out.data = _load_stage(options)
    if "preproc" in todo:
        if out.data is None:
            raise ValueError("the 'preproc' stage requires 'load'")
        out.tracedata = _preproc_stage(options, out.data)
    if "retrieve" in todo:
        if out.tracedata is None:
            raise ValueError("the 'retrieve' stage requires 'preproc'")
        # collection="file" rebuilds the aperture from the scan's own
        # window_def record, so the entry's scan file must travel with the
        # retrieval (the GUI passes its loaded path the same way).
        scan_path = (
            options.simulated.frog_path
            if options.entry == "simulated"
            else options.load.frog_path
        ) or None
        # ... and which of its collection holes the trace was loaded from, so a
        # multi-window scan does not silently get the first hole's aperture.
        scan_window = (
            options.simulated.window_key if options.entry == "simulated" else None
        )
        result = pipeline.run_retrieval(
            options.retrieve,
            out.tracedata,
            truth=None if out.data is None else out.data.get("truth"),
            rng=rng,
            callback=retrieve_callback,
            scan_path=scan_path,
            scan_window=scan_window,
        )
        out.result = pipeline.apply_post_filter(options.retrieve, result, out.tracedata)
    if "dispersion" in todo and out.result is not None:
        out.dispersed = compose_dispersion(options.dispersion, out.result)
    if "process" in todo and out.tracedata is not None:
        # Summarise the final pulse: the dispersion-compensated one if stage 4 ran.
        final = out.dispersed if out.dispersed is not None else out.result
        if final is not None:
            # An optional measured pulse energy rescales the temporal summary to
            # absolute power (0 = unspecified → normalised units).
            out.processed = pipeline.process(
                final, out.tracedata, energy=options.load.energy_j or None
            )
    if "uncertainty" in todo:
        if out.result is None or out.tracedata is None:
            raise ValueError("the 'uncertainty' stage requires 'retrieve'")
        # The bootstrap re-retrieves and so works on the pre-dispersion result,
        # mirroring the GUI's stage-4/stage-5 independence.
        out.uncertainty = pipeline.run_uncertainty(
            options.uncertainty,
            options.retrieve,
            out.result,
            out.tracedata,
            rng=rng,
            callback=uncertainty_callback,
        )
    return out
