"""Tests for the stage sequencing of :func:`croak.session.run_session`.

The per-stage mappings are exercised in ``test_session.py`` and
``test_session_pipeline.py``; here they are replaced by recorders so the
engine's own job — validating, ordering and chaining the stages — is tested on
its own, quickly.
"""

from __future__ import annotations

import pytest

from croak.session import SessionOptions, engine, run_session
from croak.session.options import _dispersion_from_mapping, _from_mapping
from croak.session.params import (
    BeamPathMirror,
    DispersionParams,
    LoadParams,
    SimulatedLoadParams,
)


@pytest.fixture
def stub_stages(monkeypatch):
    """Replace every stage mapping by a recorder returning a tagged value."""
    calls: list[tuple[str, tuple, dict]] = []

    def stub(name, value):
        def fn(*args, **kwargs):
            calls.append((name, args, kwargs))
            return value

        monkeypatch.setattr(engine.pipeline, name, fn)

    stub("assemble_load_data", {"truth": "truth"})
    stub("assemble_simulated_load_data", {"truth": None})
    stub("build_tracedata", "td")
    stub("assemble_simulated_tracedata", "td-raw")
    stub("run_retrieval", "raw-result")
    stub("apply_post_filter", "result")
    stub("process", "processed")
    stub("run_uncertainty", "uncertainty")
    monkeypatch.setattr(
        engine,
        "compose_dispersion",
        lambda p, r: calls.append(("compose_dispersion", (p, r), {})) or "dispersed",
    )
    return calls


def _names(calls):
    return [name for name, _, _ in calls]


def test_run_session_rejects_unknown_stages():
    with pytest.raises(ValueError, match=r"unknown session stage\(s\) \['plot'\]"):
        run_session(SessionOptions(), stages=("load", "plot"))


@pytest.mark.parametrize(
    ("stages", "message"),
    [
        (("preproc",), "'preproc' stage requires 'load'"),
        (("retrieve",), "'retrieve' stage requires 'preproc'"),
        (("uncertainty",), "'uncertainty' stage requires 'retrieve'"),
    ],
)
def test_run_session_requires_each_upstream_stage(stub_stages, stages, message):
    with pytest.raises(ValueError, match=message):
        run_session(SessionOptions(), stages=stages)


def test_run_session_orders_stages_and_chains_results(stub_stages):
    """Requested out of order, stages still run load → … → uncertainty."""
    options = SessionOptions(load=LoadParams(frog_path="scan.h5", energy_j=2e-6))
    out = run_session(
        options,
        stages=("uncertainty", "process", "dispersion", "retrieve", "preproc", "load"),
    )
    assert _names(stub_stages) == [
        "assemble_load_data",
        "build_tracedata",
        "run_retrieval",
        "apply_post_filter",
        "compose_dispersion",
        "process",
        "run_uncertainty",
    ]
    (_, _, retrieve_kw) = stub_stages[2]
    assert retrieve_kw["truth"] == "truth"
    assert retrieve_kw["scan_path"] == "scan.h5" and retrieve_kw["scan_window"] is None
    # the summary describes the dispersed pulse, at the measured energy
    _, process_args, process_kw = stub_stages[5]
    assert process_args == ("dispersed", "td") and process_kw == {"energy": 2e-6}
    # the bootstrap works on the pre-dispersion result
    assert stub_stages[6][1][2] == "result"
    assert (out.data, out.tracedata, out.result) == ({"truth": "truth"}, "td", "result")
    assert (out.dispersed, out.processed, out.uncertainty) == (
        "dispersed",
        "processed",
        "uncertainty",
    )


def test_run_session_simulated_raw_direct_skips_the_regrid(stub_stages):
    options = SessionOptions(
        entry="simulated",
        simulated=SimulatedLoadParams(frog_path="sim.h5", raw_direct=True),
    )
    out = run_session(options, stages=("load", "preproc", "retrieve"))
    assert _names(stub_stages)[:2] == [
        "assemble_simulated_load_data",
        "assemble_simulated_tracedata",
    ]
    assert out.tracedata == "td-raw"
    retrieve_kw = stub_stages[2][2]
    assert retrieve_kw["scan_path"] == "sim.h5"
    assert retrieve_kw["scan_window"] == options.simulated.window_key


def test_run_session_rejects_synthetic_sessions(stub_stages):
    with pytest.raises(ValueError, match="cannot replay a synthetic session"):
        run_session(SessionOptions(entry="synthetic"), stages=("load",))


def test_run_session_process_without_a_result_summarises_nothing(stub_stages):
    """Processing needs a pulse: with only a trace there is nothing to summarise."""
    out = run_session(SessionOptions(), stages=("load", "preproc", "process"))
    assert "process" not in _names(stub_stages)
    assert out.processed is None


def test_run_session_late_stages_alone_are_no_ops(stub_stages):
    out = run_session(SessionOptions(), stages=("dispersion", "process"))
    assert stub_stages == []
    assert out.result is None and out.processed is None


def test_run_session_process_without_dispersion_uses_the_retrieval(stub_stages):
    options = SessionOptions()
    run_session(options, stages=("load", "preproc", "retrieve", "process"))
    _, process_args, process_kw = stub_stages[-1]
    # no measured energy: 0 means unspecified, passed as None
    assert process_args == ("result", "td") and process_kw == {"energy": None}


# ---------------------------------------------------------------------------
# Options mapping: permissive on purpose (older/newer files)
# ---------------------------------------------------------------------------
def test_from_mapping_ignores_a_non_table_section():
    assert _from_mapping(LoadParams, "not a table") == LoadParams()


def test_dispersion_from_mapping_ignores_a_non_table_section():
    assert _dispersion_from_mapping(["junk"]) == DispersionParams()


def test_dispersion_from_mapping_rebuilds_nested_values_and_drops_unknowns():
    p = _dispersion_from_mapping(
        {
            "gdd_fs2": 12.0,
            "material_thickness_mm": {"SiO2": "0.5"},
            "mirrors": [{"name": "PC70", "bounces": 2, "colour": "gold"}, "junk"],
            "from_the_future": True,
        }
    )
    assert p.gdd_fs2 == 12.0
    assert p.material_thickness_mm == {"SiO2": 0.5}
    assert p.mirrors == [BeamPathMirror(name="PC70", bounces=2), BeamPathMirror()]
    assert not hasattr(p, "from_the_future")


def test_session_options_round_trip_through_toml(tmp_path):
    options = SessionOptions(
        entry="simulated",
        dispersion=DispersionParams(mirrors=[BeamPathMirror(name="PC70", bounces=1)]),
    )
    path = options.to_toml(str(tmp_path / "options.toml"))
    assert SessionOptions.from_toml(path) == options


def test_load_params_fixed_path_fields_are_the_calibrations():
    """Calibration files are shared across datasets, so retargeting leaves them."""
    assert LoadParams.fixed_path_fields() == ("frog_calib_path", "spec_calib_path")
    assert set(LoadParams.fixed_path_fields()).isdisjoint(
        LoadParams.dataset_path_fields()
    )


def test_from_mapping_drops_unknown_keys_and_keeps_defaults():
    p = _from_mapping(LoadParams, {"frog_path": "a.h5", "obsolete_key": 1})
    assert p.frog_path == "a.h5" and p.lam_unit == LoadParams().lam_unit
    assert not hasattr(p, "obsolete_key")


def test_session_options_with_replaces_one_stage():
    options = SessionOptions()
    changed = options.with_(load=LoadParams(frog_path="b.h5"))
    assert changed.load.frog_path == "b.h5"
    assert options.load.frog_path == ""  # the original is untouched
    assert changed.retrieve is options.retrieve
