"""The background workers: signals, throttling, Stop and failure paths.

Each worker's ``run`` is called directly on the test thread (not ``start``), with
the session pipeline replaced by a stub, so the signal logic is tested
synchronously and without a real retrieval.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from croak.gui import worker as worker_mod
from croak.gui.worker import (
    PreprocessWorker,
    RetrievalWorker,
    UncertaintyWorker,
)


class _Recorder:
    """Collect every emission of a set of signals as ``(name, args)`` tuples."""

    def __init__(self, obj, *names):
        self.calls: list[tuple[str, tuple]] = []
        for name in names:
            getattr(obj, name).connect(
                lambda *args, n=name: self.calls.append((n, args))
            )

    def named(self, name):
        return [args for n, args in self.calls if n == name]


# -- PreprocessWorker --------------------------------------------------------


def test_preprocess_worker_emits_result_with_request_id_and_mode(monkeypatch):
    seen = {}

    def build(params, data, fast_baseline):
        seen["fast"] = fast_baseline
        return ("td", params, data)

    monkeypatch.setattr(worker_mod.pipeline, "build_tracedata", build)
    w = PreprocessWorker("p", "d", 7, fast=True)
    rec = _Recorder(w, "finished_ok", "failed")
    w.run()
    assert seen["fast"] is True
    assert rec.named("finished_ok") == [(("td", "p", "d"), 7, True)]
    assert rec.named("failed") == []


def test_preprocess_worker_reports_failure_with_request_id(monkeypatch):
    def build(*_args, **_kwargs):
        raise ValueError("bad band")

    monkeypatch.setattr(worker_mod.pipeline, "build_tracedata", build)
    w = PreprocessWorker("p", "d", 3)
    rec = _Recorder(w, "finished_ok", "failed")
    w.run()
    assert rec.named("failed") == [("bad band", 3)]
    assert rec.named("finished_ok") == []


# -- RetrievalWorker ---------------------------------------------------------


def _stub_retrieval(monkeypatch, body):
    """Replace run_retrieval by ``body(callback)`` and return the kwargs it saw."""
    seen = {}

    def run_retrieval(p, td, **kwargs):
        seen.update(kwargs, p=p, td=td)
        return body(kwargs["callback"])

    monkeypatch.setattr(worker_mod.pipeline, "run_retrieval", run_retrieval)
    return seen


def test_retrieval_worker_success_forwards_arguments(monkeypatch):
    seen = _stub_retrieval(monkeypatch, lambda cb: "result")
    w = RetrievalWorker(
        "td",
        "params",
        guess_override="g",
        extras_seed="x",
        truth="t",
        scan_path="scan.h5",
        scan_window=2,
    )
    rec = _Recorder(w, "finished_ok", "failed", "stopped")
    w.run()
    ((result, elapsed),) = rec.named("finished_ok")
    assert result == "result" and elapsed >= 0.0
    assert seen["p"] == "params" and seen["td"] == "td"
    assert seen["guess_override"] == "g" and seen["extras_seed"] == "x"
    assert seen["truth"] == "t"
    assert seen["scan_path"] == "scan.h5" and seen["scan_window"] == 2
    assert rec.named("failed") == rec.named("stopped") == []


def test_retrieval_worker_failure_emits_message(monkeypatch):
    def body(_cb):
        raise RuntimeError("solver diverged")

    _stub_retrieval(monkeypatch, body)
    w = RetrievalWorker("td", "params")
    rec = _Recorder(w, "finished_ok", "failed")
    w.run()
    assert rec.named("failed") == [("solver diverged",)]
    assert rec.named("finished_ok") == []


def test_retrieval_worker_progress_and_first_preview(monkeypatch):
    def body(cb):
        cb(1, 0.5, 0.4, snapshot=lambda: "snap1")
        # Inside the throttle interval: progress still flows, no second preview.
        cb(2, 0.3, 0.3, snapshot=lambda: "snap2")
        return "done"

    _stub_retrieval(monkeypatch, body)
    w = RetrievalWorker("td", "params")
    rec = _Recorder(w, "progress", "preview")
    w.run()
    assert rec.named("progress") == [(1, 0.5, 0.4), (2, 0.3, 0.3)]
    assert rec.named("preview") == [("snap1",)]


def test_retrieval_worker_preview_after_interval(monkeypatch):
    clock = iter([0.0, 10.0, 20.0, 30.0, 40.0])
    monkeypatch.setattr(worker_mod.time, "perf_counter", lambda: next(clock))

    def body(cb):
        cb(1, 0.5, 0.5, snapshot=lambda: "a")
        cb(2, 0.4, 0.4, snapshot=lambda: "b")
        return "done"

    _stub_retrieval(monkeypatch, body)
    w = RetrievalWorker("td", "params")
    rec = _Recorder(w, "preview")
    w.run()
    # 10 s apart, far beyond the minimum interval: both previews go out.
    assert rec.named("preview") == [("a",), ("b",)]


def test_retrieval_worker_preview_disabled_and_snapshotless(monkeypatch):
    def body(cb):
        cb(1, 0.5, 0.5, snapshot=lambda: "a")
        cb(2, 0.4, 0.4)  # a solver that cannot build snapshots
        return "done"

    _stub_retrieval(monkeypatch, body)
    w = RetrievalWorker("td", "params", preview=False)
    rec = _Recorder(w, "preview", "progress")
    w.run()
    assert rec.named("preview") == []
    assert len(rec.named("progress")) == 2


def test_retrieval_worker_failed_snapshot_is_skipped(monkeypatch):
    def broken():
        raise FloatingPointError("degenerate early field")

    def body(cb):
        cb(1, 0.5, 0.5, snapshot=broken)
        cb(2, 0.4, 0.4, snapshot=lambda: "ok")
        return "done"

    _stub_retrieval(monkeypatch, body)
    w = RetrievalWorker("td", "params")
    rec = _Recorder(w, "preview", "finished_ok")
    w.run()
    # The broken snapshot neither aborts the run nor consumes the throttle slot.
    assert rec.named("preview") == [("ok",)]
    assert len(rec.named("finished_ok")) == 1


def test_retrieval_worker_stop_with_snapshot_emits_stopped(monkeypatch):
    def body(cb):
        cb(1, 0.5, 0.5, snapshot=lambda: "partial")
        raise AssertionError("the callback should have raised StopRetrieval")

    _stub_retrieval(monkeypatch, body)
    w = RetrievalWorker("td", "params", preview=False)
    w.request_stop()
    rec = _Recorder(w, "stopped", "failed")
    w.run()
    ((result, elapsed),) = rec.named("stopped")
    assert result == "partial" and elapsed >= 0.0
    assert rec.named("failed") == []


def test_retrieval_worker_stop_without_snapshot_reports_stopped(monkeypatch):
    def body(cb):
        cb(1, 0.5, 0.5)
        return "unreachable"

    _stub_retrieval(monkeypatch, body)
    w = RetrievalWorker("td", "params")
    w.request_stop()
    rec = _Recorder(w, "stopped", "failed")
    w.run()
    assert rec.named("failed") == [("stopped",)]
    assert rec.named("stopped") == []


def test_retrieval_worker_preview_interval_follows_cost():
    w = RetrievalWorker("td", "params")
    assert w.preview_interval == RetrievalWorker.PREVIEW_MIN_INTERVAL
    # Cheap previews keep the floor ...
    w.report_preview_cost(0.001)
    assert w.preview_interval == RetrievalWorker.PREVIEW_MIN_INTERVAL
    # ... an expensive one stretches it to cost / duty.
    w.report_preview_cost(1.0)
    assert w.preview_interval == pytest.approx(1.0 / RetrievalWorker.PREVIEW_DUTY)


# -- UncertaintyWorker -------------------------------------------------------


def _stub_uncertainty(monkeypatch, body):
    seen = {}

    def run_uncertainty(up, rp, result, td, *, dispersion, rng, callback):
        seen.update(up=up, rp=rp, result=result, td=td, dispersion=dispersion)
        return body(callback)

    monkeypatch.setattr(worker_mod.pipeline, "run_uncertainty", run_uncertainty)
    return seen


def test_uncertainty_worker_success_and_progress(monkeypatch):
    def body(cb):
        cb(1, 2, 0.1)
        cb(2, 2, 0.2)
        return "uresult"

    seen = _stub_uncertainty(monkeypatch, body)
    w = UncertaintyWorker("res", "td", "rp", "up", dispersion="disp")
    rec = _Recorder(w, "progress", "finished_ok", "failed")
    w.run()
    assert rec.named("progress") == [(1, 2, 0.1), (2, 2, 0.2)]
    assert rec.named("finished_ok") == [("uresult",)]
    assert seen == dict(up="up", rp="rp", result="res", td="td", dispersion="disp")


def test_uncertainty_worker_stop(monkeypatch):
    def body(cb):
        cb(1, 5, 0.1)
        return "unreachable"

    _stub_uncertainty(monkeypatch, body)
    w = UncertaintyWorker("res", "td", "rp", "up")
    w.request_stop()
    rec = _Recorder(w, "finished_ok", "failed")
    w.run()
    assert rec.named("failed") == [("stopped",)]
    assert rec.named("finished_ok") == []


def test_uncertainty_worker_failure(monkeypatch):
    def body(_cb):
        raise ValueError("no trace recorded")

    _stub_uncertainty(monkeypatch, body)
    w = UncertaintyWorker("res", "td", "rp", "up")
    rec = _Recorder(w, "failed")
    w.run()
    assert rec.named("failed") == [("no trace recorded",)]
