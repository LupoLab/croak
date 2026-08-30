"""Tests for the terminal/notebook progress reporting in :mod:`croak.progress`."""

import io
import warnings

import numpy as np
import pytest

import croak
from croak.progress import ProgressReporter, format_report, make_reporter


@pytest.fixture
def problem():
    g = croak.Grid(64, dt=0.4e-15)
    ew = croak.gaussian_pulse(g, 2.5e-15)
    delays = np.linspace(-15e-15, 15e-15, 61)
    trace = croak.maketrace(g.omega, delays, ew, "shg")
    return g, ew, delays, trace


def test_make_reporter_auto_is_silent_when_non_interactive():
    # Under pytest stderr is not a TTY -> "plain" -> auto reports nothing.
    assert make_reporter("auto", total=100) is None


def test_make_reporter_false_disables():
    assert make_reporter(False, total=100) is None
    assert make_reporter(True, total=100) is not None


def test_format_report_contains_all_fields(problem):
    g, ew, delays, trace = problem
    res = croak.retrieve(trace, g.omega, delays, "shg", guess=ew, maxiters=10)
    report = format_report(res, 1.5)
    assert "final error R" in report
    assert "retrieved FWHM" in report
    assert "transform limit" in report
    assert f"iterations       : {len(res.errors)}" in report
    assert "time taken       : 1.50 s" in report


def test_reporter_builtin_bar_draws_to_stream(problem):
    # use_tqdm=False forces the built-in carriage-return bar, whose rendering is
    # deterministic and capturable (tqdm manages its own stream).
    g, ew, delays, trace = problem
    res = croak.retrieve(trace, g.omega, delays, "shg", guess=ew, maxiters=10)
    buf = io.StringIO()
    rep = ProgressReporter(
        total=10, env="terminal", live=True, min_interval=0.0, use_tqdm=False
    )
    rep.stream = buf
    rep.callback(1, 0.5, 0.5)
    rep.callback(10, 0.01, 0.01)
    drawn = buf.getvalue()
    assert "10/10" in drawn  # progress fraction rendered
    assert "█" in drawn  # bar glyph
    # finish() emits the summary to stdout (captured by pytest's capsys elsewhere)
    rep.finish(res, 2.0)


def test_reporter_tqdm_path_runs(problem, capsys):
    # The default renderer is tqdm.auto; exercise create/update/close + report.
    g, ew, delays, trace = problem
    res = croak.retrieve(trace, g.omega, delays, "shg", guess=ew, maxiters=10)
    rep = ProgressReporter(total=10, env="terminal", live=True)
    assert rep._tqdm is not None  # tqdm is a declared dependency
    rep.callback(1, 0.5, 0.5)
    rep.callback(10, 0.01, 0.01)
    rep.finish(res, 2.0)
    assert "retrieval complete" in capsys.readouterr().out


def test_reporter_not_live_draws_nothing(problem):
    g, ew, delays, trace = problem
    res = croak.retrieve(trace, g.omega, delays, "shg", guess=ew, maxiters=10)
    buf = io.StringIO()
    rep = ProgressReporter(total=10, env="plain", live=False)
    rep.stream = buf
    rep.callback(1, 0.5, 0.5)
    assert buf.getvalue() == ""  # no live bar when not live
    rep.finish(res, 1.0)  # report still prints (to stdout), no exception


def test_retrieve_with_user_callback_skips_builtin_reporter(problem, capsys):
    g, ew, delays, trace = problem
    seen = []
    croak.retrieve(
        trace,
        g.omega,
        delays,
        "shg",
        guess=ew,
        maxiters=10,
        callback=lambda it, R, best: seen.append(it),
        progress=True,  # would normally force a report, but a callback wins
    )
    assert seen  # the user's callback was invoked
    # No built-in report printed when the caller supplies its own callback.
    assert "retrieval complete" not in capsys.readouterr().out


def test_retrieve_progress_true_prints_report(problem, capsys):
    g, ew, delays, trace = problem
    croak.retrieve(trace, g.omega, delays, "shg", guess=ew, maxiters=10, progress=True)
    out = capsys.readouterr().out
    assert "retrieval complete" in out
    assert "retrieved FWHM" in out


def test_retrieve_progress_false_is_silent(problem, capsys):
    g, ew, delays, trace = problem
    croak.retrieve(trace, g.omega, delays, "shg", guess=ew, maxiters=10, progress=False)
    assert capsys.readouterr().out == ""


def test_format_report_emits_no_warning(problem):
    # process_result computes a 2πc/ω wavelength axis that divides by zero on a
    # centred ω grid; format_report must suppress that RuntimeWarning so the
    # terminal report stays clean.
    g, ew, delays, trace = problem
    res = croak.retrieve(trace, g.omega, delays, "shg", guess=ew, maxiters=10)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        format_report(res, 1.0)


def test_reporter_falls_back_to_text_bar_without_ipywidgets(monkeypatch):
    # A notebook kernel without ipywidgets must use the text bar (tqdm.std) and
    # emit no TqdmWarning — i.e. we do not go through tqdm.auto's probing.
    monkeypatch.setattr("croak.progress._have_ipywidgets", lambda: False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rep = ProgressReporter(total=5, env="notebook", live=True)
    assert type(rep._tqdm).__module__ == "tqdm.std"
