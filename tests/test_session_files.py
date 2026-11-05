"""A session's files: saving a session folder and checking a saved result."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from test_session import _options_for

from croak import save
from croak.session import run_session
from croak.session.files import (
    OPTIONS_FILE,
    OVERVIEW_FILE,
    RESULT_FILE,
    result_mismatch,
    save_session,
)


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    """A short real retrieval of the synthetic experiment, with its options."""
    from conftest import make_synthetic_experiment, write_experiment_h5

    experiment = make_synthetic_experiment()
    path = write_experiment_h5(
        str(tmp_path_factory.mktemp("data") / "exp.h5"), experiment
    )
    options = _options_for(path, experiment)
    options = replace(options, retrieve=replace(options.retrieve, maxiters=5))
    out = run_session(
        options,
        stages=("load", "preproc", "retrieve", "process"),
        rng=np.random.default_rng(0),
    )
    return options, out


def test_session_files_save_options_alone(tmp_path, session):
    options, _ = session
    assert save_session(tmp_path, options) == [OPTIONS_FILE]
    assert save.load_options(str(tmp_path / OPTIONS_FILE))["retrieve"]["maxiters"] == 5


def test_session_files_save_result_uncertainty_and_overview(tmp_path, session):
    from croak.plotting import retrieval_plot_data

    options, out = session
    td = out.tracedata
    overview = retrieval_plot_data(out.result, measured=td.trace, processed=None)
    written = save_session(
        tmp_path,
        options,
        result=out.result,
        processed=out.processed,
        uncertainty={},  # nothing estimated: nothing appended
        overview=overview,
    )
    assert written == [RESULT_FILE, OPTIONS_FILE, OVERVIEW_FILE]
    assert (tmp_path / OVERVIEW_FILE).stat().st_size > 0
    saved = save.load_result(str(tmp_path / RESULT_FILE))
    np.testing.assert_allclose(saved["Ew_retr"], out.result.spectrum)


def test_session_files_append_uncertainty_to_the_result(tmp_path, session, monkeypatch):
    options, out = session
    appended = {}
    monkeypatch.setattr(
        save, "save_uncertainty", lambda est, path, force: appended.update(est)
    )
    save_session(tmp_path, options, result=out.result, uncertainty={"m": "estimate"})
    assert appended == {"m": "estimate"}


def test_session_files_refuse_a_missing_folder_or_orphan_uncertainty(tmp_path, session):
    options, _ = session
    with pytest.raises(FileNotFoundError, match="no such folder"):
        save_session(tmp_path / "absent", options)
    with pytest.raises(ValueError, match="need the result"):
        save_session(tmp_path, options, uncertainty={"m": object()})


def test_session_files_accept_the_result_fitted_to_this_trace(tmp_path, session):
    options, out = session
    save_session(tmp_path, options, result=out.result, processed=out.processed)
    saved = save.load_result(str(tmp_path / RESULT_FILE))
    assert result_mismatch(saved, out.tracedata, options.retrieve) == ""
    del saved["omega"]  # an older file without the axis: the trace still decides
    assert result_mismatch(saved, out.tracedata, options.retrieve) == ""


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (lambda s: s.pop("trace_meas"), "does not store the trace"),
        (lambda s: s.update(trace_meas=s["trace_meas"][:-1]), "grid but"),
        (lambda s: s.update(trace_meas=1.5 * s["trace_meas"]), "no longer reproduces"),
        (lambda s: s.update(omega=1.01 * s["omega"]), "different frequency grid"),
        (lambda s: s.update(omega=s["omega"][:-1]), "different frequency grid"),
        (lambda s: s.update(algorithm="no-such-solver"), "came from no-such-solver"),
    ],
)
def test_session_files_refuse_a_result_fitted_to_other_data(
    tmp_path, session, change, reason
):
    options, out = session
    save_session(tmp_path, options, result=out.result, processed=out.processed)
    saved = save.load_result(str(tmp_path / RESULT_FILE))
    change(saved)
    assert reason in result_mismatch(saved, out.tracedata, options.retrieve)


def test_format_options_matches_the_saved_file(tmp_path, session):
    options, _ = session
    save_session(tmp_path, options)
    text = (tmp_path / OPTIONS_FILE).read_text(encoding="utf-8")
    assert text == save.format_options(options.to_dict())
