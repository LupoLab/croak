"""Tests for the ``croak`` command-line interface (:mod:`croak.cli`)."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import h5py
import pytest
from conftest import write_experiment_h5
from test_session import _options_for

from croak import cli
from croak.cli import main
from croak.session import SessionOptions, SessionResult


def _write_options(experiment_h5: str, experiment, path: str) -> SessionOptions:
    """A small, fast options.toml for the synthetic experiment at ``experiment_h5``."""
    options = _options_for(experiment_h5, experiment)
    options.retrieve.maxiters = 30  # the CLI tests check plumbing, not convergence
    options.to_toml(path)
    return options


def test_cli_replay_in_place(tmp_path, experiment_h5, experiment):
    """`croak replay options.toml --out DIR` writes result.h5 + options.toml."""
    opts_path = str(tmp_path / "options.toml")
    _write_options(experiment_h5, experiment, opts_path)
    out = tmp_path / "out"

    rc = main(["replay", opts_path, "--out", str(out), "--seed", "0", "--force"])

    assert rc == 0
    assert (out / "result.h5").exists()
    assert (out / "options.toml").exists()
    # the saved result is a readable retrieval with a FROG error attribute
    with h5py.File(out / "result.h5", "r") as f:
        assert "error" in f


def test_cli_replay_retarget_batch(tmp_path, experiment, experiment_h5):
    """`--base-dir` (repeated) retargets and writes a per-dataset subfolder."""
    # Build the options against run01, then copy the folder to run02/run03.
    run01 = tmp_path / "run01"
    run01.mkdir()
    write_experiment_h5(run01 / "acq.h5", experiment)
    opts_path = str(tmp_path / "options.toml")
    _write_options(str(run01 / "acq.h5"), experiment, opts_path)
    runs = []
    for name in ("run02", "run03"):
        dst = tmp_path / name
        shutil.copytree(run01, dst)
        runs.append(str(dst))

    out = tmp_path / "batch"
    rc = main(
        [
            "replay",
            opts_path,
            "--base-dir",
            runs[0],
            "--base-dir",
            runs[1],
            "--out",
            str(out),
            "--seed",
            "0",
            "--force",
        ]
    )

    assert rc == 0
    assert (out / "run02" / "result.h5").exists()
    assert (out / "run03" / "result.h5").exists()
    # the retargeted options.toml points at the new dataset folder
    moved = SessionOptions.from_toml(str(out / "run02" / "options.toml"))
    assert moved.load.frog_path == str(tmp_path / "run02" / "acq.h5")


def test_cli_replay_defaults_to_base_dir(tmp_path, experiment, experiment_h5):
    """Without --out, results land inside the retargeted dataset folder."""
    run01 = tmp_path / "run01"
    run01.mkdir()
    write_experiment_h5(run01 / "acq.h5", experiment)
    opts_path = str(tmp_path / "options.toml")
    _write_options(str(run01 / "acq.h5"), experiment, opts_path)
    dst = tmp_path / "run02"
    shutil.copytree(run01, dst)

    rc = main(["replay", opts_path, "--base-dir", str(dst), "--seed", "0", "--force"])

    assert rc == 0
    assert (dst / "result.h5").exists()


def test_cli_replay_missing_base_dir_reports_failure(
    tmp_path, experiment, experiment_h5
):
    """A missing dataset folder fails loudly (non-zero exit) without crashing."""
    opts_path = str(tmp_path / "options.toml")
    _write_options(experiment_h5, experiment, opts_path)

    rc = main(["replay", opts_path, "--base-dir", str(tmp_path / "nope"), "--force"])

    assert rc == 1  # the loader raised on the absent file; batch returns failure


# ---------------------------------------------------------------------------
# Output placement, saving and reporting (stubbed: plumbing only)
# ---------------------------------------------------------------------------
def test_resolve_outdir_priority(tmp_path):
    opts = str(tmp_path / "sub" / "options.toml")
    assert cli._resolve_outdir("out", None, opts) == Path("out")
    assert cli._resolve_outdir("out", "/data/run7", opts) == Path("out") / "run7"
    assert cli._resolve_outdir(None, "/data/run7", opts) == Path("/data/run7")
    assert cli._resolve_outdir(None, None, opts) == (tmp_path / "sub").resolve()


def test_save_session_without_a_result_raises(tmp_path):
    res = SessionResult(options=SessionOptions())
    with pytest.raises(RuntimeError, match="no retrieval result to save"):
        cli._save_session(res, SessionOptions(), tmp_path, force=False)


def test_save_session_writes_the_dispersed_pulse_and_uncertainty(tmp_path, monkeypatch):
    saved = {}
    monkeypatch.setattr(
        cli,
        "save_result",
        lambda final, path, **kw: saved.update(result=(final, path, kw)),
    )
    monkeypatch.setattr(
        cli,
        "save_uncertainty",
        lambda results, path, force: saved.update(unc=(results, path, force)),
    )
    unc = SimpleNamespace(method="parametric")
    res = SessionResult(
        options=SessionOptions(),
        result="raw",
        dispersed="dispersed",
        processed="processed",
        uncertainty=unc,
    )
    path = cli._save_session(res, SessionOptions(), tmp_path, force=True)
    assert path == tmp_path / "result.h5"
    final, where, kw = saved["result"]
    assert final == "dispersed" and where == str(path)
    # no cleaned trace (tracedata) → no measured trace to store alongside
    assert kw == {"processed": "processed", "measured": None, "force": True}
    assert saved["unc"] == ({"parametric": unc}, str(path), True)
    assert (tmp_path / "options.toml").exists()


def test_report_lists_only_what_ran(capsys):
    res = SessionResult(
        options=SessionOptions(),
        uncertainty=SimpleNamespace(plus_minus=0.25e-15),
    )
    cli._report("run1", res, Path("x/result.h5"))
    line = capsys.readouterr().out
    assert line.startswith("[croak] run1:")
    assert "± 0.25 fs" in line and "R =" not in line and "FWHM" not in line
    assert line.rstrip().endswith(str(Path("x/result.h5")))


def test_cli_replay_verbose_prints_progress_and_saves_next_to_options(
    tmp_path, monkeypatch, capsys
):
    """Without --out the result lands beside the options file; -v streams R."""
    opts_path = tmp_path / "options.toml"
    SessionOptions().to_toml(str(opts_path))

    def fake_run_session(opts, *, stages, rng, retrieve_callback):
        retrieve_callback(3, 0.125, 0.1)
        return SessionResult(options=opts)

    seen = {}
    monkeypatch.setattr(cli, "run_session", fake_run_session)
    monkeypatch.setattr(
        cli,
        "_save_session",
        lambda res, opts, outdir, force: seen.setdefault("outdir", outdir) / "r.h5",
    )
    assert main(["replay", str(opts_path), "--verbose"]) == 0
    assert seen["outdir"] == tmp_path.resolve()
    assert "iter    3: R = 12.5000%" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# `croak script`
# ---------------------------------------------------------------------------
def test_cli_script_to_stdout(tmp_path, experiment_h5, experiment, capsys):
    opts_path = str(tmp_path / "options.toml")
    _write_options(experiment_h5, experiment, opts_path)
    assert main(["script", opts_path, "--seed", "4"]) == 0
    source = capsys.readouterr().out
    compile(source, "replay.py", "exec")  # a runnable script, not a template
    assert "SEED = 4" in source


def test_cli_script_to_file(tmp_path, experiment_h5, experiment, capsys):
    opts_path = str(tmp_path / "options.toml")
    _write_options(experiment_h5, experiment, opts_path)
    target = tmp_path / "rerun.py"
    assert main(["script", opts_path, "-o", str(target)]) == 0
    assert f"wrote {target}" in capsys.readouterr().out
    compile(target.read_text(), str(target), "exec")


def test_cli_script_reports_a_synthetic_session(tmp_path, capsys):
    opts_path = tmp_path / "options.toml"
    SessionOptions(entry="synthetic").to_toml(str(opts_path))
    assert main(["script", str(opts_path)]) == 1
    assert "synthetic" in capsys.readouterr().err
