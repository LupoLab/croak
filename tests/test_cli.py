"""Tests for the ``croak`` command-line interface (:mod:`croak.cli`)."""

from __future__ import annotations

import shutil

import h5py
from conftest import write_experiment_h5
from test_session import _options_for

from croak.cli import main
from croak.session import SessionOptions


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
