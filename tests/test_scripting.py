"""Tests for generated replay scripts (:mod:`croak.scripting`)."""

from __future__ import annotations

import numpy as np
import pytest
from test_session import _options_for

import croak
from croak.scripting import generate_script
from croak.session import SessionOptions, retarget, run_session
from croak.session.params import SimulatedLoadParams


def _fast_options(experiment_h5, experiment):
    options = _options_for(experiment_h5, experiment)
    options.retrieve.maxiters = 30
    return options


def test_generate_script_compiles(experiment_h5, experiment):
    """A generated script is syntactically valid Python."""
    options = _fast_options(experiment_h5, experiment)
    options.dispersion.gdd_fs2 = 25.0  # exercise the dispersion block
    source = generate_script(options, output="result.h5", seed=0)
    compile(source, "<generated>", "exec")  # raises SyntaxError on failure
    # numpy scalars are rendered as plain literals, not np.float64(...)
    assert "np.float64(" not in source


def test_generated_script_runs_and_matches_engine(tmp_path, experiment_h5, experiment):
    """Executing the script reproduces the engine's result for the same seed."""
    options = _fast_options(experiment_h5, experiment)
    out_h5 = tmp_path / "result.h5"
    source = generate_script(options, output=str(out_h5), seed=0)

    exec(  # noqa: S102 - running the generated script is the point of this test
        compile(source, "<generated>", "exec"), {"__name__": "__main__"}
    )

    assert out_h5.exists()
    script_error = float(croak.save.load_result(str(out_h5))["error_final"])
    # the engine, same seed and stages, must land on the same FROG error
    engine = run_session(
        options,
        stages=("load", "preproc", "retrieve", "dispersion", "process"),
        rng=np.random.default_rng(0),
    )
    assert engine.result is not None
    np.testing.assert_allclose(script_error, engine.result.error, rtol=0, atol=1e-12)


def test_generate_script_retargets(tmp_path, experiment_h5, experiment):
    """``base_dir`` rewrites the rendered load path to the new dataset folder."""
    options = _fast_options(experiment_h5, experiment)
    source = generate_script(options, base_dir="/data/run99")
    moved = retarget(options, "/data/run99")
    assert f"frog_path={moved.load.frog_path!r}" in source


def _simulated_options(simulated_scan_h5, **simulated_changes):
    options = SessionOptions(
        entry="simulated",
        simulated=SimulatedLoadParams(frog_path=simulated_scan_h5, **simulated_changes),
    )
    options.retrieve.maxiters = 30
    return options


def test_generate_script_uses_the_simulated_loader(simulated_scan_h5):
    """A simulated session scripts through the simulated loader, not [load]."""
    source = generate_script(_simulated_options(simulated_scan_h5))
    compile(source, "<generated>", "exec")

    assert "simulated = SimulatedLoadParams(" in source
    assert f"frog_path={simulated_scan_h5!r}" in source
    assert "data = assemble_simulated_load_data(simulated)" in source
    assert "tracedata = build_tracedata(preproc, data)" in source
    # the experimental loader has no business in this script
    assert "assemble_load_data(load)" not in source
    assert "LoadParams(" not in source.replace("SimulatedLoadParams(", "")


def test_generate_script_simulated_raw_direct_skips_the_regrid(simulated_scan_h5):
    """``raw_direct`` scripts the native-grid trace, with no regrid step."""
    source = generate_script(_simulated_options(simulated_scan_h5, raw_direct=True))
    compile(source, "<generated>", "exec")

    assert "tracedata = assemble_simulated_tracedata(simulated)" in source
    assert "build_tracedata" not in source


def test_generated_simulated_raw_script_seeds_and_saves_the_truth(
    tmp_path, simulated_truth
):
    """A truth-seeded native-grid script carries the truth end to end.

    ``raw_direct`` builds its trace without ``build_tracedata``, so the script
    still has to assemble the load bundle just for ``data["truth"]`` --- drop
    that and ``truth_init`` fails at run time, not at compile time. Executing
    the script is what catches it.
    """
    from conftest import write_simulated_h5

    path = write_simulated_h5(
        tmp_path / "complex.h5", simulated_truth, store_complex=True
    )
    options = _simulated_options(path, raw_direct=True, truth_source="source")
    options.retrieve.truth_init = True
    out_h5 = tmp_path / "result.h5"
    source = generate_script(options, output=str(out_h5), seed=0)

    exec(  # noqa: S102 - running the generated script is the point of this test
        compile(source, "<generated>", "exec"), {"__name__": "__main__"}
    )

    saved = croak.save.load_result(str(out_h5))
    # the complex truth reached the saver, so the diagnostic is reproducible
    assert "Ew_truth" in saved and "phit_truth" in saved
    engine = run_session(
        options,
        stages=("load", "preproc", "retrieve", "dispersion", "process"),
        rng=np.random.default_rng(0),
    )
    assert engine.result is not None
    np.testing.assert_allclose(
        float(saved["error_final"]), engine.result.error, rtol=0, atol=1e-12
    )


def test_generate_script_rejects_synthetic_sessions():
    """A synthetic session has no saved generator settings to script."""
    with pytest.raises(ValueError, match="cannot script a synthetic session"):
        generate_script(SessionOptions(entry="synthetic"))
