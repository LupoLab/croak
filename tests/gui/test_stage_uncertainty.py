"""Stage 6 (uncertainty): controls, retained estimates, combine, save and the run."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QObject, Qt, pyqtSignal

from croak.forward import maketrace
from croak.grid import Grid
from croak.gui import stage_uncertainty as su
from croak.gui import worker as worker_mod
from croak.gui.stage_uncertainty import StageUncertainty
from croak.gui.state import WizardState
from croak.maths import wlfreq
from croak.pulses import gaussian_pulse
from croak.retrieve import retrieve
from croak.uncertainty import UncertaintyResult


def _uresult(method: str, sd: float, seed: int = 0) -> UncertaintyResult:
    """A lightweight estimate with a temporal band, as the stage stores it."""
    rng = np.random.default_rng(seed)
    t = np.linspace(-15e-15, 15e-15, 30)
    prof = np.clip(
        np.exp(-((t / 1.5e-15) ** 2)) + 0.02 * rng.standard_normal((50, t.size)),
        0.0,
        None,
    )
    return UncertaintyResult(
        statistic="fwhm",
        point_estimate=1.5e-15,
        samples=1.5e-15 + sd * rng.standard_normal(50),
        interval_68=(1.5e-15 - sd, 1.5e-15 + sd),
        interval_95=(1.5e-15 - 2 * sd, 1.5e-15 + 2 * sd),
        interval_method="percentile",
        method=method,
        n_resamples=50,
        n_converged=50,
        frog_errors=np.empty(0),
        base_error=0.01,
        profiles=prof,
        t_profile=t,
    )


@pytest.fixture(scope="module")
def retrieval():
    g = Grid(64, dt=0.8e-15)
    ew = gaussian_pulse(g, 6e-15)
    delays = np.linspace(-30e-15, 30e-15, 40)
    trace = maketrace(g.omega, delays, ew, "pg")
    return retrieve(
        trace,
        g.omega,
        delays,
        "pg",
        algorithm="copra",
        maxiters=10,
        guess=ew,
        omega0=wlfreq(800e-9),
        progress=False,
    )


@pytest.fixture
def stage(qtbot):
    widget = StageUncertainty(WizardState())
    qtbot.addWidget(widget)
    return widget


def test_controls_write_the_parameters_and_gate_each_other(stage):
    u = stage.state.uncertainty
    stage.method.setCurrentText("thickness")
    assert u.method == "thickness"
    assert stage.thickness_sigma.isEnabled() and not stage.noise_source.isEnabled()
    stage.method.setCurrentText("parametric")
    stage.noise_source.setCurrentText("manual")
    assert u.noise_source == "manual" and stage.noise_sigma.isEnabled()
    stage.n_resamples.setValue(50)
    stage.noise_sigma.setValue(0.02)
    stage.thickness_sigma.setValue(1.5)
    stage.warm_start.setChecked(not u.warm_start)
    stage.collect_profiles.setChecked(not u.collect_profiles)
    stage.propagate.setChecked(not u.propagate_to_dispersion_point)
    stage.interval.setCurrentText("percentile")
    assert u.n_resamples == 50
    assert u.noise_sigma == pytest.approx(0.02)
    assert u.thickness_sigma_um == pytest.approx(1.5)
    assert u.warm_start == stage.warm_start.isChecked()
    assert u.collect_profiles == stage.collect_profiles.isChecked()
    assert u.propagate_to_dispersion_point == stage.propagate.isChecked()
    assert u.interval == "percentile"


def test_on_enter_without_a_result_disables_estimate(stage):
    stage.on_enter()
    assert not stage.run_btn.isEnabled()
    assert stage.status_label.text() == "Retrieve a pulse first."


def test_on_enter_lists_and_draws_the_current_estimate(stage, retrieval):
    stage.state.set_result(retrieval)
    stage.on_enter()
    assert stage.run_btn.isEnabled()
    assert "Choose options" in stage.status_label.text()
    stage.state.set_uncertainty_result(_uresult("parametric", 0.05e-15))
    stage.on_enter()
    assert stage.est_list.count() == 1
    assert stage.canvas.rendered
    assert stage.status_label.text() == stage.state.uncertainty_result.summary()


def test_estimates_list_ticks_components_and_selects_to_view(stage):
    stage.state.set_uncertainty_result(_uresult("parametric", 0.05e-15))
    stage.state.set_uncertainty_result(_uresult("combined", 0.07e-15))
    stage._refresh_estimates()
    items = [stage.est_list.item(i) for i in range(stage.est_list.count())]
    states = {i.data(Qt.ItemDataRole.UserRole): i.checkState() for i in items}
    assert states == {
        "parametric": Qt.CheckState.Checked,
        "combined": Qt.CheckState.Unchecked,
    }
    assert "parametric: 1.50 ±" in items[0].text()
    # Only one ticked: combining is not offered.
    assert not stage.combine_btn.isEnabled()
    stage.est_list.setCurrentRow(1)
    assert (
        stage.status_label.text()
        == stage.state.uncertainty_results["combined"].summary()
    )
    shown = stage.status_label.text()
    # Clearing the selection, or selecting an estimate that has since gone,
    # leaves the view as it was.
    stage._on_select_estimate(None, None)
    items[0].setData(Qt.ItemDataRole.UserRole, "missing")
    stage.est_list.setCurrentRow(0)
    assert stage.status_label.text() == shown


def test_combine_needs_two_ticked_estimates(stage):
    stage.state.set_uncertainty_result(_uresult("parametric", 0.05e-15))
    stage._refresh_estimates()
    stage._combine()
    assert stage.status_label.text() == "Tick at least two estimates to combine."


def test_combine_two_estimates_adds_a_combined_total(stage):
    stage.state.set_uncertainty_result(_uresult("parametric", 0.05e-15, seed=1))
    stage.state.set_uncertainty_result(_uresult("thickness", 0.10e-15, seed=2))
    stage._refresh_estimates()
    assert stage.combine_btn.isEnabled()
    stage._combine()
    combined = stage.state.uncertainty_results["combined"]
    # Independent sources add in quadrature, so the total exceeds either part.
    assert combined.plus_minus > stage.state.uncertainty_results["thickness"].plus_minus
    assert stage.est_list.count() == 3
    assert stage.status_label.text() == combined.summary()


def test_combine_failure_is_reported(stage, monkeypatch):
    def fail(_results):
        raise ValueError("incompatible bands")

    monkeypatch.setattr(su, "combine_uncertainties", fail)
    stage.state.set_uncertainty_result(_uresult("parametric", 0.05e-15))
    stage.state.set_uncertainty_result(_uresult("thickness", 0.10e-15))
    stage._refresh_estimates()
    stage._combine()
    assert stage.status_label.text() == "Combine failed: incompatible bands"


def test_save_needs_an_estimate(stage):
    stage._save()
    assert stage.status_label.text() == "Run an estimate first."


def test_save_cancelled_writes_nothing(stage, retrieval, monkeypatch, tmp_path):
    stage.state.set_result(retrieval)
    stage.state.set_uncertainty_result(_uresult("parametric", 0.05e-15))
    monkeypatch.setattr(
        su.QFileDialog, "getExistingDirectory", staticmethod(lambda *a: "")
    )
    stage._save()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("drawn", [False, True])
def test_save_writes_result_options_and_the_figure_if_drawn(
    stage, retrieval, monkeypatch, tmp_path, drawn
):
    stage.state.set_result(retrieval)
    stage.state.set_uncertainty_result(_uresult("parametric", 0.05e-15))
    if drawn:
        stage._draw(stage.state.uncertainty_result)
    monkeypatch.setattr(
        su.QFileDialog,
        "getExistingDirectory",
        staticmethod(lambda *a: str(tmp_path)),
    )
    stage._save()
    assert (tmp_path / "options.toml").exists()
    with h5py.File(tmp_path / "result.h5", "r") as f:
        assert len(f.keys()) > 0
    assert (tmp_path / "uncertainty.pdf").exists() == drawn
    assert stage.status_label.text().startswith("Saved result.h5")


class _FakeWorker(QObject):
    """Stands in for UncertaintyWorker: emits what the test scripts on start()."""

    progress = pyqtSignal(int, int, float)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)
    script: list = []
    instances: list = []

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args, self.kwargs = args, kwargs
        self.stopped = False
        _FakeWorker.instances.append(self)

    def start(self):
        for name, *args in self.script:
            getattr(self, name).emit(*args)

    def request_stop(self):
        self.stopped = True


@pytest.fixture
def fake_worker(monkeypatch):
    _FakeWorker.instances = []
    monkeypatch.setattr(worker_mod, "UncertaintyWorker", _FakeWorker)
    return _FakeWorker


def test_run_without_a_retrieval_does_nothing(stage, fake_worker):
    stage._run()
    assert fake_worker.instances == []
    assert stage.status_label.text() == "Retrieve a pulse first."


def test_run_reports_progress_and_shows_the_estimate(stage, retrieval, fake_worker):
    stage.state.set_result(retrieval)
    stage.state.tracedata = object()
    est = _uresult("parametric", 0.05e-15)
    statuses = []
    set_status = stage.set_status
    stage.set_status = lambda text: (statuses.append(text), set_status(text))
    fake_worker.script = [("progress", 1, 4, 0.0123), ("finished_ok", est)]
    stage._run()
    (w,) = fake_worker.instances
    assert w.kwargs["dispersion"] is stage.state.dispersion
    assert "replicate 1/4:  R = 1.2300%" in statuses
    assert stage.state.uncertainty_results["parametric"] is est
    assert stage.run_btn.isEnabled() and not stage.stop_btn.isEnabled()
    assert stage.status_label.text() == est.summary()
    # Stop forwards to the worker.
    stage._stop()
    assert w.stopped


@pytest.mark.parametrize(
    ("message", "status"),
    [
        ("stopped", "Stopped."),
        ("boom", "Uncertainty estimation failed: boom"),
    ],
)
def test_run_failure_and_stop_messages(stage, retrieval, fake_worker, message, status):
    stage.state.set_result(retrieval)
    stage.state.tracedata = object()
    fake_worker.script = [("failed", message)]
    stage._run()
    assert stage.status_label.text() == status
    assert stage.run_btn.isEnabled() and not stage.stop_btn.isEnabled()


def test_stop_without_a_run_is_harmless(stage):
    stage._stop()
