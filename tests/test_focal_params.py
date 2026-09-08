"""Tests for the session-level focal-mixture plumbing.

Covers the params-to-model builder (:func:`focal_mixture_from_params`), the
spectral-frame reweighting (:func:`apply_spectrum_frame`), and their wiring
through :func:`run_retrieval` — the route the GUI's focal-mixture group uses.
"""

from dataclasses import replace

import numpy as np
import pytest

from croak import preprocess
from croak.collection import CollectionAperture
from croak.focal import FocalMixture
from croak.session.params import RetrieveParams
from croak.session.pipeline import (
    apply_spectrum_frame,
    focal_mixture_from_params,
    run_retrieval,
)


@pytest.fixture
def tracedata(experiment):
    exp = experiment
    return preprocess.load_and_clean(
        exp.trace,
        exp.lam_nm * 1e-9,
        exp.delay_fs * 1e-15,
        exp.interaction,
        lam_min=exp.lam_min,
        lam_max=exp.lam_max,
        input_unit="delay",
        filter_fringes=False,
        dtau_min=None,
        threshold=1e-4,
    )


def _focal_params(**kw) -> RetrieveParams:
    base = dict(
        focal=True,
        smearing=False,
        focal_n_radial=3,
        focal_n_azimuth=4,
        smear_hole_diameter_mm=1.0,
        smear_hole_spacing_mm=1.0,
        focal_f_mm=100.0,
    )
    base.update(kw)
    return RetrieveParams(**base)


def test_builder_returns_none_when_off(tracedata):
    p = RetrieveParams(focal=False)
    assert focal_mixture_from_params(p, tracedata) is None


def test_builder_builds_a_mixture(tracedata):
    mix = focal_mixture_from_params(_focal_params(), tracedata)
    assert isinstance(mix, FocalMixture)
    assert mix.collection is None


def test_builder_manual_collection(tracedata):
    p = _focal_params(collection="manual", collection_diam_mm=0.5)
    mix = focal_mixture_from_params(p, tracedata)
    assert isinstance(mix.collection, CollectionAperture)


def test_builder_rejects_focal_plus_smearing(tracedata):
    p = _focal_params(smearing=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        focal_mixture_from_params(p, tracedata)


def test_builder_file_mode_needs_a_path(tracedata):
    p = _focal_params(collection="file")
    with pytest.raises(ValueError, match="scan path"):
        focal_mixture_from_params(p, tracedata)


def test_builder_rejects_unknown_collection(tracedata):
    p = _focal_params(collection="nope")
    with pytest.raises(ValueError, match="unknown collection"):
        focal_mixture_from_params(p, tracedata)


def test_spectrum_frame_is_a_noop_at_zero(tracedata):
    p = RetrieveParams(spectrum_frame_p=0.0)
    assert apply_spectrum_frame(p, tracedata) is tracedata


def test_spectrum_frame_reweights_the_spectrum(tracedata):
    if tracedata.Iomega is None:
        td = replace(tracedata, Iomega=np.ones(np.asarray(tracedata.omega).size))
    else:
        td = tracedata
    p = RetrieveParams(spectrum_frame_p=0.5)
    out = apply_spectrum_frame(p, td)
    om = np.asarray(td.omega, float)
    wfac = np.maximum((om + td.omega0_pulse) / td.omega0_pulse, 0.0)
    np.testing.assert_allclose(
        np.asarray(out.Iomega), np.asarray(td.Iomega) * wfac, rtol=1e-12
    )
    # the reweighting is monotone bluewards: more weight above the carrier
    assert out.Iomega[-1] >= td.Iomega[-1]


def test_run_retrieval_builds_the_mixture_from_params(tracedata):
    """The GUI route: p.focal set, no explicit object — must retrieve."""
    p = _focal_params(solver="lbfgs-ad", maxiters=3)
    res = run_retrieval(p, tracedata, rng=np.random.default_rng(0))
    assert res.spectrum.shape == (tracedata.grid.n,)


def test_builder_file_collection_follows_the_loaded_window(
    tmp_path, tracedata, simulated_truth
):
    """``collection="file"`` rebuilds the aperture of the window the trace was
    loaded from, not silently the first hole's (a 0.5 mm aperture on a 2.0 mm
    trace was the GUI's failure mode on multi-window campaign files)."""
    import h5py
    from conftest import write_simulated_h5

    record = {
        "type": "PhysicalMaskWindow",
        "holex": -1e-3,
        "holey": -1e-3,
        "holediam": 0.5e-3,
        "zmask": 0.1,
        "apod": "hard",
        "apod_param": 0.0,
        "delta_k": 7803.26,
        "reference_wavelength": 260e-9,
    }
    path = write_simulated_h5(
        tmp_path / "multi.h5", simulated_truth, mask_window=record
    )
    with h5py.File(path, "r+") as f:  # a second, numbered window: a 2.0 mm hole
        g = f["grid"]
        for key in ("type", "holex", "holey", "zmask", "apod", "apod_param"):
            g[f"window_def_2_{key}"] = g[f"window_def_{key}"][()]
        g["window_def_2_holediam"] = 2.0e-3
        f["Iω_win_2"] = f["Iω_win"][()]
    p = _focal_params(collection="file")
    first = focal_mixture_from_params(p, tracedata, path).collection
    second = focal_mixture_from_params(p, tracedata, path, "Iω_win_2").collection
    span = lambda ap: np.hypot(ap.x + 1e-3, ap.y + 1e-3).max()  # noqa: E731
    assert span(second) == pytest.approx(4.0 * span(first), rel=1e-6)
    # the wiring through run_retrieval accepts the window too
    assert (
        focal_mixture_from_params(
            p, tracedata, path, "Iω_win_2_reimaged"
        ).collection.nodes
        == second.nodes
    )
