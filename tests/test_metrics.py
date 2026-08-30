"""Tests for :mod:`croak.metrics`."""

import numpy as np
import pytest

from croak import metrics


def test_mu_recovers_scale():
    rng = np.random.default_rng(0)
    t_meas = rng.random((16, 20))
    t_sim = t_meas / 2.0
    w = np.ones_like(t_meas)
    assert metrics.compute_mu(t_meas, t_sim, w) == pytest.approx(2.0)


def test_identical_traces_zero_error():
    rng = np.random.default_rng(1)
    t = rng.random((16, 20))
    assert metrics.frog_error(t, t) == pytest.approx(0.0, abs=1e-12)


def test_r_and_R_consistency():
    rng = np.random.default_rng(2)
    t_meas = rng.random((8, 8))
    t_sim = t_meas + 0.01 * rng.standard_normal((8, 8))
    w = np.ones_like(t_meas)
    mu = metrics.compute_mu(t_meas, t_sim, w)
    r = metrics.compute_r(t_meas, t_sim, w, mu)
    R = metrics.compute_R(r, t_meas, w)
    assert R == pytest.approx(metrics.frog_error(t_meas, t_sim))


def test_mu_per_freq_shape_and_values():
    t_meas = np.ones((4, 6))
    t_sim = 0.5 * np.ones((4, 6))
    w = np.ones(4)
    mu = metrics.compute_mu_per_freq(t_meas, t_sim, w, freq_axis=0)
    assert mu.shape == (4,)
    assert np.allclose(mu, 2.0)


def test_frog_error_vector_weights_broadcast():
    rng = np.random.default_rng(3)
    t_meas = rng.random((5, 7))
    t_sim = rng.random((5, 7))
    wvec = np.array([1.0, 1.0, 0.0, 1.0, 1.0])
    wmat = np.broadcast_to(wvec[:, None], t_meas.shape)
    assert metrics.frog_error(t_meas, t_sim, wvec) == pytest.approx(
        metrics.frog_error(t_meas, t_sim, wmat)
    )
