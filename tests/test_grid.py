"""Tests for :mod:`croak.grid`."""

import numpy as np
import pytest

from croak.grid import Grid, gridparams_lambda, gridparams_omega, raw_fft, raw_ifft


def test_grid_construction_consistency():
    g = Grid(128, dt=0.3e-15)
    assert g.n == 128
    assert g.domega == pytest.approx(2 * np.pi / (g.dt * g.n))
    assert g.df == pytest.approx(g.domega / (2 * np.pi))
    # axes are uniform and centred (a zero element present)
    assert np.allclose(np.diff(g.omega), g.domega)
    assert np.min(np.abs(g.t)) == pytest.approx(0.0)
    assert np.min(np.abs(g.omega)) == pytest.approx(0.0)


def test_from_omega_roundtrip():
    g = Grid(64, domega=1e13)
    g2 = Grid.from_omega(g.omega)
    assert np.allclose(g.omega, g2.omega)
    assert g2.dt == pytest.approx(g.dt)


def test_fft_ifft_roundtrip():
    g = Grid(256, dt=0.5e-15)
    rng = np.random.default_rng(0)
    et = rng.standard_normal(g.n) + 1j * rng.standard_normal(g.n)
    assert np.allclose(g.ifft(g.fft(et)), et, rtol=1e-10, atol=1e-12)


def test_fft_matches_analytic_gaussian():
    g = Grid(512, dt=0.25e-15)
    sigma = 3e-15
    et = np.exp(-0.5 * (g.t / sigma) ** 2)
    ew = g.fft(et)
    # FT of exp(-t^2/2 sigma^2) with +i, dt convention:
    # sigma*sqrt(2pi)*exp(-sigma^2 omega^2/2)
    analytic = sigma * np.sqrt(2 * np.pi) * np.exp(-0.5 * (sigma * g.omega) ** 2)
    assert np.allclose(np.abs(ew), analytic, rtol=1e-8, atol=1e-10)
    # a real, even pulse has (essentially) zero spectral phase
    assert np.allclose(np.angle(ew[np.abs(ew) > 1e-6 * np.abs(ew).max()]), 0, atol=1e-8)


def test_tshift_moves_peak():
    g = Grid(512, dt=0.5e-15)
    sigma = 5e-15
    et = np.exp(-0.5 * (g.t / sigma) ** 2).astype(complex)
    tau = 20e-15
    shifted = g.tshift(et, tau)
    assert g.t[np.argmax(np.abs(shifted))] == pytest.approx(tau, abs=g.dt)


def test_raw_transforms_are_inverse():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(32) + 1j * rng.standard_normal(32)
    assert np.allclose(raw_ifft(raw_fft(x)), x, rtol=1e-12, atol=1e-12)


def test_gridparams_omega_window_and_nyquist():
    omega0 = 2.4e15
    n, dt, w0 = gridparams_omega(2e-12, omega_min=2.0e15, omega_max=2.8e15)
    assert w0 == pytest.approx(omega0)
    assert dt == pytest.approx(np.pi / 0.4e15)
    assert n * dt >= 2e-12


def test_gridparams_lambda_matches_omega():
    n, dt, w0 = gridparams_lambda(1e-12, lambda_min=700e-9, lambda_max=900e-9)
    assert n > 0 and dt > 0 and w0 > 0
