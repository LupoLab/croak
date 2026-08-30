"""Tests for :mod:`croak.forward`: forward map, thin reduction, and adjoint."""

import numpy as np
import pytest

from croak.forward import ForwardModel, maketrace, quadrature_nodes_weights
from croak.grid import Grid
from croak.maths import wlfreq


@pytest.fixture
def grid():
    return Grid(48, dt=1.0e-15)


def _gauss_spectrum(g, fwhm=8e-15):
    et = np.exp(-0.5 * (g.t / (fwhm / 2.3548)) ** 2).astype(complex)
    ew = g.fft(et)
    return ew / np.sqrt(np.max(np.abs(ew) ** 2))


def test_quadrature_thin_and_gausslegendre():
    nodes, weights = quadrature_nodes_weights(0.0, 1)
    assert nodes == pytest.approx([0.0]) and weights == pytest.approx([1.0])
    nodes, weights = quadrature_nodes_weights(10e-6, 20)
    assert weights.sum() == pytest.approx(10e-6)  # integrates the constant 1
    assert np.all((nodes >= 0) & (nodes <= 10e-6))


@pytest.mark.parametrize("interaction", ["shg", "sd", "pg"])
def test_trace_shape_and_normalization(grid, interaction):
    ew = _gauss_spectrum(grid)
    delays = np.linspace(-30e-15, 30e-15, 21)
    trace = maketrace(grid.omega, delays, ew, interaction)
    assert trace.shape == (grid.n, delays.size)
    assert trace.max() == pytest.approx(1.0)
    assert np.all(trace >= 0)


@pytest.mark.parametrize("interaction", ["shg", "sd", "pg"])
def test_thin_equals_zero_thickness_dispersive(grid, interaction):
    ew = _gauss_spectrum(grid)
    delays = np.linspace(-30e-15, 30e-15, 15)
    omega0 = wlfreq(800e-9)
    thin = maketrace(grid.omega, delays, ew, interaction)
    disp0 = maketrace(
        grid.omega,
        delays,
        ew,
        interaction,
        material="SiO2",
        thickness=0.0,
        npoints=1,
        omega0=omega0,
    )
    assert np.allclose(thin, disp0, rtol=1e-12, atol=1e-12)


def _loss_and_grad(model, ew, weights):
    """L = sum_j Re(sum_omega conj(W[:,j]) psi_j); returns (L, dL/dEw*)."""
    loss = 0.0
    ew_bar = np.zeros(model.n, dtype=complex)
    for j, tau in enumerate(model.delays):
        psi = model.signal_single(ew, tau, record=True)
        loss += float(np.real(np.sum(np.conj(weights[:, j]) * psi)))
        ew_bar += model.adjoint_single(0.5 * weights[:, j], tau)
    return loss, ew_bar


def _fd_grad(func, z, h=1e-7):
    gr = np.zeros(len(z))
    gi = np.zeros(len(z))
    for k in range(len(z)):
        zp = z.copy()
        zp[k] += h
        zm = z.copy()
        zm[k] -= h
        gr[k] = (func(zp) - func(zm)) / (2 * h)
        zp = z.copy()
        zp[k] += 1j * h
        zm = z.copy()
        zm[k] -= 1j * h
        gi[k] = (func(zp) - func(zm)) / (2 * h)
    return gr, gi


@pytest.mark.parametrize("interaction", ["shg", "sd", "pg"])
def test_adjoint_matches_fd_thin(grid, interaction):
    rng = np.random.default_rng(7)
    ew = _gauss_spectrum(grid) + 0.1 * (
        rng.standard_normal(grid.n) + 1j * rng.standard_normal(grid.n)
    )
    delays = np.array([-10e-15, 0.0, 12e-15])
    model = ForwardModel(grid.omega, delays, interaction)
    weights = rng.standard_normal((grid.n, delays.size)) + 1j * rng.standard_normal(
        (grid.n, delays.size)
    )

    _, ew_bar = _loss_and_grad(model, ew, weights)

    def loss_only(z):
        return sum(
            float(np.real(np.sum(np.conj(weights[:, j]) * model.signal_single(z, tau))))
            for j, tau in enumerate(delays)
        )

    fd_re, fd_im = _fd_grad(loss_only, ew)
    assert np.allclose(2 * np.real(ew_bar), fd_re, atol=1e-4, rtol=1e-4)
    assert np.allclose(2 * np.imag(ew_bar), fd_im, atol=1e-4, rtol=1e-4)


def test_adjoint_matches_fd_dispersive(grid):
    rng = np.random.default_rng(3)
    ew = _gauss_spectrum(grid) + 0.05 * (
        rng.standard_normal(grid.n) + 1j * rng.standard_normal(grid.n)
    )
    delays = np.array([-8e-15, 0.0, 8e-15])
    omega0 = wlfreq(800e-9)
    model = ForwardModel(
        grid.omega,
        delays,
        "pg",
        material="SiO2",
        thickness=20e-6,
        npoints=8,
        omega0=omega0,
    )
    weights = rng.standard_normal((grid.n, delays.size)) + 1j * rng.standard_normal(
        (grid.n, delays.size)
    )
    _, ew_bar = _loss_and_grad(model, ew, weights)

    def loss_only(z):
        return sum(
            float(np.real(np.sum(np.conj(weights[:, j]) * model.signal_single(z, tau))))
            for j, tau in enumerate(delays)
        )

    fd_re, fd_im = _fd_grad(loss_only, ew)
    assert np.allclose(2 * np.real(ew_bar), fd_re, atol=1e-4, rtol=1e-4)
    assert np.allclose(2 * np.imag(ew_bar), fd_im, atol=1e-4, rtol=1e-4)
