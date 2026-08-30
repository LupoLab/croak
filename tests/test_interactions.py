"""Tests for :mod:`croak.interactions`: signal forms and adjoint correctness.

The adjoints are validated against central finite differences. For a real loss
``L`` and complex variable ``v = x + i y`` the Wirtinger cotangent
``v_bar = dL/dv*`` relates to real partials by
``dL/dx = 2 Re(v_bar)`` and ``dL/dy = 2 Im(v_bar)``.
"""

import numpy as np
import pytest

from croak.interactions import PG, SD, SHG, get_interaction


def test_signal_forms():
    e = np.array([1 + 2j, -1j])
    g = np.array([0.5 + 0j, 2 + 1j])
    assert np.allclose(SHG().signal(e, g), e * g)
    assert np.allclose(SD().signal(e, g), e**2 * np.conj(g))
    assert np.allclose(PG().signal(e, g), e * np.abs(g) ** 2)


def test_get_interaction():
    assert isinstance(get_interaction("SHG"), SHG)
    assert get_interaction(PG()).name == "pg"
    with pytest.raises(ValueError):
        get_interaction("xpw")


def _real_loss(interaction, e, g, w):
    # L = Re(sum(conj(w) * signal)); then dL/ds* = 0.5 * w.
    return float(np.real(np.sum(np.conj(w) * interaction.signal(e, g))))


def _fd_grad(func, z, h=1e-7):
    """Central-difference real gradient of a real `func(z)` w.r.t. Re/Im of z."""
    gr = np.zeros_like(z, dtype=float)
    gi = np.zeros_like(z, dtype=float)
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


@pytest.mark.parametrize("interaction", [SHG(), SD(), PG()])
def test_signal_adjoint_matches_finite_differences(interaction):
    rng = np.random.default_rng(42)
    n = 6
    e = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    g = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    w = rng.standard_normal(n) + 1j * rng.standard_normal(n)

    s_bar = 0.5 * w
    e_bar, g_bar = interaction.signal_adjoint(e, g, s_bar)

    # adjoint prediction of the real partials
    pred_e_re, pred_e_im = 2 * np.real(e_bar), 2 * np.imag(e_bar)
    pred_g_re, pred_g_im = 2 * np.real(g_bar), 2 * np.imag(g_bar)

    fd_e_re, fd_e_im = _fd_grad(lambda z: _real_loss(interaction, z, g, w), e)
    fd_g_re, fd_g_im = _fd_grad(lambda z: _real_loss(interaction, e, z, w), g)

    assert np.allclose(pred_e_re, fd_e_re, atol=1e-4, rtol=1e-4)
    assert np.allclose(pred_e_im, fd_e_im, atol=1e-4, rtol=1e-4)
    assert np.allclose(pred_g_re, fd_g_re, atol=1e-4, rtol=1e-4)
    assert np.allclose(pred_g_im, fd_g_im, atol=1e-4, rtol=1e-4)
