"""Tests for the :class:`~croak.result.RetrievalResult` derived quantities."""

from __future__ import annotations

import numpy as np

from croak.grid import Grid
from croak.maths import wlfreq
from croak.pulses import gaussian_pulse
from croak.result import RetrievalResult


def _result(**kw) -> RetrievalResult:
    g = Grid(64, dt=1e-15)
    base = dict(
        spectrum=gaussian_pulse(g, 8e-15, phases=(30e-30,)),
        grid=g,
        delays=np.zeros(3),
        interaction="pg",
        algorithm="copra",
        error=1.234e-3,
        errors=[0.1, 0.01, 1.234e-3],
        omega0=wlfreq(800e-9),
    )
    base.update(kw)
    return RetrievalResult(**base)


def test_phases_are_unwrapped_arguments():
    r = _result()
    np.testing.assert_allclose(
        np.exp(1j * r.phase_omega), np.exp(1j * np.angle(r.spectrum))
    )
    np.testing.assert_allclose(np.exp(1j * r.phase_t), np.exp(1j * np.angle(r.field)))
    # unwrapped: no jump of more than pi between neighbours
    assert np.all(np.abs(np.diff(r.phase_omega)) <= np.pi)
    assert np.all(np.abs(np.diff(r.phase_t)) <= np.pi)


def test_repr_is_a_concise_summary():
    assert repr(_result()) == (
        "RetrievalResult(algorithm='copra', interaction='pg', error=1.234e-03, "
        "iterations=3)"
    )
