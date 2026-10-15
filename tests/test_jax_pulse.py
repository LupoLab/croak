"""Tests for the JAX pulse parameterisations in :mod:`croak._jax_pulse`.

The retrieval-level behaviour is covered by the solver tests; these pin the
helpers' edge cases and argument checks directly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from croak import metrics_jax as mj
from croak._jax_pulse import (
    ExtraParamSpec,
    _second_diff_penalty_grad,
    _spectrum_support,
    augment,
    build_parameterisation,
    ls_parameterisation,
    make_param_grad_fns,
    make_spline_phase_basis,
)
from croak.grid import Grid
from croak.pulses import gaussian_pulse

GRID = Grid(32, dt=1e-15)


def test_second_differences_of_fewer_than_three_points_vanish():
    a = jnp.asarray([1.0, 2.0])
    np.testing.assert_array_equal(
        _second_diff_penalty_grad(a, normalize_by_max=True), np.zeros(2)
    )
    assert float(mj.phase_penalty(a)) == 0.0


def test_phase_only_grad_fns_match_autodiff():
    """Phase-only: no amplitude block, and the phase gradient is exact."""
    amp = np.abs(gaussian_pulse(GRID, 6e-15))
    phase = np.linspace(-1.0, 1.0, GRID.n) ** 2
    fns = make_param_grad_fns(amp, phase, phase_only=True)
    u = jnp.asarray(phase)
    np.testing.assert_array_equal(fns.amplitude_penalty_grad(u), np.zeros(GRID.n))
    target = jnp.asarray(amp)
    np.testing.assert_array_equal(
        fns.spectral_penalty_grad(u, target), np.zeros(GRID.n)
    )
    # the phase penalty has no max-normalisation, so its gradient is exact; the
    # interior entries cancel to ~1e-17, so compare on the gradient's own scale
    expected = jax.grad(mj.phase_penalty)(u)
    np.testing.assert_allclose(
        fns.phase_penalty_grad(u),
        expected,
        atol=1e-12 * float(jnp.max(jnp.abs(expected))),
    )


@pytest.mark.parametrize(
    ("amplitude", "threshold"),
    [
        (np.zeros(8), 1e-3),  # no light at all
        (np.array([], dtype=float), 1e-3),  # nothing to look at
        (np.ones(8), 1.0),  # nothing strictly above the peak
    ],
)
def test_spectrum_support_falls_back_to_the_whole_axis(amplitude, threshold):
    assert _spectrum_support(amplitude, threshold) == (0, amplitude.size)


def test_spline_basis_rejects_a_support_narrower_than_the_spline():
    amp = np.zeros(GRID.n)
    amp[10:12] = 1.0  # two lit bins cannot carry a cubic (needs 4)
    with pytest.raises(ValueError, match="support is too narrow"):
        make_spline_phase_basis(amp, np.zeros(GRID.n), GRID.omega, n_nodes=6)


@pytest.mark.parametrize("builder", [build_parameterisation, ls_parameterisation])
def test_parameterisations_reject_unknown_bases(builder):
    ew = gaussian_pulse(GRID, 6e-15)
    with pytest.raises(ValueError, match="unknown phase_basis 'fourier'"):
        builder(ew, GRID.omega, phase_only=False, phase_basis="fourier")


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        (ExtraParamSpec(fit_thickness=True), "requires thickness0 > 0"),
        (ExtraParamSpec(fit_tau0=True), "requires scale_tau > 0"),
        (
            ExtraParamSpec(fit_smearing=True, scale_smear=0.0),
            "requires scale_smear > 0",
        ),
        (
            ExtraParamSpec(fit_smearing=True, smear_scale0=0.0),
            "requires smear_scale0 > 0",
        ),
    ],
)
def test_augment_rejects_non_positive_scales(spec, message):
    with pytest.raises(ValueError, match=message):
        augment(np.zeros(4), spec)
