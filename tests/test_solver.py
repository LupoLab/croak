"""Tests for the shared helpers of :mod:`croak.solver` and :mod:`croak.retrieve`."""

from __future__ import annotations

import warnings

import nlopt
import numpy as np
import pytest

from croak.grid import Grid
from croak.lbfgs import LBFGS
from croak.pulses import ComplexPulse, gaussian_pulse
from croak.retrieve import ALGORITHMS, algorithm_params
from croak.solver import (
    NLOPT_ROUNDOFF_R,
    adapt_callback,
    build_weights,
    guard_nonfinite,
    reg_defaults,
    reg_family,
    resolve_guess,
    spectral_target_amplitude,
    split_smear_value,
    warn_nlopt_salvage,
)

GRID = Grid(32, dt=1e-15)


def test_resolve_guess_accepts_a_pulse_and_normalises_it():
    ew = 5.0 * gaussian_pulse(GRID, 6e-15)
    guess = resolve_guess(GRID, ComplexPulse(GRID, ew))
    np.testing.assert_allclose(guess, ew / np.abs(ew).max())


def test_resolve_guess_rejects_a_wrong_length_spectrum():
    with pytest.raises(ValueError, match=r"guess spectrum must have shape \(32,\)"):
        resolve_guess(GRID, np.ones(31))


def test_resolve_guess_leaves_an_all_zero_spectrum_alone():
    np.testing.assert_array_equal(resolve_guess(GRID, np.zeros(32)), np.zeros(32))


def test_build_weights_rejects_a_wrong_length():
    np.testing.assert_array_equal(build_weights(3, None), np.ones(3))
    with pytest.raises(ValueError, match=r"weights must have shape \(3,\)"):
        build_weights(3, np.ones(4))


def test_adapt_callback_passes_through_an_uninspectable_callable(monkeypatch):
    """A callable without a signature is assumed to follow the documented form."""

    def opaque(*args, **kwargs):
        pass

    def no_signature(obj):
        raise ValueError("no signature found")

    monkeypatch.setattr("croak.solver.inspect.signature", no_signature)
    assert adapt_callback(opaque) is opaque


def test_guard_nonfinite_with_no_gradient_buffer():
    """Gradient-free solvers pass no (or an empty) gradient buffer."""
    state: dict = {}
    with pytest.warns(RuntimeWarning, match="non-finite"):
        assert guard_nonfinite(float("inf"), None, None, state) is not None
    assert guard_nonfinite(float("nan"), None, np.empty(0), state) is not None


@pytest.mark.parametrize(
    ("solver", "family", "defaults"),
    [
        ("lbfgs-ad", "gradient", (1e-2, 3e-2)),
        ("LM", "lm", (1e-5, 3e-5)),
        ("lm-optx", "lm", (1e-5, 3e-5)),
    ],
)
def test_reg_family_and_defaults(solver, family, defaults):
    assert reg_family(solver) == family
    assert reg_defaults(solver) == defaults


def test_warn_nlopt_salvage_warns_while_still_improving():
    with pytest.warns(RuntimeWarning, match=r"NLopt stopped early \(RoundoffLimited\)"):
        warn_nlopt_salvage(nlopt.RoundoffLimited(), 3.4e-2)


@pytest.mark.parametrize("best_err", [0.0, 3e-17, NLOPT_ROUNDOFF_R])
def test_warn_nlopt_salvage_is_silent_at_roundoff(best_err):
    """At roundoff the line-search failure is convergence, not an early stop."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_nlopt_salvage(nlopt.RoundoffLimited(), best_err)


def test_algorithm_params_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown algorithm 'gerchberg'"):
        algorithm_params("gerchberg")


def test_algorithm_params_skips_classes_without_their_own_init(monkeypatch):
    """A subclass that only inherits ``__init__`` advertises its parent's options."""

    class TunedLBFGS(LBFGS):
        """A solver subclass that adds behaviour but no constructor."""

    monkeypatch.setitem(ALGORITHMS, "tuned-lbfgs", TunedLBFGS)
    assert algorithm_params("tuned-lbfgs") == algorithm_params("lbfgs")


def test_algorithm_params_of_a_solver_that_only_forwards(monkeypatch):
    """A class whose whole MRO forwards ``**kwargs`` names no options of its own."""

    class Forwarder:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setitem(ALGORITHMS, "forwarder", Forwarder)
    assert algorithm_params("forwarder") == set()


@pytest.mark.parametrize(
    ("intensity", "expected"),
    [
        (None, None),
        ([4.0, 1.0, -1.0], [1.0, 0.5, 0.0]),  # negatives clipped before the root
        ([0.0, 0.0], [0.0, 0.0]),  # nothing to normalise by
        ([], []),
    ],
)
def test_spectral_target_amplitude(intensity, expected):
    out = spectral_target_amplitude(intensity)
    if expected is None:
        assert out is None
    else:
        np.testing.assert_allclose(out, expected)


def test_split_smear_value_forms():
    assert split_smear_value(1.5) == (1.5, None)
    assert split_smear_value((1.2, 0.8)) == (1.2, 0.8)
