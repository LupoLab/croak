"""Tests for the known-truth retrieval errors (:mod:`croak.truth_metrics`)."""

import numpy as np
import pytest

from croak.grid import Grid
from croak.maths import wlfreq
from croak.processing import TruthPulse, oversample
from croak.pulses import gaussian_pulse
from croak.result import RetrievalResult
from croak.truth_metrics import (
    eps_complex_field,
    eps_spectral_intensity,
    eps_temporal_intensity,
    truth_errors,
)

OMEGA0 = float(wlfreq(800e-9))


@pytest.fixture
def grid():
    # Wide enough that a chirped 20 fs pulse's tails stay inside the window:
    # a shifted pulse would otherwise lose energy off the edge and score that
    # truncation as error, which is a property of the window, not the metric.
    return Grid(512, dt=1e-15)


@pytest.fixture
def spectrum(grid):
    return gaussian_pulse(grid, 20e-15, phases=[300e-30])


@pytest.fixture
def truth(grid, spectrum):
    return TruthPulse.from_spectrum(grid, spectrum, OMEGA0)


def _result(grid, spectrum) -> RetrievalResult:
    return RetrievalResult(
        spectrum=np.asarray(spectrum, dtype=complex),
        grid=grid,
        delays=np.linspace(-50e-15, 50e-15, 16),
        interaction="pg",
        algorithm="test",
        error=0.0,
        omega0=OMEGA0,
    )


def _temporal(grid, spectrum):
    """Oversampled ``(t, |E(t)|^2)`` for a spectrum, as truth_errors does."""
    t_over, e_over = oversample(grid.t, grid.ifft(spectrum), 8)
    return t_over, np.abs(e_over) ** 2


def test_truth_errors_are_zero_for_the_generating_field(grid, spectrum, truth):
    """A retrieval that returned the true field scores zero on all three."""
    errs = truth_errors(_result(grid, spectrum), truth)
    assert errs.eps_It < 1e-12
    # the LS (sin-theta) form cannot resolve a perfect match below
    # ~sqrt(eps) ~ 1.5e-8 (same caveat as eps_complex_field)
    assert errs.eps_Iw == pytest.approx(0.0, abs=1e-7)
    assert errs.eps_Ew == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize("scale", [1.0, 3.7])
@pytest.mark.parametrize("phase", [0.0, 1.1])
@pytest.mark.parametrize("delay_fs", [0.0, 0.0625, 7.0])
def test_truth_errors_ignore_the_unobservable_gauges(
    grid, spectrum, truth, scale, phase, delay_fs
):
    """Amplitude scale, absolute phase and delay must all cost nothing.

    These are exactly the three transformations a PNPS measurement cannot
    resolve, so a metric that charged for them would report a retrieval as wrong
    for a choice it was never able to make. The 0.0625 fs delay is half of the
    oversampled time step — the worst case for the sub-sample alignment.
    """
    gauged = (
        scale
        * spectrum
        * np.exp(1j * phase)
        * np.exp(1j * grid.omega * delay_fs * 1e-15)
    )
    errs = truth_errors(_result(grid, gauged), truth)
    assert errs.eps_It < 1e-4
    # the LS (sin-theta) form cannot resolve a perfect match below
    # ~sqrt(eps) ~ 1.5e-8 (same caveat as eps_complex_field)
    assert errs.eps_Iw == pytest.approx(0.0, abs=1e-7)
    # eps_Ew cannot resolve below ~sqrt(machine eps): see its Notes section.
    assert errs.eps_Ew == pytest.approx(0.0, abs=1e-7)


def test_sub_sample_alignment_beats_nearest_bin_on_a_half_step_delay(
    grid, spectrum, truth
):
    """The refined alignment removes a floor the nearest-sample one leaves.

    A half-step delay is invisible physics but costs ``~dt/2sigma`` if the
    alignment can only land on a sample. The refinement must put it far below
    anything a real retrieval would produce.
    """
    step = float(np.mean(np.diff(truth.t)))
    shifted = spectrum * np.exp(1j * grid.omega * 0.5 * step)
    t_over, it = _temporal(grid, shifted)

    # The nearest-sample alignment this function deliberately does not use.
    t = t_over - t_over[int(np.argmax(it))]
    t0 = truth.t - truth.t[int(np.argmax(truth.It))]
    ir = np.interp(t0, t, it, left=0.0, right=0.0)
    i0 = truth.It / np.trapezoid(truth.It, t0)
    ir = ir / np.trapezoid(ir, t0)
    lag = int(np.argmax(np.correlate(i0, ir, "full"))) - (ir.size - 1)
    nearest_bin = float(np.sqrt(np.sum((np.roll(ir, lag) - i0) ** 2) / np.sum(i0**2)))

    refined = eps_temporal_intensity(t_over, it, truth.t, truth.It)
    assert nearest_bin > 1e-3  # the floor being removed
    assert refined < 0.02 * nearest_bin


def test_eps_spectral_intensity_is_blind_to_phase(grid, spectrum):
    """A pure phase error costs nothing in the spectral-intensity metric.

    That is the point of reporting it beside the complex error: together they
    say *which half* of the field is wrong.
    """
    rng = np.random.default_rng(0)
    rephased = np.abs(spectrum) * np.exp(1j * 2 * np.pi * rng.random(grid.n))
    assert eps_spectral_intensity(rephased, spectrum) == pytest.approx(0.0, abs=1e-12)
    assert eps_complex_field(rephased, spectrum) > 0.5


def test_eps_complex_field_grows_with_a_phase_error(grid, spectrum):
    """The complex error increases monotonically with added spectral phase."""
    scaled = grid.omega / 1e14
    values = [
        eps_complex_field(
            np.abs(spectrum) * np.exp(1j * (np.angle(spectrum) + a * scaled**2)),
            spectrum,
        )
        for a in (0.0, 0.25, 0.5, 1.0)
    ]
    assert values[0] == pytest.approx(0.0, abs=1e-12)
    assert all(b > a for a, b in zip(values, values[1:], strict=False))
    assert all(0.0 <= v <= 1.0 for v in values)


def test_eps_temporal_intensity_rejects_mismatched_axes(truth):
    with pytest.raises(ValueError, match="match the length"):
        eps_temporal_intensity([0.0, 1.0, 2.0], [1.0, 2.0], truth.t, truth.It)


def test_spectral_errors_reject_a_different_grid(grid, spectrum):
    with pytest.raises(ValueError, match="same grid"):
        eps_spectral_intensity(spectrum, spectrum[:-1])
    with pytest.raises(ValueError, match="same grid"):
        eps_complex_field(spectrum, spectrum[:-1])


def test_intensity_only_truth_reports_the_temporal_error_alone(grid, spectrum, truth):
    """A legacy truth with no complex field still scores what it can.

    ``eps_It`` needs only the intensity envelope, so it is always available; the
    spectral pair is reported absent rather than guessed at from intensities.
    """
    intensity_only = TruthPulse.from_intensity(truth.t, truth.It)
    errs = truth_errors(_result(grid, spectrum), intensity_only)
    assert errs.eps_It < 1e-12
    assert errs.eps_Iw is None and errs.eps_Ew is None


def test_truth_errors_separate_a_wrong_pulse_from_a_gauge(grid, spectrum, truth):
    """A genuinely different pulse scores non-zero on every metric."""
    wrong = gaussian_pulse(grid, 30e-15, phases=[-500e-30])
    errs = truth_errors(_result(grid, wrong), truth)
    assert errs.eps_It > 0.05
    assert errs.eps_Iw is not None and errs.eps_Iw > 0.05
    assert errs.eps_Ew is not None and errs.eps_Ew > 0.05


def test_frame_p_undoes_the_retrieval_frame(grid, spectrum, truth):
    """A frame-corrected retrieval scores zero once its exponent is passed.

    The focal-mixture protocol reframes the measured spectrum by
    ``((omega+omega0)/omega0)**(2 p)`` in intensity before retrieval
    (``apply_spectrum_frame``), so a perfect phase-only retrieval carries the
    truth's amplitude times ``wfac**p``. Without ``frame_p`` that footprint is
    scored as error; with it, all three errors return to the unframed values.
    """
    p = 0.5
    om = np.asarray(grid.omega, float)
    wfac = np.maximum((om + OMEGA0) / OMEGA0, 0.0)
    framed = _result(grid, np.asarray(spectrum, complex) * wfac**p)

    naive = truth_errors(framed, truth)
    assert naive.eps_Iw is not None and naive.eps_Iw > 0.01

    fixed = truth_errors(framed, truth, frame_p=p)
    assert fixed.eps_It < 1e-9
    assert fixed.eps_Iw is not None and fixed.eps_Iw < 1e-9
    assert fixed.eps_Ew is not None and fixed.eps_Ew < 1e-6
