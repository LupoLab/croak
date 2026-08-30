"""Tests for :mod:`croak.processing`."""

import numpy as np
import pytest

from croak import processing
from croak.forward import maketrace
from croak.grid import Grid
from croak.maths import wlfreq
from croak.processing import TruthPulse
from croak.pulses import gaussian_pulse
from croak.retrieve import retrieve


def test_truth_pulse_from_spectrum_intensity_fwhm():
    """from_spectrum reports the intensity FWHM and a peak-centred, unit envelope."""
    g = Grid(256, dt=0.4e-15)
    omega0 = float(wlfreq(800e-9))
    f = 12e-15
    ew = gaussian_pulse(g, f)
    truth = TruthPulse.from_spectrum(g, ew, omega0)
    assert truth.fwhm / 1e-15 == pytest.approx(f / 1e-15, rel=2e-2)
    assert truth.It.max() == pytest.approx(1.0)
    assert truth.t[int(np.argmax(truth.It))] == pytest.approx(0.0, abs=1e-18)
    assert truth.lam is not None and np.all(np.diff(truth.lam) > 0)  # ascending λ
    assert truth.phi_t is not None
    assert truth.phi_w is not None
    np.testing.assert_allclose(truth.spectrum_on_grid(g, omega0), ew)


def test_truth_spectrum_on_grid_requires_a_complex_field():
    """An intensity-only truth fails clearly when requested as a solver seed."""
    g = Grid(32, dt=1e-15)
    truth = TruthPulse.from_intensity(g.t, np.exp(-((g.t / 5e-15) ** 2)))
    with pytest.raises(ValueError, match="no complex spectrum"):
        truth.spectrum_on_grid(g, wlfreq(800e-9))


def test_truth_pulse_from_intensity_measures_fwhm():
    """from_intensity normalises, peak-centres, and measures the FWHM when absent."""
    t = np.linspace(-60e-15, 60e-15, 400)
    It = 3.0 * np.exp(-0.5 * (t / (10e-15 / 2.3548)) ** 2)  # 10 fs FWHM, off-unit scale
    truth = TruthPulse.from_intensity(t + 5e-15, It)
    assert truth.It.max() == pytest.approx(1.0)
    assert truth.t[int(np.argmax(truth.It))] == pytest.approx(0.0, abs=1e-18)
    assert truth.fwhm / 1e-15 == pytest.approx(10.0, rel=1e-2)


@pytest.fixture
def chirped_result():
    g = Grid(128, dt=0.5e-15)
    omega0 = wlfreq(800e-9)
    ew = gaussian_pulse(g, 6e-15, phases=[20e-30])  # GDD = 20 fs^2 (fits window)
    delays = np.linspace(-50e-15, 50e-15, 80)
    trace = maketrace(g.omega, delays, ew, "pg")
    res = retrieve(
        trace,
        g.omega,
        delays,
        "pg",
        algorithm="copra",
        maxiters=80,
        guess=ew,
        omega0=omega0,
    )
    return res, trace, g, ew


def test_marginals_normalised():
    trace = np.random.default_rng(0).random((16, 20))
    wmarg, tmarg = processing.marginals(trace)
    assert wmarg.shape == (16,) and tmarg.shape == (20,)
    assert wmarg.max() == pytest.approx(1.0)
    assert wmarg.min() >= 0.0


def test_residuals_zero_for_identical():
    trace = np.random.default_rng(1).random((8, 8))
    assert np.allclose(processing.residuals(trace, trace), 0.0, atol=1e-12)


def test_oversample_increases_resolution():
    g = Grid(64, dt=1e-15)
    field = np.exp(-0.5 * (g.t / 5e-15) ** 2).astype(complex)
    to, eo = processing.oversample(g.t, field, 8)
    assert to.size == 64 * 8
    # peak intensity preserved approximately
    assert np.abs(eo).max() == pytest.approx(np.abs(field).max(), rel=1e-2)


def test_process_result_recovers_gdd(chirped_result):
    res, trace, g, ew = chirped_result
    pr = processing.process_result(res, measured=trace)
    # chirped pulse: retrieved GDD close to the 20 fs^2 used (sign may flip with
    # the time-reversal ambiguity, so compare magnitude)
    assert abs(pr.gdd_fs2) == pytest.approx(20.0, rel=0.25)
    # transform-limited pulse is shorter than the chirped retrieved pulse
    assert pr.fwhm_tl < pr.fwhm_retr
    assert pr.residual is not None
    assert pr.trace_meas is not None


def test_process_result_wavelength_axis(chirped_result):
    res, trace, g, ew = chirped_result
    pr = processing.process_result(res)
    lam_peak = processing.peak_wavelength(pr.wavelength, pr.Ilam)
    assert 700e-9 < lam_peak < 950e-9


def test_spectrogram_shape(chirped_result):
    res, trace, g, ew = chirped_result
    tc, lam, S = processing.spectrogram(res, n_t=64, lam_min=700e-9, lam_max=950e-9)
    assert tc.size == 64
    assert S.shape == (lam.size, 64)
    assert S.max() == pytest.approx(1.0)


def test_process_result_no_energy_keeps_normalised(chirped_result):
    res, trace, g, ew = chirped_result
    pr = processing.process_result(res, measured=trace)
    # default: no absolute power, plots stay normalised
    assert pr.energy is None
    assert pr.peak_power is None
    assert pr.peak_power_tl is None
    assert pr.It_retr.max() == pytest.approx(1.0)


def test_process_result_peak_power_integrates_to_energy(chirped_result):
    res, trace, g, ew = chirped_result
    energy = 100e-6  # 100 µJ
    pr = processing.process_result(res, measured=trace, energy=energy)
    assert pr.energy == energy
    assert pr.peak_power is not None and pr.peak_power_tl is not None
    # the displayed absolute power (It_retr * peak_power) integrates back to E
    recovered = np.trapezoid(pr.It_retr * pr.peak_power, pr.t_over)
    assert recovered == pytest.approx(energy, rel=1e-6)
    # peak power = energy / effective duration
    assert pr.peak_power == pytest.approx(
        energy / np.trapezoid(pr.It_retr, pr.t_over), rel=1e-9
    )


def test_process_result_tl_peak_power_is_higher(chirped_result):
    # the transform-limited pulse carries the same energy in a shorter time, so it
    # has the higher (maximum-achievable) peak power for a chirped input
    res, trace, g, ew = chirped_result
    pr = processing.process_result(res, measured=trace, energy=50e-6)
    assert pr.peak_power_tl > pr.peak_power


def test_process_result_zero_energy_is_unspecified(chirped_result):
    res, trace, g, ew = chirped_result
    pr = processing.process_result(res, measured=trace, energy=0.0)
    assert pr.energy is None
    assert pr.peak_power is None


def test_post_filter_windows_spectrum_and_time(chirped_result):
    res, trace, g, ew = chirped_result
    # no windows -> spectrum unchanged
    same = processing.post_filter(res)
    assert np.allclose(same.spectrum, res.spectrum)
    # spectral window suppresses out-of-band energy
    lam = res.wavelength
    band = (lam > 770e-9) & (lam < 830e-9)
    filt = processing.post_filter(res, lam_lims=(770e-9, 830e-9))
    out_of_band_before = np.abs(res.spectrum[~band]).sum()
    out_of_band_after = np.abs(filt.spectrum[~band]).sum()
    assert out_of_band_after < out_of_band_before
    assert filt.spectrum.shape == res.spectrum.shape
    # temporal window returns a valid spectrum
    filt_t = processing.post_filter(res, tau_lims=(-20e-15, 20e-15))
    assert filt_t.spectrum.shape == res.spectrum.shape
    assert np.all(np.isfinite(filt_t.spectrum))


def test_spectral_helpers_recover_a_known_gaussian():
    """Peak wavelength and spectral FWHM on an analytic spectrum, either direction.

    The retrieval grid's wavelength axis descends (it is 2πc/(ω+ω₀) over an
    ascending ω), so both orientations must give the same answer.
    """
    lam = np.linspace(700e-9, 900e-9, 2001)
    intensity = np.exp(-(((lam - 800e-9) / 20e-9) ** 2))
    # a Gaussian exp(-(x/w)^2) has FWHM = 2 w sqrt(ln 2)
    expected = 2 * 20e-9 * np.sqrt(np.log(2))

    for x, y in ((lam, intensity), (lam[::-1], intensity[::-1])):
        assert processing.peak_wavelength(x, y) == pytest.approx(800e-9, abs=1e-11)
        assert processing.spectral_fwhm(x, y) == pytest.approx(expected, rel=1e-3)


def test_spectral_helpers_ignore_unphysical_wavelengths():
    """Bins from the axis's zero crossing are dropped, not folded into the stats.

    ``wavelength = 2πc/(ω+ω₀)`` blows up where ``ω+ω₀`` crosses zero and is
    negative over the whole left half when the result carries no carrier, so the
    helpers must ignore ``inf``/``nan``/non-positive entries.
    """
    lam = np.linspace(700e-9, 900e-9, 2001)
    intensity = np.exp(-(((lam - 800e-9) / 20e-9) ** 2))
    junk_lam = np.concatenate([[np.inf, -1e-9, np.nan, 0.0], lam])
    # the junk bins carry the *largest* intensity, so a missing mask would show
    junk_int = np.concatenate([[9.0, 9.0, 9.0, 9.0], intensity])

    assert processing.peak_wavelength(junk_lam, junk_int) == pytest.approx(
        800e-9, abs=1e-11
    )
    assert processing.spectral_fwhm(junk_lam, junk_int) == pytest.approx(
        processing.spectral_fwhm(lam, intensity), rel=1e-9
    )


def test_spectral_helpers_nan_when_undefined():
    """No usable bin, or a level never crossed, gives nan rather than raising."""
    assert np.isnan(processing.peak_wavelength([np.inf, -1.0], [1.0, 2.0]))
    assert np.isnan(processing.spectral_fwhm([np.inf, -1.0], [1.0, 2.0]))
    # a monotone ramp never falls back to half maximum on the left
    lam = np.linspace(700e-9, 900e-9, 64)
    assert np.isnan(processing.spectral_fwhm(lam, np.linspace(1.0, 2.0, 64)))


def test_truth_pulse_peak_power():
    """TruthPulse.peak_power is energy / effective duration, guarded for None."""
    t = np.linspace(-40e-15, 40e-15, 4001)
    It = np.exp(-((t / 5e-15) ** 2))
    truth = processing.TruthPulse.from_intensity(t, It)

    tau_eff = float(np.trapezoid(truth.It, truth.t))
    assert truth.peak_power(1e-6) == pytest.approx(1e-6 / tau_eff, rel=1e-12)
    assert truth.peak_power(None) is None
    assert truth.peak_power(0.0) is None
    assert truth.peak_power(-1.0) is None


# ---------------------------------------------------------------------------
# Direction-of-time ambiguity
# ---------------------------------------------------------------------------
def _asymmetric_setup():
    """A visibly time-asymmetric pulse and its known truth."""
    grid = Grid(128, dt=0.4e-15)
    omega0 = float(wlfreq(800e-9))
    # third-order phase makes the temporal profile clearly one-sided, so a
    # reversal is detectable rather than a coin toss
    ew = gaussian_pulse(grid, 6e-15, phases=[30e-30, 500e-45])
    delays = np.linspace(-40e-15, 40e-15, 90)
    truth = TruthPulse.from_spectrum(grid, ew, omega0)
    return grid, omega0, ew, delays, truth


def test_conjugating_the_spectrum_leaves_an_shg_trace_invariant():
    """The premise of the ambiguity: SHG cannot see the flip, SD and PG can."""
    grid, _, ew, delays, _ = _asymmetric_setup()
    from croak.metrics import trace_error

    for interaction, ambiguous in (("shg", True), ("sd", False), ("pg", False)):
        direct = maketrace(grid.omega, delays, ew, interaction)
        flipped = maketrace(grid.omega, delays, np.conj(ew), interaction)
        same = trace_error(direct, flipped) < 1e-12
        assert same is ambiguous, f"{interaction}: invariance was {same}"


def test_resolve_time_direction_flips_a_reversed_shg_retrieval():
    """A time-reversed SHG solution is turned back to match the truth."""
    grid, omega0, ew, delays, truth = _asymmetric_setup()
    trace = maketrace(grid.omega, delays, ew, "shg")
    # Start from the exactly-reversed pulse: a legitimate SHG solution, and the
    # branch the solver is free to return.
    reversed_result = retrieve(
        trace,
        grid.omega,
        delays,
        "shg",
        algorithm="copra",
        maxiters=5,
        omega0=omega0,
        progress=False,
        guess=np.conj(ew),
    )
    resolved, flipped = processing.resolve_time_direction(reversed_result, truth)
    assert flipped
    # the resolved profile now tracks the truth far better than the input did
    before = np.interp(reversed_result.t, truth.t, truth.It, left=0.0, right=0.0)
    corr = processing._peak_normalised_correlation
    assert corr(resolved.intensity_t, before) > corr(
        reversed_result.intensity_t, before
    )
    # conjugation is exact, so the trace and error it was fitted with still hold
    assert resolved.error == pytest.approx(reversed_result.error, rel=1e-12)


def test_resolve_time_direction_leaves_unambiguous_geometries_alone():
    """PG and SD fix the direction of time, so the result must not be touched."""
    grid, omega0, ew, delays, truth = _asymmetric_setup()
    for interaction in ("pg", "sd"):
        trace = maketrace(grid.omega, delays, ew, interaction)
        result = retrieve(
            trace,
            grid.omega,
            delays,
            interaction,
            algorithm="copra",
            maxiters=5,
            omega0=omega0,
            progress=False,
            guess=np.conj(ew),
        )
        resolved, flipped = processing.resolve_time_direction(result, truth)
        assert not flipped
        assert resolved is result


def test_resolve_time_direction_keeps_an_already_correct_result():
    """No flip when the retrieval already matches, and none on a tie."""
    grid, omega0, ew, delays, truth = _asymmetric_setup()
    trace = maketrace(grid.omega, delays, ew, "shg")
    result = retrieve(
        trace,
        grid.omega,
        delays,
        "shg",
        algorithm="copra",
        maxiters=5,
        omega0=omega0,
        progress=False,
        guess=ew,
    )
    resolved, flipped = processing.resolve_time_direction(result, truth)
    assert not flipped
    assert resolved is result

    # A time-symmetric pulse scores equally either way; the tolerance must make
    # that stable rather than letting rounding decide.
    sym_ew = gaussian_pulse(grid, 6e-15)
    sym_truth = TruthPulse.from_spectrum(grid, sym_ew, omega0)
    sym_trace = maketrace(grid.omega, delays, sym_ew, "shg")
    sym = retrieve(
        sym_trace,
        grid.omega,
        delays,
        "shg",
        algorithm="copra",
        maxiters=5,
        omega0=omega0,
        progress=False,
        guess=sym_ew,
    )
    _, sym_flipped = processing.resolve_time_direction(sym, sym_truth)
    assert not sym_flipped
