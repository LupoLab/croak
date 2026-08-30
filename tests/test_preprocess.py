"""Tests for :mod:`croak.preprocess`."""

import numpy as np
import pytest
from conftest import make_synthetic_experiment

from croak import preprocess
from croak.constants import C
from croak.maths import planck_taper, wlfreq
from croak.retrieve import retrieve


def test_planck_taper_shape():
    x = np.linspace(0, 10, 1001)
    w = planck_taper(x, 2, 3, 7, 8)
    assert np.all(w >= 0) and np.all(w <= 1)
    assert w[x < 2].max() == pytest.approx(0.0)
    assert w[x > 8].max() == pytest.approx(0.0)
    assert np.allclose(w[(x > 3.5) & (x < 6.5)], 1.0)


def test_filter_spectrum_windows():
    lam = np.linspace(600e-9, 1000e-9, 400)
    Ilam = np.ones_like(lam)
    out = preprocess.filter_spectrum(lam, Ilam, 700e-9, 900e-9)
    assert out.max() == pytest.approx(1.0)
    assert out[lam < 650e-9].max() < 1e-3
    assert out[lam > 950e-9].max() < 1e-3


def test_filter_frog_trace_runs(experiment):
    exp = experiment
    lam = exp.lam_nm * 1e-9
    tau = exp.delay_fs * 1e-15
    tau2, filt = preprocess.filter_frog_trace(
        exp.trace, lam, tau, filter_fringes=False, filter_dc=True, dtau_min=0.3e-15
    )
    assert filt.shape == exp.trace.shape
    assert filt.max() == pytest.approx(1.0)
    # centring puts the delay-marginal peak at zero (robust to hot pixels)
    peak_delay = tau2[int(np.argmax(filt.sum(axis=0)))]
    assert peak_delay == pytest.approx(0.0)


def _tone_amplitude(t, s, f):
    """Crude single-frequency amplitude estimate (works for non-uniform t)."""
    return 2.0 / len(t) * np.abs(np.sum(s * np.exp(-2j * np.pi * f * t)))


def test_delay_filter_dc_matches_fft_uniform():
    # On a uniform axis the NUFFT DC removal must equal the plain rfft version.
    n = 128
    tau = np.linspace(-60e-15, 60e-15, n)
    lam = np.linspace(700e-9, 900e-9, 6)
    rng = np.random.default_rng(0)
    sig = rng.standard_normal((lam.size, n))
    out = preprocess._delay_filter(
        sig, tau, lam, filter_dc=True, filter_fringes=False, dtau_min=None
    )
    spec = np.fft.rfft(sig, axis=1)
    spec[:, 0] = 0.0
    ref = np.fft.irfft(spec, n=n, axis=1)
    assert np.allclose(out, ref, atol=1e-6 * np.abs(sig).max())


def test_delay_filter_lowpass_attenuates_uniform_and_nonuniform():
    n = 256
    base = np.linspace(-128e-15, 128e-15, n)
    dt = base[1] - base[0]
    lam = np.array([800e-9])
    f_lo, f_hi = 0.04e15, 0.30e15  # both below Nyquist 1/(2 dt)
    cut = 0.15e15
    dtau_min = 1.0 / cut  # low-pass keeps |f| <= cut

    rng = np.random.default_rng(1)
    for jitter in (0.0, 0.3):  # uniform, then non-uniform delays
        tau = np.sort(base + jitter * dt * rng.standard_normal(n))
        sig = (np.cos(2 * np.pi * f_lo * tau) + np.cos(2 * np.pi * f_hi * tau))[None, :]
        out = preprocess._delay_filter(
            sig, tau, lam, filter_dc=False, filter_fringes=False, dtau_min=dtau_min
        )[0]
        # low tone preserved, high tone strongly suppressed
        assert _tone_amplitude(tau, out, f_lo) > 0.8
        assert _tone_amplitude(tau, out, f_hi) < 0.1


# ---------------------------------------------------------------------------
# Takeda-FTSI de-fringing (defringe_carrier)
# ---------------------------------------------------------------------------
def test_defringe_carrier_recovers_slow_envelope():
    # A row I = a + 2 Re(c e^{iωτ}) with slow a, c and the carrier f_c = c/λ.
    # Keeping the DC band must recover the slow envelope a and drop the fringe.
    n = 256
    dt = 0.4e-15  # Nyquist 1.25e15 Hz >> f_c(800nm)=3.75e14 Hz, no aliasing
    tau = (np.arange(n) - n // 2) * dt
    lam = np.array([800e-9])
    fc = C / lam[0]
    a = np.exp(-0.5 * (tau / 8e-15) ** 2)  # slow envelope, incl. DC
    row = a * (1.0 + np.cos(2 * np.pi * fc * tau))  # c = a/2 -> cross term = a cos
    out = preprocess.defringe_carrier(row[None, :], lam, tau, carrier_fraction=0.5)[0]
    core = slice(n // 4, 3 * n // 4)  # interior, away from periodic edges
    assert np.allclose(out[core], a[core], atol=2e-2 * a.max())
    # the fringe tone at f_c is essentially gone
    assert _tone_amplitude(tau, out, fc) < 0.05 * _tone_amplitude(tau, row, fc)


@pytest.mark.parametrize("jitter", [0.0, 0.3])
def test_defringe_carrier_nonuniform_matches_uniform(jitter):
    n = 256
    dt = 0.4e-15
    base = (np.arange(n) - n // 2) * dt
    rng = np.random.default_rng(2)
    tau = np.sort(base + jitter * dt * rng.standard_normal(n))
    lam = np.array([800e-9])
    fc = C / lam[0]
    a = np.exp(-0.5 * (tau / 8e-15) ** 2)
    row = a * (1.0 + np.cos(2 * np.pi * fc * tau))
    out = preprocess.defringe_carrier(row[None, :], lam, tau, carrier_fraction=0.5)[0]
    core = slice(n // 4, 3 * n // 4)
    # the fringe is removed on both grids; the envelope is recovered exactly on a
    # uniform grid and faithfully (the NUFFT round-trip is only approximate on a
    # non-uniform grid, as for the existing delay filters) on a jittered one.
    assert _tone_amplitude(tau, out, fc) < 0.1 * _tone_amplitude(tau, row, fc)
    rel_l2 = np.linalg.norm(out[core] - a[core]) / np.linalg.norm(a[core])
    assert rel_l2 < (1e-3 if jitter == 0.0 else 0.1)


def test_defringe_carrier_preserves_signal_band():
    # The property DC removal destroys: the slow low-frequency content (including
    # the per-row mean) must survive de-fringing.
    n = 256
    dt = 0.4e-15
    tau = (np.arange(n) - n // 2) * dt
    lam = np.array([800e-9])
    fc = C / lam[0]
    f_lo = 0.02e15  # 2e13 Hz, well below the 0.5*f_c ~ 1.87e14 cutoff
    a = 1.0 + 0.5 * np.cos(2 * np.pi * f_lo * tau)  # DC + slow low tone
    row = a * (1.0 + np.cos(2 * np.pi * fc * tau))
    out = preprocess.defringe_carrier(row[None, :], lam, tau, carrier_fraction=0.5)[0]
    # low-frequency tone and DC mean preserved
    assert _tone_amplitude(tau, out, f_lo) > 0.9 * _tone_amplitude(tau, a, f_lo)
    assert np.mean(out) == pytest.approx(np.mean(a), rel=0.05)
    # DC removal, by contrast, would gut the mean
    dc = preprocess._delay_filter(
        row[None, :], tau, lam, filter_dc=True, filter_fringes=False, dtau_min=None
    )[0]
    assert abs(np.mean(dc)) < 0.1 * abs(np.mean(a))


def test_defringe_carrier_raises_on_alias():
    # Delay step too coarse: f_c(400nm) exceeds the delay Nyquist -> aliasing.
    n = 128
    dt = 2e-15  # Nyquist 2.5e14 Hz < f_c(400nm)=7.49e14 Hz
    tau = (np.arange(n) - n // 2) * dt
    lam = np.array([400e-9])
    a = np.exp(-0.5 * (tau / 12e-15) ** 2)
    row = a * (1.0 + np.cos(2 * np.pi * (C / lam[0]) * tau))
    with pytest.raises(ValueError, match="alias"):
        preprocess.defringe_carrier(row[None, :], lam, tau)


def test_defringe_carrier_clamp_on_alias():
    n = 128
    dt = 2e-15
    tau = (np.arange(n) - n // 2) * dt
    lam = np.array([400e-9])
    fc = C / lam[0]
    a = np.exp(-0.5 * (tau / 12e-15) ** 2)
    row = a * (1.0 + np.cos(2 * np.pi * fc * tau))
    out = preprocess.defringe_carrier(row[None, :], lam, tau, on_alias="clamp")[0]
    # clamp low-passes below the apparent (folded) carrier -> aliased tone drops
    f_s = 1.0 / dt
    f_app = abs(fc - round(fc / f_s) * f_s)
    assert _tone_amplitude(tau, out, f_app) < 0.5 * _tone_amplitude(tau, row, f_app)


@pytest.mark.parametrize("frac", [0.0, 1.5])
def test_defringe_carrier_rejects_bad_fraction(frac):
    tau = np.linspace(-50e-15, 50e-15, 64)
    lam = np.array([800e-9])
    row = np.ones((1, 64))
    with pytest.raises(ValueError, match="carrier_fraction"):
        preprocess.defringe_carrier(row, lam, tau, carrier_fraction=frac)


def test_filter_frog_trace_defringe_runs(experiment):
    # The synthetic fixture is sampled coarsely in delay (carrier aliases at the
    # signal wavelengths), so de-fringe runs with on_alias="clamp".
    exp = experiment
    lam = exp.lam_nm * 1e-9
    tau = exp.delay_fs * 1e-15
    tau2, filt = preprocess.filter_frog_trace(
        exp.trace, lam, tau, filter_fringes=False, defringe=True, on_alias="clamp"
    )
    assert filt.shape == exp.trace.shape
    assert filt.max() == pytest.approx(1.0)
    peak_delay = tau2[int(np.argmax(filt.sum(axis=0)))]
    assert peak_delay == pytest.approx(0.0)


def test_filter_fringes_and_defringe_mutually_exclusive(experiment):
    exp = experiment
    lam = exp.lam_nm * 1e-9
    tau = exp.delay_fs * 1e-15
    with pytest.raises(ValueError, match="at most one"):
        preprocess.filter_frog_trace(
            exp.trace, lam, tau, filter_fringes=True, defringe=True
        )


# ---------------------------------------------------------------------------
# Signal-aware baseline removal (arpls_baseline)
# ---------------------------------------------------------------------------
def _shelf_peak(n=600):
    """A broad, slowly-varying ASYMMETRIC background (high at negative delay)
    under a compact Gaussian signal peak."""
    tau = np.linspace(-300e-15, 300e-15, n)
    shelf = 0.4 * np.exp(np.minimum(-tau / 150e-15, 0.0))  # 0.4 at τ<0, decays for τ>0
    peak = np.exp(-0.5 * (tau / 8e-15) ** 2)
    return tau, shelf, peak


def test_arpls_baseline_recovers_asymmetric_shelf():
    # The free-form smooth baseline must ride under an asymmetric background — the
    # whole point. Check the wings (away from the peak, where the background is
    # observable); right under the peak the background is necessarily interpolated.
    tau, shelf, peak = _shelf_peak()
    rng = np.random.default_rng(0)
    y = shelf + peak + 0.01 * rng.standard_normal(tau.size)
    z = preprocess.arpls_baseline(y, smoothness=1e2)
    wings = np.abs(tau) > 150e-15
    assert np.allclose(z[wings], shelf[wings], atol=5e-2)
    cleaned = y - z
    assert np.abs(cleaned[wings]).max() < 0.1  # background gone, only noise left


def test_arpls_baseline_recovers_constant():
    # A flat background is the zero-curvature limit — recovered essentially exactly.
    n = 512
    tau = np.linspace(-300e-15, 300e-15, n)
    base = 0.3 * np.ones(n)
    peak = np.exp(-0.5 * (tau / 8e-15) ** 2)
    z = preprocess.arpls_baseline(base + peak, smoothness=1e2)
    cleaned = base + peak - z
    wings = np.abs(tau) > 80e-15
    assert cleaned[wings].mean() == pytest.approx(0.0, abs=2e-2)
    assert cleaned[int(np.argmin(np.abs(tau)))] == pytest.approx(1.0, abs=5e-2)


def test_arpls_baseline_preserves_peak_area():
    tau, shelf, peak = _shelf_peak()
    z = preprocess.arpls_baseline(shelf + peak, smoothness=1e2)
    cleaned = shelf + peak - z
    near = np.abs(tau) < 60e-15  # a window around the peak
    area_true = np.trapezoid(peak[near], tau[near])
    area_clean = np.trapezoid(np.clip(cleaned[near], 0.0, None), tau[near])
    assert area_clean == pytest.approx(area_true, rel=0.15)


def test_arpls_tau_exclude_prevents_peak_suckup():
    # A broad tall peak on a flat background with a soft penalty bows the baseline
    # up into the peak; holding out |τ|<tau_exclude keeps it down at the true floor.
    n = 512
    tau = np.linspace(-200e-15, 200e-15, n)
    y = 0.05 + np.exp(-0.5 * (tau / 25e-15) ** 2)
    peak_i = int(np.argmin(np.abs(tau)))
    z_no = preprocess.arpls_baseline(y, smoothness=0.1)
    z_ex = preprocess.arpls_baseline(y, smoothness=0.1, tau=tau, tau_exclude=60e-15)
    assert z_no[peak_i] > 0.3  # unguarded baseline is pulled well up into the peak
    assert z_ex[peak_i] == pytest.approx(
        0.05, abs=3e-2
    )  # hold-out keeps it at the floor


def test_arpls_baseline_nonuniform_2d_shape():
    tau, shelf, peak = _shelf_peak(n=128)
    trace = np.outer(np.array([0.3, 1.0, 0.6]), shelf + peak)  # (3, 128)
    z = preprocess.arpls_baseline(trace, smoothness=1e2)
    assert z.shape == trace.shape


@pytest.mark.parametrize(
    "kw",
    [
        {"smoothness": 0.0},
        {"smoothness": -1.0},
        {"ratio": 0.0},
        {"max_iter": 0},
    ],
)
def test_arpls_baseline_rejects_bad_params(kw):
    y = np.ones(64)
    base = {"smoothness": 1e2}
    with pytest.raises(ValueError):
        preprocess.arpls_baseline(y, **{**base, **kw})


def test_arpls_tau_exclude_requires_tau():
    with pytest.raises(ValueError, match="tau"):
        preprocess.arpls_baseline(np.ones(64), tau_exclude=10e-15)


def test_filter_frog_trace_baseline_runs(experiment):
    exp = experiment
    lam = exp.lam_nm * 1e-9
    tau = exp.delay_fs * 1e-15
    tau2, filt = preprocess.filter_frog_trace(
        exp.trace, lam, tau, filter_fringes=False, baseline=True
    )
    assert filt.shape == exp.trace.shape
    assert filt.max() == pytest.approx(1.0)
    assert tau2[int(np.argmax(filt.sum(axis=0)))] == pytest.approx(0.0)


def test_filter_frog_trace_baseline_clip_removes_negatives(experiment):
    exp = experiment
    lam = exp.lam_nm * 1e-9
    tau = exp.delay_fs * 1e-15
    _, kept = preprocess.filter_frog_trace(
        exp.trace, lam, tau, filter_fringes=False, baseline=True, baseline_clip=False
    )
    _, clipped = preprocess.filter_frog_trace(
        exp.trace, lam, tau, filter_fringes=False, baseline=True, baseline_clip=True
    )
    assert kept.min() < 0.0  # negatives kept by default (MLE-correct)
    assert clipped.min() >= 0.0  # the flag clips them


def test_arpls_baseline_downsample_matches_full():
    # The baseline is smooth, so a coarse-grid solve matches the full-resolution
    # one to a few percent — the basis of the GUI's fast interactive preview.
    _, shelf, peak = _shelf_peak(n=600)
    trace = np.outer(np.array([0.3, 1.0, 0.6]), shelf + peak)  # (3, 600)
    z_full = preprocess.arpls_baseline(trace, smoothness=1e2)
    z_coarse = preprocess.arpls_baseline(trace, smoothness=1e2, downsample=4)
    assert z_coarse.shape == trace.shape
    assert np.abs(z_coarse - z_full).max() < 5e-2 * np.abs(z_full).max()


def test_arpls_baseline_skip_below_zeros_dead_rows():
    _, shelf, peak = _shelf_peak(n=400)
    sig = shelf + peak
    trace = np.stack([sig, 1e-4 * sig, 0.5 * sig])  # row 1 is essentially dead
    z = preprocess.arpls_baseline(trace, smoothness=1e2, skip_below=0.01)
    assert np.all(z[1] == 0.0)  # below-threshold row skipped -> zero baseline
    assert np.abs(z[0]).max() > 0.1  # signal-bearing rows still get a baseline
    assert np.abs(z[2]).max() > 0.1


@pytest.mark.parametrize(
    "kw", [{"downsample": 0}, {"skip_below": 1.0}, {"skip_below": -0.1}]
)
def test_arpls_baseline_rejects_bad_speed_params(kw):
    with pytest.raises(ValueError):
        preprocess.arpls_baseline(np.ones(64), **kw)


def test_load_and_clean_fast_baseline_threads_through(experiment):
    # The coarse-grid / row-skip speed controls thread through load_and_clean to a
    # valid trace. (Numerical equivalence of the coarse baseline to the exact one
    # is covered by test_arpls_baseline_downsample_matches_full.)
    exp = experiment
    lam = exp.lam_nm * 1e-9
    delays = exp.delay_fs * 1e-15
    td = preprocess.load_and_clean(
        exp.trace,
        lam,
        delays,
        exp.interaction,
        lam_min=exp.lam_min,
        lam_max=exp.lam_max,
        input_unit="delay",
        filter_fringes=False,
        baseline=True,
        baseline_downsample=4,
        baseline_skip_below=0.01,
    )
    assert td.trace.shape[0] == td.grid.n
    assert np.isfinite(td.trace).all()
    assert td.trace.max() == pytest.approx(1.0)


def test_resample_spectrum_positive_and_peaked():
    lam = np.linspace(700e-9, 900e-9, 500)
    Ilam = np.exp(-0.5 * ((lam - 800e-9) / 20e-9) ** 2)
    omega0 = wlfreq(800e-9)
    omega = np.linspace(-0.3e15, 0.3e15, 256) + omega0
    Iw = preprocess.resample_spectrum(lam, Ilam, omega)
    assert np.all(Iw >= 0)  # edges are tapered to zero
    # peak near the carrier (omega ~ omega0, i.e. centre)
    assert abs(omega[np.argmax(Iw)] - omega0) < 0.05e15


def test_regrid_builds_grid(experiment):
    exp = experiment
    lam = exp.lam_nm * 1e-9
    tau = exp.delay_fs * 1e-15
    tau2, grid, w0p, w0t, trace, Iw, taper_loss = preprocess.regrid(
        tau,
        lam,
        exp.trace,
        None,
        None,
        exp.interaction,
        lam_min=exp.lam_min,
        lam_max=exp.lam_max,
        taper_warn_level=0.0,
    )
    assert trace.shape == (grid.n, tau2.size)
    assert trace.max() == pytest.approx(1.0)
    assert w0t == pytest.approx(w0p)  # PG: scale 1
    assert Iw is None
    assert 0.0 <= taper_loss <= 1.0


def test_load_and_clean_produces_retrievable_trace(experiment):
    exp = experiment
    td = preprocess.load_and_clean(
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
    assert td.trace.shape == (td.grid.n, td.delays.size)
    assert td.trace.max() == pytest.approx(1.0)
    assert td.omega0_pulse > 0
    # the cleaned trace should be retrievable (resampling adds a few % residual)
    res = retrieve(
        td.trace,
        td.omega,
        td.delays,
        td.interaction,
        algorithm="copra",
        maxiters=250,
    )
    assert res.error < 7e-2


def test_tau_crop_reduces_delays(experiment):
    exp = experiment
    full = preprocess.load_and_clean(
        exp.trace,
        exp.lam_nm * 1e-9,
        exp.delay_fs * 1e-15,
        exp.interaction,
        lam_min=exp.lam_min,
        lam_max=exp.lam_max,
        input_unit="delay",
        filter_fringes=False,
        dtau_min=None,
    )
    cropped = preprocess.load_and_clean(
        exp.trace,
        exp.lam_nm * 1e-9,
        exp.delay_fs * 1e-15,
        exp.interaction,
        lam_min=exp.lam_min,
        lam_max=exp.lam_max,
        input_unit="delay",
        filter_fringes=False,
        dtau_min=None,
        tau_crop=(-20e-15, 20e-15),  # narrower than the ±60 fs scan
    )
    assert cropped.delays.size < full.delays.size
    assert cropped.delays.min() >= -20e-15 and cropped.delays.max() <= 20e-15


def test_dc_filter_keeps_negatives_until_threshold(experiment):
    """DC filtering produces negatives that are kept (not floored) by default,
    and only removed when a threshold is explicitly applied."""
    exp = experiment
    common = dict(
        lam_min=exp.lam_min,
        lam_max=exp.lam_max,
        input_unit="delay",
        filter_fringes=False,
        filter_dc=True,
        dtau_min=None,
    )
    kept = preprocess.load_and_clean(
        exp.trace, exp.lam_nm * 1e-9, exp.delay_fs * 1e-15, exp.interaction, **common
    )
    assert kept.trace.min() < 0  # negatives survive into the regridded trace
    thr = preprocess.load_and_clean(
        exp.trace,
        exp.lam_nm * 1e-9,
        exp.delay_fs * 1e-15,
        exp.interaction,
        threshold=1e-3,
        **common,
    )
    assert thr.trace.min() >= 0  # an explicit threshold removes them


def test_threshold_off_by_default_and_applied_at_end(experiment):
    exp = experiment
    common = dict(
        lam_min=exp.lam_min,
        lam_max=exp.lam_max,
        input_unit="delay",
        filter_fringes=False,
        dtau_min=None,
    )
    none = preprocess.load_and_clean(
        exp.trace, exp.lam_nm * 1e-9, exp.delay_fs * 1e-15, exp.interaction, **common
    )
    default = preprocess.load_and_clean(
        exp.trace, exp.lam_nm * 1e-9, exp.delay_fs * 1e-15, exp.interaction, **common
    )
    thr = preprocess.load_and_clean(
        exp.trace,
        exp.lam_nm * 1e-9,
        exp.delay_fs * 1e-15,
        exp.interaction,
        threshold=0.5,
        **common,
    )
    # default is "off" (None): identical to explicitly passing threshold=None
    assert np.array_equal(default.trace, none.trace)
    # thresholding only removes the low tail, leaving more exact zeros + peak 1
    assert (thr.trace == 0).sum() > (none.trace == 0).sum()
    assert none.trace.max() == pytest.approx(1.0)
    assert thr.trace.max() == pytest.approx(1.0)
    assert thr.trace[(thr.trace > 0)].min() >= 0.5


def test_tracedata_stores_measurement_windows(experiment):
    exp = experiment
    lamm = (exp.lam_min * 1.05, exp.lam_max * 0.95)
    taum = (-30e-15, 30e-15)
    td = preprocess.load_and_clean(
        exp.trace,
        exp.lam_nm * 1e-9,
        exp.delay_fs * 1e-15,
        exp.interaction,
        lam_min=exp.lam_min,
        lam_max=exp.lam_max,
        input_unit="delay",
        filter_fringes=False,
        dtau_min=None,
        lamm_lims=lamm,
        tau_lims=taum,
    )
    assert td.lamm_lims == lamm
    assert td.tau_lims == taum
    # λm defaults to the full measurement extent, τm to None when unset
    td2 = preprocess.load_and_clean(
        exp.trace,
        exp.lam_nm * 1e-9,
        exp.delay_fs * 1e-15,
        exp.interaction,
        lam_min=exp.lam_min,
        lam_max=exp.lam_max,
        input_unit="delay",
        filter_fringes=False,
        dtau_min=None,
    )
    assert td2.tau_lims is None
    assert td2.lamm_lims == pytest.approx(
        (exp.lam_nm.min() * 1e-9, exp.lam_nm.max() * 1e-9)
    )


def test_load_and_clean_position_unit(experiment):
    exp = experiment
    # express the delay axis as a stage position z = c*tau/2
    from croak.constants import C

    z = exp.delay_fs * 1e-15 * C / 2.0
    td = preprocess.load_and_clean(
        exp.trace,
        exp.lam_nm * 1e-9,
        z,
        exp.interaction,
        lam_min=exp.lam_min,
        lam_max=exp.lam_max,
        input_unit="position",
        filter_fringes=False,
        dtau_min=None,
    )
    assert td.trace.max() == pytest.approx(1.0)


def test_suggested_tau_exclude_is_twice_the_marginal_fwhm():
    """The suggestion tracks the delay-marginal width, from a trace or a marginal."""
    tau = np.linspace(-50e-15, 50e-15, 401)
    # exp(-(t/w)^2) has FWHM = 2 w sqrt(ln 2)
    width = 2 * 4e-15 * np.sqrt(np.log(2))
    marginal = np.exp(-((tau / 4e-15) ** 2))

    assert preprocess.suggested_tau_exclude(marginal, tau) == pytest.approx(
        2 * width, rel=1e-3
    )
    # a 2-D trace is summed over wavelength first, giving the same answer
    trace = np.outer(np.array([0.2, 1.0, 0.5]), marginal)
    assert preprocess.suggested_tau_exclude(trace, tau) == pytest.approx(
        2 * width, rel=1e-3
    )
    # a broad pedestal under the peak must not shrink the measured width
    assert preprocess.suggested_tau_exclude(marginal + 3.0, tau) == pytest.approx(
        2 * width, rel=1e-3
    )
    assert preprocess.suggested_tau_exclude(marginal, tau, factor=1.0) == pytest.approx(
        width, rel=1e-3
    )


def test_suggested_tau_exclude_is_capped_and_validated():
    """The hold-out can never approach the whole axis, and needs a real peak."""
    tau = np.linspace(-50e-15, 50e-15, 401)
    # a marginal as wide as the axis would otherwise ask for more than exists
    broad = np.exp(-((tau / 60e-15) ** 2))
    assert preprocess.suggested_tau_exclude(broad, tau) == pytest.approx(
        0.4 * 0.5 * 100e-15
    )
    # arpls_baseline rejects a hold-out covering the axis; the cap keeps us clear
    preprocess.arpls_baseline(
        broad, tau=tau, tau_exclude=preprocess.suggested_tau_exclude(broad, tau)
    )

    with pytest.raises(ValueError, match="no discernible peak"):
        preprocess.suggested_tau_exclude(np.ones_like(tau), tau)
    with pytest.raises(ValueError, match="tau length"):
        preprocess.suggested_tau_exclude(np.ones(10), tau)


def test_arpls_hold_out_prevents_the_single_row_stripe():
    """No row may lose its peak while its neighbours keep theirs.

    Regression for a thin stripe at one wavelength that appeared only with baseline
    removal on: without a τ hold-out the arPLS baseline bows up into the τ≈0 peak,
    and how far it bows depends on how well each row constrains the fit — so a row
    whose peak sits low against its background is gutted while its neighbours are
    barely touched. The stripe is that row-to-row contrast, not a uniform loss.
    """
    tau = np.linspace(-50e-15, 50e-15, 400)
    peak = np.exp(-((tau / 5e-15) ** 2))
    # asymmetric leakage background, taller than the signal and skewed to -τ
    shelf = 2.0 * np.exp(-(((tau + 7.2e-15) / 12e-15) ** 2))
    rows = np.array([shelf + a * peak for a in (1.0, 0.9, 0.12, 0.9, 1.0)])

    def kept(tau_exclude):
        z = preprocess.arpls_baseline(
            rows, smoothness=1e2, tau=tau, tau_exclude=tau_exclude
        )
        return (rows - z).max(axis=1) / rows.max(axis=1)

    unguarded = kept(None)
    assert unguarded.min() < 0.1, unguarded  # the low-contrast row is gutted...
    assert unguarded.max() > 0.9, unguarded  # ...while its neighbours are fine
    guarded = kept(preprocess.suggested_tau_exclude(rows, tau))
    assert guarded.min() > 0.8, guarded  # the hold-out saves it


def test_arpls_baseline_reweights_before_declaring_convergence():
    """A low-contrast peak must not be lost to a premature exit.

    The convergence test compares successive *reweighted* solves. Measuring the
    first solve against the data instead asks "how far does smoothing move this
    row", which a low-contrast row answers with "barely" — it then exits having
    never run the asymmetric reweighting that is the algorithm, and keeps far less
    of its peak (measured 0.13 of it, against 0.32 once the first solve is made
    ineligible to converge).
    """
    tau = np.linspace(-50e-15, 50e-15, 512)
    # a narrow peak two thousandths of the pedestal it sits on
    amplitude = 0.002
    row = 1.0 + amplitude * np.exp(-((tau / 3e-15) ** 2))
    cleaned = row - preprocess.arpls_baseline(row, smoothness=1e2, ratio=1e-3)
    assert cleaned.max() / amplitude > 0.2, cleaned.max() / amplitude


def test_filters_are_off_by_default(experiment):
    """Every filter removes real signal, so none may switch itself on."""
    exp = experiment
    lam = exp.lam_nm * 1e-9
    tau = exp.delay_fs * 1e-15

    default = preprocess.filter_frog_trace(exp.trace, lam, tau)[1]
    explicit = preprocess.filter_frog_trace(
        exp.trace, lam, tau, filter_fringes=False, filter_dc=False, dtau_min=None
    )[1]
    np.testing.assert_allclose(default, explicit, rtol=0, atol=0)

    # ...and each one, switched on, actually changes the trace — so the defaults
    # are off rather than the filters being no-ops. The dtau_min cutoff must sit
    # *below* the delay Nyquist frequency or there is nothing above it to remove:
    # this fixture is stepped at ~1.5 fs, so 5 fs cuts at 200 THz against a 329
    # THz Nyquist.
    for kw in ({"filter_dc": True}, {"filter_fringes": True}, {"dtau_min": 5e-15}):
        changed = preprocess.filter_frog_trace(exp.trace, lam, tau, **kw)[1]
        assert not np.allclose(changed, default), kw


def test_dtau_min_zero_disables_the_low_pass(experiment):
    """0 means "no minimum resolvable step", matching the GUI, not a divide by zero."""
    exp = experiment
    lam = exp.lam_nm * 1e-9
    tau = exp.delay_fs * 1e-15
    zero = preprocess.filter_frog_trace(exp.trace, lam, tau, dtau_min=0.0)[1]
    none = preprocess.filter_frog_trace(exp.trace, lam, tau, dtau_min=None)[1]
    np.testing.assert_allclose(zero, none, rtol=0, atol=0)


def test_load_and_clean_defaults_apply_no_filtering(experiment):
    """The one-call entry point inherits the same off-by-default policy."""
    exp = experiment
    lam = exp.lam_nm * 1e-9
    tau = exp.delay_fs * 1e-15
    # taper_warn_level=0: this test is about the filter defaults, and the narrow
    # band it uses to stay quick would otherwise trip the band-clipping warning.
    common = dict(
        lam_min=700e-9, lam_max=900e-9, input_unit="delay", taper_warn_level=0.0
    )
    default = preprocess.load_and_clean(exp.trace, lam, tau, "pg", **common)
    explicit = preprocess.load_and_clean(
        exp.trace,
        lam,
        tau,
        "pg",
        filter_fringes=False,
        filter_dc=False,
        dtau_min=None,
        **common,
    )
    np.testing.assert_allclose(default.trace, explicit.trace, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Sub-sample tau=0 centring
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("offset_fs", [0.0, 0.19, -0.37, 0.48])
def test_marginal_peak_delay_resolves_below_one_sample(offset_fs):
    """The peak is located to a small fraction of a step, not snapped to one."""
    tau = np.linspace(-10e-15, 10e-15, 21)  # 1 fs steps
    step = float(np.diff(tau)[0])
    true = offset_fs * 1e-15
    marginal = np.exp(-(((tau - true) / 3e-15) ** 2))

    estimate = preprocess.marginal_peak_delay(tau, marginal)
    # snapping to the nearest sample would be wrong by up to half a step; the
    # parabolic vertex must do far better than that
    assert abs(estimate - true) < 0.05 * step


def test_marginal_peak_delay_falls_back_at_the_edges():
    """A peak on the first or last sample cannot be interpolated; keep it."""
    tau = np.linspace(0.0, 10e-15, 11)
    rising = np.linspace(0.0, 1.0, 11)
    assert preprocess.marginal_peak_delay(tau, rising) == pytest.approx(tau[-1])
    assert preprocess.marginal_peak_delay(tau, rising[::-1]) == pytest.approx(tau[0])


def test_centring_puts_the_true_peak_at_zero_sub_sample(experiment):
    """filter_frog_trace centres on the interpolated peak, not the peak sample."""
    exp = experiment
    lam = exp.lam_nm * 1e-9
    tau = exp.delay_fs * 1e-15
    step = float(np.diff(tau)[0])

    tau_out, filt = preprocess.filter_frog_trace(exp.trace, lam, tau)
    # where the marginal actually peaks on the returned axis
    peak = preprocess.marginal_peak_delay(tau_out, filt.sum(axis=0))
    assert abs(peak) < 0.05 * step

    # the sample-snapped answer would sit up to half a step away, so the axis
    # generally does *not* have a sample sitting exactly at zero
    nearest_sample = tau_out[int(np.argmin(np.abs(tau_out)))]
    assert abs(nearest_sample) <= 0.5 * step


# ---------------------------------------------------------------------------
# Spectral taper loss: warn when the band clips the trace
# ---------------------------------------------------------------------------
def _narrowband_experiment():
    """A long, unchirped pulse — narrow enough that a wide band really is wide."""
    return make_synthetic_experiment(
        fwhm=30e-15, phases=(), n=384, dt=0.4e-15, delay_max=120e-15
    )


def test_regrid_reports_the_taper_loss(experiment):
    """regrid returns how much of the trace its edge taper removed."""
    exp = experiment
    _, _, _, _, _, _, loss = preprocess.regrid(
        exp.delay_fs * 1e-15,
        exp.lam_nm * 1e-9,
        exp.trace,
        None,
        None,
        exp.interaction,
        lam_min=exp.lam_min,
        lam_max=exp.lam_max,
        taper_warn_level=0.0,
    )
    assert 0.0 <= loss <= 1.0


def test_taper_loss_falls_as_the_band_widens():
    """Widening the band moves the taper out onto empty trace."""
    exp = _narrowband_experiment()
    lam = exp.lam_nm * 1e-9
    tau = exp.delay_fs * 1e-15
    losses = [
        preprocess.load_and_clean(
            exp.trace,
            lam,
            tau,
            "pg",
            lam_min=lo * 1e-9,
            lam_max=hi * 1e-9,
            input_unit="delay",
            taper_warn_level=0.0,
        ).taper_loss
        for lo, hi in [(770, 835), (700, 930), (650, 1000)]
    ]
    assert losses[0] > losses[1] >= losses[2]
    assert losses[2] < 1e-4  # a genuinely generous band loses nothing


def test_load_and_clean_warns_when_the_band_clips_the_trace():
    """A band tight enough to taper live signal must say so."""
    exp = _narrowband_experiment()
    with pytest.warns(UserWarning, match=r"edge taper removed .*% of the measured"):
        preprocess.load_and_clean(
            exp.trace,
            exp.lam_nm * 1e-9,
            exp.delay_fs * 1e-15,
            "pg",
            lam_min=770e-9,
            lam_max=835e-9,
            input_unit="delay",
        )


def test_load_and_clean_is_quiet_on_a_generous_band(recwarn):
    """A band chosen with room to spare must not warn."""
    exp = _narrowband_experiment()
    preprocess.load_and_clean(
        exp.trace,
        exp.lam_nm * 1e-9,
        exp.delay_fs * 1e-15,
        "pg",
        lam_min=650e-9,
        lam_max=1000e-9,
        input_unit="delay",
    )
    assert not [w for w in recwarn if "edge taper" in str(w.message)]


def test_taper_warn_level_zero_silences_the_check(recwarn):
    """The check is opt-out for callers who know their band is tight."""
    exp = _narrowband_experiment()
    td = preprocess.load_and_clean(
        exp.trace,
        exp.lam_nm * 1e-9,
        exp.delay_fs * 1e-15,
        "pg",
        lam_min=770e-9,
        lam_max=835e-9,
        input_unit="delay",
        taper_warn_level=0.0,
    )
    assert not [w for w in recwarn if "edge taper" in str(w.message)]
    assert td.taper_loss > 0.01  # the loss is still measured, just not announced


# ---------------------------------------------------------------------------
# uniform_core_indices
# ---------------------------------------------------------------------------
class TestUniformCoreIndices:
    def test_uniform_axis_is_kept_whole(self):
        x = np.linspace(-25e-15, 25e-15, 200)
        idx = preprocess.uniform_core_indices(x)
        assert idx.size == x.size
        np.testing.assert_array_equal(idx, np.arange(x.size))

    def test_campaign_core_plus_wings(self):
        """The 04-style axis: a 200-point core with 1 fs-step wings."""
        core = np.linspace(-25e-15, 25e-15, 200)
        wing = np.arange(26e-15, 40.5e-15, 1e-15)
        x = np.sort(np.concatenate([-wing[::-1], core, wing]))
        idx = preprocess.uniform_core_indices(x)
        np.testing.assert_allclose(x[idx], core)

    def test_one_sided_wings(self):
        core = np.arange(0.0, 10.0, 0.5)
        x = np.concatenate([core, [11.0, 13.0, 15.0]])
        idx = preprocess.uniform_core_indices(x)
        np.testing.assert_allclose(x[idx], core)

    def test_tiny_axes_pass_through(self):
        for n in (0, 1, 2):
            x = np.arange(n, dtype=float)
            np.testing.assert_array_equal(
                preprocess.uniform_core_indices(x), np.arange(n)
            )

    def test_result_is_contiguous(self):
        rng = np.random.default_rng(0)
        x = np.sort(rng.uniform(-1, 1, 37))
        idx = preprocess.uniform_core_indices(x)
        assert np.all(np.diff(idx) == 1)
