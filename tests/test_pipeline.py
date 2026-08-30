"""Tests for :mod:`croak.pipeline` and the solver omega0/callback plumbing."""

import numpy as np
import pytest

from croak import preprocess
from croak.focal import focal_mixture
from croak.pipeline import initial_guess, retrieve_from_tracedata
from croak.smearing import square_boxcars_kernel


@pytest.fixture
def tracedata(experiment):
    exp = experiment
    return preprocess.load_and_clean(
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


@pytest.mark.parametrize("mode", ["auto", "tl", "perfect", "random"])
def test_initial_guess_modes(tracedata, mode):
    ew = initial_guess(tracedata, mode=mode, rng=np.random.default_rng(0))
    assert ew.shape == (tracedata.grid.n,)
    assert np.iscomplexobj(ew)


def test_explicit_array_guess_is_used(tracedata):
    """Passing a spectrum array as the guess (the GUI 'reuse result' path)."""
    seed = initial_guess(tracedata, mode="perfect", perfect_fwhm=8e-15)
    res = retrieve_from_tracedata(tracedata, algorithm="copra", maxiters=5, guess=seed)
    assert res.spectrum.shape == seed.shape


def test_truth_initial_guess_uses_the_known_complex_spectrum(tracedata):
    """The public pipeline's truth mode aligns and returns the known field."""
    from croak.processing import TruthPulse

    seed = initial_guess(tracedata, mode="perfect", perfect_fwhm=8e-15)
    truth = TruthPulse.from_spectrum(
        tracedata.grid, seed, tracedata.omega0_pulse, oversampling=2
    )
    actual = initial_guess(tracedata, mode="truth", truth=truth)
    np.testing.assert_allclose(actual, seed)


def test_truth_initial_guess_requires_truth(tracedata):
    """Selecting truth without a complex truth object fails before optimisation."""
    with pytest.raises(ValueError, match="requires a known TruthPulse"):
        initial_guess(tracedata, mode="truth")


def test_unknown_algorithm_raises(tracedata):
    with pytest.raises(ValueError, match="unknown algorithm"):
        retrieve_from_tracedata(tracedata, algorithm="nope", maxiters=1)


@pytest.mark.parametrize(
    "algorithm",
    [
        "copra",
        "copra-jax",
        "lbfgs",
        "lbfgs-hand",
        "lbfgs-ad",
        "lbfgs-optx",
        "lm",
        "lm-optx",
    ],
)
def test_retrieve_from_tracedata(tracedata, algorithm):
    res = retrieve_from_tracedata(
        tracedata, algorithm=algorithm, maxiters=200, rng=np.random.default_rng(0)
    )
    assert res.algorithm == algorithm
    assert res.omega0 == pytest.approx(tracedata.omega0_pulse)
    assert res.error < 8e-2
    # wavelength axis is sensible (near 800 nm carrier)
    lam_peak = res.wavelength[np.argmax(res.intensity_omega)]
    assert 700e-9 < lam_peak < 950e-9


def test_reg_spectrum_warns_when_no_independent_spectrum(tracedata):
    # The synthetic experiment has no independent spectrum (Iomega is None), so
    # reg_spectrum must be ignored with a warning and give the same result as 0.
    assert tracedata.Iomega is None
    kw = dict(algorithm="lbfgs", maxiters=120)
    r0 = retrieve_from_tracedata(
        tracedata, reg_spectrum=0.0, rng=np.random.default_rng(0), **kw
    )
    with pytest.warns(RuntimeWarning, match="independent spectrum"):
        r1 = retrieve_from_tracedata(
            tracedata, reg_spectrum=0.3, rng=np.random.default_rng(0), **kw
        )
    assert np.allclose(r0.spectrum, r1.spectrum)


@pytest.mark.parametrize(
    "algorithm", ["lbfgs", "lbfgs-hand", "lbfgs-ad", "lbfgs-optx", "lm", "lm-optx"]
)
def test_reg_spectrum_forwarded(tracedata, algorithm):
    # Give the trace an independent spectrum (proxy from the frequency marginal)
    # and confirm reg_spectrum actually changes the retrieved spectrum.
    td = tracedata
    td.Iomega = td.trace.sum(axis=1)
    td.Iomega = td.Iomega / td.Iomega.max()
    kw = dict(algorithm=algorithm, maxiters=150)
    r0 = retrieve_from_tracedata(
        td, reg_spectrum=0.0, rng=np.random.default_rng(0), **kw
    )
    r1 = retrieve_from_tracedata(
        td, reg_spectrum=0.5, rng=np.random.default_rng(0), **kw
    )
    assert not np.allclose(r0.spectrum, r1.spectrum)


def test_callback_receives_progress(tracedata):
    seen = []
    retrieve_from_tracedata(
        tracedata,
        algorithm="copra",
        maxiters=10,
        callback=lambda it, R, best: seen.append((it, R, best)),
        rng=np.random.default_rng(0),
    )
    assert len(seen) == 10
    assert seen[0][0] == 1 and seen[-1][0] == 10


def test_omega0_recorded_via_retrieve():
    from croak import retrieve
    from croak.forward import maketrace
    from croak.grid import Grid
    from croak.maths import wlfreq
    from croak.pulses import gaussian_pulse

    g = Grid(64, dt=0.5e-15)
    ew = gaussian_pulse(g, 6e-15)
    delays = np.linspace(-30e-15, 30e-15, 40)
    trace = maketrace(g.omega, delays, ew, "pg")
    w0 = wlfreq(800e-9)
    res = retrieve(
        trace,
        g.omega,
        delays,
        "pg",
        algorithm="copra",
        maxiters=20,
        guess=ew,
        omega0=w0,
    )
    assert res.omega0 == pytest.approx(w0)


def test_smearing_rejected_for_unsupported_algorithms(tracedata):
    """A smearing kernel must never be dropped silently — it is a model term.

    COPRA's magnitude-replacement projection and the hand-written adjoints cannot
    represent the incoherent sum over quadrature nodes, so asking for smearing with
    them has to fail loudly rather than return a confidently wrong pulse.

    ``copra-jax`` is deliberately absent from the list: its *global* stage is an
    ordinary descent step and can carry the kernel, so it accepts one and applies
    it there (its local projection still cannot use it). The NumPy ``copra`` and
    the hand-written adjoints take no kernel at all.
    """
    kernel = square_boxcars_kernel(
        "pg", hole_diameter=1e-3, hole_spacing=0.5e-3, wavelength=800e-9
    )
    for algorithm in ("copra", "lbfgs", "lbfgs-hand"):
        with pytest.raises(ValueError, match="cannot model geometrical smearing"):
            retrieve_from_tracedata(
                tracedata, algorithm=algorithm, maxiters=2, smearing=kernel
            )


def test_smearing_forwarded_to_supported_algorithms(tracedata):
    kernel = square_boxcars_kernel(
        "pg", hole_diameter=1e-3, hole_spacing=0.5e-3, wavelength=800e-9
    )
    res = retrieve_from_tracedata(
        tracedata, algorithm="lbfgs-ad", maxiters=3, smearing=kernel
    )
    assert res.spectrum.shape == (tracedata.grid.n,)


def test_focal_rejected_for_unsupported_algorithms(tracedata):
    """A focal mixture is a model term and must fail loudly, like smearing.

    The projection-based and hand-adjoint solvers cannot represent the
    node-resolved chromatic mixture, so asking for one has to raise rather
    than silently retrieve with the ideal-geometry model.
    """
    mix = focal_mixture(
        hole_diameter=1e-3,
        hole_spacing=0.5e-3,
        f_foc=0.1,
        wavelength=800e-9,
        n_radial=3,
        n_azimuth=4,
    )
    for algorithm in ("copra", "lbfgs", "lbfgs-hand"):
        with pytest.raises(ValueError, match="cannot model a chromatic focal"):
            retrieve_from_tracedata(
                tracedata, algorithm=algorithm, maxiters=2, focal=mix
            )


def test_focal_forwarded_to_supported_algorithms(tracedata):
    """`focal=` reaches the AD solvers through the high-level pipeline.

    Regression test for the plumbing gap where the solver constructors accepted
    a mixture but ``retrieve_from_tracedata`` (and therefore the whole session
    layer) silently could not pass one.
    """
    mix = focal_mixture(
        hole_diameter=1e-3,
        hole_spacing=0.5e-3,
        f_foc=0.1,
        wavelength=800e-9,
        n_radial=3,
        n_azimuth=4,
    )
    res = retrieve_from_tracedata(
        tracedata, algorithm="lbfgs-ad", maxiters=3, focal=mix
    )
    assert res.spectrum.shape == (tracedata.grid.n,)
