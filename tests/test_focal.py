"""Tests for the chromatic focal mixture (``croak.focal``)."""

from __future__ import annotations

import numpy as np
import pytest

from croak.focal import airy_amplitude, focal_mixture, reduced_moments

C_LIGHT = 299_792_458.0

#: Closed forms from croak.smearing for the square BOXCARS PG/TG geometry, in
#: units of (d/D) lambda/c.
PG_SIGMA_P = 0.49135
PG_SIGMA_THETA = 0.54933
PG_RHO = 1.0 / np.sqrt(5.0)

#: The SD constants. The PG arm ordering is easy to permute into these by
#: accident, which is what makes the moment test worth having.
SD_SIGMA_P = 0.69487
SD_SIGMA_THETA = 0.34743


def _mix(gap=1.0e-3, n_radial=32, n_azimuth=8, r_max_units=6.0):
    return focal_mixture(
        hole_diameter=1.0e-3,
        hole_spacing=gap,
        f_foc=0.1,
        wavelength=260e-9,
        n_radial=n_radial,
        n_azimuth=n_azimuth,
        r_max_units=r_max_units,
    )


def test_airy_amplitude_removable_singularity():
    """2 J1(u)/u -> 1 at the origin, and matches a series expansion near it."""
    assert airy_amplitude(np.array([0.0]))[0] == pytest.approx(1.0)
    u = np.array([1e-6, 1e-3])
    assert airy_amplitude(u) == pytest.approx(1.0 - u**2 / 8.0, rel=1e-9)


def test_airy_amplitude_first_zero():
    """First zero of 2 J1(u)/u at u = 3.8317 (the Airy dark ring)."""
    assert abs(airy_amplitude(np.array([3.8317059702075125]))[0]) < 1e-9


def test_reduced_moments_match_pg_closed_forms():
    """The mixture samples the SAME focal average the reduced kernel models.

    This is the module's central self-test: the second moments of (p, theta) under
    the |A|^6 weight must reproduce croak.smearing's closed forms. It caught a
    wrong arm assignment during development, which silently produced the SD
    constants instead.
    """
    for gap in (0.5e-3, 1.0e-3, 2.0e-3):
        mix = _mix(gap=gap)
        q = reduced_moments(mix, 260e-9)
        dD = (gap + 1.0e-3) / 2.0 / 1.0e-3
        unit = dD * 260e-9 / C_LIGHT
        assert q["sigma_p"] / unit == pytest.approx(PG_SIGMA_P, rel=2e-3)
        assert q["sigma_theta"] / unit == pytest.approx(PG_SIGMA_THETA, rel=2e-3)
        assert q["rho"] == pytest.approx(PG_RHO, rel=1e-6)


def test_moments_are_not_the_sd_constants():
    """Guard the arm ordering explicitly: PG must not collapse onto SD."""
    q = reduced_moments(_mix(), 260e-9)
    unit = 260e-9 / C_LIGHT
    assert q["sigma_p"] / unit != pytest.approx(SD_SIGMA_P, rel=1e-2)
    assert q["sigma_theta"] / unit != pytest.approx(SD_SIGMA_THETA, rel=1e-2)
    assert abs(q["rho"]) > 0.1  # SD has rho = 0 exactly


def test_widths_scale_linearly_with_dD():
    """sigma ~ d/D at fixed wavelength: the mask ratio is the only geometry knob."""
    unit = 260e-9 / C_LIGHT
    got = []
    for gap in (0.5e-3, 1.0e-3, 2.0e-3):
        q = reduced_moments(_mix(gap=gap), 260e-9)
        dD = (gap + 1.0e-3) / 2.0 / 1.0e-3
        got.append(q["sigma_p"] / unit / dD)
    assert np.std(got) / np.mean(got) < 2e-3


def test_focal_length_cancels():
    """alpha ~ 1/f and sigma_r ~ f, so the widths must not depend on f."""
    unit = 260e-9 / C_LIGHT
    a = (
        reduced_moments(
            focal_mixture(
                hole_diameter=1e-3,
                hole_spacing=1e-3,
                f_foc=0.1,
                wavelength=260e-9,
                n_radial=32,
                n_azimuth=8,
            ),
            260e-9,
        )["sigma_p"]
        / unit
    )
    b = (
        reduced_moments(
            focal_mixture(
                hole_diameter=1e-3,
                hole_spacing=1e-3,
                f_foc=0.25,
                wavelength=260e-9,
                n_radial=32,
                n_azimuth=8,
            ),
            260e-9,
        )["sigma_p"]
        / unit
    )
    assert a == pytest.approx(b, rel=1e-6)


def test_spectral_filter_is_chromatic_and_scales_with_r_omega():
    """The filter depends on r and omega only through their product.

    This is the physics the reduced kernel drops: the mask maps to transverse
    wavevector as x = k z c / omega, so the focal pattern breathes with frequency
    and the local spectrum is reshaped by radius.
    """
    mix = _mix(n_radial=32, n_azimuth=4)
    omega = 2.0 * np.pi * C_LIGHT / np.array([200e-9, 260e-9, 400e-9])
    filt = mix.spectral_filter(omega)
    assert filt.shape == (mix.nodes, 3)

    # The scaling law itself: A depends on r*omega alone, so halving the frequency
    # is exactly doubling the radius. Compare a mixture at 2x the radial extent
    # against the same mixture at half the frequency.
    wide = focal_mixture(
        hole_diameter=1.0e-3,
        hole_spacing=1.0e-3,
        f_foc=0.1,
        wavelength=260e-9,
        n_radial=32,
        n_azimuth=4,
        r_max_units=12.0,
    )
    w0 = 2.0 * np.pi * C_LIGHT / 260e-9
    assert wide.spectral_filter(np.array([0.5 * w0]))[:, 0] == pytest.approx(
        mix.spectral_filter(np.array([w0]))[:, 0], rel=1e-9
    ), "A(2r, w) must equal A(r, 2w)"

    # Chromatic variation must be appreciable where the amplitude actually is --
    # inside the first lobe, not out in the tail where A is ~0 for every colour.
    bright = np.argsort(np.abs(filt[:, 1]))[-len(filt) // 4 :]
    spread = np.ptp(filt[bright], axis=1) / np.abs(filt[bright, 1])
    assert spread.max() > 0.10, "filter should reshape the spectrum across the band"


def test_quadrature_converged_at_default_resolution():
    """The shipped default must be converged in the radial direction."""
    unit = 260e-9 / C_LIGHT
    coarse = reduced_moments(_mix(n_radial=32), 260e-9)["sigma_p"] / unit
    fine = reduced_moments(_mix(n_radial=64), 260e-9)["sigma_p"] / unit
    assert coarse == pytest.approx(fine, rel=1e-3)


def test_rejects_degenerate_quadrature():
    with pytest.raises(ValueError, match="n_radial and n_azimuth"):
        _mix(n_radial=1)


def test_gaussian_profile_moments():
    """A Gaussian focal weight gives sigma_r = w0/(2 sqrt 3) exactly.

    |A|^6 = exp(-6 r^2 / w0^2) for a Gaussian beam, whose rms in one Cartesian
    component is w0/(2 sqrt 3) -- the 4.78 um that sets the effective gap used
    for the Gaussian control runs.
    """
    w0 = 16.5521e-6
    mix = focal_mixture(
        hole_diameter=1.0e-3,
        hole_spacing=1.0e-3,
        f_foc=0.1,
        wavelength=260e-9,
        n_radial=48,
        n_azimuth=8,
        r_max_units=4.0,
        profile="gaussian",
        waist=w0,
    )
    q = reduced_moments(mix, 260e-9)
    alpha = 2.0 * 1.0e-3 / (0.1 * C_LIGHT)  # |alpha_3 - alpha_2| for this layout
    assert q["sigma_p"] / alpha == pytest.approx(w0 / (2.0 * np.sqrt(3.0)), rel=1e-4)
    assert q["rho"] == pytest.approx(PG_RHO, rel=1e-6)


def test_gaussian_profile_is_achromatic():
    """The Gaussian control's defining property: no chromatic focal structure."""
    mix = focal_mixture(
        hole_diameter=1.0e-3,
        hole_spacing=1.0e-3,
        f_foc=0.1,
        wavelength=260e-9,
        n_radial=8,
        n_azimuth=4,
        profile="gaussian",
        waist=16.5521e-6,
    )
    omega = 2.0 * np.pi * C_LIGHT / np.array([200e-9, 260e-9, 400e-9])
    filt = mix.spectral_filter(omega)
    assert np.ptp(filt, axis=1).max() < 1e-15


def test_gaussian_profile_requires_waist():
    with pytest.raises(ValueError, match="needs `waist`"):
        focal_mixture(
            hole_diameter=1.0e-3,
            hole_spacing=1.0e-3,
            f_foc=0.1,
            wavelength=260e-9,
            profile="gaussian",
        )


def test_reduced_kernel_is_exact_for_a_gaussian_weight():
    """The effective-gap surrogate used for the Gaussian control is EXACT.

    For a Gaussian focal weight |A|^6 is an isotropic Gaussian, so (p, theta) --
    being linear in r -- are exactly jointly Gaussian, which is precisely the
    reduced kernel's assumption. Rescaling the gap to match sigma_r therefore
    introduces no approximation, with or without chirp.

    This is worth pinning down: the Gaussian control's forward-model residual was
    once attributed to this surrogate, and it is not the cause.
    """
    from croak.forward_jax import make_param_trace_fn
    from croak.smearing import square_boxcars_kernel

    n, nd = 128, 41
    lam0 = 260e-9
    om0 = 2.0 * np.pi * C_LIGHT / lam0
    w0 = 16.5521e-6
    om = np.fft.fftshift(np.fft.fftfreq(n, d=0.5e-15)) * 2.0 * np.pi
    delays = np.linspace(-25e-15, 25e-15, nd)
    kern = square_boxcars_kernel(
        "pg", hole_diameter=1.0e-3, hole_spacing=0.496e-3, wavelength=lam0, npoints=9
    )
    mix = focal_mixture(
        hole_diameter=1.0e-3,
        hole_spacing=1.0e-3,
        f_foc=0.1,
        wavelength=lam0,
        n_radial=48,
        n_azimuth=8,
        r_max_units=4.0,
        profile="gaussian",
        waist=w0,
    )
    t_k = make_param_trace_fn(
        om,
        delays,
        "pg",
        smearing=kern,
        material="SiO2",
        thickness=20e-6,
        npoints=10,
        omega0=om0,
        normalize=True,
    )
    t_m = make_param_trace_fn(
        om,
        delays,
        "pg",
        focal=mix,
        material="SiO2",
        thickness=20e-6,
        npoints=10,
        omega0=om0,
        normalize=True,
    )
    for gdd in (0.0, 2.0e-30):
        ew = np.exp(-((om / (2 * np.pi * 40e12)) ** 2)).astype(complex) * np.exp(
            0.5j * gdd * om**2
        )
        a = np.asarray(t_m(ew, 20e-6, 0.0))
        b = np.asarray(t_k(ew, 20e-6, 0.0))
        mu = float(np.dot(a.ravel(), b.ravel()) / np.dot(b.ravel(), b.ravel()))
        assert np.sqrt(np.mean((a - mu * b) ** 2)) / a.max() < 1e-7


def test_focal_reaches_the_solvers(tmp_path):
    """``focal=`` is accepted by every solver that accepts ``smearing=``.

    The chromatic mixture was reachable from the forward model long before it
    was reachable from a retrieval, so croak could measure the error freezing
    the kernel at the carrier makes and could not correct it. This pins the
    plumbing: the parameter is advertised, stored, and forwarded.
    """
    from croak.retrieve import ALGORITHMS, algorithm_params

    for name in ("lbfgs-ad", "warm-lbfgs", "lm", "lm-optx", "lbfgs-optx", "cma-es"):
        params = algorithm_params(name)
        assert "smearing" in params, name
        assert "focal" in params, name

    mix = focal_mixture(
        hole_diameter=1.0e-3,
        hole_spacing=1.0e-3,
        f_foc=0.1,
        wavelength=260e-9,
        n_radial=8,
        n_azimuth=6,
        r_max_units=4.0,
    )
    solver = ALGORITHMS["lbfgs-ad"](focal=mix)
    assert solver.focal is mix
    assert solver.smearing is None


def test_focal_retrieval_beats_the_reduced_kernel_at_1fs():
    """Retrieving a chromatic trace with the chromatic model removes the bias.

    Freezing the kernel width at the carrier is a -16%/+24% spread across a
    1 fs deep-UV band. Here the trace is generated with the chromatic mixture
    and retrieved both ways from the same start; the chromatic model must land
    closer to the truth than the reduced kernel does.
    """
    from croak.forward_jax import make_param_trace_fn
    from croak.grid import Grid
    from croak.maths import fwhm
    from croak.pulses import gaussian_pulse
    from croak.retrieve import retrieve
    from croak.smearing import square_boxcars_kernel

    lam0, z = 260e-9, 20e-6
    om0 = 2.0 * np.pi * C_LIGHT / lam0
    g = Grid(128, dt=0.8e-15)
    delays = np.linspace(-20e-15, 20e-15, 41)
    ew = np.asarray(gaussian_pulse(g, 1e-15), dtype=complex)
    kern = square_boxcars_kernel(
        "pg", hole_diameter=1e-3, hole_spacing=1e-3, wavelength=lam0, npoints=9
    )
    mix = focal_mixture(
        hole_diameter=1e-3,
        hole_spacing=1e-3,
        f_foc=0.1,
        wavelength=lam0,
        n_radial=20,
        n_azimuth=8,
        r_max_units=4.0,
    )
    common = dict(
        omega0=om0,
        material="SiO2",
        thickness=z,
        npoints=20,
        R_omega=True,
        progress=False,
    )
    meas = np.asarray(
        make_param_trace_fn(
            g.omega,
            delays,
            "pg",
            focal=mix,
            material="SiO2",
            thickness=z,
            npoints=20,
            omega0=om0,
            normalize=True,
        )(ew, z, 0.0),
        float,
    )

    def duration(spec):
        s = np.asarray(spec)
        pad = np.concatenate(
            [np.zeros(3 * s.size, complex), s, np.zeros(3 * s.size, complex)]
        )
        et = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(pad)))
        dt = 2.0 * np.pi / (pad.size * float(g.omega[1] - g.omega[0]))
        t = (np.arange(pad.size) - pad.size // 2) * dt
        return float(fwhm(t, np.abs(et) ** 2))

    truth = duration(ew)
    kw = dict(guess=np.abs(ew), maxiters=120, rng=np.random.default_rng(0))
    reduced = retrieve(
        meas, g.omega, delays, "pg", algorithm="lbfgs-ad", smearing=kern, **common, **kw
    )
    chromatic = retrieve(
        meas, g.omega, delays, "pg", algorithm="lbfgs-ad", focal=mix, **common, **kw
    )
    err_reduced = abs(duration(reduced.spectrum) / truth - 1)
    err_chromatic = abs(duration(chromatic.spectrum) / truth - 1)
    # The chromatic model matches how the trace was made, so it must fit it
    # better AND land closer to the truth.
    assert chromatic.error < reduced.error
    assert err_chromatic < err_reduced


def test_focal_rejects_a_fitted_kernel_width():
    """``fit_smearing`` has no meaning for an explicit mixture, and says so."""
    from croak.forward_jax import make_param_trace_fn

    mix = focal_mixture(
        hole_diameter=1e-3,
        hole_spacing=1e-3,
        f_foc=0.1,
        wavelength=260e-9,
        n_radial=8,
        n_azimuth=6,
        r_max_units=4.0,
    )
    om = np.linspace(-2e15, 2e15, 32)
    with pytest.raises(ValueError, match="fit_smearing has no effect"):
        make_param_trace_fn(
            om, np.linspace(-1e-14, 1e-14, 5), "pg", focal=mix, fit_smearing=True
        )


def test_mixture_records_its_arm_positions():
    """The mask hole centres are kept, and the signal leaves by the fourth corner.

    The incoherent sum needs only the arm *differences* that make ``(p, theta)``, so
    the absolute positions used to be discarded. A finite collection aperture needs
    them: they fix the phase-matched direction, and hence where the aperture must sit.
    """
    mix = _mix(gap=1.0e-3)
    d = 0.5 * (1.0e-3 + 1.0e-3)
    assert mix.arms is not None
    assert mix.arms.shape == (3, 2)
    assert mix.arms.tolist() == [[-d, d], [d, -d], [d, d]]
    # r_s = r1 + r2 - r3, the corner diagonally opposite the conjugated arm.
    assert mix.signal_position.tolist() == [-d, -d]


def test_mixture_without_arms_refuses_to_name_a_signal_direction():
    """A mixture that never recorded its arms says so rather than guessing."""
    import dataclasses

    with pytest.raises(ValueError, match="no arm positions"):
        _ = dataclasses.replace(_mix(), arms=None).signal_position


def test_mixture_from_nodes_agrees_with_the_polar_builder():
    """The Cartesian entry point shares the ``(p, theta)`` algebra, it does not copy it.

    ``mixture_from_nodes`` exists so the collection tests can lay nodes on a Cartesian
    grid (to compare against a plain 2-D FFT, and to check Parseval exactly). If it
    re-derived the arm offsets, a sign error there would hide a sign error in the
    model it is meant to check.
    """
    from croak.focal import mixture_from_nodes

    polar = _mix(n_radial=6, n_azimuth=4)
    cart = mixture_from_nodes(
        x=polar.r * np.cos(polar.phi),
        y=polar.r * np.sin(polar.phi),
        area=polar.area,
        arms=polar.arms,
        hole_diameter=polar.hole_diameter,
        f_foc=polar.f_foc,
    )
    assert cart.p == pytest.approx(polar.p)
    assert cart.theta == pytest.approx(polar.theta)
    assert cart.r == pytest.approx(polar.r)

    with pytest.raises(ValueError, match=r"shape \(3, 2\)"):
        mixture_from_nodes(
            x=np.zeros(2),
            y=np.zeros(2),
            area=np.ones(2),
            arms=np.zeros((2, 2)),
            hole_diameter=1e-3,
            f_foc=0.1,
        )


def test_make_trace_fn_honours_focal_without_smearing():
    """``make_trace_fn(focal=...)`` must not silently fall through to the plain model.

    The mixture reaches retrieval through ``make_param_trace_fn``, so a ``focal=``
    dropped here was invisible: the single-delay entry point returned an unsmeared
    trace and said nothing.
    """
    from croak.forward_jax import make_param_trace_fn, make_trace_fn

    lam0 = 260e-9
    om0 = 2.0 * np.pi * C_LIGHT / lam0
    om = np.fft.fftshift(np.fft.fftfreq(32, d=0.6e-15)) * 2.0 * np.pi
    delays = np.linspace(-8e-15, 8e-15, 5)
    ew = np.exp(-((om / (2 * np.pi * 150e12)) ** 2)).astype(complex)
    mix = focal_mixture(
        hole_diameter=1e-3,
        hole_spacing=1e-3,
        f_foc=0.1,
        wavelength=lam0,
        n_radial=12,
        n_azimuth=8,
        r_max_units=4.0,
    )
    got = np.asarray(make_trace_fn(om, delays, "pg", omega0=om0, focal=mix)(ew))
    want = np.asarray(
        make_param_trace_fn(om, delays, "pg", omega0=om0, focal=mix)(ew, 0.0, 0.0)
    )
    assert np.array_equal(got, want)
    plain = np.asarray(make_trace_fn(om, delays, "pg", omega0=om0)(ew))
    assert not np.allclose(got / got.max(), plain / plain.max())
