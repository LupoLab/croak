"""Geometric time smearing: kernel statistics, the reduction, and the forward models."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.fft
from numpy.polynomial.hermite_e import hermegauss
from scipy.integrate import quad
from scipy.special import j1

from croak import Grid, maketrace, maketrace_jax
from croak.maths import wlfreq
from croak.pulses import gaussian_pulse
from croak.smearing import (
    AIRY6_RMS_COEFF,
    SmearingKernel,
    delay_frequency_grid,
    kernel_from_arms,
    square_boxcars_delay_width,
    square_boxcars_kernel,
    square_boxcars_spacing,
)

C_LIGHT = 299792458.0

# Reference DUV instrument: 1 mm holes, 0.5 mm edge-to-edge, 260 nm carrier.
HOLE_DIAMETER = 1.0e-3
HOLE_SPACING = 0.5e-3
WAVELENGTH = 260e-9
# Hole-centre offset along one Cartesian direction.
D_OFFSET = 0.5 * (HOLE_SPACING + HOLE_DIAMETER)

# Mask hole positions in each interaction's role order, for a folded square BOXCARS
# with the signal at (-d, -d) and therefore the conjugated arm at (+d, +d).
ARMS = {
    "pg": ((-D_OFFSET, D_OFFSET), (D_OFFSET, -D_OFFSET), (D_OFFSET, D_OFFSET)),
    "sd": ((D_OFFSET, -D_OFFSET), (-D_OFFSET, D_OFFSET), (D_OFFSET, D_OFFSET)),
}
# Which arms carry the scanned delay, as multipliers of tau in role order.
MECHANICAL_DELAY = {"pg": (0.0, 1.0, 1.0), "sd": (0.0, 0.0, 1.0)}


@pytest.fixture
def setup():
    """A small chirped test pulse on a coarse grid, with a uniform delay axis."""
    g = Grid(96, dt=0.6e-15)
    ew = gaussian_pulse(g, 3.0e-15) * np.exp(0.4j * (g.omega / g.omega.max() * 6) ** 2)
    delays = np.linspace(-25e-15, 25e-15, 51)
    return g, ew, delays


def reference_kernel(interaction):
    """The square-BOXCARS kernel for the reference instrument."""
    return square_boxcars_kernel(
        interaction,
        hole_diameter=HOLE_DIAMETER,
        hole_spacing=HOLE_SPACING,
        wavelength=WAVELENGTH,
    )


# --- kernel statistics -------------------------------------------------------------


def test_airy6_rms_coefficient_matches_quadrature():
    # sigma_r = C lambda f / D with C = sqrt(<|u|^2>/2)/pi and the focal weight
    # |2 J1(u)/u|^6 (the product of three focal amplitudes, squared).
    def g(u):
        return 1.0 if u == 0.0 else (2.0 * j1(u) / u) ** 6

    num = quad(lambda u: u**3 * g(u), 0, np.inf, limit=800)[0]
    den = quad(lambda u: u * g(u), 0, np.inf, limit=800)[0]
    coeff = np.sqrt(num / den / 2.0) / np.pi
    assert coeff == pytest.approx(AIRY6_RMS_COEFF, rel=1e-9)


@pytest.mark.parametrize(
    ("interaction", "coeff_p", "coeff_delta", "rho"),
    [
        # sigma = coeff * (d/D) * lambda / c. See docs/explanation/forward_model.md.
        ("pg", 2.0 * AIRY6_RMS_COEFF, np.sqrt(5.0) * AIRY6_RMS_COEFF, 1 / np.sqrt(5.0)),
        ("sd", 2 * np.sqrt(2) * AIRY6_RMS_COEFF, np.sqrt(2) * AIRY6_RMS_COEFF, 0.0),
    ],
)
def test_square_boxcars_kernel_matches_closed_form(
    interaction, coeff_p, coeff_delta, rho
):
    k = reference_kernel(interaction)
    unit = (D_OFFSET / HOLE_DIAMETER) * WAVELENGTH / C_LIGHT
    assert k.sigma_p == pytest.approx(coeff_p * unit, rel=1e-12)
    assert k.sigma_delta == pytest.approx(coeff_delta * unit, rel=1e-12)
    assert k.rho == pytest.approx(rho, abs=1e-12)


def test_square_boxcars_sd_reproduces_design_note_ratio():
    """For the SD layout the two parameters decouple and ``sigma_p == 2 sigma_delta``.

    This is the special structure of the design note: with the *conjugated* arm on the
    delay stage the coefficient vectors of ``p`` and ``delta`` are orthogonal. It does
    **not** hold for the PG layout, where the delay sits on an unconjugated arm.
    """
    k = reference_kernel("sd")
    assert k.rho == pytest.approx(0.0, abs=1e-12)
    assert k.sigma_p == pytest.approx(2.0 * k.sigma_delta, rel=1e-12)
    assert k.sigma_delta * 1e15 == pytest.approx(0.2260, abs=5e-4)


def test_square_boxcars_pg_is_correlated_and_wider():
    """The PG (ModelPNPS) layout gives a correlated, wider delay kernel."""
    pg, sd = reference_kernel("pg"), reference_kernel("sd")
    assert pg.rho == pytest.approx(1 / np.sqrt(5.0), rel=1e-12)
    assert pg.sigma_delta / sd.sigma_delta == pytest.approx(np.sqrt(2.5), rel=1e-12)
    assert pg.sigma_delta * 1e15 == pytest.approx(0.3573, abs=5e-4)


def test_square_boxcars_kernel_matches_explicit_arms():
    for interaction, arms in ARMS.items():
        expected = reference_kernel(interaction)
        got = kernel_from_arms(
            *arms,
            interaction=interaction,
            hole_diameter=HOLE_DIAMETER,
            wavelength=WAVELENGTH,
        )
        assert got == expected


def test_square_boxcars_probe_corner_choice_is_immaterial():
    """The square's mirror symmetry makes the two probe/gate assignments equivalent."""
    swapped = kernel_from_arms(
        (D_OFFSET, -D_OFFSET),
        (-D_OFFSET, D_OFFSET),
        (D_OFFSET, D_OFFSET),
        interaction="pg",
        hole_diameter=HOLE_DIAMETER,
        wavelength=WAVELENGTH,
    )
    assert swapped == reference_kernel("pg")


def test_kernel_width_scales_with_mask_ratio_not_focal_length():
    """sigma is linear in d/D; the focal length never enters the API at all."""
    wide = square_boxcars_kernel(
        "pg",
        hole_diameter=HOLE_DIAMETER,
        hole_spacing=1.5e-3,
        wavelength=WAVELENGTH,
    )
    narrow = reference_kernel("pg")
    d_wide = 0.5 * (1.5e-3 + HOLE_DIAMETER)
    assert wide.sigma_delta / narrow.sigma_delta == pytest.approx(d_wide / D_OFFSET)
    assert wide.rho == pytest.approx(narrow.rho)


def test_square_boxcars_spacing_inverts_the_delay_width():
    sigma = square_boxcars_delay_width(
        "pg",
        hole_diameter=HOLE_DIAMETER,
        hole_spacing=HOLE_SPACING,
        wavelength=WAVELENGTH,
    )
    assert sigma == pytest.approx(reference_kernel("pg").sigma_delta)
    spacing = square_boxcars_spacing(
        "pg", hole_diameter=HOLE_DIAMETER, sigma_delta=sigma, wavelength=WAVELENGTH
    )
    assert spacing == pytest.approx(HOLE_SPACING, rel=1e-12)


def test_reverse_delay_negates_the_correlation():
    forward = reference_kernel("pg")
    reversed_ = square_boxcars_kernel(
        "pg",
        hole_diameter=HOLE_DIAMETER,
        hole_spacing=HOLE_SPACING,
        wavelength=WAVELENGTH,
        reverse_delay=True,
    )
    assert reversed_.rho == pytest.approx(-forward.rho)
    assert reversed_.sigma_p == pytest.approx(forward.sigma_p)


def test_nodes_are_antisymmetric_and_weights_normalised():
    k = SmearingKernel(sigma_p=0.4e-15, sigma_delta=0.3e-15, rho=0.5, npoints=7)
    p, w = k.nodes_weights()
    # Exact antisymmetry is what lets the forward model reuse each shifted field twice.
    assert np.array_equal(p, -p[::-1])
    assert np.array_equal(w, w[::-1])
    assert w.sum() == pytest.approx(1.0)
    # Second moment of the quadrature reproduces sigma_p.
    assert np.sqrt(np.sum(w * p**2)) == pytest.approx(k.sigma_p, rel=1e-12)


def test_degenerate_shape_parameter_collapses_to_one_node():
    k = SmearingKernel(sigma_p=0.0, sigma_delta=0.3e-15, npoints=5)
    p, w = k.nodes_weights()
    assert p.shape == (1,)
    assert w == pytest.approx(np.ones(1))
    mu, sigma = k.conditional_delay(p)
    assert mu == pytest.approx(np.zeros(1))
    assert sigma == pytest.approx(k.sigma_delta)


def test_conditional_delay_matches_bivariate_normal():
    k = SmearingKernel(sigma_p=0.4e-15, sigma_delta=0.3e-15, rho=0.6, npoints=5)
    p, _ = k.nodes_weights()
    mu, sigma = k.conditional_delay(p)
    assert mu == pytest.approx(k.rho * (k.sigma_delta / k.sigma_p) * p)
    assert sigma == pytest.approx(k.sigma_delta * np.sqrt(1 - k.rho**2))
    # The conditional decomposition must reproduce the marginal variance of delta.
    _, w = k.nodes_weights()
    assert np.sqrt(np.sum(w * mu**2) + sigma**2) == pytest.approx(k.sigma_delta)


def test_scaled_multiplies_both_widths():
    k = reference_kernel("pg").scaled(2.5)
    base = reference_kernel("pg")
    assert k.sigma_p == pytest.approx(2.5 * base.sigma_p)
    assert k.sigma_delta == pytest.approx(2.5 * base.sigma_delta)
    assert k.rho == pytest.approx(base.rho)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sigma_p": -1e-15, "sigma_delta": 0.0}, "non-negative"),
        ({"sigma_p": 1e-15, "sigma_delta": -1e-15}, "non-negative"),
        ({"sigma_p": 1e-15, "sigma_delta": 1e-15, "rho": 1.0}, "rho"),
        ({"sigma_p": 1e-15, "sigma_delta": 1e-15, "npoints": 0}, "npoints"),
    ],
)
def test_kernel_rejects_invalid_parameters(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SmearingKernel(**kwargs)


def test_scaled_rejects_negative_factor():
    with pytest.raises(ValueError, match="non-negative"):
        reference_kernel("pg").scaled(-1.0)


def test_kernel_from_arms_rejects_shg_and_bad_geometry():
    with pytest.raises(ValueError, match="three-arm"):
        kernel_from_arms(
            (0.0, 0.0),
            (1e-3, 0.0),
            (0.0, 1e-3),
            interaction="shg",
            hole_diameter=HOLE_DIAMETER,
            wavelength=WAVELENGTH,
        )
    with pytest.raises(ValueError, match="hole_diameter"):
        kernel_from_arms(
            *ARMS["pg"], interaction="pg", hole_diameter=0.0, wavelength=WAVELENGTH
        )
    with pytest.raises(ValueError, match="wavelength"):
        kernel_from_arms(
            *ARMS["pg"], interaction="pg", hole_diameter=HOLE_DIAMETER, wavelength=0.0
        )
    with pytest.raises(ValueError, match="2-vector"):
        kernel_from_arms(
            (0.0, 0.0, 0.0),
            (1e-3, 0.0),
            (0.0, 1e-3),
            interaction="pg",
            hole_diameter=HOLE_DIAMETER,
            wavelength=WAVELENGTH,
        )


def test_square_boxcars_rejects_negative_spacing_and_shg():
    with pytest.raises(ValueError, match="hole_spacing"):
        square_boxcars_kernel(
            "pg",
            hole_diameter=HOLE_DIAMETER,
            hole_spacing=-1e-3,
            wavelength=WAVELENGTH,
        )
    with pytest.raises(ValueError, match="three-arm"):
        square_boxcars_kernel(
            "shg",
            hole_diameter=HOLE_DIAMETER,
            hole_spacing=HOLE_SPACING,
            wavelength=WAVELENGTH,
        )


# --- the delay axis ----------------------------------------------------------------


def test_delay_frequency_grid_requires_a_usable_delay_axis():
    good = np.linspace(-25e-15, 25e-15, 51)
    omega = delay_frequency_grid(good, 0.3e-15)
    assert omega.shape == good.shape
    with pytest.raises(ValueError, match="at least two points"):
        delay_frequency_grid(np.array([0.0]), 0.3e-15)
    with pytest.raises(ValueError, match="uniformly spaced"):
        delay_frequency_grid(np.array([0.0, 1e-15, 3e-15]), 0.3e-15)
    with pytest.raises(ValueError, match="too narrow"):
        delay_frequency_grid(np.linspace(-1e-15, 1e-15, 11), 0.5e-15)


# --- the reduction ------------------------------------------------------------------

FOCAL_LENGTH = (
    0.1  # arbitrary: the model is focal-length independent, so it must cancel
)


def focal_plane_reference(g, ew, delays, interaction, positions, weights):
    """Brute-force incoherent sum of the three-replica signal over the focal plane.

    Builds the local signal directly as ``E(t-s1) E(t-s2) conj(E(t-s3))`` with each arm
    carrying its own pulse-front tilt ``alpha_j . r`` on top of its mechanical delay,
    and accumulates ``|FT|^2`` over the supplied focal-plane quadrature. This is the
    definition the ``(p, delta)`` reduction is supposed to reproduce, computed without
    using any part of :mod:`croak.smearing`.

    Parameters
    ----------
    positions : numpy.ndarray
        ``(K, 2)`` focal-plane sample points (m).
    weights : numpy.ndarray
        ``(K,)`` quadrature weights, summing to 1.
    """
    alpha = [
        -np.asarray(r, float) / (FOCAL_LENGTH * C_LIGHT) for r in ARMS[interaction]
    ]
    mech = MECHANICAL_DELAY[interaction]
    omega_bin = scipy.fft.ifftshift(g.omega)
    ew_bin = scipy.fft.ifftshift(ew)

    def replica(shift):
        """``E(t - shift)`` in the time domain, thin medium."""
        return np.asarray(scipy.fft.fft(ew_bin * np.exp(1j * omega_bin * shift)))

    out = np.zeros((g.omega.size, delays.size))
    for r, weight in zip(positions, weights, strict=True):
        tilt = [a @ r for a in alpha]
        for j, tau in enumerate(delays):
            shifts = [mech[k] * tau + tilt[k] for k in range(3)]
            sig = replica(shifts[0]) * replica(shifts[1]) * np.conj(replica(shifts[2]))
            psi = scipy.fft.fftshift(scipy.fft.ifft(sig))
            out[:, j] += weight * np.abs(psi) ** 2
    return out


def gaussian_focal_nodes(wavelength, n=25):
    """Product Gauss-Hermite nodes for the isotropic Gaussian focal weight."""
    sigma_r = AIRY6_RMS_COEFF * wavelength * FOCAL_LENGTH / HOLE_DIAMETER
    x, w = hermegauss(n)
    x, w = sigma_r * x, w / np.sqrt(2 * np.pi)
    positions = np.array([(xi, yi) for xi in x for yi in x])
    weights = np.array([wi * wj for wi in w for wj in w])
    return positions, weights


def airy6_focal_nodes(wavelength, n_radial=200, n_angle=12, u_max=40.0):
    """Polar quadrature nodes for the exact ``|2 J1(u)/u|**6`` focal weight.

    The Airy amplitude oscillates (zeros at ``u = 3.83, 7.02, ...``) and its sixth power
    decays only as ``u**-9``, so the radial rule needs both fine sampling and a long
    tail. ``u = pi D r / (lambda f)``.
    """
    x, w = np.polynomial.legendre.leggauss(n_radial)
    u = 0.5 * u_max * (x + 1.0)
    du = 0.5 * u_max * w
    amp = np.where(u == 0.0, 1.0, 2.0 * j1(u) / np.where(u == 0.0, 1.0, u))
    radial_weight = amp**6 * u * du  # the Jacobian r dr becomes u du
    scale = wavelength * FOCAL_LENGTH / (np.pi * HOLE_DIAMETER)
    phi = 2.0 * np.pi * np.arange(n_angle) / n_angle
    positions = np.array(
        [(scale * ui * np.cos(p), scale * ui * np.sin(p)) for ui in u for p in phi]
    )
    weights = np.array([rw / n_angle for rw in radial_weight for _ in phi])
    return positions, weights / weights.sum()


@pytest.mark.parametrize("interaction", ["pg", "sd"])
def test_smeared_trace_matches_focal_plane_quadrature(setup, interaction):
    """The whole chain against a brute-force incoherent sum over the focal spot.

    Agreement validates the ``(p, delta)`` reduction, the geometry to covariance map,
    the ``p`` quadrature and the exact ``delta`` convolution all at once — and would
    fail if the arms were assigned to the wrong roles.
    """
    g, ew, delays = setup
    positions, weights = gaussian_focal_nodes(WAVELENGTH)
    reference = focal_plane_reference(g, ew, delays, interaction, positions, weights)
    kernel = square_boxcars_kernel(
        interaction,
        hole_diameter=HOLE_DIAMETER,
        hole_spacing=HOLE_SPACING,
        wavelength=WAVELENGTH,
        npoints=15,
    )
    model = maketrace(
        g.omega, delays, ew, interaction, smearing=kernel, normalize=False
    )
    assert np.max(np.abs(model - reference)) / reference.max() < 1e-6


@pytest.mark.parametrize("interaction", ["pg", "sd"])
def test_gaussian_focal_weight_approximates_the_airy_weight(interaction):
    """Quantify the one modelling approximation: Gaussian instead of ``|Airy|**6``.

    The kernel matches the exact focal weight's second moments, and the leading trace
    correction is quadratic in ``(p, delta)``, so the two agree to first order. The
    ``|Airy|**6`` weight has heavier tails, and this test pins how much that is worth:
    about one part in ``1e5`` of the trace peak for the reference geometry, i.e. orders
    of magnitude below any realistic measurement noise.
    """
    g = Grid(48, dt=1.2e-15)
    # Chirped: for a transform-limited (real) field the PG and SD operators coincide,
    # so an unchirped pulse would not distinguish the two layouts at all.
    ew = gaussian_pulse(g, 3.0e-15) * np.exp(0.5j * (g.omega / g.omega.max() * 6) ** 2)
    delays = np.linspace(-20e-15, 20e-15, 13)
    gauss = focal_plane_reference(
        g, ew, delays, interaction, *gaussian_focal_nodes(WAVELENGTH)
    )
    airy = focal_plane_reference(
        g, ew, delays, interaction, *airy6_focal_nodes(WAVELENGTH)
    )
    residual = np.max(np.abs(gauss - airy)) / airy.max()
    assert residual < 1e-4


# --- the forward models -------------------------------------------------------------


@pytest.mark.parametrize("interaction", ["pg", "sd"])
@pytest.mark.parametrize("dispersive", [False, True])
def test_zero_kernel_reproduces_the_unsmeared_trace(setup, interaction, dispersive):
    g, ew, delays = setup
    kw = (
        dict(material="SiO2", thickness=15e-6, npoints=16, omega0=wlfreq(800e-9))
        if dispersive
        else {}
    )
    base = maketrace(g.omega, delays, ew, interaction, **kw)
    zero = maketrace(
        g.omega,
        delays,
        ew,
        interaction,
        smearing=SmearingKernel(sigma_p=0.0, sigma_delta=0.0),
        **kw,
    )
    assert np.allclose(base, zero, rtol=0.0, atol=1e-14)


@pytest.mark.parametrize("interaction", ["pg", "sd"])
@pytest.mark.parametrize("dispersive", [False, True])
def test_jax_trace_matches_numpy_smeared(setup, interaction, dispersive):
    g, ew, delays = setup
    kw = (
        dict(material="SiO2", thickness=15e-6, npoints=16, omega0=wlfreq(800e-9))
        if dispersive
        else {}
    )
    kernel = square_boxcars_kernel(
        interaction,
        hole_diameter=HOLE_DIAMETER,
        hole_spacing=HOLE_SPACING,
        wavelength=800e-9,
    )
    a = maketrace(g.omega, delays, ew, interaction, smearing=kernel, **kw)
    b = maketrace_jax(g.omega, delays, ew, interaction, smearing=kernel, **kw)
    assert np.allclose(a, b, atol=1e-10, rtol=1e-10)


def _delay_marginal_moments(trace, delays):
    """Total, mean and variance of the delay marginal ``sum_omega T(omega, tau)``."""
    m = trace.sum(axis=0)
    total = float(m.sum())
    weights = m / total
    mean = float(weights @ delays)
    return total, mean, float(weights @ (delays - mean) ** 2)


@pytest.mark.parametrize("interaction", ["pg", "sd"])
def test_delay_blur_adds_exactly_sigma_delta_squared(setup, interaction):
    """With only the delay offset active the kernel is an exact, sharp invariant.

    ``delta`` translates the trace along ``tau``, so integrating over it convolves the
    delay marginal with a normalised Gaussian: the total is conserved and the variance
    grows by exactly ``sigma_delta**2``. ``sigma_p = 0`` isolates that from the shape
    parameter, which does *not* conserve the total (separating the two replicas reduces
    their overlap, so the signal genuinely weakens).
    """
    g, ew, delays = setup
    sigma_delta = 1.2e-15
    kernel = SmearingKernel(sigma_p=0.0, sigma_delta=sigma_delta)
    base = maketrace(g.omega, delays, ew, interaction, normalize=False)
    smeared = maketrace(
        g.omega, delays, ew, interaction, smearing=kernel, normalize=False
    )
    total_a, mean_a, var_a = _delay_marginal_moments(base, delays)
    total_b, mean_b, var_b = _delay_marginal_moments(smeared, delays)
    assert total_b == pytest.approx(total_a, rel=1e-12)
    assert mean_b == pytest.approx(mean_a, abs=1e-18)
    assert var_b - var_a == pytest.approx(sigma_delta**2, rel=2e-3)


@pytest.mark.parametrize("interaction", ["pg", "sd"])
def test_smearing_broadens_the_delay_marginal(setup, interaction):
    """The full kernel must widen the trace along tau — it is an instrument response."""
    g, ew, delays = setup
    kernel = square_boxcars_kernel(
        interaction,
        hole_diameter=HOLE_DIAMETER,
        hole_spacing=HOLE_SPACING,
        wavelength=800e-9,
    )
    base = maketrace(g.omega, delays, ew, interaction, normalize=False)
    smeared = maketrace(
        g.omega, delays, ew, interaction, smearing=kernel, normalize=False
    )
    _, _, var_a = _delay_marginal_moments(base, delays)
    _, _, var_b = _delay_marginal_moments(smeared, delays)
    assert var_b > var_a


def test_smearing_rejects_shg_and_nonuniform_delays(setup):
    g, ew, delays = setup
    kernel = SmearingKernel(sigma_p=0.3e-15, sigma_delta=0.3e-15)
    with pytest.raises(ValueError, match="three-arm"):
        maketrace(g.omega, delays, ew, "shg", smearing=kernel)
    with pytest.raises(ValueError, match="uniformly spaced"):
        maketrace(g.omega, np.sort(np.append(delays, 1e-15)), ew, "pg", smearing=kernel)


def test_hand_adjoint_refuses_smearing(setup):
    from croak.forward import ForwardModel

    g, ew, delays = setup
    model = ForwardModel(
        g.omega, delays, "pg", smearing=SmearingKernel(0.3e-15, 0.3e-15)
    )
    with pytest.raises(NotImplementedError, match="autodiff"):
        model.adjoint_single(np.zeros(g.omega.size, dtype=complex), 0.0)


# --- split (p, delta) channel scaling ------------------------------------------------


def test_split_scale_pair_matches_joint_scalar(setup):
    """A pair ``(s, s)`` must reproduce the single joint multiplier exactly."""
    import jax.numpy as jnp

    from croak.forward_jax import make_param_trace_fn

    g, ew, delays = setup
    trace = make_param_trace_fn(g.omega, delays, "pg", smearing=reference_kernel("pg"))
    joint = np.asarray(trace(jnp.asarray(ew), 0.0, 0.0, 1.3))
    pair = np.asarray(trace(jnp.asarray(ew), 0.0, 0.0, jnp.array([1.3, 1.3])))
    np.testing.assert_allclose(pair, joint, rtol=0.0, atol=1e-14)


def test_split_scale_channels_act_independently(setup):
    """Scaling p and delta separately must produce distinct traces (chirped pulse)."""
    import jax.numpy as jnp

    from croak.forward_jax import make_param_trace_fn

    g, ew, delays = setup
    trace = make_param_trace_fn(g.omega, delays, "pg", smearing=reference_kernel("pg"))
    t_p = np.asarray(trace(jnp.asarray(ew), 0.0, 0.0, jnp.array([1.5, 1.0])))
    t_d = np.asarray(trace(jnp.asarray(ew), 0.0, 0.0, jnp.array([1.0, 1.5])))
    assert np.max(np.abs(t_p - t_d)) > 1e-4


def test_augment_split_requires_fit_smearing_and_positive_centres():
    from croak._jax_pulse import ExtraParamSpec, augment

    with pytest.raises(ValueError, match="fit_smearing_split requires fit_smearing"):
        augment(np.zeros(4), ExtraParamSpec(fit_smearing_split=True))
    with pytest.raises(ValueError, match="smear_delta0"):
        augment(
            np.zeros(4),
            ExtraParamSpec(
                fit_smearing=True, fit_smearing_split=True, smear_delta0=0.0
            ),
        )


def test_augment_split_unpacks_both_channels():
    import jax.numpy as jnp

    from croak._jax_pulse import ExtraParamSpec, augment

    spec = ExtraParamSpec(
        fit_smearing=True, fit_smearing_split=True, smear_scale0=1.1, smear_delta0=0.9
    )
    aug = augment(np.zeros(6), spec)
    assert aug.u0.size == 8 and aug.idx_smear_delta == aug.idx_smear + 1
    u = aug.u0.copy()
    u[aug.idx_smear] = 0.2
    u[aug.idx_smear_delta] = -0.1
    _, _, _, smear = aug.unpack(jnp.asarray(u))
    np.testing.assert_allclose(np.asarray(smear), [1.3, 0.8], atol=1e-14)
    assert aug.physical(u)[2] == pytest.approx((1.3, 0.8))


def test_fit_smearing_split_recovers_channel_scales():
    """End-to-end: a chirped pulse's split-scaled kernel is recovered exactly.

    The pulse must be chirped: for a transform-limited gate the ``p`` channel
    only attenuates the signal (absorbed by ``mu``), leaving the trace shape
    unchanged, so ``scale_p`` would be unidentifiable (the report's "TL gate
    means p does nothing" degeneracy).
    """
    import jax.numpy as jnp

    from croak.forward_jax import make_param_trace_fn
    from croak.lbfgs_ad import LBFGSAD

    g = Grid(128, dt=0.5e-15)
    delays = np.linspace(-15e-15, 15e-15, 48)
    kernel = square_boxcars_kernel(
        "pg", hole_diameter=1e-3, hole_spacing=1.5e-3, wavelength=260e-9, npoints=5
    )
    ew = gaussian_pulse(g, 2.0e-15, phases=[8e-30])
    trace = make_param_trace_fn(g.omega, delays, "pg", smearing=kernel)
    t_meas = np.asarray(trace(jnp.asarray(ew), 0.0, 0.0, jnp.array([1.4, 0.8])))

    solver = LBFGSAD(
        maxiters=300, smearing=kernel, fit_smearing=True, fit_smearing_split=True
    )
    result = solver.run(t_meas, g.omega, delays, "pg", guess=ew)
    assert result.smear_scale == pytest.approx(1.4, abs=0.02)
    assert result.smear_scale_delta == pytest.approx(0.8, abs=0.02)


def test_joint_fit_reports_no_delta_scale():
    """Without the split flag the result's delta channel stays ``None``."""
    import jax.numpy as jnp

    from croak.forward_jax import make_param_trace_fn
    from croak.lbfgs_ad import LBFGSAD

    g = Grid(96, dt=0.6e-15)
    delays = np.linspace(-15e-15, 15e-15, 31)
    kernel = square_boxcars_kernel(
        "pg", hole_diameter=1e-3, hole_spacing=1.5e-3, wavelength=260e-9, npoints=5
    )
    ew = gaussian_pulse(g, 2.5e-15, phases=[6e-30])
    trace = make_param_trace_fn(g.omega, delays, "pg", smearing=kernel)
    t_meas = np.asarray(trace(jnp.asarray(ew), 0.0, 0.0, 0.9))

    solver = LBFGSAD(maxiters=150, smearing=kernel, fit_smearing=True)
    result = solver.run(t_meas, g.omega, delays, "pg", guess=ew)
    assert result.smear_scale == pytest.approx(0.9, abs=0.02)
    assert result.smear_scale_delta is None


def test_collinear_sd_kernel_has_no_p_channel():
    """A two-beam SD has sigma_p identically zero, not merely small.

    ``p`` is proportional to the difference of the two UNCONJUGATED arms' tilts,
    and in a two-beam self-diffraction those arms are the same beam, so the
    gate-shape channel does not exist. This matters because the boxcars closed
    form for ``"sd"`` assumes two separate unconjugated holes and returns a
    ``sigma_p`` of order 0.4 fs at this geometry — applying it to a two-beam
    measurement would model a blur the experiment does not have.
    """
    from croak.smearing import collinear_sd_kernel, square_boxcars_kernel

    D, gap, lam = 1.9e-3, 1.0e-3, 235e-9
    k = collinear_sd_kernel(hole_diameter=D, hole_spacing=gap, wavelength=lam)
    assert k.sigma_p == 0.0
    assert k.rho == 0.0
    # sigma_theta = 0.49135 (d/D) lambda/c
    d_over_D = 0.5 * (gap + D) / D
    assert k.sigma_delta == pytest.approx(
        0.49135 * d_over_D * lam / 299_792_458.0, rel=1e-4
    )
    # and it is genuinely different from the three-hole closed form
    boxcars = square_boxcars_kernel(
        "sd", hole_diameter=D, hole_spacing=gap, wavelength=lam
    )
    assert boxcars.sigma_p > 0.4e-15


def test_session_selects_the_two_beam_sd_kernel():
    """`smear_layout="sd2"` must reach the kernel builder, not just the params."""
    import numpy as np

    from croak.session.params import RetrieveParams
    from croak.session.pipeline import smearing_kernel

    class _TD:
        interaction = "sd"
        omega0_pulse = 2 * np.pi * 299_792_458.0 / 235e-9

    common = dict(smearing=True, smear_hole_diameter_mm=1.9, smear_hole_spacing_mm=1.0)
    sd2 = smearing_kernel(RetrieveParams(**common, smear_layout="sd2"), _TD())
    boxcars = smearing_kernel(RetrieveParams(**common), _TD())
    assert sd2.sigma_p == 0.0
    assert boxcars.sigma_p > 0.0
