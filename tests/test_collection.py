"""Tests for the finite collection aperture (``croak.collection``).

The load-bearing test is :func:`test_matches_a_dense_fourier_reference`: it builds the
focal-plane signal from the arm tilts DIRECTLY -- no ``(p, theta)`` anywhere -- takes a
dense 2-D FFT of it, windows it in k and integrates, sharing nothing with the production
path but the physical constants. That one comparison pins the derivation, both Fourier
sign conventions, the aperture centring and the absolute normalisation at once.
"""

from __future__ import annotations

import numpy as np
import pytest

from croak.collection import (
    CollectionAperture,
    aperture_offset,
    mask_hole_aperture,
    mask_transmission,
    resolve_tanh_width,
    transform_phases,
)
from croak.focal import _ARMS_PG, airy_amplitude, focal_mixture, mixture_from_nodes
from croak.forward_jax import make_param_trace_fn

C_LIGHT = 299_792_458.0

#: The reference instrument (04_production_gap1000_kerr_raman.jl): 1 mm holes at 1 mm
#: edge-to-edge gap, f = 100 mm, collected through a 0.5 mm hole at the signal corner
#: (-d, -d) in a mask plane 100 mm from the focus, tanh-apodised.
LAMBDA0 = 260e-9
OMEGA0 = 2.0 * np.pi * C_LIGHT / LAMBDA0
MASK_D = 1.0e-3
MASK_GAP = 1.0e-3
F_FOC = 0.1
ARM_OFFSET = 0.5 * (MASK_GAP + MASK_D)
HOLE_DIAM = 0.5e-3
Z_MASK = 0.1
#: The simulator's resolved default for this grid: 3 * dk * z * c / w0 = 96.87 um.
TANH_WIDTH = 9.687034277198214e-05

HOLE = dict(
    hole_x=-ARM_OFFSET,
    hole_y=-ARM_OFFSET,
    hole_diameter=HOLE_DIAM,
    z_mask=Z_MASK,
    apod="tanh",
    apod_param=TANH_WIDTH,
)


def _grid(n=48, dt=0.5e-15, n_delay=3, span=6e-15):
    """A small centred frequency grid, delay axis and ~2 fs Gaussian test pulse."""
    omega = np.fft.fftshift(np.fft.fftfreq(n, d=dt)) * 2.0 * np.pi
    delays = np.linspace(-span, span, n_delay)
    width = 2.0 * np.sqrt(2.0 * np.log(2.0)) / 2.0e-15
    ew = np.exp(-((omega / width) ** 2)).astype(complex)
    return omega, delays, ew


def _mixture(mode=None, n_radial=32, n_azimuth=24, r_max_units=4.0, **hole):
    """A BOXCARS mixture, optionally collected through the reference hole."""
    aperture = (
        None if mode is None else mask_hole_aperture(mode=mode, **{**HOLE, **hole})
    )
    return focal_mixture(
        hole_diameter=MASK_D,
        hole_spacing=MASK_GAP,
        f_foc=F_FOC,
        wavelength=LAMBDA0,
        n_radial=n_radial,
        n_azimuth=n_azimuth,
        r_max_units=r_max_units,
        collection=aperture,
    )


def _trace(mix, omega, delays, ew, normalize=True):
    """Evaluate the thin-medium PG trace of a mixture."""
    fn = make_param_trace_fn(
        omega, delays, "pg", omega0=OMEGA0, normalize=normalize, focal=mix
    )
    return np.asarray(fn(ew, 0.0, 0.0), dtype=float)


def _dense_reference(ew, omega, delays, *, n_grid=48, dr=4.5e-6, pad=3, mode):
    """Apertured trace from a dense Cartesian focal grid and a plain 2-D FFT.

    Independent of :mod:`croak.collection`: the three arm fields are built at each
    focal position from their own arrival offsets ``r_j . r / (f c)``, with the shift
    phase taken against the **absolute** frequency so each arm carries its physical
    carrier, and the product is transformed, windowed and summed directly.
    """
    omega = np.asarray(omega, dtype=float)
    omega_abs = omega + OMEGA0
    omega_bin = np.fft.ifftshift(omega)
    ew_bin = np.fft.ifftshift(np.asarray(ew, dtype=complex))

    axis = (np.arange(n_grid) - n_grid // 2) * dr
    gx, gy = np.meshgrid(axis, axis, indexing="ij")
    rx, ry = gx.ravel(), gy.ravel()
    filt_bin = np.fft.ifftshift(
        airy_amplitude(
            MASK_D
            * np.hypot(rx, ry)[:, None]
            * omega_abs[None, :]
            / (2 * F_FOC * C_LIGHT)
        ),
        axes=1,
    )
    alpha = np.array(_ARMS_PG) * ARM_OFFSET / (F_FOC * C_LIGHT)
    dt = np.array([a[0] * rx + a[1] * ry for a in alpha])

    def arm(shift):
        phase = np.exp(1j * (omega_bin + OMEGA0)[None, :] * shift[:, None])
        return np.fft.fft(ew_bin[None, :] * filt_bin * phase, axis=1)

    psis = np.empty((len(delays), rx.size, omega.size), dtype=complex)
    for i, tau in enumerate(delays):
        a1, a2, a3 = arm(dt[0]), arm(tau + dt[1]), arm(tau + dt[2])
        psis[i] = np.fft.fftshift(np.fft.ifft(a1 * a2 * np.conj(a3), axis=1), axes=1)

    n_pad = pad * n_grid
    lo = (n_pad - n_grid) // 2
    field = np.zeros((len(delays), n_pad, n_pad, omega.size), dtype=complex)
    field[:, lo : lo + n_grid, lo : lo + n_grid, :] = psis.reshape(
        len(delays), n_grid, n_grid, omega.size
    )
    sk = (
        np.fft.fftshift(
            np.fft.fft2(np.fft.ifftshift(field, axes=(1, 2)), axes=(1, 2)), axes=(1, 2)
        )
        * dr**2
    )
    kax = np.fft.fftshift(np.fft.fftfreq(n_pad, d=dr)) * 2.0 * np.pi
    kxg, kyg = np.meshgrid(kax, kax, indexing="ij")
    # A wavevector maps to the mask-plane position x = k z c / w (ModelPNPS.makemask).
    scale = Z_MASK * C_LIGHT / omega_abs
    radius = np.hypot(
        kxg[:, :, None] * scale[None, None, :] + ARM_OFFSET,
        kyg[:, :, None] * scale[None, None, :] + ARM_OFFSET,
    )
    w = mask_transmission(
        radius, hole_diameter=HOLE_DIAM, apod="tanh", apod_param=TANH_WIDTH
    )
    dk = (kax[1] - kax[0]) ** 2 / (2.0 * np.pi) ** 2
    if mode == "reimaged":
        return (np.abs(np.sum(sk * w[None], axis=(1, 2)) * dk) ** 2).T
    return (np.sum(np.abs(sk) ** 2 * w[None] ** 2, axis=(1, 2)) * dk).T


# --- the window itself --------------------------------------------------------------


def test_mask_transmission_matches_the_simulator():
    """The three apodisations reproduce ``ModelPNPS.makemask`` at the obvious radii."""
    r = np.array([0.0, 0.5 * HOLE_DIAM, HOLE_DIAM])
    hard = mask_transmission(r, hole_diameter=HOLE_DIAM, apod="hard")
    assert hard.tolist() == [1.0, 1.0, 0.0]

    sg = mask_transmission(r, hole_diameter=HOLE_DIAM, apod="supergauss")
    assert sg == pytest.approx([1.0, np.exp(-1.0), np.exp(-(2.0**16))], abs=1e-12)

    tanh = mask_transmission(
        r, hole_diameter=HOLE_DIAM, apod="tanh", apod_param=TANH_WIDTH
    )
    expected = 0.5 * (1.0 - np.tanh((r - 0.5 * HOLE_DIAM) / TANH_WIDTH))
    assert tanh == pytest.approx(expected)
    # The reference edge is 19% of the diameter: soft enough that a top hat is wrong
    # by a factor of two at the rim and still passes 0.6% at r = D.
    assert tanh[0] == pytest.approx(0.9943, abs=1e-4)
    assert tanh[2] == pytest.approx(0.0057, abs=1e-4)


def test_resolve_tanh_width_reproduces_the_simulator_default():
    """3 dk z c / w0, with w0 the GRID POINT nearest the carrier -- not 2 pi c / l0."""
    delta_k = 7803.260441107285
    width = resolve_tanh_width(delta_k=delta_k, z_mask=Z_MASK, omega_reference=OMEGA0)
    assert width == pytest.approx(3.0 * delta_k * Z_MASK * C_LIGHT / OMEGA0)
    # On the reference file's own grid the carrier IS a grid point, giving 96.87 um.
    assert width == pytest.approx(TANH_WIDTH, rel=1e-12)
    with pytest.raises(ValueError, match="must all be positive"):
        resolve_tanh_width(delta_k=delta_k, z_mask=Z_MASK, omega_reference=0.0)


def test_aperture_quadrature_integrates_the_hole():
    """The mask-plane quadrature reproduces the hole's transmitted area.

    A check of the polar quadrature independent of any field: the weights and the
    transmission must integrate ``W^2`` over the disc. The default node count is good
    to ~1e-4 here -- which the trace comparisons show is ample -- and a denser one
    converges, confirming the split-panel construction is accurate rather than merely
    close.
    """
    r = np.linspace(0.0, 2.5 * 0.5 * HOLE_DIAM, 200001)
    w = mask_transmission(
        r, hole_diameter=HOLE_DIAM, apod="tanh", apod_param=TANH_WIDTH
    )
    exact = float(2.0 * np.pi * np.trapezoid(w**2 * r, r))

    default = mask_hole_aperture(**HOLE)
    assert float(np.sum(default.weight * default.transmission**2)) == pytest.approx(
        exact, rel=1e-3
    )
    dense = mask_hole_aperture(n_radial=24, n_azimuth=8, **HOLE)
    assert float(np.sum(dense.weight * dense.transmission**2)) == pytest.approx(
        exact, rel=1e-9
    )


# --- geometry -----------------------------------------------------------------------


def test_aperture_sits_on_the_phase_matched_direction():
    """A BOXCARS instrument collecting at the signal corner has zero k offset.

    Phase matching sends the signal to ``r1 + r2 - r3``, the fourth corner, and that is
    where the collection hole is drilled -- so in the carrier-removed frame the aperture
    is centred at k = 0. Asserting it rather than hardcoding it catches an arm
    permutation, which is the failure `reduced_moments` guards against for the kernel.
    """
    mix = _mixture(mode="integrated")
    assert mix.signal_position == pytest.approx([-ARM_OFFSET, -ARM_OFFSET])
    assert aperture_offset(mix) == pytest.approx([0.0, 0.0], abs=1e-15)

    moved = _mixture(mode="integrated", hole_x=-ARM_OFFSET + 1e-4)
    assert aperture_offset(moved) == pytest.approx([1e-4, 0.0], abs=1e-12)


def test_transform_phase_removes_the_signal_carrier():
    """The reinstated phase is w_centred * p, i.e. w_abs * p minus the carrier term.

    The centred/absolute distinction is the one that cannot be argued from the code:
    croak applies every shift as ``exp(i w s)`` on a centred spectrum, so the three
    replicas together drop ``exp(-i w0 p)`` -- invisible to ``|Psi|^2``, fatal to a
    coherent sum.
    """
    mix = _mixture(mode="integrated", n_radial=4, n_azimuth=4)
    omega = np.linspace(-1e15, 1e15, 5)
    ph = transform_phases(mix, omega, OMEGA0)
    # At the aperture centre the node's ramp vanishes and only the p term survives.
    centre = int(np.argmax(mix.collection.transmission))
    total = (omega + OMEGA0)[None, :] * ph.chromatic_delay[centre][
        :, None
    ] - ph.static_phase[centre][:, None]
    offset = aperture_offset(mix)
    assert np.allclose(offset, 0.0, atol=1e-15)
    rx = mix.r * np.cos(mix.phi)
    ry = mix.r * np.sin(mix.phi)
    ramp = (
        (mix.collection.x[centre] + ARM_OFFSET) * rx
        + (mix.collection.y[centre] + ARM_OFFSET) * ry
    ) / (C_LIGHT * Z_MASK)
    expected = (
        omega[None, :] * mix.p[:, None] - (omega + OMEGA0)[None, :] * ramp[:, None]
    )
    assert total == pytest.approx(expected, rel=1e-10, abs=1e-9)


# --- correctness --------------------------------------------------------------------


def test_parseval_is_exact_over_a_full_k_grid():
    """An unbounded aperture reproduces the incoherent sum EXACTLY, not asymptotically.

    On a Cartesian focal grid the DFT Parseval identity is exact, so summing
    ``|S~|^2 dk / (2 pi)^2`` over the whole reciprocal grid must return
    ``sum_k a_k |Psi_k|^2`` to round-off. This is the normalisation test: getting the
    1/(2 pi)^2 or the area weights wrong shows up here and nowhere else.
    """
    omega, delays, ew = _grid(n=32, n_delay=3)
    n_grid, dr = 24, 6.0e-6
    axis = (np.arange(n_grid) - n_grid // 2) * dr
    gx, gy = np.meshgrid(axis, axis, indexing="ij")
    arms = np.array(_ARMS_PG) * ARM_OFFSET
    common = dict(
        x=gx.ravel(),
        y=gy.ravel(),
        area=np.full(gx.size, dr**2),
        arms=arms,
        hole_diameter=MASK_D,
        f_foc=F_FOC,
    )
    kax = np.fft.fftfreq(n_grid, d=dr) * 2.0 * np.pi
    kx, ky = np.meshgrid(kax, kax, indexing="ij")
    full_k = CollectionAperture(
        x=kx.ravel(),
        y=ky.ravel(),
        weight=np.full(kx.size, (kax[1] - kax[0]) ** 2),
        transmission=np.ones(kx.size),
        z_mask=Z_MASK,
        chromatic=False,
    )
    incoherent = _trace(
        mixture_from_nodes(**common), omega, delays, ew, normalize=False
    )
    apertured = _trace(
        mixture_from_nodes(collection=full_k, **common),
        omega,
        delays,
        ew,
        normalize=False,
    )
    assert apertured == pytest.approx(incoherent, rel=1e-10)


@pytest.mark.parametrize("mode", ["integrated", "reimaged"])
def test_matches_a_dense_fourier_reference(mode):
    """The quadrature model reproduces a dense 2-D FFT of the exact focal field.

    Absolute, with no fitted scale: the reference shares no code with the production
    path, so agreement fixes the derivation, both transform sign conventions, the
    chromatic aperture mapping and the normalisation simultaneously.
    """
    omega, delays, ew = _grid(n=48, n_delay=3)
    ref = _dense_reference(ew, omega, delays, mode=mode)
    got = _trace(
        _mixture(mode=mode, r_max_units=4.0), omega, delays, ew, normalize=False
    )
    assert got.max() / ref.max() == pytest.approx(1.0, rel=2e-2)
    assert np.sqrt(np.mean((got - ref) ** 2)) / ref.max() < 2e-3


def test_collection_none_is_the_incoherent_sum():
    """The default path is untouched: no aperture means the model croak always had."""
    omega, delays, ew = _grid()
    mix = _mixture()
    assert mix.collection is None
    psis_sum = _trace(mix, omega, delays, ew, normalize=False)
    fn = make_param_trace_fn(omega, delays, "pg", omega0=OMEGA0, focal=mix)
    assert np.array_equal(np.asarray(fn(ew, 0.0, 0.0), dtype=float), psis_sum)


def test_quadrature_converged_at_default_resolution():
    """Default aperture nodes reproduce a far denser quadrature.

    The coherent integrand is NOT the incoherent one, so the mixture's own convergence
    (``tests/test_focal.py``, tuned for the |A|^6 weight) does not carry over. Measured
    on the reference geometry with a 1 fs pulse, ``n_azimuth = 8`` -- what the paper's
    diagnostics used, and converged there for the incoherent sum -- costs 4e-5 of peak
    against 3e-7 at 16. The ``focal_mixture`` default of 16 is enough; that the coarser
    setting is not is a change of guidance the collection model forces. (The margin
    shrinks with bandwidth, so this test asserts the default's convergence rather than
    the coarse setting's failure, which is pulse-dependent.)
    """
    omega, delays, ew = _grid(n=48, n_delay=5)
    for mode in ("integrated", "reimaged"):
        dense = focal_mixture(
            hole_diameter=MASK_D,
            hole_spacing=MASK_GAP,
            f_foc=F_FOC,
            wavelength=LAMBDA0,
            n_radial=48,
            n_azimuth=32,
            r_max_units=5.0,
            collection=mask_hole_aperture(
                n_radial=12, n_azimuth=32, pad=3.0, mode=mode, **HOLE
            ),
        )
        ref = _trace(dense, omega, delays, ew)
        got = _trace(_mixture(mode=mode, n_radial=24, n_azimuth=16), omega, delays, ew)
        assert np.sqrt(np.mean((got - ref) ** 2)) / ref.max() < 1e-4


def test_aperture_narrows_the_effective_blur():
    """Collecting through a hole makes the instrument's blur NARROWER, monotonically.

    The reduced kernel is a full-collection (incoherent) average, so under an aperture
    the trace prefers a kernel below the geometric width -- as measured on the 3D
    reference traces (1.10 full, 0.80-0.90 apertured). The ordering
    full >= integrated >= reimaged is the model reproducing that from geometry alone.
    """
    from croak.smearing import square_boxcars_kernel

    omega, delays, ew = _grid(n=64, dt=0.4e-15, n_delay=21, span=12e-15)
    scales = np.arange(0.6, 1.35, 0.05)

    def best_scale(meas):
        errors = []
        for s in scales:
            kern = square_boxcars_kernel(
                "pg",
                hole_diameter=MASK_D,
                hole_spacing=MASK_GAP,
                wavelength=LAMBDA0,
                npoints=11,
            ).scaled(float(s))
            fn = make_param_trace_fn(
                omega, delays, "pg", omega0=OMEGA0, normalize=True, smearing=kern
            )
            model = np.asarray(fn(ew, 0.0, 0.0), dtype=float)
            mu = float(np.sum(meas * model) / np.sum(model * model))
            errors.append(np.sqrt(np.mean((meas - mu * model) ** 2)))
        return float(scales[int(np.argmin(errors))])

    full = best_scale(_trace(_mixture(), omega, delays, ew))
    apertured = best_scale(_trace(_mixture(mode="integrated"), omega, delays, ew))
    reimaged = best_scale(_trace(_mixture(mode="reimaged"), omega, delays, ew))
    assert full >= apertured >= reimaged
    assert reimaged < full


def test_collection_is_differentiable_and_jittable():
    """``focal=`` with an aperture stays usable inside AD -- the point of the model."""
    import jax
    import jax.numpy as jnp

    omega, delays, ew = _grid(n=32, n_delay=3)
    mix = _mixture(mode="integrated", n_radial=12, n_azimuth=8)
    fn = make_param_trace_fn(
        omega, delays, "pg", omega0=OMEGA0, focal=mix, normalize=False
    )

    def loss(scale):
        return jnp.sum(fn(jnp.asarray(ew) * scale, 0.0, 0.0))

    grad = float(jax.grad(loss)(1.0))
    h = 1e-4
    fd = (float(loss(1.0 + h)) - float(loss(1.0 - h))) / (2.0 * h)
    assert grad == pytest.approx(fd, rel=1e-4)
    # The trace is homogeneous of degree 6 in the field, so d/ds at s = 1 is 6 T.
    assert grad == pytest.approx(6.0 * float(loss(1.0)), rel=1e-6)
    assert np.allclose(
        np.asarray(jax.jit(fn)(ew, 0.0, 0.0)), np.asarray(fn(ew, 0.0, 0.0))
    )


# --- error paths --------------------------------------------------------------------


def test_collection_rejects_self_diffraction():
    """Only PG is derived: SD drops ``-w theta``, not ``+w p``, so it must not run."""
    omega, delays, _ = _grid()
    with pytest.raises(ValueError, match="derived only for PG"):
        make_param_trace_fn(
            omega, delays, "sd", omega0=OMEGA0, focal=_mixture(mode="integrated")
        )


def test_collection_requires_arm_positions():
    """A mixture without arms cannot say where the signal goes, and says so."""
    ap = mask_hole_aperture(**HOLE)
    mix = mixture_from_nodes(
        x=np.array([1e-5]),
        y=np.array([2e-5]),
        area=np.array([1e-10]),
        arms=np.array(_ARMS_PG) * ARM_OFFSET,
        hole_diameter=MASK_D,
        f_foc=F_FOC,
        collection=ap,
    )
    import dataclasses

    with pytest.raises(ValueError, match="mask hole positions"):
        transform_phases(dataclasses.replace(mix, arms=None), np.array([0.0]), OMEGA0)


def test_aperture_rejects_bad_input():
    """Enumerations and shapes are checked at construction, not at trace time."""
    with pytest.raises(ValueError, match="apod must be one of"):
        mask_transmission(np.array([0.0]), hole_diameter=HOLE_DIAM, apod="planck")
    with pytest.raises(ValueError, match="needs `apod_param`"):
        mask_transmission(np.array([0.0]), hole_diameter=HOLE_DIAM, apod="tanh")
    with pytest.raises(ValueError, match="hole_diameter must be positive"):
        mask_transmission(np.array([0.0]), hole_diameter=0.0, apod="hard")
    with pytest.raises(ValueError, match="mode must be one of"):
        mask_hole_aperture(mode="camera", **HOLE)
    with pytest.raises(ValueError, match="pad must be >= 1"):
        mask_hole_aperture(pad=0.5, **HOLE)
    with pytest.raises(ValueError, match="n_radial and n_azimuth"):
        mask_hole_aperture(n_radial=1, **HOLE)
    with pytest.raises(ValueError, match="1-D of the same length"):
        CollectionAperture(
            x=np.zeros(3),
            y=np.zeros(2),
            weight=np.zeros(3),
            transmission=np.zeros(3),
            z_mask=Z_MASK,
        )
    with pytest.raises(ValueError, match="positive z_mask"):
        CollectionAperture(
            x=np.zeros(3),
            y=np.zeros(3),
            weight=np.zeros(3),
            transmission=np.zeros(3),
            z_mask=0.0,
        )


def test_aperture_offset_needs_a_chromatic_hole():
    """A fixed k window has no mask-plane position, and the error says what to do."""
    import dataclasses

    fixed = CollectionAperture(
        x=np.zeros(2),
        y=np.zeros(2),
        weight=np.ones(2),
        transmission=np.ones(2),
        z_mask=Z_MASK,
        chromatic=False,
    )
    with pytest.raises(ValueError, match="no mask-plane offset"):
        aperture_offset(dataclasses.replace(_mixture(), collection=fixed))
    with pytest.raises(ValueError, match="no collection aperture"):
        aperture_offset(_mixture())


# --- reading the aperture back off a scan file --------------------------------------


def _reference_window_record():
    """The reference instrument's own ``window_def_*`` record, apod width unresolved."""
    return {
        "type": "PhysicalMaskWindow",
        "holex": -ARM_OFFSET,
        "holey": -ARM_OFFSET,
        "holediam": HOLE_DIAM,
        "zmask": Z_MASK,
        "apod": "tanh",
        "apod_param": "default",
        "delta_k": 7803.260441107285,
        "reference_wavelength": LAMBDA0,
    }


def test_aperture_from_scan_rebuilds_the_recorded_window(tmp_path, simulated_truth):
    """A scan file's own window is rebuilt losslessly, ``"default"`` width included."""
    from conftest import write_simulated_h5

    from croak.collection import aperture_from_scan
    from croak.io import read_simulated_mask_window, read_simulated_scan

    record = _reference_window_record()
    path = write_simulated_h5(
        tmp_path / "windowed.h5", simulated_truth, mask_window=record
    )

    spec = read_simulated_mask_window(path)
    assert spec is not None
    assert spec.window_type == "PhysicalMaskWindow"
    assert (spec.hole_x, spec.hole_y) == (-ARM_OFFSET, -ARM_OFFSET)
    assert spec.hole_diameter == HOLE_DIAM
    assert spec.apod == "tanh"
    assert spec.apod_param is None  # the file said "default"; it must be re-derived
    assert spec.delta_k == pytest.approx(record["delta_k"])
    # The full loader carries the same record, so a retrieval can reach it.
    assert read_simulated_scan(path).mask_window == spec

    ap = aperture_from_scan(path)
    assert ap.mode == "integrated"
    assert ap.z_mask == Z_MASK
    resolved = resolve_tanh_width(
        delta_k=spec.delta_k, z_mask=spec.z_mask, omega_reference=spec.omega_reference
    )
    direct = mask_hole_aperture(
        hole_x=-ARM_OFFSET,
        hole_y=-ARM_OFFSET,
        hole_diameter=HOLE_DIAM,
        z_mask=Z_MASK,
        apod="tanh",
        apod_param=resolved,
    )
    assert ap.transmission == pytest.approx(direct.transmission)
    assert ap.weight == pytest.approx(direct.weight)


def test_aperture_from_scan_needs_a_recorded_window(simulated_scan_h5):
    """A file predating the window record says so instead of guessing an aperture."""
    from croak.collection import aperture_from_scan
    from croak.io import read_simulated_mask_window

    assert read_simulated_mask_window(simulated_scan_h5) is None
    with pytest.raises(KeyError, match="cannot be rebuilt"):
        aperture_from_scan(simulated_scan_h5)
