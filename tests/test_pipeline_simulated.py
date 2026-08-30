"""Tests for the simulated-trace loader (de-Jacobian, corrections, retrieval)."""

import numpy as np
import pytest

from croak import io, preprocess
from croak.maths import wlfreq
from croak.pipeline import initial_guess
from croak.session import (
    PreprocParams,
    RetrieveParams,
    SimulatedLoadParams,
    assemble_simulated_load_data,
    assemble_simulated_tracedata,
    build_tracedata,
    run_retrieval,
)


# ---------------------------------------------------------------------------
# omega_to_lambda_density helper
# ---------------------------------------------------------------------------
def test_omega_to_lambda_density_1d():
    lam = np.array([300e-9, 400e-9, 500e-9])
    intensity = np.array([1.0, 2.0, 3.0])
    out = preprocess.omega_to_lambda_density(intensity, lam)
    np.testing.assert_allclose(out, intensity / lam**2)


def test_omega_to_lambda_density_2d_broadcasts_over_delay():
    lam = np.array([300e-9, 400e-9])
    trace = np.array([[1.0, 2.0], [4.0, 8.0]])
    out = preprocess.omega_to_lambda_density(trace, lam)
    np.testing.assert_allclose(out, trace / (lam**2)[:, None])


def test_jacobian_round_trips_through_regrid_factor():
    """``÷λ²`` then the regrid's implicit ``×λ²`` recovers the ω-density."""
    lam = np.linspace(200e-9, 600e-9, 50)
    I_omega = np.linspace(1.0, 5.0, 50)
    recovered = preprocess.omega_to_lambda_density(I_omega, lam) * lam**2
    np.testing.assert_allclose(recovered, I_omega)


# ---------------------------------------------------------------------------
# assemble_simulated_load_data
# ---------------------------------------------------------------------------
def _params(path, **overrides):
    base = dict(
        frog_path=path,
        third_order=False,
        reverse_trace=False,
        use_spectrum=False,
        lam_min_nm=140.0,
        lam_max_nm=900.0,
    )
    base.update(overrides)
    return SimulatedLoadParams(**base)


def _band_flip(scan, lam_min_nm=140.0, lam_max_nm=900.0):
    """Replicate assemble_simulated_load_data's crop -> ascending-λ ordering."""
    keep = scan.omega > 0
    keep &= scan.omega <= 2 * np.pi * 299792458.0 / (lam_min_nm * 1e-9)
    keep &= scan.omega >= 2 * np.pi * 299792458.0 / (lam_max_nm * 1e-9)
    return np.flatnonzero(keep)[::-1]


def test_load_data_shape_and_band(simulated_scan_h5):
    data = _params(simulated_scan_h5, lam_min_nm=250.0, lam_max_nm=500.0)
    out = assemble_simulated_load_data(data)
    assert out["input_unit"] == "delay"
    assert out["interaction"] == "pg"
    assert out["trace"].shape == (out["lam"].size, out["scanaxis"].size)
    assert np.all(np.diff(out["lam"]) > 0)  # ascending λ
    assert out["lam"].min() >= 250e-9 and out["lam"].max() <= 500e-9
    assert out["trace"].max() == pytest.approx(1.0)


def test_load_data_carries_known_truth(simulated_scan_h5, simulated_truth):
    """The simulated loader surfaces the scan's reference pulse as a TruthPulse."""
    from croak.processing import TruthPulse

    out = assemble_simulated_load_data(_params(simulated_scan_h5))
    truth = out["truth"]
    assert isinstance(truth, TruthPulse)
    assert truth.fwhm == pytest.approx(simulated_truth.tau_fwhm)
    assert truth.It.max() == pytest.approx(1.0)
    assert truth.lam is not None and np.all(np.diff(truth.lam) > 0)


def test_complex_modelpnps_truth_is_preserved_for_retrieval(tmp_path, simulated_truth):
    """Complex ModelPNPS fields survive loading in croak's Fourier convention.

    The point of the truth seed is a forward-model diagnostic, so the bar is not
    "a small error" but "bit-for-bit the field that generated the trace": the
    stored half-window shift and opposite Fourier sign must both be undone.
    """
    from conftest import write_simulated_h5

    path = write_simulated_h5(
        tmp_path / "complex_truth.h5", simulated_truth, store_complex=True
    )
    params = _params(path, truth_source="source", use_spectrum=True)
    data = assemble_simulated_load_data(params)
    truth = data["truth"]
    td = assemble_simulated_tracedata(params)

    assert truth.omega is not None
    assert truth.phi_w is not None
    assert truth.phi_t is not None
    seed = truth.spectrum_on_grid(td.grid, td.omega0_pulse)
    np.testing.assert_allclose(seed, truth.Eomega)
    # The native grid is the generating grid, so the seed is the generating
    # field itself — no interpolation, no convention residue.
    np.testing.assert_allclose(seed, simulated_truth.ew, atol=1e-12)


def test_truth_seeded_native_grid_run_has_no_forward_model_residual(
    tmp_path, simulated_truth
):
    """The whole point of the truth seed: on the native grid the error is zero.

    Nothing between the stored field and the forward model is lossy here — the
    grid is the generating grid, the delays are the generating delays — so a
    correct forward model has to reproduce the trace to rounding. Anything that
    quantises time zero to a delay bin, or slips a Fourier convention, shows up
    immediately as a residual many orders of magnitude above this bar.
    """
    from conftest import write_simulated_h5

    from croak.forward import maketrace
    from croak.metrics import frog_error

    path = write_simulated_h5(
        tmp_path / "complex_truth.h5", simulated_truth, store_complex=True
    )
    params = _params(path, truth_source="source", use_spectrum=True)
    truth = assemble_simulated_load_data(params)["truth"]
    td = assemble_simulated_tracedata(params)

    seed = truth.spectrum_on_grid(td.grid, td.omega0_pulse)
    seeded = frog_error(
        td.trace, maketrace(td.grid.omega, td.delays, seed, td.interaction)
    )
    assert seeded < 1e-12

    # A session run routed through ``truth_init`` starts from that same seed, so
    # it cannot come out worse than the seed it was handed.
    res = run_retrieval(
        RetrieveParams(solver="copra", maxiters=1, truth_init=True),
        td,
        truth=truth,
        rng=np.random.default_rng(0),
    )
    assert res.error < 1e-12


def test_native_grid_delay_zero_is_not_quantised_to_a_bin(tmp_path, simulated_truth):
    """Time zero is centred to sub-sample precision, so a true zero is preserved.

    The fixture's delay axis is even and symmetric, so its true zero falls
    *between* the two central bins. Reading the marginal peak off as the largest
    sample would move the axis half a step and silently mismatch the model
    against the measurement; :func:`croak.preprocess.marginal_peak_delay`
    interpolates the peak, so an axis already centred on zero is left alone.
    """
    from conftest import write_simulated_h5

    path = write_simulated_h5(tmp_path / "scan.h5", simulated_truth)
    td = assemble_simulated_tracedata(_params(path))

    step = float(np.diff(simulated_truth.tau).mean())
    assert np.min(np.abs(simulated_truth.tau)) > 0.4 * step  # zero is between bins
    np.testing.assert_allclose(td.delays, simulated_truth.tau, atol=1e-3 * step)


def test_native_grid_delay_zero_is_recovered_from_an_offset_axis(
    tmp_path, simulated_truth
):
    """A scan recorded about a non-zero delay origin is still centred on zero."""
    import dataclasses

    from conftest import write_simulated_h5

    offset = dataclasses.replace(simulated_truth, tau=simulated_truth.tau + 7.3e-15)
    path = write_simulated_h5(tmp_path / "offset.h5", offset)
    td = assemble_simulated_tracedata(_params(path))

    step = float(np.diff(simulated_truth.tau).mean())
    np.testing.assert_allclose(td.delays, simulated_truth.tau, atol=1e-3 * step)


def test_truth_seed_survives_the_regrid_onto_a_different_grid(
    tmp_path, simulated_truth
):
    """Off the native grid the truth seed is interpolated, and still lands on it.

    The regrid path gives the retrieval a grid the stored field was never
    sampled on, so ``spectrum_on_grid`` interpolates amplitude and *unwrapped*
    phase. Interpolating the wrapped phase instead would scramble the seed; the
    check is that the truth still starts orders of magnitude closer than the
    measured-amplitude starts it is supposed to beat.
    """
    from conftest import write_simulated_h5

    from croak.forward import maketrace
    from croak.metrics import frog_error

    path = write_simulated_h5(
        tmp_path / "complex_truth.h5", simulated_truth, store_complex=True
    )
    data = assemble_simulated_load_data(
        _params(
            path,
            truth_source="source",
            use_spectrum=True,
            lam_min_nm=simulated_truth.lam_min / 1e-9,
            lam_max_nm=simulated_truth.lam_max / 1e-9,
        )
    )
    td = build_tracedata(
        PreprocParams(
            lam_min_nm=simulated_truth.lam_min / 1e-9,
            lam_max_nm=simulated_truth.lam_max / 1e-9,
            lamm_min_nm=simulated_truth.lam_min / 1e-9,
            lamm_max_nm=simulated_truth.lam_max / 1e-9,
        ),
        data,
    )
    truth = data["truth"]
    assert truth.omega is not None and td.grid.n != truth.omega.size  # regridded

    def seed_error(mode):
        ew = initial_guess(td, mode=mode, truth=truth)
        return frog_error(
            td.trace, maketrace(td.grid.omega, td.delays, ew, td.interaction)
        )

    truth_error = seed_error("truth")
    assert truth_error < 1e-3
    assert truth_error < 0.05 * min(seed_error("auto"), seed_error("tl"))


def test_complex_truth_phases_recover_the_generating_chirp(tmp_path, simulated_truth):
    """The loaded truth phases carry the generating field's real, signed chirp.

    An ``is not None`` check cannot see a convention slip: a missed conjugation
    flips the sign of the chirp and a missed half-window shift adds a π-per-bin
    tilt, and both still draw a perfectly plausible curve. Fitting the GDD pins
    sign *and* magnitude against the fixture's own :data:`conftest.SIMULATED_GDD`.

    The temporal phase of a purely GDD-chirped Gaussian is quadratic too. With
    field ``E(ω) ∝ exp(-a²ω²/2 + i·b₂ω²/2)`` and ``a² = fwhm²/(4 ln 2)``,
    croak's ``E(t) = ∫E(ω)e^{-iωt}dω`` gives ``φ(t) = -b₂t²/(2(a⁴ + b₂²))`` ---
    note the sign reversal, which is the ``e^{-iωt}`` convention showing up in
    the time domain.
    """
    from conftest import SIMULATED_GDD, write_simulated_h5

    path = write_simulated_h5(
        tmp_path / "complex_truth.h5", simulated_truth, store_complex=True
    )
    truth = assemble_simulated_load_data(
        _params(path, truth_source="source", use_spectrum=True)
    )["truth"]
    assert truth.lam is not None and truth.Iw is not None
    assert truth.phi_w is not None and truth.phi_t is not None

    # Spectral: a quadratic fit about the spectral peak returns φ'' = GDD.
    band = truth.Iw > 0.05
    omega = 2.0 * np.pi * 299792458.0 / truth.lam[band]
    omega -= omega[int(np.argmax(truth.Iw[band]))]
    quad_w = np.polyfit(omega, truth.phi_w[band], 2)
    assert 2.0 * quad_w[0] == pytest.approx(SIMULATED_GDD, rel=1e-6)

    # Temporal: quadratic, with the sign reversed by the Fourier convention.
    support = truth.It > 0.05
    quad_t = np.polyfit(truth.t[support], truth.phi_t[support], 2)
    a_sq = simulated_truth.tau_fwhm**2 / (4.0 * np.log(2.0))
    expected = -SIMULATED_GDD / (2.0 * (a_sq**2 + SIMULATED_GDD**2))
    assert quad_t[0] == pytest.approx(expected, rel=1e-6)
    residual = np.abs(np.polyval(quad_t, truth.t[support]) - truth.phi_t[support]).max()
    assert residual < 1e-9  # a pure quadratic: no leftover tilt or wrap


def test_truth_init_without_a_complex_field_raises(tmp_path, simulated_truth):
    """An intensity-only file cannot supply a truth seed, and says so."""
    from conftest import write_simulated_h5

    path = write_simulated_h5(tmp_path / "intensity_only.h5", simulated_truth)
    params = _params(path)
    truth = assemble_simulated_load_data(params)["truth"]
    td = assemble_simulated_tracedata(params)

    assert truth.Eomega is None
    with pytest.raises(ValueError, match="no complex spectrum"):
        run_retrieval(
            RetrieveParams(solver="copra", maxiters=1, truth_init=True),
            td,
            truth=truth,
        )


def test_truth_source_selects_beamlet_envelope(tmp_path, simulated_truth):
    """``truth_source='beamlet'`` overlays ``It_beamlet`` (peak-normalised)."""
    import h5py
    from conftest import write_simulated_h5

    path = write_simulated_h5(tmp_path / "scan.h5", simulated_truth)
    # A genuinely different envelope (asymmetric ramp), so the choice is visible
    # in the peak region — not just a tiny pedestal that ``allclose`` would miss.
    ramp = 1.0 + 0.8 * np.linspace(0.0, 1.0, simulated_truth.It.size)
    it_beamlet = simulated_truth.It * ramp
    with h5py.File(path, "a") as f:
        f["grid"]["It_beamlet"] = it_beamlet

    beamlet = assemble_simulated_load_data(_params(path, truth_source="beamlet"))[
        "truth"
    ]
    source = assemble_simulated_load_data(_params(path, truth_source="source"))["truth"]
    # The loader peak-normalises but does not reorder It, so it equals the chosen
    # raw envelope scaled to unit peak — beamlet for one, source for the other.
    np.testing.assert_allclose(beamlet.It, it_beamlet / it_beamlet.max())
    np.testing.assert_allclose(source.It, simulated_truth.It / simulated_truth.It.max())


def test_de_jacobian_is_the_only_transform_when_corrections_off(
    simulated_scan_h5, simulated_truth
):
    """With third-order/mask/reverse off, ``trace × λ²`` == the raw ω-density."""
    out = assemble_simulated_load_data(_params(simulated_scan_h5))
    recovered = out["trace"] * (out["lam"] ** 2)[:, None]
    recovered = recovered / recovered.max()

    scan = io.read_simulated_scan(simulated_scan_h5)
    keep = scan.omega > 0
    keep &= scan.omega <= 2 * np.pi * 299792458.0 / (140e-9)
    keep &= scan.omega >= 2 * np.pi * 299792458.0 / (900e-9)
    flip = np.flatnonzero(keep)[::-1]
    expected = scan.trace[flip, :]
    expected = expected / expected.max()
    np.testing.assert_allclose(recovered, expected, rtol=1e-9, atol=1e-9)


def test_reverse_trace_mirrors_delay_columns(simulated_scan_h5):
    fwd = assemble_simulated_load_data(_params(simulated_scan_h5, reverse_trace=False))
    rev = assemble_simulated_load_data(_params(simulated_scan_h5, reverse_trace=True))
    # τ is a symmetric grid, so reversal keeps the axis but flips the columns
    np.testing.assert_allclose(rev["scanaxis"], fwd["scanaxis"], atol=1e-30)
    np.testing.assert_allclose(rev["trace"], fwd["trace"][:, ::-1], rtol=1e-9)


def test_third_order_scales_rows_by_lambda_power(simulated_scan_h5):
    exp = 4.7
    no = assemble_simulated_load_data(_params(simulated_scan_h5, third_order=False))
    yes = assemble_simulated_load_data(
        _params(simulated_scan_h5, third_order=True, third_order_exp=exp)
    )
    # both share the de-Jacobian; the ratio per row is (λ/µm)^exp up to a constant
    lam = no["lam"]
    sig = no["trace"].max(axis=1) > 1e-3
    ratio = yes["trace"][sig].sum(axis=1) / no["trace"][sig].sum(axis=1)
    model = (lam[sig] / 1e-6) ** exp
    ratio = ratio / ratio[0]
    model = model / model[0]
    np.testing.assert_allclose(ratio, model, rtol=1e-6)


def test_spectrum_source_defaults_to_beamlet():
    assert SimulatedLoadParams().spectrum_source == "beamlet"


def test_spectrum_source_selects_reference(simulated_scan_h5):
    """'beamlet'/'source' pick the matching de-Jacobianed reference spectrum."""
    scan = io.read_simulated_scan(simulated_scan_h5)
    flip = _band_flip(scan)
    lam = 2 * np.pi * 299792458.0 / scan.omega[flip]
    for source, ref in (("beamlet", scan.Iomega_beamlet), ("source", scan.Iomega)):
        out = assemble_simulated_load_data(
            _params(simulated_scan_h5, use_spectrum=True, spectrum_source=source)
        )
        expected = ref[flip] / lam**2
        expected = expected / expected.max()
        np.testing.assert_allclose(out["Ilam_spec"], expected, rtol=1e-9)
    # the post-mask beamlet differs from the pre-mask source (chromatic vignette)
    beam = assemble_simulated_load_data(
        _params(simulated_scan_h5, use_spectrum=True, spectrum_source="beamlet")
    )
    src = assemble_simulated_load_data(
        _params(simulated_scan_h5, use_spectrum=True, spectrum_source="source")
    )
    assert not np.allclose(beam["Ilam_spec"], src["Ilam_spec"])


def test_use_spectrum_provides_independent_spectrum(simulated_scan_h5):
    without = assemble_simulated_load_data(
        _params(simulated_scan_h5, use_spectrum=False)
    )
    assert without["lam_spec"] is None and without["Ilam_spec"] is None
    with_spec = assemble_simulated_load_data(
        _params(simulated_scan_h5, use_spectrum=True)
    )
    assert with_spec["lam_spec"] is not None
    np.testing.assert_allclose(with_spec["lam_spec"], with_spec["lam"])
    assert with_spec["Ilam_spec"].max() == pytest.approx(1.0)


def test_empty_band_raises(simulated_scan_h5):
    with pytest.raises(ValueError, match="no frequency bins"):
        assemble_simulated_load_data(
            _params(simulated_scan_h5, lam_min_nm=1.0, lam_max_nm=2.0)
        )


def test_interaction_passthrough(simulated_scan_h5):
    out = assemble_simulated_load_data(_params(simulated_scan_h5, interaction="sd"))
    assert out["interaction"] == "sd"


def test_z_thickness_um_selects_slice(simulated_scan_multi_h5):
    """``z_thickness_um`` routes through to the nearest saved thickness slice."""
    exit_data = assemble_simulated_load_data(_params(simulated_scan_multi_h5))
    thin_data = assemble_simulated_load_data(
        _params(simulated_scan_multi_h5, z_thickness_um=10.0)
    )
    # a thinner slice differs in shape, so the normalised traces are not equal
    assert thin_data["trace"].shape == exit_data["trace"].shape
    assert not np.allclose(thin_data["trace"], exit_data["trace"])
    # asking for the exit thickness explicitly reproduces the default (exit) load
    exit_explicit = assemble_simulated_load_data(
        _params(simulated_scan_multi_h5, z_thickness_um=40.0)
    )
    np.testing.assert_allclose(exit_explicit["trace"], exit_data["trace"])


# ---------------------------------------------------------------------------
# assemble_simulated_tracedata — raw, native-grid, straight-to-retrieval path
# ---------------------------------------------------------------------------
def test_raw_tracedata_on_native_grid(simulated_scan_h5):
    """The direct TraceData sits on the simulation's native FFT grid (no regrid)."""
    td = assemble_simulated_tracedata(
        SimulatedLoadParams(frog_path=simulated_scan_h5, use_spectrum=True)
    )
    scan = io.read_simulated_scan(simulated_scan_h5)
    assert td.grid.n == scan.omega.size
    # grid.omega + carrier reproduces the file's absolute ω axis exactly
    np.testing.assert_allclose(
        td.grid.omega + td.omega0_pulse, scan.omega, rtol=0, atol=1e-3 * td.grid.domega
    )
    assert td.trace.shape == (scan.omega.size, scan.tau.size)
    assert td.trace.max() == pytest.approx(1.0)
    assert td.Iomega is not None
    assert td.interaction == "pg"
    assert td.omega0_trace == pytest.approx(td.omega0_pulse)  # pg: no doubling


def test_raw_tracedata_ignores_third_order(simulated_scan_h5):
    """The raw path is fully raw — the third-order toggle has no effect."""
    base = SimulatedLoadParams(frog_path=simulated_scan_h5)
    a = assemble_simulated_tracedata(replace_third_order(base, False))
    b = assemble_simulated_tracedata(replace_third_order(base, True))
    np.testing.assert_allclose(a.trace, b.trace)


def replace_third_order(p, on):
    from dataclasses import replace

    return replace(p, third_order=on, third_order_exp=4.7)


def test_raw_tracedata_round_trip_retrieval(simulated_scan_h5):
    """The native-grid raw trace retrieves to a small error (self-consistent)."""
    td = assemble_simulated_tracedata(
        SimulatedLoadParams(frog_path=simulated_scan_h5, use_spectrum=True)
    )
    res = run_retrieval(
        RetrieveParams(solver="copra", maxiters=200),
        td,
        rng=np.random.default_rng(0),
    )
    assert res.error < 2e-2


# ---------------------------------------------------------------------------
# End-to-end: a self-consistent trace must retrieve to a small error
# ---------------------------------------------------------------------------
def test_round_trip_retrieval(simulated_scan_h5, simulated_truth):
    """Loading a maketrace-generated scan and retrieving recovers a low R.

    The trace is produced by croak's own PG forward model, so the whole load →
    de-Jacobian → regrid → retrieve chain is self-consistent: a wrong Jacobian
    would leave an irreducible λ²-shaped residual and a large error.
    """
    data = assemble_simulated_load_data(
        _params(
            simulated_scan_h5,
            lam_min_nm=simulated_truth.lam_min / 1e-9,
            lam_max_nm=simulated_truth.lam_max / 1e-9,
            use_spectrum=True,
        )
    )
    pp = PreprocParams(
        lam_min_nm=simulated_truth.lam_min / 1e-9,
        lam_max_nm=simulated_truth.lam_max / 1e-9,
        lamm_min_nm=simulated_truth.lam_min / 1e-9,
        lamm_max_nm=simulated_truth.lam_max / 1e-9,
    )
    td = build_tracedata(pp, data)
    assert td.Iomega is not None
    # carrier lands near the 350 nm pulse wavelength
    lam0 = wlfreq(td.omega0_pulse)
    assert 300e-9 < lam0 < 400e-9
    res = run_retrieval(
        RetrieveParams(solver="copra", maxiters=200),
        td,
        rng=np.random.default_rng(0),
    )
    assert res.error < 2e-2


def test_uniform_delay_core_drops_the_wings(tmp_path, simulated_truth):
    """`uniform_delay_core=True` keeps only the uniformly spaced core.

    The fixture's axis is uniform, so the file is post-edited into the
    campaign shape: coarse wing delays appended on both sides (zero-signal
    columns, as the real Raman wings nearly are at the trace edge).
    """
    import h5py
    from conftest import write_simulated_h5

    path = write_simulated_h5(tmp_path / "wings.h5", simulated_truth)
    with h5py.File(path, "a") as f:
        tau = f["scanvariables/τ"][()]
        step = 4.0 * (tau[1] - tau[0])  # coarser wings, clearly non-uniform
        wing_lo = tau[0] - step * np.arange(3, 0, -1)
        wing_hi = tau[-1] + step * np.arange(1, 4)
        tau_new = np.concatenate([wing_lo, tau, wing_hi])
        for key in ("Iω_win", "Iω_win_reimaged"):
            w = f[key][()]
            pad = np.zeros((3,) + w.shape[1:], dtype=w.dtype)
            del f[key]
            f[key] = np.concatenate([pad, w, pad], axis=0)
        del f["scanvariables/τ"]
        f["scanvariables/τ"] = tau_new

    n_core = simulated_truth.tau.size
    with_wings = assemble_simulated_load_data(_params(path))
    assert with_wings["scanaxis"].size == n_core + 6
    core_only = assemble_simulated_load_data(_params(path, uniform_delay_core=True))
    assert core_only["scanaxis"].size == n_core
    d = np.diff(core_only["scanaxis"])
    np.testing.assert_allclose(d, d[0], rtol=1e-9)
    # the kept columns are the core's, not the zero-padded wings
    assert core_only["trace"].max() > 0
