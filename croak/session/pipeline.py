"""Pure stage mappings: parameter dataclasses → core croak calls.

Each function turns one stage's :mod:`croak.session.params` dataclass into the
corresponding call on croak's public API (``assemble_load_data`` → arrays,
``build_tracedata`` → :func:`~croak.preprocess.load_and_clean`, ``run_retrieval``
→ :func:`~croak.pipeline.retrieve_from_tracedata`, ``run_uncertainty`` →
:func:`~croak.uncertainty.estimate_fwhm_uncertainty`). They are Qt-free and
side-effect-light (only the loaders read files), so the GUI workers and the
headless engine call exactly the same mapping — there is one obvious way to turn
a saved session into a retrieval.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace

import numpy as np

from .. import io, preprocess
from ..collection import aperture_from_scan, mask_hole_aperture
from ..constants import C
from ..covariance import covariance_uncertainty
from ..focal import FocalMixture, focal_mixture
from ..forward import quadrature_nodes_weights
from ..grid import Grid
from ..interactions import get_interaction
from ..pipeline import retrieve_from_tracedata
from ..preprocess import TraceData
from ..processing import (
    ProcessedResult,
    TruthPulse,
    normalize_spectral_phase,
    normalize_temporal_phase,
    post_filter,
    process_result,
    shiftnorm,
)
from ..result import RetrievalResult
from ..smearing import SmearingKernel, collinear_sd_kernel, square_boxcars_kernel
from ..uncertainty import (
    NoiseModel,
    UncertaintyResult,
    estimate_fwhm_uncertainty,
    noise_from_residual,
)
from .dispersion import has_dispersion, transfer_function
from .params import (
    DispersionParams,
    LoadParams,
    PreprocParams,
    RetrieveParams,
    SimulatedLoadParams,
    UncertaintyParams,
)

__all__ = [
    "assemble_load_data",
    "assemble_simulated_load_data",
    "build_tracedata",
    "resolve_guess",
    "extra_param_centres",
    "run_retrieval",
    "apply_post_filter",
    "process",
    "bootstrap_solver_kwargs",
    "smearing_kernel",
    "focal_mixture_from_params",
    "apply_spectrum_frame",
    "generation_envelope",
    "covariance_kwargs",
    "run_uncertainty",
]


# ---------------------------------------------------------------------------
# Load: LoadParams -> SI arrays
# ---------------------------------------------------------------------------


def _resolve_reverse_trace(p: SimulatedLoadParams, scan) -> bool:
    """Resolve the delay-axis reversal: explicit setting, else the file marker.

    Files that declare ``delay_convention == "gate"`` are already stored in
    the gate-delay (paper) frame and must not be reversed; legacy files
    (marker absent, ``"legacy"``) use the probe-delayed frame and are negated.
    """
    if p.reverse_trace is not None:
        return p.reverse_trace
    return scan.delay_convention != "gate"


def _load_spectrum(path: str, p: LoadParams) -> tuple[np.ndarray, np.ndarray]:
    """Load (λ_m, intensity) from a CSV (columns) or HDF5/NPZ (datasets)."""
    ext = os.path.splitext(path)[1].lower()
    factor = io.unit_to_si(p.spec_lam_unit)
    if ext in (".csv", ".txt"):
        mat = io.load_csv(path)
        lam = mat[:, p.spec_lam_col] * factor
        intensity = mat[:, p.spec_int_col]
    else:
        fd = io.list_datasets(path)
        lam = io.to_1d(fd.load(p.spec_lam_name), p.spec_lam_name) * factor
        intensity = io.reduce_to_1d(fd.load(p.spec_int_name), p.spec_int_name, lam.size)
    if lam.size != intensity.size:
        raise ValueError("spectrum λ and intensity lengths differ; pick matching data")
    return lam, intensity


def _load_frog_bg(path, lam_name, int_name, lam_factor, lam_frog_si) -> np.ndarray:
    """Load a FROG background spectrum sliced to the main FROG's λ range."""
    fd = io.list_datasets(path)
    lam_bg = io.to_1d(fd.load(lam_name), lam_name) * lam_factor
    bg = io.reduce_to_1d(fd.load(int_name), int_name, lam_bg.size)
    lo, hi = lam_frog_si.min(), lam_frog_si.max()
    sliced = bg[(lam_bg >= lo) & (lam_bg <= hi)]
    if sliced.size != lam_frog_si.size:
        raise ValueError(
            f"FROG background λ overlap has {sliced.size} points but the trace has "
            f"{lam_frog_si.size}; the wavelength grids must match where they overlap"
        )
    return sliced


def assemble_load_data(p: LoadParams) -> dict:
    """Build a ``load_data`` dict (SI arrays) from the load parameters.

    Returns ``{lam, scanaxis, input_unit, trace (Nlambda, Ndelay), lam_spec,
    Ilam_spec, interaction}``. The scale curve (calibration × 3rd-order) is
    applied eagerly to the trace, so everything downstream sees calibrated
    counts and the calibration never has to be re-applied.
    """
    fd = io.list_datasets(p.frog_path)
    lam = fd.load(p.lam_name) * io.unit_to_si(p.lam_unit)
    scan = fd.load(p.scan_name) * io.unit_to_si(p.scan_unit)
    trace = fd.load(p.ifrog_name)

    # orient to (Nlambda, Ndelay)
    if trace.shape == (scan.size, lam.size):
        trace = trace.T
    if p.transpose:
        trace = trace.T

    # centre + sort the scan axis (keeps the delay window around 0)
    scan = scan - scan.mean()
    order = np.argsort(scan)
    scan = scan[order]
    trace = trace[:, order]

    # FROG background subtraction (own λ axis, sliced to overlap)
    if p.frog_bg_use and p.frog_bg_path:
        bg = _load_frog_bg(
            p.frog_bg_path,
            p.frog_bg_lam_name,
            p.frog_bg_int_name,
            io.unit_to_si(p.lam_unit),
            lam,
        )
        trace = preprocess.background_subtract(trace, bg[:, None])

    mx = trace.max()
    if mx > 0:
        trace = trace / mx

    # scale curve: calibration × 3rd-order, applied eagerly
    scalecurve = np.ones(lam.shape)
    if p.frog_calib_use and p.frog_calib_path:
        cal = io.load_csv(p.frog_calib_path)
        scalecurve = scalecurve * preprocess.calibration_curve_scale(
            lam * io.unit_to_si(p.frog_calib_lam_unit) / io.unit_to_si(p.lam_unit),
            cal[:, 0],
            cal[:, 1],
        )
    if p.third_order:
        scalecurve = scalecurve * preprocess.third_order_scale(lam, p.third_order_exp)
    if not np.allclose(scalecurve, 1.0):
        trace = trace * scalecurve[:, None]
        mx2 = trace.max()
        if mx2 > 0:
            trace = trace / mx2

    # independent spectrum (+ background + calibration)
    lam_spec = Ilam_spec = None
    if p.use_spec and p.spec_path:
        lam_spec, Ilam_spec = _load_spectrum(p.spec_path, p)
        if p.spec_bg_use and p.spec_bg_path:
            _, bg = _load_spectrum(p.spec_bg_path, p)
            if bg.size == Ilam_spec.size:
                Ilam_spec = Ilam_spec - bg
        if p.spec_calib_use and p.spec_calib_path:
            cal = io.load_csv(p.spec_calib_path)
            Ilam_spec = Ilam_spec * preprocess.calibration_curve_scale(
                lam_spec
                * io.unit_to_si(p.spec_calib_lam_unit)
                / io.unit_to_si(p.spec_lam_unit),
                cal[:, 0],
                cal[:, 1],
            )
        m = np.max(Ilam_spec)
        if m > 0:
            Ilam_spec = Ilam_spec / m

    return dict(
        lam=lam,
        scanaxis=scan,
        input_unit="position" if p.scan_type == "position" else "delay",
        trace=trace,
        lam_spec=lam_spec,
        Ilam_spec=Ilam_spec,
        interaction=p.interaction,
    )


def assemble_simulated_load_data(p: SimulatedLoadParams) -> dict:
    r"""Build a ``load_data`` dict from a numerically simulated FROG scan.

    Reads a Luna ``scansave`` HDF5 file (:func:`croak.io.read_simulated_scan`) and
    returns the **same** ``{lam, scanaxis, input_unit, trace, lam_spec,
    Ilam_spec, interaction}`` shape as :func:`assemble_load_data`, so the
    simulated trace flows into :func:`build_tracedata` and the retrieval exactly
    like a measured one.

    For a multi-thickness scan (one that stores ``/grid/zsave``), ``p`` selects
    which propagation thickness to load: ``z_thickness_um`` (µm) picks the nearest
    saved slice, and falls back to the substrate-exit slice when ``None`` (also
    the only behaviour for the legacy single-thickness format).

    The simulated trace is an angular-frequency density :math:`I_\omega`; the
    measured-data pipeline (regrid) multiplies the loaded trace by
    :math:`\lambda^2`. To land on the correct :math:`I_\omega` the trace is
    divided by :math:`\lambda^2` here
    (:func:`croak.preprocess.omega_to_lambda_density`), so the round-trip is
    undistorted. The pipeline is: read → drop ``ω ≤ 0`` and crop to the band →
    optional ``×(λ/µm)^n`` third-order correction → ``÷ λ²`` → optional delay
    reversal → normalise. The chosen reference spectrum (post-mask ``beamlet`` by
    default, or the pre-mask ``source``) is offered as the independent spectrum;
    the mask vignetting is handled by that choice, not by a trace correction.

    Parameters
    ----------
    p : SimulatedLoadParams
        The simulated-load options (file, window, band, corrections, …).

    Returns
    -------
    dict
        ``load_data`` with SI arrays; ``trace`` is ``(Nlambda, Ndelay)`` and
        ``input_unit`` is ``"delay"``.

    Raises
    ------
    ValueError
        If the requested wavelength band contains no positive-frequency bins.
    """
    # Select the propagation thickness: the µm-valued z_thickness_um (the GUI
    # selector) wins over the raw z_index when set; otherwise the exit slice.
    z_thickness = None if p.z_thickness_um is None else p.z_thickness_um * 1e-6
    scan = io.read_simulated_scan(
        p.frog_path,
        window_key=p.window_key,
        z_index=p.z_index,
        z_thickness=z_thickness,
        truth_source=p.truth_source,
    )

    # Keep positive frequencies inside the requested band; ω = 2πc/λ, so the band
    # [lam_min, lam_max] maps to a frequency window. Ascending λ is the reverse of
    # ascending ω, so flip the kept slices onto an ascending-λ axis.
    omega = scan.omega
    keep = omega > 0.0
    keep &= omega <= 2.0 * np.pi * C / (p.lam_min_nm * 1e-9)
    keep &= omega >= 2.0 * np.pi * C / (p.lam_max_nm * 1e-9)
    if not np.any(keep):
        raise ValueError(
            f"no frequency bins in the band {p.lam_min_nm}–{p.lam_max_nm} nm"
        )
    flip = np.flatnonzero(keep)[::-1]  # ascending ω -> ascending λ
    lam = 2.0 * np.pi * C / omega[flip]
    trace = scan.trace[flip, :]

    # Pick the reference spectrum. The mask vignetting is a linear filter on the
    # gating beam, so it belongs in the reference spectrum, not in a trace
    # correction: the post-mask "beamlet" spectrum is the one that actually gates.
    # Fall back to the source spectrum if the file has no beamlet (e.g. Gaussian
    # beam runs, or spectrum_source="source").
    if p.spectrum_source == "beamlet" and scan.Iomega_beamlet is not None:
        ref_spectrum = scan.Iomega_beamlet[flip]
    else:
        ref_spectrum = scan.Iomega[flip]

    # Third-order (χ³) efficiency correction: divide out the ∝ ω^n generation
    # response, i.e. multiply by (λ/µm)^n — equivalently dividing the trace by
    # ω^n, up to a constant that the retrieval's overall scale factor absorbs.
    if p.third_order:
        trace = trace * preprocess.third_order_scale(lam, p.third_order_exp)[:, None]

    # ω-density -> λ-density so the regrid's ×λ² returns the original I_ω.
    trace = preprocess.omega_to_lambda_density(trace, lam)

    # Delay axis: optionally flip the sign convention, then sort ascending and
    # centre (the regrid recentres on the marginal peak regardless).
    scanaxis = -scan.tau if _resolve_reverse_trace(p, scan) else scan.tau.astype(float)
    order = np.argsort(scanaxis)
    scanaxis = scanaxis[order]
    trace = trace[:, order]
    scanaxis = scanaxis - scanaxis.mean()

    # Optionally restrict to the uniformly spaced delay core (wing points at a
    # coarser step break the smearing kernel's delay-axis convolution).
    if p.uniform_delay_core:
        idx = preprocess.uniform_core_indices(scanaxis)
        if idx.size < scanaxis.size:
            scanaxis = scanaxis[idx] - scanaxis[idx].mean()
            trace = trace[:, idx]

    mx = trace.max()
    if mx > 0:
        trace = trace / mx

    # Independent spectrum: the chosen reference (also ω-density -> λ-density).
    lam_spec = Ilam_spec = None
    if p.use_spectrum:
        lam_spec = lam
        Ilam_spec = preprocess.omega_to_lambda_density(ref_spectrum, lam)
        m = Ilam_spec.max()
        if m > 0:
            Ilam_spec = Ilam_spec / m

    return dict(
        lam=lam,
        scanaxis=scanaxis,
        input_unit="delay",
        trace=trace,
        lam_spec=lam_spec,
        Ilam_spec=Ilam_spec,
        interaction=p.interaction,
        truth=_simulated_truth(scan, p.truth_source),
        geometry=simulated_geometry(scan),
    )


@dataclass(frozen=True)
class SimulatedGeometry:
    r"""What a simulated scan says about the instrument that produced it.

    The retrieval needs three numbers that are properties of the *simulation*,
    not of the trace: how far the signal propagated (the slab thickness), and
    the mask hole diameter and edge-to-edge spacing that set the smearing
    kernel through :math:`d/D = (\text{spacing} + D)/2D`. Carrying them by hand
    is how a sweep ended up running a gap-500 kernel against a gap-1000 trace,
    which cost ~2% of retrieved duration while *lowering* nothing observable in
    the trace error. Reading them off the file removes that failure mode.

    Every field is ``None`` when the file does not record it: older files
    predate the geometry metadata, and the honest answer is then "unknown",
    not a plausible default.
    """

    #: Propagation distance of the selected slice (µm) — the slab thickness the
    #: dispersive forward model should use.
    thickness_um: float | None = None
    #: Mask hole diameter D (mm).
    mask_diameter_mm: float | None = None
    #: Edge-to-edge gap between holes (mm).
    mask_spacing_mm: float | None = None
    #: Kernel layout implied by the simulated interaction geometry:
    #: ``"boxcars"`` for the four-hole TG mask, ``"sd2"`` for two-beam SD.
    layout: str | None = None
    #: Focusing focal length (mm) — needed only by the explicit chromatic
    #: focal mixture; the reduced kernel is f-independent.
    f_foc_mm: float | None = None

    @property
    def has_mask(self) -> bool:
        """Whether the mask geometry (both numbers) is known."""
        return self.mask_diameter_mm is not None and self.mask_spacing_mm is not None


def simulated_geometry(scan: io.SimulatedScan) -> SimulatedGeometry:
    """Instrument geometry recorded in a simulated scan (fields may be ``None``)."""
    thickness_um = None
    if scan.z_positions is not None and scan.z_index is not None:
        try:
            thickness_um = float(scan.z_positions[scan.z_index]) * 1e6
        except IndexError, TypeError:
            thickness_um = None
    layout = None
    if scan.geometry is not None:
        layout = {"sd": "sd2"}.get(scan.geometry.lower(), "boxcars")
    return SimulatedGeometry(
        thickness_um=thickness_um,
        mask_diameter_mm=(
            None if scan.mask_diameter is None else scan.mask_diameter * 1e3
        ),
        mask_spacing_mm=(
            None if scan.mask_spacing is None else scan.mask_spacing * 1e3
        ),
        layout=layout,
        f_foc_mm=(None if scan.f_foc is None else scan.f_foc * 1e3),
    )


def _simulated_truth_spectrum(
    scan: io.SimulatedScan, truth_source: str
) -> np.ndarray | None:
    """Return the selected stored complex field in croak's convention.

    Two corrections are needed before the file's ``Eomega_beamlet`` means what
    it looks like, and both are easy to miss because the result is plausible
    either way:

    * the field is stored centred in its own time window, which is a linear
      spectral phase of very nearly pi per sample — right at the aliasing limit,
      so unwrapping it as stored yields nonsense (on a +2 fs^2 test trace it
      fits +27 fs^2). Undoing that half-window shift first drops the largest
      in-band phase step from 3.14 rad to 0.40;
    * the simulation stores E(omega) in the opposite Fourier convention to
      croak's ``e^{+i w t}``, so the phase must be conjugated. Checked three
      ways on the +2 fs^2 arm: the stored field fits -2.000 fs^2, a retrieval
      of that trace returns +2.014, and the conjugated truth is the one that
      scores a small complex error.

    The returned field deliberately retains its complex values rather than only
    its display phase: it is both the source for the truth phase overlays and the
    exact initial spectrum used by ``RetrieveParams(truth_init=True)``.
    """
    if truth_source == "beamlet" and scan.Iomega_beamlet is not None:
        E = scan.Eomega_beamlet
    else:
        E = scan.Eomega
    if E is None:
        return None
    E = np.asarray(E, dtype=complex) * ((-1.0) ** np.arange(np.size(E)))
    return np.conj(E).astype(np.complex128)


def _truth_temporal_phase(
    scan: io.SimulatedScan, Eomega: np.ndarray
) -> np.ndarray | None:
    """Derive the selected truth's temporal phase on its stored time axis."""
    if not np.any(np.abs(Eomega) > 0):
        return None
    domega = float(np.mean(np.diff(scan.omega)))
    grid = Grid(Eomega.size, domega=domega)
    field = shiftnorm(grid, grid.ifft(Eomega))
    # ModelPNPS may store an oversampled temporal truth. Interpolate the complex
    # envelope (not its wrapped phase) onto that peak-centred axis, then apply the
    # exact gauge normalisation used by process_result.
    target_t = np.asarray(scan.t, dtype=float)
    target_t = target_t - target_t[int(np.argmax(scan.It))]
    real = np.interp(target_t, grid.t, field.real, left=0.0, right=0.0)
    imag = np.interp(target_t, grid.t, field.imag, left=0.0, right=0.0)
    target_field = real + 1j * imag
    intensity = np.asarray(scan.It, dtype=float)
    peak = float(intensity.max())
    mask = intensity > 0.01 * peak if peak > 0 else np.zeros_like(intensity, dtype=bool)
    if np.count_nonzero(mask) < 2:
        return None
    return normalize_temporal_phase(target_t, target_field, mask)


def _simulated_truth(scan: io.SimulatedScan, truth_source: str) -> TruthPulse:
    """Known ground-truth pulse from a simulated scan's stored reference.

    The simulation stores the reference temporal intensity (already selected onto
    ``scan.It`` for the chosen ``truth_source``) and the matching input spectrum,
    used for the retrieval-screen overlay and the saved output. So the overlaid
    pulse is self-consistent, the spectral panel uses the beamlet spectrum
    (``scan.Iomega_beamlet``) when the beamlet truth is shown and the file
    provides it, else the source spectrum. The spectrum uses the photon (×ω²)
    convention of the retrieved spectral panel; only positive frequencies carry a
    wavelength.
    """
    pos = scan.omega > 0.0
    omega_pos = scan.omega[pos]
    # The recorded τfwhm is the *source* input FWHM, so only trust it for the
    # source truth; for the beamlet, measure the FWHM from its own envelope and
    # show the matching beamlet spectrum so the overlaid pulse is self-consistent.
    if truth_source == "beamlet" and scan.Iomega_beamlet is not None:
        Iomega, fwhm = scan.Iomega_beamlet, None
    else:
        Iomega, fwhm = scan.Iomega, scan.tau_fwhm
    Eomega = _simulated_truth_spectrum(scan, truth_source)
    phi = None
    phi_t = None
    if Eomega is not None:
        Ep = Eomega[pos]
        phase_intensity = np.abs(Ep) ** 2
        if np.any(phase_intensity > 0):
            mask = phase_intensity > 0.01 * phase_intensity.max()
            phi = normalize_spectral_phase(scan.omega[pos], Ep, mask, phase_intensity)
        phi_t = _truth_temporal_phase(scan, Eomega)
    return TruthPulse.from_intensity(
        scan.t,
        scan.It,
        lam=2.0 * np.pi * C / omega_pos,
        Iw=Iomega[pos] * omega_pos**2,
        phi_w=phi,
        phi_t=phi_t,
        omega=scan.omega if Eomega is not None else None,
        Eomega=Eomega,
        fwhm=fwhm,
    )


def assemble_simulated_tracedata(p: SimulatedLoadParams) -> TraceData:
    r"""Build a :class:`~croak.preprocess.TraceData` directly from a simulated scan.

    Loads the **as-simulated** angular-frequency-density trace onto the
    simulation's **native FFT grid** and packages it as a retrieval-ready
    :class:`~croak.preprocess.TraceData`, bypassing both the
    :func:`assemble_simulated_load_data` ``÷λ²`` and the :func:`build_tracedata`
    regrid/filtering. This is the "skip marginal-check and preprocess" path: it
    lets the retrieval be sanity-checked with no doubt about those steps.

    It is **fully raw** — no third-order ``(λ/µm)^n`` correction — but still
    honours the window/thickness selection, the delay-sign convention
    (``reverse_trace``) and the known-spectrum choice (``use_spectrum`` /
    ``spectrum_source``). ``lam_min_nm``/``lam_max_nm`` are ignored; the whole
    native grid is kept.

    Parameters
    ----------
    p : SimulatedLoadParams
        The simulated-load options.

    Returns
    -------
    TraceData
        On the simulation's native grid (``grid.omega + omega0_pulse ==
        scan.omega``); ``trace`` is the raw ω-density ``(Nomega, Ndelay)``
        normalised to unit peak.

    Raises
    ------
    ValueError
        If the file's ``ω`` axis is not a uniform FFT grid centred on the carrier
        (the retrieval needs a uniform baseband grid).
    """
    z_thickness = None if p.z_thickness_um is None else p.z_thickness_um * 1e-6
    scan = io.read_simulated_scan(
        p.frog_path,
        window_key=p.window_key,
        z_index=p.z_index,
        z_thickness=z_thickness,
        truth_source=p.truth_source,
    )

    # Reconstruct the simulation's native FFT grid: Grid(n, domega) rebuilds the
    # centred baseband ω axis, and grid.omega + ω0 must reproduce scan.omega — a
    # hard check that the file really is a uniform FFT grid centred on the carrier
    # (the retrieval represents the pulse on exactly such a grid).
    omega = scan.omega
    n = omega.size
    domega = float(np.mean(np.diff(omega)))
    grid = Grid(n, domega=domega)
    omega0_pulse = float(scan.omega0)
    if not np.allclose(grid.omega + omega0_pulse, omega, rtol=0.0, atol=1e-3 * domega):
        raise ValueError(
            "simulated scan ω-axis is not a uniform FFT grid centred on ω0; "
            "cannot load it directly into the retrieval (use the regrid path)"
        )
    inter = get_interaction(p.interaction)
    omega0_trace = omega0_pulse * inter.omega0_scale  # 1× for pg/sd (no doubling)

    # Raw ω-density trace on the native grid; honour only the delay-sign
    # convention (no third-order, no λ² conversion, no resampling).
    trace = scan.trace.astype(float)
    delays = -scan.tau if _resolve_reverse_trace(p, scan) else scan.tau.astype(float)
    order = np.argsort(delays)
    delays = delays[order]
    trace = trace[:, order]
    # Centre delay=0 on the delay-marginal peak, to sub-sample precision and by
    # the same helper regrid uses. Reading the peak off as the largest *sample*
    # quantises time zero to the delay step, and a scan whose true zero falls
    # between two bins — an even, symmetric delay axis does exactly that — is
    # then modelled up to half a step away from where it was measured. That is
    # a pure time translation, which a retrieval absorbs into a linear spectral
    # phase, but an *un-translated* known field pays for it in full: it floored
    # the truth-seeded forward-model check at R ~ 5e-3 instead of ~1e-16.
    delays = delays - preprocess.marginal_peak_delay(delays, trace.sum(axis=0))
    mx = float(np.max(np.abs(trace)))
    if mx > 0:
        trace = trace / mx

    # Independent spectrum on the native grid (already == grid.omega + ω0_pulse).
    ref_full = None
    if p.use_spectrum:
        if p.spectrum_source == "beamlet" and scan.Iomega_beamlet is not None:
            ref_full = scan.Iomega_beamlet.astype(float)
        else:
            ref_full = scan.Iomega.astype(float)
    Iomega = None
    if ref_full is not None:
        m = float(np.max(ref_full))
        Iomega = ref_full / m if m > 0 else ref_full

    # Wavelength-domain provenance (positive-ω bins, ascending λ) and the plot
    # extents (the signal band, for sensible spectral axis limits).
    pos = omega > 0.0
    lam_pos = 2.0 * np.pi * C / omega[pos]
    sp = np.argsort(lam_pos)
    lam_frog = lam_pos[sp]
    Ifrog = trace[pos][sp]
    marg = Ifrog.sum(axis=1)
    sig = lam_frog[marg > 1e-2 * marg.max()] if marg.max() > 0 else lam_frog
    lam_min, lam_max = float(sig.min()), float(sig.max())

    lam_spec = Ilam_spec = None
    if ref_full is not None:
        lam_spec = lam_frog
        Ilam_spec = preprocess.omega_to_lambda_density(ref_full[pos][sp], lam_frog)
        ms = float(np.max(Ilam_spec))
        if ms > 0:
            Ilam_spec = Ilam_spec / ms

    return TraceData(
        interaction=inter.name,
        grid=grid,
        delays=delays,
        trace=trace,
        omega0_pulse=omega0_pulse,
        omega0_trace=omega0_trace,
        Iomega=Iomega,
        lam_min=lam_min,
        lam_max=lam_max,
        lamm_lims=(lam_min, lam_max),
        tau_lims=(float(delays.min()), float(delays.max())),
        lam_frog=lam_frog,
        tau_meas=delays,
        Ifrog_meas=Ifrog,
        tau_filt=delays,
        Ifrog_filt=Ifrog,
        lam_spec=lam_spec,
        Ilam_spec_meas=Ilam_spec,
        Ilam_spec_filt=Ilam_spec,
        scalecurve=np.ones(lam_frog.shape),
    )


# ---------------------------------------------------------------------------
# Preprocess: PreprocParams + load_data -> TraceData
# ---------------------------------------------------------------------------
#: Coarse-grid downsample and row-skip used for the GUI's *interactive* baseline
#: preview (``fast_baseline=True``). The baseline is smooth, so solving on every
#: 4th delay and skipping near-zero rows is ~3× faster at ~1% baseline difference;
#: the committed/headless path stays exact.
_FAST_BASELINE_DOWNSAMPLE = 4
_FAST_BASELINE_SKIP_BELOW = 0.01


def build_tracedata(
    p: PreprocParams, data: dict, *, fast_baseline: bool = False
) -> TraceData:
    """Clean and regrid a loaded trace into a :class:`~croak.preprocess.TraceData`.

    Maps the (nm/fs) preprocess parameters onto
    :func:`croak.preprocess.load_and_clean` SI arguments.

    ``fast_baseline`` (off by default) solves the arPLS baseline on a coarse delay
    grid and skips near-zero rows — a ~3× speedup at ~1% baseline difference, for
    the GUI's interactive preview. Headless replay and direct callers get the
    exact, full-resolution baseline.
    """
    ds = _FAST_BASELINE_DOWNSAMPLE if fast_baseline else 1
    skip = _FAST_BASELINE_SKIP_BELOW if fast_baseline else 0.0
    return preprocess.load_and_clean(
        data["trace"],
        data["lam"],
        data["scanaxis"],
        data["interaction"],
        lam_min=p.lam_min_nm * 1e-9,
        lam_max=p.lam_max_nm * 1e-9,
        lamm_lims=(p.lamm_min_nm * 1e-9, p.lamm_max_nm * 1e-9),
        input_unit=data["input_unit"],
        trange=p.trange_fs * 1e-15,
        tau_crop=(p.tau_min_fs * 1e-15, p.tau_max_fs * 1e-15),
        tau_lims=(p.taum_min_fs * 1e-15, p.taum_max_fs * 1e-15),
        filter_fringes=p.filter_fringes,
        filter_dc=p.filter_dc,
        dtau_min=p.dtau_min_fs * 1e-15 if p.dtau_min_fs > 0 else None,
        defringe=p.defringe,
        defringe_fraction=p.defringe_fraction,
        on_alias=p.defringe_on_alias,
        baseline=p.baseline,
        baseline_smoothness=p.baseline_smoothness,
        baseline_ratio=p.baseline_ratio,
        baseline_clip=p.baseline_clip,
        baseline_tau_exclude=(
            p.baseline_tau_exclude_fs * 1e-15 if p.baseline_tau_exclude_fs > 0 else None
        ),
        baseline_downsample=ds,
        baseline_skip_below=skip,
        resample_t=p.resample_t,
        subsample=p.subsample_tau if p.use_subsample else None,
        threshold=p.threshold if p.use_threshold else None,
        marginal_correct=p.marginal_correct,
        prefilter=p.prefilter,
        lam_spec=data["lam_spec"],
        Ilam_spec=data["Ilam_spec"],
    )


# ---------------------------------------------------------------------------
# Retrieve: RetrieveParams + TraceData -> RetrievalResult
# ---------------------------------------------------------------------------
def resolve_guess(p: RetrieveParams, override=None):
    """Pick the initial-guess mode (or an explicit spectrum) from the parameters.

    Initial-condition flags are mutually exclusive. An explicit ``override``
    supplies the spectrum for ``reuse_result``; without a compatible previous
    result that GUI-only choice falls back to ``"auto"``.
    """
    selected = [
        name
        for enabled, name in (
            (p.random_init, "random"),
            (p.perfect_init, "perfect"),
            (p.tl_init, "tl"),
            (p.truth_init, "truth"),
            (p.reuse_result, "reuse"),
        )
        if enabled
    ]
    if len(selected) > 1:
        raise ValueError(
            "initial guesses are mutually exclusive; selected " + ", ".join(selected)
        )
    if selected == ["reuse"]:
        return override if override is not None else "auto"
    if override is not None:
        return override
    return selected[0] if selected else "auto"


def smearing_kernel(p: RetrieveParams, td: TraceData) -> SmearingKernel | None:
    """Build the stage-4 geometric-smearing kernel, or ``None`` when it is off.

    The mask geometry is the stored truth (the displayed delay width is a derived
    GUI convenience), and the carrier comes from the measurement itself, so the
    kernel is always evaluated at the wavelength the trace was actually taken at.
    """
    if not p.smearing:
        return None
    if p.smear_layout == "sd2":
        # Two-beam self-diffraction: sigma_p is identically zero, so this is
        # NOT the "sd" branch of the boxcars closed form (which assumes two
        # separate unconjugated holes and would add a ~0.4 fs p channel).
        kernel = collinear_sd_kernel(
            hole_diameter=p.smear_hole_diameter_mm * 1e-3,
            hole_spacing=p.smear_hole_spacing_mm * 1e-3,
            wavelength=2.0 * np.pi * C / td.omega0_pulse,
            npoints=p.smear_nodes,
        )
    else:
        kernel = square_boxcars_kernel(
            td.interaction,
            hole_diameter=p.smear_hole_diameter_mm * 1e-3,
            hole_spacing=p.smear_hole_spacing_mm * 1e-3,
            wavelength=2.0 * np.pi * C / td.omega0_pulse,
            npoints=p.smear_nodes,
        )
    # A fixed width multiplier, distinct from fit_smearing: the gap can only
    # reach d/D >= 0.5, so narrower kernels are otherwise unreachable.
    return kernel if p.smear_scale == 1.0 else kernel.scaled(p.smear_scale)


def generation_envelope(p: RetrieveParams):
    r"""Build the generation envelope g(z) at the depth-quadrature nodes, or ``None``.

    The Gaussian-crossing form: a Rayleigh amplitude droop and beamlet
    walk-through, with a net Gouy phase,

    .. math::

        g(z) = (1+u^2)^{-1/2}\\, e^{-((z-z_f)/L_w)^2/(1+u^2)}\\,
               e^{i\\gamma \arctan u},\\qquad u = (z - z_f)/z_R,

    evaluated at the Gauss--Legendre nodes of the depth quadrature. All lengths
    are in-medium values, measured from the slab entrance (``envelope_*`` fields
    of :class:`~croak.session.params.RetrieveParams`). Negligible for slabs much
    thinner than ``z_R``; see ``docs/explanation/forward_model.md``.
    """
    if not (p.envelope and p.dispersive):
        return None
    if p.envelope_zr_um <= 0 or p.envelope_lw_um <= 0:
        raise ValueError(
            "the generation envelope requires envelope_zr_um > 0 and envelope_lw_um > 0"
        )
    nodes, _ = quadrature_nodes_weights(
        p.thickness_um * 1e-6, p.npoints, "gausslegendre"
    )
    z_f = p.envelope_zf_um * 1e-6
    u = (nodes - z_f) / (p.envelope_zr_um * 1e-6)
    amp = (1.0 + u**2) ** -0.5 * np.exp(
        -(((nodes - z_f) / (p.envelope_lw_um * 1e-6)) ** 2) / (1.0 + u**2)
    )
    return amp * np.exp(1j * p.envelope_gouy * np.arctan(u))


def extra_param_centres(
    p: RetrieveParams, seed: RetrievalResult | None
) -> tuple[float, float, float, float]:
    """Return the starting centres ``(thickness, tau0, smear_scale, smear_delta)``.

    Without a ``seed`` these are the nominal settings: the slab thickness from
    the parameters, a zero delay-zero offset and an unscaled smearing kernel.

    With a ``seed`` (the previous retrieval, when reusing it) each extra that is
    **being fitted again** starts from the value that fit found, so re-running is
    a true warm start: the reused spectrum was optimised jointly with those
    extras, so restarting them from their priors would re-fit it against, e.g., a
    differently-referenced delay axis. An extra whose ``fit_*`` flag is off keeps
    the explicit nominal value instead — an unticked box means "hold this at what
    I set", not "hold it at whatever the last fit drifted to".

    Parameters
    ----------
    p : RetrieveParams
        Retrieval settings (supplies the nominal thickness and the fit flags).
    seed : RetrievalResult or None
        Previous result to warm-start from, or ``None`` for a cold start.

    Returns
    -------
    tuple of float
        Thickness (m), delay-zero offset (s), smearing-width multiplier, and the
        delay-channel multiplier of a split smearing fit (``fit_smearing_split``;
        equal to the joint multiplier's default ``1.0`` otherwise).
    """
    thickness = p.thickness_um * 1e-6 if p.dispersive else 0.0
    tau0 = 0.0
    smear_scale = 1.0
    smear_delta = 1.0
    if seed is None:
        return thickness, tau0, smear_scale, smear_delta
    if p.fit_thickness and p.dispersive and seed.thickness is not None:
        thickness = float(seed.thickness)
    if p.fit_tau0:
        tau0 = float(seed.tau0)
    if p.fit_smearing and p.smearing and seed.smear_scale is not None:
        smear_scale = float(seed.smear_scale)
    if (
        p.fit_smearing_split
        and p.fit_smearing
        and p.smearing
        and seed.smear_scale_delta is not None
    ):
        smear_delta = float(seed.smear_scale_delta)
    return thickness, tau0, smear_scale, smear_delta


def focal_mixture_from_params(
    p: RetrieveParams, td: TraceData, scan_path: str | None = None
) -> FocalMixture | None:
    """Build the chromatic focal mixture from ``RetrieveParams``, or ``None``.

    The focal-model counterpart of :func:`smearing_kernel`: the mask geometry
    comes from the same ``smear_hole_*`` fields (it is the same mask), the
    carrier from the measurement, and the focusing focal length — which the
    reduced kernel never needs but the explicit mixture does — from
    ``focal_f_mm``. The collection model rides on the mixture per
    ``p.collection``:

    - ``"off"``: incoherent full-beam sum (exact only for full collection);
    - ``"file"``: the aperture rebuilt from the simulated scan's own
      ``window_def_*`` record (needs ``scan_path``);
    - ``"manual"``: a hole of ``collection_diam_mm`` at the phase-matched
      BOXCARS corner ``(-d, -d)``, ``d = (gap + D)/2``, hard-edged — the
      physical pinhole of an experiment. (``"file"`` reproduces a simulated
      scan's soft tanh edge exactly, which exists there as a grid-resolution
      device, not physics.)

    Raises
    ------
    ValueError
        If ``p.smearing`` is also set (the mixture replaces the reduced
        kernel), if the interaction is not PG (the mixture's arm layout is
        the four-hole BOXCARS TG/PG one), if ``collection="file"`` without a
        ``scan_path``, or on an unknown ``collection`` mode.
    """
    if not p.focal:
        return None
    if p.smearing:
        raise ValueError(
            "focal and smearing are mutually exclusive: the chromatic focal "
            "mixture replaces the reduced kernel — disable one of them"
        )
    if td.interaction != "pg":
        raise ValueError(
            f"the focal mixture models the four-hole BOXCARS TG/PG layout; "
            f"interaction is {td.interaction!r}"
        )
    lam0 = 2.0 * np.pi * C / td.omega0_pulse
    collection = None
    if p.collection == "file":
        if not scan_path:
            raise ValueError(
                "collection='file' rebuilds the aperture from the scan's own "
                "window_def record and needs the scan path"
            )
        collection = aperture_from_scan(scan_path, mode=p.collection_mode)
    elif p.collection == "manual":
        d = (p.smear_hole_spacing_mm + p.smear_hole_diameter_mm) / 2.0 * 1e-3
        collection = mask_hole_aperture(
            hole_x=-d,
            hole_y=-d,
            hole_diameter=p.collection_diam_mm * 1e-3,
            z_mask=p.focal_f_mm * 1e-3,
            apod="hard",
            mode=p.collection_mode,
        )
    elif p.collection != "off":
        raise ValueError(
            f"unknown collection mode {p.collection!r}; "
            "expected 'off', 'file' or 'manual'"
        )
    return focal_mixture(
        hole_diameter=p.smear_hole_diameter_mm * 1e-3,
        hole_spacing=p.smear_hole_spacing_mm * 1e-3,
        f_foc=p.focal_f_mm * 1e-3,
        wavelength=lam0,
        n_radial=p.focal_n_radial,
        n_azimuth=p.focal_n_azimuth,
        r_max_units=p.focal_rmax_units,
        collection=collection,
    )


def apply_spectrum_frame(p: RetrieveParams, td: TraceData) -> TraceData:
    """Reweight the measured spectrum into the retrieval frame, if requested.

    Multiplies ``td.Iomega`` by ``((omega + omega0)/omega0)**(2 p)`` with
    ``p = spectrum_frame_p``, moving the initial guess and the
    ``reg_spectrum`` target together into the frame the forward model's
    ``ew`` lives in. The focal mixture's ``ew`` is the ON-AXIS focal field,
    bluer than a spatially integrated spectrum;
    the reduced kernel's implicit frame is the integrated one, so leave the
    exponent at 0 there. A no-op when the exponent is 0 or there is no
    independent spectrum.
    """
    if p.spectrum_frame_p == 0.0 or td.Iomega is None:
        return td
    om = np.asarray(td.omega, float)
    wfac = np.maximum((om + td.omega0_pulse) / td.omega0_pulse, 0.0)
    iom = np.asarray(td.Iomega, float) * wfac ** (2.0 * p.spectrum_frame_p)
    return dataclass_replace(td, Iomega=iom)


def run_retrieval(
    p: RetrieveParams,
    td: TraceData,
    *,
    guess_override=None,
    extras_seed: RetrievalResult | None = None,
    truth: TruthPulse | None = None,
    rng: np.random.Generator | None = None,
    callback=None,
    focal=None,
    scan_path: str | None = None,
) -> RetrievalResult:
    """Run :func:`~croak.pipeline.retrieve_from_tracedata` from a ``RetrieveParams``.

    Parameters
    ----------
    p : RetrieveParams
        Retrieval settings (solver, regularisation, dispersive slab, …).
    td : TraceData
        Cleaned, regridded trace.
    guess_override : array_like or Pulse, optional
        Explicit initial spectrum (reuse-previous-result); overrides the modes.
    extras_seed : RetrievalResult, optional
        Previous result to warm-start the fitted extra parameters from — passed
        together with ``guess_override`` when reusing a result. See
        :func:`extra_param_centres`.
    truth : TruthPulse, optional
        Known pulse from the load stage. Required when ``p.truth_init`` is set;
        its complex spectrum is aligned to ``td`` and used as the solver seed.
    rng : numpy.random.Generator, optional
        Generator for the random guess / solver; a fresh default is used if None.
    callback : callable, optional
        ``callback(iteration, R, best_R, snapshot=...)`` progress hook; the
        optional ``snapshot`` is a zero-argument in-progress-result builder (or
        ``None``) for a live full-plot preview.
    focal : FocalMixture, optional
        Explicit chromatic focal-plane mixture (optionally with a collection
        aperture), forwarded to
        :func:`~croak.pipeline.retrieve_from_tracedata`. When ``None`` (the
        default) and ``p.focal`` is set, the mixture is built from the params
        by :func:`focal_mixture_from_params` — the route the GUI uses; pass
        an explicit object to override it from a script. The mixture replaces
        the reduced kernel, so ``p.smearing`` must be off either way.
    scan_path : str, optional
        Path of the loaded scan file, needed only for
        ``p.collection == "file"`` (the aperture is rebuilt from the file's
        own ``window_def_*`` record).
    """
    rng = np.random.default_rng() if rng is None else rng
    if focal is None:
        focal = focal_mixture_from_params(p, td, scan_path)
    td = apply_spectrum_frame(p, td)
    thickness, tau0, smear_scale, smear_delta = extra_param_centres(p, extras_seed)
    return retrieve_from_tracedata(
        td,
        algorithm=p.solver,
        full=p.full,
        R_omega=p.R_omega,
        guess=resolve_guess(p, guess_override),
        perfect_fwhm=p.perfect_init_fs * 1e-15,
        truth=truth,
        material=p.material if p.dispersive else None,
        thickness=thickness,
        npoints=p.npoints,
        smearing=smearing_kernel(p, td),
        focal=focal,
        depth_weight=generation_envelope(p),
        reg_amp=p.reg_amp,
        reg_phase=p.reg_phase,
        reg_spectrum=p.reg_spectrum,
        reg_time=p.reg_time,
        time_window=(p.reg_time_lo_fs * 1e-15, p.reg_time_hi_fs * 1e-15),
        # warm-start centres for the extras (nominal unless reusing a result)
        tau0=tau0,
        smear_scale=smear_scale,
        smear_scale_delta=smear_delta,
        # thickness fitting needs a dispersive slab (the solver guards SHG)
        fit_thickness=p.fit_thickness and p.dispersive,
        fit_tau0=p.fit_tau0,
        # the smearing strength is only fittable when a kernel is being modelled
        fit_smearing=p.fit_smearing and p.smearing,
        fit_smearing_split=p.fit_smearing_split and p.fit_smearing and p.smearing,
        polish=p.polish_two_phase,
        phase_basis=p.phase_basis,
        n_nodes=p.n_nodes,
        strategy=p.cma_strategy,
        popsize=p.cma_popsize,
        std_init=p.cma_std_init,
        maxiters=p.maxiters,
        reltol=p.reltol,
        abstol=p.abstol,
        alpha=p.copra_alpha,
        stall_patience=p.copra_stall_patience,
        rng=rng,
        callback=callback,
    )


# ---------------------------------------------------------------------------
# Post-retrieval cleanup: optional windowing + processing for plots/FWHM
# ---------------------------------------------------------------------------
def apply_post_filter(
    p: RetrieveParams, result: RetrievalResult, td: TraceData
) -> RetrievalResult:
    """Window the result to the measurement λ/τ ranges if the toggles are set.

    Mirrors the retrieve stage's post-filter pane: a spectral and/or temporal
    window to the measurement extents (``td.lamm_lims`` / ``td.tau_lims``), not
    the full grid. Returns the result unchanged when neither toggle is on.
    """
    lam_lims = td.lamm_lims if p.post_filter_lam else None
    tau_lims = td.tau_lims if p.post_filter_tau else None
    if lam_lims is None and tau_lims is None:
        return result
    return post_filter(result, lam_lims=lam_lims, tau_lims=tau_lims)


def process(
    result: RetrievalResult, td: TraceData, *, energy: float | None = None
) -> ProcessedResult:
    """Process a result against its measured trace (FWHM, dispersion fit, marginals).

    ``energy`` is an optional measured pulse energy (J); when given, the result
    carries absolute peak power (see :func:`croak.processing.process_result`).
    """
    return process_result(
        result, measured=td.trace, Iomega_meas=td.Iomega, energy=energy
    )


# ---------------------------------------------------------------------------
# Uncertainty: UncertaintyParams + RetrieveParams + result -> UncertaintyResult
# ---------------------------------------------------------------------------
def bootstrap_solver_kwargs(p: RetrieveParams, td: TraceData) -> dict:
    """Solver kwargs for the uncertainty bootstrap (matching the retrieval).

    :func:`~croak.uncertainty.estimate_fwhm_uncertainty` rebuilds the solver
    internally and filters these by what the chosen algorithm accepts, so it is
    safe to pass the full candidate set.
    """
    return {
        "algorithm": p.solver,
        "maxiters": p.maxiters,
        "reltol": p.reltol,
        "abstol": p.abstol,
        "material": p.material if p.dispersive else None,
        "thickness": p.thickness_um * 1e-6 if p.dispersive else 0.0,
        "npoints": p.npoints,
        "smearing": smearing_kernel(p, td),
        "depth_weight": generation_envelope(p),
        "phase_only": not p.full,
        "phase_basis": p.phase_basis,
        "n_nodes": p.n_nodes,
        "R_omega": p.R_omega,
        "reg_amp": p.reg_amp,
        "reg_phase": p.reg_phase,
        "reg_spectrum": p.reg_spectrum,
        "spectrum_target": td.Iomega if p.reg_spectrum > 0 and p.full else None,
        "reg_time": p.reg_time,
        "time_window": (p.reg_time_lo_fs * 1e-15, p.reg_time_hi_fs * 1e-15),
        "fit_thickness": p.fit_thickness and p.dispersive,
        "fit_tau0": p.fit_tau0,
        "fit_smearing": p.fit_smearing and p.smearing,
        "fit_smearing_split": p.fit_smearing_split and p.fit_smearing and p.smearing,
        "polish": p.polish_two_phase,
        "strategy": p.cma_strategy,
        "popsize": p.cma_popsize,
        "std_init": p.cma_std_init,
    }


def covariance_kwargs(p: RetrieveParams, td: TraceData | None = None) -> dict:
    """Forward-model + parameterisation kwargs for the covariance estimator.

    Mirrors the dispersive-slab, smearing and pulse-parameterisation settings of
    the retrieval (the subset :func:`croak.covariance.covariance_uncertainty`
    consumes), so the analytic covariance is rebuilt on exactly the model that was
    fitted. ``td`` is needed only to rebuild the smearing kernel; without it the
    kernel is omitted.
    """
    return {
        "material": p.material if p.dispersive else None,
        "thickness": p.thickness_um * 1e-6 if p.dispersive else 0.0,
        "npoints": p.npoints,
        "smearing": smearing_kernel(p, td) if td is not None else None,
        "phase_only": not p.full,
        "phase_basis": p.phase_basis,
        "n_nodes": p.n_nodes,
        "R_omega": p.R_omega,
        "fit_thickness": p.fit_thickness and p.dispersive,
        "fit_tau0": p.fit_tau0,
        "fit_smearing": p.fit_smearing and p.smearing and td is not None,
        "fit_smearing_split": (
            p.fit_smearing_split and p.fit_smearing and p.smearing and td is not None
        ),
    }


def run_uncertainty(
    up: UncertaintyParams,
    rp: RetrieveParams,
    result: RetrievalResult,
    td: TraceData,
    *,
    dispersion: DispersionParams | None = None,
    rng: np.random.Generator | None = None,
    callback=None,
) -> UncertaintyResult:
    """Run an FWHM-uncertainty estimate from the saved parameters.

    Dispatches on ``up.method``: the resampling bootstraps (parametric / delay /
    frequency), the substrate-thickness systematic, or the analytic ``covariance``
    estimator. When ``up.propagate_to_dispersion_point`` is set and ``dispersion``
    applies any dispersion, every replicate is propagated to that beamline point
    via :func:`~croak.session.dispersion.transfer_function` before the FWHM/band is
    taken.
    """
    rng = np.random.default_rng() if rng is None else rng
    measured = td.trace

    propagation = None
    propagation_label = None
    if (
        up.propagate_to_dispersion_point
        and dispersion is not None
        and has_dispersion(dispersion)
    ):
        propagation = transfer_function(dispersion, result.grid, result.omega0)
        propagation_label = "dispersion-stage point"

    if up.method == "covariance":
        noise = (
            noise_from_residual(measured, result)
            if up.noise_source == "residual"
            else NoiseModel(sigma=up.noise_sigma)
        )
        return covariance_uncertainty(
            result,
            measured,
            noise=noise,
            propagation=propagation,
            propagation_label=propagation_label,
            n_samples=up.n_resamples,
            collect_profiles=up.collect_profiles,
            interval=up.interval,
            rng=rng,
            callback=callback,
            omega0=result.omega0,
            **covariance_kwargs(rp, td),
        )

    kw = bootstrap_solver_kwargs(rp, td)
    noise = None
    extra: dict = {}
    if up.method == "parametric":
        noise = (
            noise_from_residual(measured, result)
            if up.noise_source == "residual"
            else NoiseModel(sigma=up.noise_sigma)
        )
    elif up.method == "thickness":
        # Central thickness from the retrieval settings; 1σ from the uncertainty
        # stage. The dispersive pair is injected per draw, so drop it from kw to
        # avoid colliding with the explicit `material=` keyword below.
        extra = {
            "central_thickness": rp.thickness_um * 1e-6,
            "thickness_sigma": up.thickness_sigma_um * 1e-6,
            "material": rp.material,
        }
        # Bootstrapping the thickness systematic and *fitting* the thickness are
        # alternative ways to account for it — never both, so drop the fit here.
        kw = {
            k: v
            for k, v in kw.items()
            if k not in ("material", "thickness", "fit_thickness", "polish")
        }
    return estimate_fwhm_uncertainty(
        result,
        measured,
        method=up.method,
        noise=noise,
        n_resamples=up.n_resamples,
        warm_start=up.warm_start,
        collect_profiles=up.collect_profiles,
        interval=up.interval,
        propagation=propagation,
        propagation_label=propagation_label,
        rng=rng,
        callback=callback,
        **extra,
        **kw,
    )
