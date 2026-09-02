"""Per-stage parameter dataclasses for a retrieval session.

These plain dataclasses describe a complete retrieval — loading, preprocessing,
retrieval, dispersion compensation and uncertainty — independently of any user
interface. The GUI (:mod:`croak.gui`) binds its controls to them and the headless
engine (:mod:`croak.session.engine`) replays them; both share this one schema, so
a session saved from the GUI can be rerun from a script and vice versa.

The dataclasses are deliberately Qt-free: they carry only data and a little
classification logic (which load paths follow the dataset folder vs. which are
fixed, used by :mod:`croak.session.retarget`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "LoadParams",
    "SimulatedLoadParams",
    "PreprocParams",
    "RetrieveParams",
    "BeamPathMirror",
    "DispersionParams",
    "UncertaintyParams",
    "REG_DEFAULTS",
    "reg_defaults",
    "reg_family",
]


# Re-exported from croak.solver, their canonical home since the solver
# constructors themselves resolve these defaults; kept here because the GUI
# and session code historically import them from this module.
from ..solver import (  # noqa: E402  (re-export)
    REG_DEFAULTS,
    reg_defaults,
    reg_family,
)


@dataclass
class LoadParams:
    """Stage 1: which files/datasets/units make up the measured trace.

    Holds the FROG trace file and its wavelength/scan datasets and units, the
    optional independent spectrum, the optional backgrounds and spectral-response
    calibration curves, and the optional measured pulse energy used to rescale
    the temporal plots to absolute power.
    """

    frog_path: str = ""
    ifrog_name: str = ""
    lam_name: str = ""
    scan_name: str = ""
    transpose: bool = False
    lam_unit: str = "nm"
    scan_unit: str = "fs"
    scan_type: str = "delay"
    interaction: str = "shg"
    third_order: bool = False
    third_order_exp: float = 4.0
    # Measured pulse energy (J), e.g. from a power meter. Optional measurement
    # metadata, not derived from the trace: when > 0 the temporal envelope plots
    # are rescaled from normalised "a.u." to absolute instantaneous power (and a
    # peak power is reported); 0 means unspecified (keep normalised units).
    energy_j: float = 0.0
    # independent spectrum (CSV columns or HDF5/NPZ datasets)
    use_spec: bool = False
    spec_path: str = ""
    spec_is_csv: bool = True
    spec_lam_name: str = ""  # HDF5/NPZ dataset name
    spec_int_name: str = ""
    spec_lam_col: int = 0  # CSV column index
    spec_int_col: int = 1
    spec_lam_unit: str = "nm"
    # FROG background (own λ axis + spectrum/trace dataset)
    frog_bg_use: bool = False
    frog_bg_path: str = ""
    frog_bg_lam_name: str = ""
    frog_bg_int_name: str = ""
    # FROG spectral-response calibration curve (λ, scale CSV)
    frog_calib_use: bool = False
    frog_calib_path: str = ""
    frog_calib_lam_unit: str = "nm"
    # spectrum background / calibration
    spec_bg_use: bool = False
    spec_bg_path: str = ""
    spec_calib_use: bool = False
    spec_calib_path: str = ""
    spec_calib_lam_unit: str = "nm"

    @staticmethod
    def dataset_path_fields() -> tuple[str, ...]:
        """Path fields that live *inside* the per-dataset folder.

        These move together when an :func:`~croak.session.retarget.retarget`
        points an existing session at a new dataset folder: the measured trace,
        the independent spectrum, and their backgrounds all share that folder and
        keep their filenames across acquisitions.
        """
        return ("frog_path", "spec_path", "frog_bg_path", "spec_bg_path")

    @staticmethod
    def fixed_path_fields() -> tuple[str, ...]:
        """Path fields that stay put when retargeting (shared calibration files)."""
        return ("frog_calib_path", "spec_calib_path")


@dataclass
class SimulatedLoadParams:
    r"""Stage 1 (alternative): load a numerically simulated FROG scan.

    Describes how to turn a Luna ``scansave`` HDF5 file (see
    :func:`croak.io.read_simulated_scan`) into the same ``load_data`` dict the
    measured-trace loader produces. Unlike :class:`LoadParams` the file layout is
    fixed, so there are no dataset-name/unit pickers — only the physics options:
    which trace window, the loaded wavelength band, the third-order efficiency
    correction, the delay-sign convention, the nonlinear interaction, and which
    independent spectrum (if any) to feed the retrieval.

    Notes
    -----
    The simulated trace is stored as an angular-frequency density
    :math:`I_\omega`; the loader divides it by :math:`\lambda^2` so the
    wavelength pipeline's :math:`\times\lambda^2` regrid recovers it undistorted
    (see :func:`croak.preprocess.omega_to_lambda_density`).

    The chromatic vignetting imprinted by the physical input mask is a linear
    filter on the *gating beam*, not a trace artefact — so it belongs in the
    reference spectrum: choose ``spectrum_source="beamlet"`` (the post-mask
    ``grid/Iω_beamlet``) so the retrieval is constrained by the spectrum that
    actually gates. The leftover ``ω``-dependence is then close to the physical
    radiated-field factor (``≈ ω²`` for the on-axis re-imaged trace), which the
    third-order correction divides out.
    """

    frog_path: str = ""
    #: ``"Iω_win"`` (integrated over all k) or ``"Iω_win_reimaged"`` (on-axis).
    window_key: str = "Iω_win"
    #: Propagation slice index (``-1`` = the last saved slice, i.e. substrate exit); the
    #: low-level/back-compat knob, used only when ``z_thickness_um`` is ``None``.
    z_index: int = -1
    #: Material thickness (µm) to load from a multi-thickness scan. When set, the
    #: nearest saved ``/grid/zsave`` slice is loaded (takes precedence over
    #: ``z_index``); ``None`` falls back to ``z_index`` (the exit slice), which
    #: also covers the legacy single-thickness files that lack ``zsave``.
    z_thickness_um: float | None = None
    # wavelength band to load (the unphysical ω ≤ 0 bins are always dropped); the
    # retrieval grid range is chosen later in the preprocess stage.
    lam_min_nm: float = 140.0
    lam_max_nm: float = 800.0
    # third-order (χ³) efficiency correction: divide out the ∝ ω^n generation
    # response, equivalently multiply by (λ/µm)^n. The true response is a *curve*
    # (radiated ω² × chromatic collection), so a single exponent is only an
    # approximation — the better routes are the retrieval's per-frequency Rω
    # scaling or phase-only retrieval (see the manual). The default 2 matches the
    # radiated-field ω² factor; the marginal-check stage's Auto refines it.
    third_order: bool = True
    third_order_exp: float = 2.0
    # Delay-sign convention. ``None`` (default) auto-detects from the file's
    # ``/grid/delay_convention`` marker: files stored in the gate-delay (paper)
    # frame load unreversed; legacy marker-less files are negated as before.
    # An explicit ``True``/``False`` overrides the marker.
    reverse_trace: bool | None = None
    #: Keep only the longest uniformly spaced contiguous stretch of the delay
    #: axis (:func:`croak.preprocess.uniform_core_indices`). Campaign scans
    #: often extend a uniform core with coarser wing points (e.g. 1 fs-step
    #: wings to ±40 fs for the Raman wake); the smearing kernel's delay-axis
    #: convolution needs a uniform grid, so retrievals that model smearing
    #: must drop the wings. Off by default: the wings are real data and every
    #: model without a smearing kernel can use them.
    uniform_delay_core: bool = False
    #: ``"pg"`` (transient-grating/PG) or ``"sd"``; both keep the carrier.
    interaction: str = "pg"
    # feed the simulation's known input spectrum as the independent spectrum
    # (enables spectral regularisation); turn off for a blind test
    use_spectrum: bool = True
    #: which spectrum when ``use_spectrum``: ``"beamlet"`` (post-mask
    #: ``grid/Iω_beamlet``, the transverse-integrated beam that gates — the
    #: 1-D models' frame), ``"beamlet_reimaged"`` (the ON-AXIS beamlet
    #: ``grid/Iω_beamlet_reimaged``, stored by pnps files — the chromatic focal
    #: mixture's own frame, so no ``spectrum_frame_p`` is then needed) or
    #: ``"source"`` (pre-mask ``grid/Iω``, the ideal input). Falls back along
    #: reimaged → beamlet → source when the file lacks the requested one.
    spectrum_source: str = "beamlet"
    #: which stored pulse to overlay as the time-domain truth (and to seed
    #: ``truth_init`` from): ``"beamlet"`` (post-mask ``grid/It_beamlet``, the
    #: beam that gates — default), ``"beamlet_reimaged"`` (its on-axis
    #: counterpart, ``grid/It_beamlet_reimaged`` — the like-for-like truth for
    #: a focal-mixture retrieval) or ``"source"`` (pre-mask ``grid/It``, the
    #: ideal input). Falls back when the file stores fewer. See
    #: :func:`croak.io.read_simulated_truth_keys`.
    truth_source: str = "beamlet"
    #: load the as-simulated ω-density trace on its native grid straight into the
    #: retrieval, skipping the marginal-check and preprocess (filtering + regrid)
    #: stages. Fully raw: no third-order correction; window/thickness/delay-sign/
    #: known-spectrum selections are still honoured. For sanity-checking the
    #: retrieval free of any preprocessing doubt (see
    #: :func:`croak.session.pipeline.assemble_simulated_tracedata`).
    raw_direct: bool = False


@dataclass
class PreprocParams:
    """Stage 2: the grid, windowing and filtering used to clean and regrid.

    Wavelengths are in nm and delays in fs (converted to SI when applied). The
    grid fields set the retrieval ω/τ grid; the windowing fields the measurement
    λm/τm extents; the filtering fields the fringe/DC/low-pass and thresholding.
    """

    # -- grid --
    # retrieval grid wavelength range (fundamental; = trace λ × ω0scale)
    lam_min_nm: float = 700.0
    lam_max_nm: float = 900.0
    lam_bound_min_nm: float = 350.0  # slider bounds (data-driven)
    lam_bound_max_nm: float = 1100.0
    trange_fs: float = 1000.0  # time grid range; always on, default 2× scan delay range
    # delay range to use (hard crop of the delay axis)
    tau_min_fs: float = -500.0
    tau_max_fs: float = 500.0
    use_subsample: bool = False
    subsample_tau: int = 1
    resample_t: bool = False
    prefilter: bool = True  # anti-alias low-pass before downsampling the spectrum
    # -- windowing --
    # measurement spectral window (trace λ domain)
    lamm_min_nm: float = 300.0
    lamm_max_nm: float = 1100.0
    lamm_bound_min_nm: float = 300.0
    lamm_bound_max_nm: float = 1100.0
    # measurement delay window (τm; always on, crops the delay axis)
    taum_min_fs: float = -500.0
    taum_max_fs: float = 500.0
    tau_bound_min_fs: float = -1000.0
    tau_bound_max_fs: float = 1000.0
    # -- filtering --
    filter_fringes: bool = False
    filter_dc: bool = False
    dtau_min_fs: float = 0.0
    # Takeda-FTSI de-fringe (carrier low-pass); mutually exclusive with
    # filter_fringes. defringe_fraction sets the cutoff as a fraction of the
    # carrier f_c = c/λ.
    defringe: bool = False
    defringe_fraction: float = 0.5
    # Nyquist-guard mode when the de-fringe carrier aliases ("raise"/"warn"/
    # "clamp"). Defaults to "clamp" (never fails; identical to "raise" when the
    # delay is sampled finely enough that the fringe does not alias).
    defringe_on_alias: str = "clamp"
    # arPLS signal-aware baseline removal (broad leakage |E_b|^2 along delay).
    # Applied right after de-fringing, before τ=0 centring; asymmetric-baseline
    # aware. baseline_tau_exclude_fs=0 ⇒ no peak hold-out; clip off by default.
    baseline: bool = False
    baseline_smoothness: float = 1e2
    baseline_ratio: float = 1e-3
    baseline_clip: bool = False
    baseline_tau_exclude_fs: float = 0.0
    use_threshold: bool = False
    threshold: float = 1e-4  # noise floor as a fraction of the peak (e.g. 5e-3)
    marginal_correct: bool = False


@dataclass
class RetrieveParams:
    """Stage 3: the solver, initial guess, dispersive slab and regularisation.

    Covers the algorithm and its tolerances, the guess mode, optional dispersive
    propagation through a slab (the forward model, distinct from the stage-4
    dispersion *compensation*), the regularisation weights, the phase
    parameterisation, the CMA-ES global-search knobs and the post-filter toggles.
    """

    #: "warm-lbfgs" is the extended model seeded from COPRA's local projection
    #: sweep: same forward model as "lbfgs-ad", better starting point. It is the
    #: default because it is close to neutral where the landscape is benign and
    #: substantially better where it is not, so it is the safer choice for a
    #: user who does not want to reason about initial guesses.
    solver: str = "warm-lbfgs"
    #: 300 iterations and a 1e-5 relative tolerance: the settings every result
    #: in the smearing/solver study was measured with. 100/1e-4 stopped some
    #: dispersive retrievals short of convergence.
    maxiters: int = 300
    reltol: float = 1e-5
    abstol: float = 1e-8
    full: bool = True
    R_omega: bool = True
    random_init: bool = False
    perfect_init: bool = False
    #: Seed from the known complex field carried by a synthetic or ModelPNPS
    #: load. The selected truth must contain complex spectral datasets; intensity-
    #: only legacy files cannot provide this initial condition.
    truth_init: bool = False
    #: Start from the measured spectrum with flat phase: its transform limit.
    #: Preferred over ``perfect_init`` for near-transform-limited pulses --- the
    #: two perform equivalently, but this one uses the amplitude that was
    #: actually measured and assumes no duration. Ignored if ``random_init`` or
    #: ``perfect_init`` is set.
    tl_init: bool = False
    reuse_result: bool = False  # seed from the previous retrieval's spectrum
    # Live full-plot preview during retrieval (GUI only): redraw the 12-panel view
    # ~1/s as the solver converges, rather than only the convergence curve. When
    # off, the cheap convergence curve is shown. Ignored by the headless pipeline.
    live_preview: bool = True
    perfect_init_fs: float = 10.0
    dispersive: bool = False  # dispersive propagation through a slab
    material: str = "SiO2-Franta"
    thickness_um: float = 10.0
    # Default node count kept consistent with the tied defaults below:
    # thickness_um / delta_z_um = 10 µm / 1 µm = 10.
    npoints: int = 10
    # Nominal depth step (µm) the GUI uses to pick ``npoints`` for the slab
    # integral: ``npoints = round(thickness_um / delta_z_um)``. The two are tied
    # so editing the thickness grows ``npoints`` to hold the step fixed; editing
    # ``npoints`` back-computes the step. Only ``npoints`` reaches the forward
    # model (:class:`croak.forward.ForwardModel`) — ``delta_z_um`` is the GUI knob.
    delta_z_um: float = 1.0
    # Geometric time smearing of a non-collinear BOXCARS geometry (PG/SD only,
    # autodiff solvers only). The mask hole diameter and edge-to-edge spacing are
    # the physical inputs; the resulting delay-offset width is tied to them via
    # croak.smearing.square_boxcars_delay_width at the measured carrier, so editing
    # either geometry field updates the width and editing the width back-computes
    # the spacing — the same coupling as thickness / Delta z / npoints above. Only
    # the derived kernel reaches the forward model.
    smearing: bool = False
    smear_hole_diameter_mm: float = 1.0
    smear_hole_spacing_mm: float = 0.5
    smear_sigma_delay_fs: float = 0.357
    #: COPRA step-size heuristic (Geib's alpha). Measured to be inert: a factor
    #: of five in it moves the complex error by a few percent, non-monotonically.
    copra_alpha: float = 0.25
    #: Iterations without improvement before COPRA's local projection sweep
    #: hands over to its global descent. Not a smooth knob but a switch between
    #: the two stages, and with a smeared model the stages differ in whether
    #: they can represent the physics at all: the local projection cannot (a
    #: mixture has no single signal to project onto) while the global descent
    #: can. A very large value never leaves the local sweep.
    copra_stall_patience: int = 5
    smear_nodes: int = 5
    #: Multiplier on the geometric kernel's widths, applied as a FIXED value
    #: (unlike ``fit_smearing``, which optimises it). Exists because the mask
    #: geometry can only express d/D = (gap + D)/2D >= 0.5, so a kernel
    #: narrower than that — which Gaussian-beam data wants, see below — is not
    #: reachable by adjusting the gap at all.
    #:
    #: 1.0 is correct for the apertured instrument: scanning the width against
    #: the 3D simulation at the known truth puts the optimum at 0.85-1.00 at
    #: both 9.5 and 20 um and at both d/D = 1.0 and 1.5. Gaussian-beam traces
    #: want about 0.5, on top of the 0.748 focal-weight rescale their effective
    #: gap already carries.
    smear_scale: float = 1.0
    #: Aperture layout the kernel is built from. ``"boxcars"`` is the four-hole
    #: BOXCARS mask used for TG/PG (and for a three-hole SD). ``"sd2"`` is a
    #: genuine TWO-beam self-diffraction: two collinear apertures, the probe
    #: entering twice and the gate once with conjugation.
    #:
    #: The distinction is not cosmetic. In a two-beam SD the two unconjugated
    #: arms are the SAME beam, so ``p``—which is proportional to the difference
    #: of their tilts—is identically zero and the smearing collapses to a pure
    #: delay-axis convolution. Building such a measurement's kernel from the
    #: boxcars closed form instead attaches a spurious ``sigma_p`` of order
    #: 0.4 fs that the experiment does not have.
    smear_layout: str = "boxcars"
    #: Chromatic focal mixture (``croak.focal``) instead of the reduced
    #: smearing kernel — the full model. Each focal-plane node
    #: carries the chromatic Airy filter ``A(r, omega)`` and its own definite
    #: ``(p, theta)``, so the r-dependent (bluer-on-axis) local spectra that
    #: the reduced kernel averages away are modelled explicitly. Mutually
    #: exclusive with ``smearing`` (the mixture REPLACES the kernel; asking
    #: for both raises). Autodiff solvers only. Roughly 40x the reduced
    #: kernel's cost per iteration at the default 24x16 node grid.
    focal: bool = False
    focal_n_radial: int = 24
    #: 16, not 8: the coherent (apertured) sum needs the finer azimuthal
    #: quadrature — 8 costs 4e-5 of trace peak.
    focal_n_azimuth: int = 16
    focal_rmax_units: float = 4.0
    #: Focusing focal length (mm). The reduced kernel never needs it (f
    #: cancels between spot size and tilt); the explicit mixture does. For
    #: simulated scans it is stored in the file (``f_foc``, files written
    #: after 2026-08-26) and adopted with the rest of the geometry.
    focal_f_mm: float = 100.0
    #: Collection model riding on the mixture: ``"off"`` sums incoherently
    #: over the focal plane (exact only for full-beam collection), ``"file"``
    #: rebuilds the collection aperture from a simulated scan's own
    #: ``window_def_*`` record, ``"manual"`` builds a centred hole at the
    #: phase-matched corner from ``collection_diam_mm``. A tightly apertured
    #: instrument sits far from the incoherent limit and the mixture
    #: over-blurs without this.
    collection: str = "off"
    collection_mode: str = "integrated"
    collection_diam_mm: float = 0.5
    #: Spectral-frame exponent ``p``: the measured spectrum used for the
    #: initial guess and the ``reg_spectrum`` target is reweighted by
    #: ``(omega/omega0)^(2p)`` in intensity before retrieval. The focal
    #: mixture's ``ew`` is the ON-AXIS focal field, while a measured/beamlet
    #: spectrum is spatially integrated — bluer on axis by omega^2 for an
    #: aperture-diffraction beamlet, reduced towards 0 as the collection
    #: opens (measured: ~1.0 at gap 0.5 mm, ~0.5 at 1.0 mm).
    #: Leave 0 for the reduced kernel, whose implicit frame is the
    #: mixture-averaged one. GDD/span are insensitive to it; durations move
    #: by a few %.
    spectrum_frame_p: float = 0.0
    # extra fitted parameters (lbfgs-ad / lm / lm-optx / cma-es): slab thickness
    # (dispersive PG/SD only), a delay-zero offset and the smearing strength;
    # two-phase polish is a local-solver option (cma-es frees the extras from the
    # start).
    # Generation envelope g(z) on the depth quadrature (lbfgs-ad only): the
    # transverse-geometry factor of the crossing focused beams — focal droop
    # over the Rayleigh range, beamlet walk-through over the overlap length,
    # and a net Gouy phase. Negligible for slabs much thinner than the
    # (in-medium) Rayleigh range; see docs/explanation/forward_model.md.
    # Lengths in um from the slab entrance; in-medium values.
    envelope: bool = False
    envelope_zr_um: float = 0.0
    envelope_lw_um: float = 0.0
    envelope_zf_um: float = 0.0
    envelope_gouy: float = 1.0
    fit_thickness: bool = False
    fit_tau0: bool = False
    fit_smearing: bool = False
    # fit the gate-shape (p) and delay (delta) kernel widths as two independent
    # multipliers instead of the single joint one — the diagnostic for which
    # smearing channel deviates from the geometric prediction; needs fit_smearing
    fit_smearing_split: bool = False
    polish_two_phase: bool = False
    #: Where the model's delay zero sits (AD solvers): ``"coincidence"`` (gate–probe
    #: coincidence, the forward model's native convention) or ``"marginal_peak"``
    #: (every model trace re-centred on its own delay-marginal peak, the convention
    #: the preprocessing gives the data — no free parameter). The latter is the
    #: single-cycle protocol: a coherently collected trace peaks tens of as from
    #: coincidence, growing with the chirp, and a fixed origin costs ~10 % of
    #: duration while a free ``fit_tau0`` wanders on chirped pulses (FROG N102).
    delay_origin: str = "coincidence"
    #: ``|τ0| ≤ tau0_bound_fs`` (fs) for a fitted delay offset; 0 = unbounded.
    tau0_bound_fs: float = 0.0
    #: Amplitude-smoothness weight (2nd difference of ``|E(w)|``, max-normalised).
    #: Used *together with* ``reg_spectrum``: 0.03 alongside a spectrum weight of
    #: 0.01 beat every single-penalty setting tested, on every metric at once and
    #: across eight arms at two depths -- 12% better in complex error, 2.4 points
    #: better in retrieved duration, and 2.5x smoother in the worst case, than
    #: ``reg_spectrum`` = 0.1 alone. Adding it never hurt in any configuration
    #: measured, noiseless or noisy. Same objective-scale caveat as
    #: ``reg_spectrum`` below.
    reg_amp: float = 0.03
    reg_phase: float = 0.0
    #: Spectrum-divergence weight. 1e-2 is the measured optimum for the
    #: gradient solvers: scanning it over three decades on a dispersive
    #: TG-FROG trace moves the trace error by only 14% but the pulse error by
    #: 44%, with a clear interior optimum here. The penalty does not remove
    #: misfit so much as move it out of the spectral amplitude, where it is
    #: constrained, and into the phase, where it is not -- so both a weight
    #: that is too small (amplitude error dominates) and one too large (phase
    #: error dominates) are worse. Do NOT tune this by watching the trace
    #: error: it barely responds.
    #:
    #: NOTE this value is for the L-BFGS/gradient family, which minimises the
    #: scalar R + lambda P. The LM solvers add the penalty as residual rows
    #: (R^2 + lambda' P), so the same number is far stronger there; see
    #: :data:`REG_DEFAULTS` for the per-family values and the lambda' = 2 R
    #: lambda mapping between them. croak does not translate automatically, so
    #: reset both weights when changing `solver` to "lm"/"lm-optx" (the GUI
    #: does this for you).
    reg_spectrum: float = 1e-2
    # temporal (pedestal) penalty (LBFGS-AD / cma-es) + its time window (fs)
    reg_time: float = 0.0
    reg_time_lo_fs: float = -500.0
    reg_time_hi_fs: float = 500.0
    # phase parameterisation (AD solvers + cma-es): pointwise or reduced B-spline
    phase_basis: str = "pointwise"
    n_nodes: int = 20
    # global search (cma-es)
    cma_strategy: str = "cma"
    cma_popsize: int = 0  # 0 -> evosax default
    cma_std_init: float = 0.5
    # post-retrieval cleanup (toggled on the result pane, no re-run needed)
    post_filter_lam: bool = False
    post_filter_tau: bool = False


@dataclass(frozen=True)
class BeamPathMirror:
    """One mirror in the dispersion stage's beam-path stack (a serialisable row).

    A *stable* description of a single mirror reflection, so the stack survives a
    save/reload (unlike the live registry keys the GUI uses at runtime). A
    built-in mirror is named (``name`` in :data:`croak.mirrors.BUILTIN_COATINGS`
    or :data:`croak.mirrors.BUILTIN_MIRRORS`); a custom mirror leaves ``name``
    empty and points ``custom_file`` at a CSV of wavelength + phase/GDD.

    Parameters
    ----------
    name : str
        Built-in coating/mirror name; empty for a custom-file mirror.
    polarization : {"s", "p"}
        Polarisation for coatings (ignored by phase-only mirrors).
    bounces : int
        Number of reflections (0 disables the row).
    direction : {"remove", "add"}
        ``"remove"`` back-propagates the mirror out of the pulse (negative bounce
        count); ``"add"`` propagates it in.
    custom_file : str
        CSV path for a custom mirror (empty for a built-in).
    custom_lam_col, custom_val_col : int
        0-based wavelength and value column indices in ``custom_file``.
    custom_unit : str
        Wavelength unit of ``custom_file`` (e.g. ``"nm"``).
    custom_datatype : {"phase", "gdd"}
        Whether the value column is spectral phase (rad) or GDD (fs²).
    """

    name: str = ""
    polarization: str = "s"
    bounces: int = 0
    direction: str = "remove"
    custom_file: str = ""
    custom_lam_col: int = 0
    custom_val_col: int = 1
    custom_unit: str = "nm"
    custom_datatype: str = "phase"


@dataclass
class DispersionParams:
    """Stage 4: post-retrieval dispersion compensation applied to the pulse.

    Taylor coefficients (fs², fs³, fs⁴), an optional refractiveindex.info
    material (shelf/book/page + thickness), an optional active gas (pressure +
    path), built-in material thicknesses (mm by name), and the beam-path mirror
    stack. This stage tunes/compresses the retrieved pulse; it does not re-run
    retrieval.

    Notes
    -----
    Scalar fields are declared before the ``material_thickness_mm`` table and the
    ``mirrors`` array-of-tables because TOML requires a table's scalar keys to
    precede its sub-tables.
    """

    gdd_fs2: float = 0.0
    tod_fs3: float = 0.0
    fod_fs4: float = 0.0
    # refractiveindex.info material (single active page + thickness)
    ri_shelf: str = ""
    ri_book: str = ""
    ri_page: str = ""
    ri_thickness_mm: float = 0.0
    # active gas: pressure is baked into n(λ); the path length acts as thickness
    gas_name: str = ""
    gas_pressure_bar: float = 1.0
    gas_path_cm: float = 0.0
    # built-in Sellmeier/tabulated materials: {name: thickness_mm}
    material_thickness_mm: dict[str, float] = field(default_factory=dict)
    # beam-path mirror stack (signed by each row's direction/bounces)
    mirrors: list[BeamPathMirror] = field(default_factory=list)


@dataclass
class UncertaintyParams:
    """Stage 5: the FWHM-uncertainty bootstrap configuration.

    The ``method`` selects a statistical bootstrap — ``"parametric"`` (add noise
    and re-retrieve) or ``"delay"``/``"frequency"`` (resample the trace) — the
    substrate-thickness *systematic* ``"thickness"`` (Monte-Carlo over the
    measured thickness prior), or the fast analytic ``"covariance"`` (linearised
    Gauss–Newton covariance; see :func:`croak.covariance.covariance_uncertainty`).
    The remaining fields tune the resampling, noise model and confidence interval.
    """

    method: str = "parametric"
    n_resamples: int = 200
    # noise model for the parametric/covariance methods: from the residual, or manual σ
    noise_source: str = "residual"  # "residual" | "manual"
    noise_sigma: float = 0.01  # fraction of peak (manual source)
    warm_start: bool = True
    collect_profiles: bool = True
    interval: str = "bias-corrected"  # "bias-corrected" | "percentile"
    # 1σ of the measured substrate thickness (µm), for the "thickness" method; the
    # central value is taken from RetrieveParams.thickness_um
    thickness_sigma_um: float = 0.0
    # evaluate the FWHM/band at the dispersion-stage beamline point instead of the
    # measurement plane — propagate every replicate through the stage-4 dispersion
    propagate_to_dispersion_point: bool = False
