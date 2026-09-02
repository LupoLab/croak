# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to follow
semantic versioning.

## [Unreleased]

### Fixed

- **`collection="file"` on a multi-window scan used the first hole's record for
  every window.** The session and the GUI now forward the loaded `window_key`
  (`run_retrieval(..., scan_window=...)`, `focal_mixture_from_params(...,
  scan_window)`), so retrieving `Iω_win_5` rebuilds the 2.0 mm aperture it was
  recorded through rather than the 0.5 mm one.

### Added

- **`delay_origin="marginal_peak"`: the model trace re-centred on its own
  delay-marginal peak** (`croak.delay_origin`; AD solvers, session params, GUI
  "Delay origin"). The preprocessing centres the data on their marginal peak;
  the forward model puts delay zero at gate–probe coincidence. For coherently
  collected single-cycle traces the two differ by tens of attoseconds, growing
  with the chirp, and a retrieval with a fixed origin pays for the misalignment
  by broadening the pulse (~10 % at the production hole, FROG N101/N102).
  Re-centring the model by the same sub-sample parabolic rule gives both sides
  one convention with no free parameter; unlike `fit_tau0` it does not wander on
  chirped pulses. Exact Fourier translation, two FFTs per evaluation.
- **`tau0_bound`** (`RetrieveParams.tau0_bound_fs`, GUI "τ₀ bound"): a box bound
  on a fitted delay offset — the stopgap for chirped pulses where a free τ₀
  ran to hundreds of attoseconds.
- **GUI "Auto frame"** in the focal group: sets the spectral-frame exponent from
  the model and the loaded spectrum source (mixture → on-axis; 1-D kernel →
  ≈0.5; `beamlet_reimaged` source → 0 for the mixture).
- **pnps collection windows read natively.** `mask_transmission` gains the compact
  raised-cosine edge `"rcos"`, and the window record's `weighting` scalar is
  honoured: `"quadrature"` (pnps) treats the profile as the integration weight over
  a hard hole — it enters the collected energy ONCE and is built at
  `quadrature_hole_diameter()` so it encloses exactly the nominal hole area — where
  `"amplitude"` (ModelPNPS/Luna, and the default for records without the scalar)
  keeps squaring a field filter. `CollectionAperture`, `mask_hole_aperture` and
  `aperture_from_scan` carry `weighting`; `MaskWindowSpec.weighting` records it. Until
  now a pnps file's `rcos` window was refused, and modelling it as the squared tanh
  under-counted a 0.5 mm aperture by a quarter (FROG N101).
- **The on-axis beamlet truth as a source.** pnps files store the on-axis beamlet
  alongside the transverse-integrated one (`Iω_beamlet_reimaged`,
  `Eω_beamlet_reimaged`, `It_beamlet_reimaged`); `SimulatedScan` now carries them and
  `truth_source` / `spectrum_source` accept `"beamlet_reimaged"` (with fallback
  reimaged → beamlet → source). This is the frame the chromatic focal mixture's `ew`
  represents, so with it a focal-mixture retrieval needs no `spectrum_frame_p`
  reweighting and can be seeded (`truth_init`) and scored against the truth it
  actually reconstructs. `read_simulated_truth_keys` lists it when present.

- **fc-z: the depth-resolved chromatic focal mixture** (`focal_mixture(...,
  evolve_profiles=True, material=..., thickness=..., npoints=...)`). Every
  entrance-face model freezes the chromatic beamlet profiles at the slab
  entrance; measurement on the 3-D reference traces localised the dominant
  at-truth residual to exactly that frozen physics. With fc-z each arm at
  each depth node applies its own linearly evolved complex profile — the
  aperture's k-space disc propagated by the transverse part of the in-glass
  `k_z` and evaluated at the arm's walked radius `-z r_j/(f n0)` — in place
  of the single shared entrance filter; the `(p, theta)` arrival offsets stay
  depth-independent (exact). Parameter-free once the geometry is given.
  Cost is ~fc: the focal path already batches the depth axis, so the per-arm
  `(N, Q)` filter slices ride the existing batched FFTs; only a seconds-level
  numpy table build (`croak.focal.evolved_arm_filters`, shape `(3, K, Q, N)`)
  is new. A zero-evolution table is bit-identical to the entrance filter (the
  difference-form build cancels its own quadrature at z = 0) and the trace
  reduces at machine precision; the evolved profiles match an independent
  plane-wave propagator to 1e-6 of peak at 40 um; the evolved three-beam
  moments reproduce the measured beamlet-companion table (walk sign, arm
  order and conjugation pinned by measurement); the Gaussian-control geometry
  is a null, matching the measured absence of the arrival advance there. PG
  only; `fit_thickness` unsupported in v1; the mixture's slab spec must match
  the trace map's (verified loudly). `depth_transverse` composes orthogonally.

### Added

- **Depth-resolved transverse dephasing in the collected focal model**
  (`depth_transverse=True` on `croak.forward_jax.make_param_trace_fn`, forwarded
  by `lbfgs-ad` and `retrieve_from_tracedata`; other solvers refuse it loudly).
  croak's depth quadrature adds the signal generated at every depth with the
  same transverse phase — fully coherently in transverse wavevector — which is
  what makes the collection model over-correct on thick slabs. The flag applies
  the missing per-depth phase `[k_z(w, k_perp) - k_z(w, 0)](L - z_q)` at each
  aperture node's absolute wavevector *inside* the collection transform, before
  the coherent depth sum (full square-root `k_z`, full Sellmeier `n(w)`;
  parameter-free). Off (the default) is bit-identical to the previous model.
  Validated against a per-depth dense 2-D FFT reference with the exact
  `k_z(w, k_perp)`: agreement improves 25–91× to 1.4–5.7e-5 of peak, and a
  flipped sign would be worse than no correction — the sign is test-pinned.
  Requires `focal` with a `collection` and a dispersive slab; full collection is
  representable as a window much wider than the signal footprint. Costs ≈2.5×
  the collected trace at 10 depth nodes (one aperture contraction per node), at
  unchanged memory. See "Where it stops being right" in
  `docs/howto/collection_aperture.md`.

### Changed

- **The settled regularisation weights are now the library-wide defaults.**
  The penalty-capable solvers (`lbfgs`, `lbfgs-hand`, `lbfgs-ad`,
  `lbfgs-optx`, `cma-es` with `reg_spectrum=0.01`/`reg_amp=0.03`, and
  `lm`/`lm-optx` with the LM-mapped `1e-5`/`3e-5`) and
  `retrieve_from_tracedata` default `reg_spectrum`/`reg_amp` to ``None`` =
  "use the settled weight", matching the GUI and the companion paper's
  protocol. The spectral-match term is inert unless a target spectrum is
  supplied, so retrievals without an independent spectrum gain only the mild
  amplitude-smoothness term. Pass `reg_amp=0, reg_spectrum=0` for the previous
  unregularised behaviour. `REG_DEFAULTS`/`reg_family`/`reg_defaults` moved
  from `croak.session.params` to `croak.solver` (still re-exported from the
  old location).

## [0.1.0] - 2026-08-31

### Changed

- **`warm-lbfgs` is now the default solver everywhere.** `croak.retrieve`
  (previously `"copra"`), `retrieve_from_tracedata` (previously `"lbfgs"`) and
  the coverage-calibration harness (previously `"copra"`) now default to
  `"warm-lbfgs"`, matching the session pipeline, the GUI and the companion
  paper's production protocol. Pass `algorithm=` explicitly to keep the old
  behaviour.

### Added

- **Companion-paper validation data and examples.** Two reduced datasets from
  the paper's first-principles 3D instrument simulations ship in
  `examples/data/` (the 1 fs fused-silica thickness series and the single-cycle
  RDW pulse), with worked retrievals in `examples/example_paper_thickness.py`
  and `examples/example_paper_rdw.py`, and `tools/reduce_scansave.py` to cut
  such datasets from full `scansave` scans. The documentation's
  [Validation](docs/explanation/validation.md) page now summarises the paper's
  study — model hierarchy, thickness series, geometry and collection results,
  solver protocol and metrics guidance — with its figures.

- **GUI-only install guide.** `docs/getting_started/gui.md` walks a non-Python
  user from nothing to a running wizard, linked from the top of the README.

- **Release metadata.** README badges (tests, coverage, docs, ruff, pyright,
  Python, licence), a `CITATION.cff`, and Codecov upload in CI.

- **Known-truth pulse errors.** New `croak.truth_metrics` scores a retrieval
  against a *known* pulse rather than against the measured trace: `eps_It`
  (temporal intensity), `eps_Iw` (spectral intensity) and `eps_Ew` (Geib's
  complex-field epsilon). Each removes exactly the gauge freedoms a PNPS
  measurement cannot determine — energy scale, absolute phase, delay — and no
  others. The Retrieve stage's numeric read-out shows all three beside the FROG
  error whenever a known pulse is loaded. Read together they localise a failure a
  trace error hides: `eps_Iw` small with `eps_Ew` large is a correct amplitude
  with a wrong phase.

- **Live preview during COPRA.** `copra` and `copra-jax` now hand the GUI an
  in-progress result each iteration, so the full 12-panel preview redraws during
  their sweeps instead of only showing a convergence curve. This matters most
  for `warm-lbfgs`, whose 300-iteration COPRA warm-up previously left the
  preview blank for most of the run. The snapshot reuses the iterate the sweep
  already computed, so it costs no extra forward pass.

- **Complex ModelPNPS truth as a retrieval seed.** Complex source/beamlet fields
  are retained on `TruthPulse`; `initial_guess(..., mode="truth")`,
  `retrieve_from_tracedata(..., guess="truth", truth=...)`, and
  `RetrieveParams(truth_init=True)` route the known solution into the solver for
  direct forward-model residual checks. The Retrieve GUI now has one mutually
  exclusive initial-guess selector (including **Truth** and **Reuse previous**),
  replacing the conflicting independent checkboxes. Known temporal and spectral
  phases are dotted, named in both legends, and saved with the complex truth.

- **Headless install: the GUI is now an extra.** `pyqt6`/`superqt` moved from
  the core dependencies to `croak[gui]`; everything except the wizard —
  retrieval, preprocessing, processing, uncertainty, plotting, the `croak`
  CLI — runs without Qt, so compute nodes install without it (`uv sync
  --no-dev`, or plain `pip install croak`). Launching `croak-gui` without the
  extra raises a message saying how to get it. The dev group still carries Qt
  (the suite includes the GUI tests), so `uv sync` in a checkout is unchanged.

- **Multi-aperture scans: every stored window is selectable.**
  `io.read_simulated_window_keys` now discovers all trace windows in the
  file — the canonical trio first, then the numbered windows of an
  aperture-series scan (`Iω_win_2` … with their `_reimaged` partners, one
  propagation reduced through several collection holes) — so the GUI's
  trace-window selector offers them all.

- **Aperture selection for multi-window scans** (`window=` on
  `aperture_from_scan` / `read_simulated_mask_window`, plus
  `io.window_def_prefix`). An aperture-series scan records one
  `window_def_N_*` record per collection hole; the selector (the trace-window
  dataset name, `_reimaged` accepted, or the bare index) rebuilds that hole's
  aperture, so the focal+collection model can retrieve any window of the
  series against its own recorded geometry.

- **Uniform-delay-core selection at load** (`SimulatedLoadParams.
  uniform_delay_core`, a checkbox on the simulated-load stage, and
  `preprocess.uniform_core_indices`). Campaign scans often extend a uniform
  delay core with coarser wing points (e.g. 1 fs-step wings to ±40 fs for the
  Raman wake); the smearing kernel's delay-axis convolution needs a uniform
  grid, so such scans could not previously be retrieved with the kernel
  modelled. The option keeps the longest uniformly spaced contiguous stretch
  of the axis (off by default — the wings are real data for every
  kernel-free model).

- **`focal=` now reaches the high-level pipeline.** `retrieve_from_tracedata`
  (and therefore `croak.session.pipeline.run_retrieval`, via a new keyword)
  accepts a `FocalMixture`, forwarding it to the solvers that can model it and
  raising for those that cannot — closing the gap where the solver
  constructors accepted a mixture but every session-level entry point silently
  could not pass one.

- **The focal mixture in `RetrieveParams` and the GUI.** New scalar fields
  (`focal`, `focal_n_radial`/`focal_n_azimuth`/`focal_rmax_units`,
  `focal_f_mm`, `collection`, `collection_mode`, `collection_diam_mm`,
  `spectrum_frame_p`) plus `session.pipeline.focal_mixture_from_params` — the
  focal counterpart of `smearing_kernel` — which `run_retrieval` invokes
  automatically when `focal` is set (pass `scan_path` for
  `collection="file"`, which rebuilds the aperture from the scan's own
  `window_def_*` record; `"manual"` builds a hard-edged pinhole at the
  phase-matched corner). `apply_spectrum_frame` reweights the measured
  spectrum (initial guess and `reg_spectrum` target together) by
  `(ω/ω₀)^{2p}` into the mixture's on-axis frame. The GUI's retrieve stage
  gains a *Chromatic focal mixture (advanced)* group, mutually exclusive with
  the reduced kernel, with the focusing focal length adopted from simulated
  scans' geometry records (`io.SimulatedScan` now reads `f_foc`).

- **Finite collection aperture for the focal mixture** (new `croak.collection`,
  reached through `focal_mixture(..., collection=...)`). `croak.focal` sums
  *incoherently* over focal position, which by Parseval is exact only if the
  detector collects the whole signal beam. Most non-collinear instruments pick the
  signal out of the BOXCARS pattern with a hole — a filter in transverse
  wavevector — and a typical DUV instrument's hole is **six times narrower
  than the signal's own k content**, so it sits far closer to the coherent limit
  than to the incoherent one. The new path transforms the focal field to
  transverse k, applies the aperture there and integrates, in either the
  `"integrated"` (spectrometer behind the hole) or `"reimaged"` (on-axis re-imaged
  pixel, the fully coherent limit) model.

  The `(p, theta)` reduction survives intact: croak's per-node build and the exact
  focal-plane signal differ by a rigid time shift, so the incoherent sum discards
  exactly one phase, and after the signal's phase-matched carrier is removed that
  phase is `exp(+i w p)` — *w times one of the two reduced smearing parameters*
  (`-w theta` for SD, which is not yet implemented). Validated against a dense 2-D
  FFT of the focal field built directly from the arm tilts: 3.5e-7 of peak with no
  fitted scale, plus an exact Parseval test on a Cartesian grid. Costs +5 % of
  runtime at production shape, stays differentiable, and reaches every solver that
  already accepted `focal=`. `collection=None` (the default) leaves the existing
  incoherent sum untouched.

- **`aperture_from_scan` / `croak.io.MaskWindowSpec`.** Simulated scans record the
  collection window they were taken through as flattened `/grid/window_def_*`
  scalars; these read it back and rebuild the aperture, resolving the simulator's
  `"default"` apodisation width from the file's own transverse k-grid. That matters:
  for the reference instrument it is a 96.9 µm `tanh` edge on a 500 µm hole — 19 %
  of the diameter — which a top hat does not approximate.

- **`croak.focal.mixture_from_nodes`** builds a mixture on arbitrary focal-plane
  nodes, and `FocalMixture` now records the mask hole positions it was built from
  (`arms`, plus a `signal_position` property). The incoherent sum needed only the
  arm *differences* that make `(p, theta)`; the aperture needs to know where the
  signal actually goes.

- **Faster `lm-optx` linear solve** (`linear_solver` on
  `croak.optimistix_lm.OptxLM`). Each Levenberg–Marquardt iteration solves
  `[J; sqrt(lambda) I] d = [r; 0]`; the new default `"normal"` forms
  `J^T J + lambda I` and takes its Cholesky factor instead of running
  Optimistix's QR on the stacked operator (still available as `"qr"`). The Gram
  matrix is one BLAS `gemm` at near-peak throughput while a tall-skinny
  Householder QR is panel-bound, so a 512-point PG retrieval drops from 2.9 s to
  1.2 s per iteration at an identical final error. Both routes rely on
  `lambda > 0` — the FROG Jacobian is strongly rank-deficient either way — but
  the normal equations square the condition number, so `"qr"` is kept as an
  escape hatch.

- **Complex simulated-scan truth spectra.** `read_simulated_scan` now loads
  the complex vignetted-beamlet spectrum (`Eω_beamlet_re`/`_im`, written by
  newer ModelPNPS runs) and the complex source spectrum onto
  `SimulatedScan.Eomega_beamlet` / `.Eomega` (both `None` for
  intensity-only legacy files). The beamlet phase carries any input chirp
  exactly, enabling complex-field retrieval-error metrics and direct
  truth-GDD measurement instead of intensity-only comparisons.

- **Generation-envelope weights** (`depth_weight` on
  `croak.forward_jax.make_param_trace_fn`, the `lbfgs-ad` solver and
  `retrieve_from_tracedata`, which raises rather than silently dropping it on
  unsupported solvers): optional complex per-node weights multiplying the
  depth quadrature, modelling the transverse geometry a 1D depth integral
  cannot see (focal-spot evolution across the slab, beamlet walk-through, Gouy
  phase). The session layer exposes the analytic Gaussian-crossing form via
  `RetrieveParams.envelope*` and `croak.session.pipeline.generation_envelope`.
  Uniform weights are a verified no-op; the envelope matters only when the
  slab is not much thinner than the beams' effective depth of focus
  (`docs/explanation/forward_model.md`) — note that for aperture-masked
  beamlets that scale is set by diffraction (`~lambda (f/D)^2`), not by the
  imaged-source Rayleigh range.

- **Split-channel smearing fit** (`fit_smearing_split`, AD solvers): fit the
  gate-shape (`p`) and delay (`delta`) widths of the geometric-smearing kernel
  as two independent multipliers instead of the single joint one — the
  diagnostic for *which* channel deviates from the geometric prediction. The
  split is exact (the bivariate kernel scales channel-wise at fixed `rho`);
  the fitted values land in `RetrievalResult.smear_scale` (p) and the new
  `smear_scale_delta` (delta), are saved/reloaded by `croak.save`, and
  `parameter_covariance` reports `sigma_smear_delta` alongside `sigma_smear`.
  Note the `p` multiplier is only identifiable for a chirped gate (a
  transform-limited gate's `p` response is absorbed by the intensity scale).

- **`croak.save.save_result(..., group=…)`** writes the unchanged flat result
  schema into a named HDF5 group instead of the file root, appending rather than
  truncating, so a parameter study (a thickness series, a solver comparison) can
  keep every run in one file — one group each, nesting with `"full/z00"`-style
  names. `force=True` then replaces only that group. `load_result` already
  recurses into subgroups, so such a file reads back as a nested dict.
- **Retrieval-quality diagnostics for spectral energy at the band edges.** The
  outermost frequency bins of the retrieval grid carry no measurement (`regrid`
  tapers them to zero), so the field there is nearly unconstrained and — since
  intensity is bounded below by zero — can only drift upward. Two new measures
  catch it:
  - {func}`croak.processing.edge_energy_fraction`, also carried on
    `ProcessedResult.edge_energy` and printed in the `plot_retrieval` spectrum
    panel (red above `croak.plotting.EDGE_ENERGY_WARN`, 1 %).
  - `TraceData.taper_loss`, the fraction of the measured trace the edge taper
    removed. `load_and_clean`/`regrid` now warn above 1 % (`taper_warn_level`),
    which is the load-time signature of a `lam_min`/`lam_max` band that clips.
- `croak.preprocess.TAPER_COLLAR_BINS` names the taper width, previously an
  unexplained literal `10` in two places.

### Changed

- **Simulated-scan delay convention is now auto-detected.**
  `SimulatedLoadParams.reverse_trace` defaults to `None` (auto): scansave
  files that carry the new `/grid/delay_convention = "gate"` marker (written
  by ModelPNPS runs that store the trace directly in the gate-delay/paper
  frame) load without delay-axis reversal, while legacy marker-less files are
  negated exactly as before. An explicit `True`/`False` still overrides.
  `SimulatedScan` gained the `delay_convention` attribute.

- `regrid` returns a seventh element, the taper loss.
- **Renamed `ProcessedResult.Iw_photon` to `ProcessedResult.Ilam`.** The
  quantity is the spectral intensity as a *wavelength density* (`|Ẽ(ω)|²ω²`,
  the λ→ω Jacobian applied), not a photon-flux weighting; the new name pairs it
  with `ProcessedResult.wavelength` and matches the saved `Ilam_retr` dataset.
- Documented that `R_omega=True` must not be used without `reg_spectrum` or
  `phase_only`: per-frequency scaling discards the frequency marginal, which is
  what pins the retrieved spectral amplitude, and it *lowers* the reported trace
  error while doing so. On a perfect synthetic PG trace it raises the edge energy
  from 0.02 % to 5.4 %.

### Fixed

- **Three-argument progress callbacks work with every solver.** The documented
  signature is `callback(iteration, R, best_R, snapshot=None)`, but the shorter
  form worked with `copra`/`copra-jax` (which never passed a snapshot) and failed
  with `lbfgs-ad` — so which forms were legal depended on the solver, and giving
  COPRA a snapshot to offer would have broken existing code. `Retriever.run` now
  adapts a callback that cannot receive a snapshot instead of raising.

- **`warm-lbfgs` reported only half of its own convergence.** The COPRA warm-up's
  error log was discarded, so the curve began partway down with no account of the
  work that got it there, and progress restarted from iteration 1 at the
  handover. Both stages are now spliced into one history, with the join recorded
  in the new `RetrievalResult.stage_boundaries` and drawn as a dashed rule —
  necessary because the halves do not count the same unit of work (a COPRA
  iteration is a full local sweep; an L-BFGS step is one function evaluation).
  Its results also identify themselves as `warm-lbfgs` instead of inheriting
  `lbfgs-ad`, which had sent an uncertainty bootstrap off to re-run replicates
  with the cold solver.

- **Native-grid simulated loads quantised time zero to a delay bin.**
  `assemble_simulated_tracedata` located the delay marginal's peak by taking its
  largest *sample*, so a scan whose true zero delay falls between two bins — an
  even, symmetric delay axis does exactly that — was modelled up to half a delay
  step away from where it was measured. It now uses
  `preprocess.marginal_peak_delay`, the sub-sample helper `regrid` already used
  and which its own docstring describes this defect in. A retrieval absorbed the
  shift into a linear spectral phase, so retrieved pulses were unaffected beyond
  a time translation, but any comparison against an un-translated known field
  paid for it in full: the truth-seeded forward-model check floored at `R ~ 5e-3`
  and now reaches `~1e-16`.

- **`make_trace_fn(focal=...)` was silently ignored** when no `smearing` kernel was
  also passed: the single-delay entry point fell through to the unsmeared model and
  said nothing. Retrievals were unaffected (they go through
  `make_param_trace_fn`), but a forward-model call was.

- **NLopt failures no longer abort retrievals.** The `lbfgs` and `lbfgs-ad`
  solvers previously died with a bare `nlopt.runtime_error` when NLopt's
  internal line search failed (observed for roughly one in six random starts
  on large dispersive grids). They now return the best iterate found, with a
  one-time warning; a non-finite objective or gradient is additionally trapped
  and replaced by a large finite penalty (`croak.solver.guard_nonfinite`) so
  the line search backtracks.

- Two docstring examples that did not match their own output
  (`peak_wavelength`, `marginal_peak_delay`).
- A batch of docstring/docs statements that had drifted from the code:
  `croak.processing.edge_energy_fraction` cited a function that does not exist
  (`taper_collar_signal`; the real load-time guard is `regrid`'s
  `taper_warn_level` warning plus `TraceData.taper_loss`); the `croak.lm` module
  docstring opened by claiming the MINPACK `method="lm"` default that the rest
  of the docstring and the code contradict (the default is `"trf"` with the JAX
  Jacobian); `retrieve_from_tracedata` omitted `"copra-jax"` and `"cma-es"`
  from its documented algorithm set and described `R_omega` as L-BFGS-only (all
  solvers support it, as `docs/explanation/retrieval_theory.md` now also
  says); `compute_mu_per_freq`/`mu_per_freq` now document that the weights
  cancel identically (the per-frequency factors are weight-independent; a
  zero-weight row falls back to `1.0`); `croak/__init__.py` and the README
  claimed all three geometries propagate dispersively (dispersive SHG raises —
  PG/SD only); `docs/explanation/forward_model.md` listed only the Sellmeier
  materials (missing `SiO2-Franta`); and `croak.solver`'s docstring described
  `Retriever` as the base of two solvers rather than of all of them.

## [0.1.0] — unreleased

First public release.

croak was developed privately before being open-sourced, and that history was
squashed for release, so this changelog starts here. The design rationale worth
keeping lives in the documentation: see [Explanation](docs/explanation/) for
the physics and algorithms, and the [how-to guides](docs/howto/) for the
practical trade-offs.

### Added

Everything. The initial release provides:

- **Retrieval.** Nine solvers over one forward model, selected by an `algorithm=`
  string: `copra`, `copra-jax`, `lbfgs`, `lbfgs-hand`, `lbfgs-ad`, `lbfgs-optx`,
  `lm`, `lm-optx` and `cma-es`. Four algorithm families — COPRA, L-BFGS,
  Levenberg–Marquardt and a derivative-free CMA-ES global search — with the rest
  on-device JAX reimplementations.
- **Forward model.** SHG, SD and PG (transient-grating) interactions, thin or
  propagated through a dispersive medium by Gauss–Legendre depth quadrature.
- **Gradients both ways.** A JAX model differentiable end to end, plus a
  hand-derived Wirtinger adjoint for the core model, cross-checked against each
  other to ~1e-6.
- **Geometric time smearing.** The pulse-front-tilt instrument response of a
  non-collinear BOXCARS geometry, as a two-parameter kernel that can be computed
  from the mask or fitted from the data.
- **Materials and dispersion.** Built-in Sellmeier materials, tabulated measured
  index, pressure-scaled gases, seven chirped-mirror designs and five beam-path
  coatings, plus an optional bridge to refractiveindex.info.
- **Workflow.** Loading (HDF5/NPZ/CSV, unit and axis detection, simulated scans),
  preprocessing (fringe and DC filtering, Takeda de-fringing, arPLS baseline
  removal, regridding), marginal consistency checks, post-processing, dispersion
  tuning, plotting and saving.
- **Uncertainty.** Parametric and resampling bootstraps, a substrate-thickness
  systematic, fast Laplace covariance, coverage calibration, and propagation of
  an interval to another point in the beamline.
- **Interfaces.** A library-first API, a PyQt6 wizard GUI, a headless session
  replay engine, and a `croak` CLI that replays a saved session or generates a
  standalone script from it.

[Unreleased]: https://github.com/LupoLab/croak/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/LupoLab/croak/releases/tag/v0.1.0
