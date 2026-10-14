(features)=
# Functionality and capabilities

croak — **Complete Retrieval by Optimisation and Adjoint Kernels** — retrieves
the complex field of an ultrashort pulse from a frequency-resolved optical gating
(FROG) trace. A spectrometer records intensity; croak estimates the spectral
amplitude and phase by comparing a predicted nonlinear measurement with the data.
From that field you can calculate a temporal profile, duration and spectral
phase, and explore propagation through materials or mirrors.

The usual workflow is to inspect and calibrate the data, choose the measurement
model, retrieve from suitable starting fields, and assess the result. The
[experimental-workflow tutorial](../tutorials/04_experimental_workflow.md) introduces
those steps. The [user guide](../user_guide/index.md) explains their assumptions,
and the [gallery](../examples/index.md) provides worked applications.

## What a retrieval can tell you

A retrieval estimates a field **within a specified model and reference plane**.
A small trace residual is evidence for agreement with the supplied trace, not a
guarantee of field accuracy. Finite bandwidth, noise, calibration, model mismatch
and optimisation can all affect the answer. Absolute phase and time origin are
not determined; SHG also has a time-direction ambiguity. An energy measurement
is needed to assign absolute power. See [conventions](conventions.md) and
[interpreting results](../user_guide/results.md).

croak implements SHG, SD and PG FROG, including a three-arm TG description using
the PG interaction. The general PNPS formalism also describes d-scan and related
methods, but those measurement operators are not bundled implementations.

## Retrieval

Ten solvers sit behind one {func}`croak.retrieve` call and all return a
{class}`~croak.result.RetrievalResult`. They are four algorithm families plus
on-device reimplementations:

| `algorithm` | Class | Gradient | Notes |
|---|---|---|---|
| `"warm-lbfgs"` | {class}`~croak.warm_lbfgs.WarmLBFGS` | JAX autodiff | the default; `lbfgs-ad` seeded from a COPRA local sweep |
| `"copra"` | {class}`~croak.copra.COPRA` | analytic adjoint | projection and gradient stages |
| `"copra-jax"` | {class}`~croak.copra_jax.COPRAJax` | analytic adjoint | COPRA on-device |
| `"lbfgs"` | {class}`~croak.lbfgs.LBFGS` | analytic adjoint | NumPy gradient-based retrieval |
| `"lbfgs-hand"` | {class}`~croak.lbfgs_hand.LBFGSHand` | analytic adjoint | JAX-compiled analytic gradient |
| `"lbfgs-ad"` | {class}`~croak.lbfgs_ad.LBFGSAD` | JAX autodiff | widest model coverage — see below |
| `"lbfgs-optx"` | {class}`~croak.optimistix_lbfgs.OptxLBFGS` | JAX autodiff | whole loop on-device |
| `"lm"` | {class}`~croak.lm.LM` | JAX Jacobian | residual-Jacobian refinement |
| `"lm-optx"` | {class}`~croak.optimistix_lm.OptxLM` | JAX Jacobian | on-device Levenberg–Marquardt |
| `"cma-es"` | {class}`~croak.cmaes.CMAES` | none (global) | population search; needs `croak[evo]` |

Options common to most solvers: amplitude-and-phase or `phase_only` retrieval,
per-frequency `R_omega` intensity scaling, second-difference smoothness penalties
on spectral amplitude (`reg_amp`) and phase (`reg_phase`), a spectral-match
penalty against an independent measurement (`reg_spectrum`), and a temporal
energy penalty outside a window (`reg_time`, on `lbfgs-ad` and `cma-es`). The
amplitude and spectral penalties default to the package weights
(0.03 and 0.01 for gradient solvers; 3e-5 and 1e-5 for LM); pass `0` to disable. See
[Regularisation](../howto/regularisation.md).

A cubic B-spline phase basis (`phase_basis="bspline"`) acts as a structural
regulariser and is what makes the global search tractable. Warm starts, random
restarts and a two-stage COPRA → LM polish are described in
[Solver selection](../howto/solver_selection.md);
{func}`croak.retrieve.algorithm_params` reports which options a given solver
accepts.

Further reading: [Retrieval algorithms](../explanation/algorithms.md),
[Retrieval as least squares](../explanation/retrieval_theory.md).

## Gradients

The forward model is written in JAX and is differentiable end to end, so
**automatic differentiation is the general route**. It supplies the gradients
for everything the closed-form adjoint does not cover:

- geometric time smearing,
- fitted medium thickness, delay-zero and smearing width,
- the B-spline phase basis and the temporal penalty,
- the residual Jacobian that Levenberg–Marquardt and the Laplace covariance need,
- implemented interaction kernels expressed in the JAX model.

For the core model — all three thin-medium interactions, plus dispersive PG and SD, pointwise
parameterisation, standard penalties — croak also carries a hand-derived
**Wirtinger adjoint**, so {class}`~croak.copra.COPRA` and
{class}`~croak.lbfgs.LBFGS` run in pure NumPy with no JAX on the hot path. The
two routes are checked on the interactions and options they share, including
complex-field perturbations and regularised objectives. These derivative checks
are separate from validation of the physical measurement model.

Further reading: [Gradients](../explanation/gradients.md).

## Forward model

- **Interactions** — SHG, SD and PG (the transient-grating/TG-FROG kernel), each
  with analytic adjoints; see [interaction physics](../explanation/interactions.md).
- **Dispersive propagation** — a Gauss–Legendre depth integral through a material
  slab, propagating both the input replicas and the generated signal. Available
  on every solver, for PG and SD (the SHG signal is not at the fundamental
  frequency). The thin medium is the same code path with zero thickness and one
  quadrature node.
- **Geometric time smearing** — {mod}`croak.smearing` represents an achromatic, fully collected focal average using two timing coordinates: a
  gate-splitting parameter that changes the signal shape, and a delay offset that
  is a pure convolution along τ. {func}`~croak.smearing.square_boxcars_kernel`
  builds the kernel for a folded square BOXCARS mask,
  {func}`~croak.smearing.kernel_from_arms` for an arbitrary three-arm layout.
  `fit_smearing=True` fits its magnitude from the data on supported solvers.
- **Chromatic focal fields and collection** — an explicit focal mixture retains
  the frequency dependence of local beamlet fields. The collection model adds
  their complex signal fields through a finite aperture before squaring. Optional
  depth evolution accounts for beam profiles changing within the slab. These
  models require measured or otherwise justified geometry; additional model
  detail does not guarantee a better field estimate.

Further reading: [The forward model](../explanation/forward_model.md),
[Interactions](../explanation/interactions.md),
[Geometric smearing](../howto/geometric_smearing.md).

## Materials, gases and mirrors

- **Sellmeier materials** — SiO₂, BK7, CaF₂, BaF₂, MgF₂, plus `SiO2-Franta` from
  tabulated measured data via a smoothing spline in log-λ.
- **Gases** — helium and air, with ideal-gas pressure/temperature scaling of the
  linear susceptibility; registered as ordinary materials, so optical path length
  plays the role of thickness.
- **Chirped mirrors and coatings** — bundled mirror designs and beam-path
  coatings, applying both the per-bounce spectral phase and the measured
  reflectivity. A negative bounce count back-propagates, removing a mirror from
  the pulse.
- **Your own** — {func}`croak.materials.register_material` and
  {func}`croak.mirrors.mirror_from_arrays` make a custom index or measured mirror
  work everywhere a built-in does; the optional `ridb` extra browses
  refractiveindex.info.
- **Dispersion tuning** — {func}`croak.dispersion.apply_dispersion` applies Taylor
  GDD/TOD/FOD, material thicknesses and a mirror stack in one call, with
  auto-tuners that maximise temporal peak power.

Further reading: [Materials and mirrors](../howto/materials_and_mirrors.md),
[Dispersion tuning](../tutorials/05_dispersion_tuning.md).

## Loading and preprocessing

- **Formats** — HDF5, NumPy `.npz` and delimited text, with dataset
  auto-detection by name, unit handling and position→delay conversion
  ({mod}`croak.io`).
- **Axis conventions** — the trace orientation (`transpose`) and the delay sign
  (`reverse_trace`, mirroring the trace about τ = 0) are per-load switches on
  both the measured and the simulated loader.
- **Simulated scans** — traces written by a propagation code, including
  multi-thickness scans selectable by thickness or slice index.
- **Corrections** — background subtraction, calibration curves, χ⁽³⁾ efficiency
  scaling.
- **Filtering** — fringe band-stop, DC removal and delay low-pass, all on a
  **non-uniform FFT**, so a jittered or unevenly stepped delay axis stays
  correctly calibrated.
- **De-fringing** — {func}`~croak.preprocess.defringe_carrier` removes
  interferometric fringes by a smooth low-pass below the optical carrier, which
  is *computed from the wavelength axis rather than fitted*, with a Nyquist guard.
- **Baseline removal** — {func}`~croak.preprocess.arpls_baseline` fits a smooth
  asymmetric lower envelope per wavelength row, handling a background that is
  asymmetric about τ = 0, with peak protection to reduce signal subtraction. Inspect broad wings
  and compare with a measured background when available.
- **Regridding** — wavelength to a uniform angular-frequency grid with the
  correct Jacobian and anti-aliasing; the SHG carrier doubling is handled
  automatically, and τ = 0 is centred on the delay marginal to sub-sample
  precision.
- **Band diagnostics** — the grid's edge taper gradually suppresses frequency
  rows near its boundaries. A narrow `lam_min`/`lam_max` can remove real signal
  and weaken constraints near the edge. `TraceData.taper_loss` reports how much of the
  trace the taper removed, and the loader warns above 1 %.

Further reading: [Preprocessing](../howto/preprocessing.md),
[Grids and filtering](../howto/grids_and_filtering.md),
[Loading simulated traces](../howto/loading_simulated.md).

## Analysis and plotting

- **Marginal checks** — {mod}`croak.marginal_checks` tests the measured frequency
  marginal against the prediction *before* retrieval: centroid and width anchors
  per geometry, and automatic choice of the third-order efficiency exponent by
  centroid matching.
- **Ambiguity resolution** — an SHG trace cannot distinguish a pulse from its
  mirror image, and the flip also inverts the sign of the retrieved spectral
  phase. With a known asymmetric temporal intensity,
  {func}`croak.processing.resolve_time_direction` can select the matching branch.
  A symmetric intensity cannot distinguish it; a separate measurement or known
  spectral phase is then needed to establish the sign of spectral phase.
- **Post-processing** — {func}`croak.processing.process_result` returns oversampled temporal
  intensity and phase with the linear ramp removed, the transform-limited
  reference, fitted GDD/TOD/FOD, marginals, weighted residuals, and absolute peak
  power when a pulse energy is supplied.
- **Edge-energy quality flag** — {func}`croak.processing.edge_energy_fraction` reports how
  much spectral energy sits near the grid boundaries. It flags results for
  a wider-grid check; it does not measure identifiability or certify accuracy.
  Compare the auxiliary spectrum and residual as well, particularly when fitting
  a free row response with `R_omega`.
- **Plotting** — a twelve-panel retrieval summary built from twelve independent
  panels (also laid out as three 2×2 pages), a six-panel filter before/after
  view, spectrograms, uncertainty and thickness-sensitivity plots. Every
  composite is built from single-axis functions, so the GUI embeds the same plotting functions.

Further reading: [Marginal checks](../howto/marginal_checks.md),
[Post-processing](../howto/postprocessing.md).

## Uncertainty

{func}`croak.uncertainty.estimate_fwhm_uncertainty` dispatches four
resampling-based estimators by name:

| `method` | Kind |
|---|---|
| `"parametric"` | Monte-Carlo over a detector noise model, re-retrieving each replicate |
| `"delay"` / `"frequency"` | resampling (block) bootstraps, using empirical resampling assumptions |
| `"thickness"` | systematic — propagates a measured substrate-thickness prior |

Two more estimators are their own functions rather than dispatcher methods:

- {func}`croak.covariance.covariance_uncertainty` — a local Gaussian
  covariance approximation from the JAX residual Jacobian. Its pseudo-inverse
  drops weak directions, which can include physical uncertainty as well as gauges. It needs **no re-retrieval** and can analyse results from NumPy solvers when
  their model and parameterisation can be reconstructed by the covariance path.
- {func}`croak.uncertainty.combine_uncertainties` — convolves the centred sample
  distributions of *independent* contributions, preserving skew (and reducing to
  quadrature in the Gaussian limit).

Also: standard errors on fitted thickness, delay-zero and smearing width;
coverage calibration against synthetic ground truth; and propagation of an
interval to another point in the beamline through a known transfer function.

Further reading: [Estimating FWHM uncertainty](../howto/uncertainty.md),
[Uncertainty of the retrieved duration](../explanation/uncertainty_estimation.md).

## Interfaces

- **Library first** — every stage is callable without the GUI.
- **The wizard** — a six-stage PyQt6 application (Load → Marginal check →
  Preprocess → Retrieve → Dispersion → Uncertainty) with live retrieval previews and a numeric
  read-out of retrieved, measured, transform-limited and (for synthetic
  sessions) true pulse parameters. Laid out for 1366×768 displays and up.
  Importing `croak` does not import Qt.
- **Sessions** — the session engine replays a saved TOML session end to end, so the
  interfaces use shared stage operations. Retain external inputs, calibrations,
  software version and random seeds to reproduce a result.
- **CLI** — `croak replay` reruns a session headlessly, including batch
  retargeting across datasets; `croak script` generates a standalone, editable
  Python script from it.

Further reading: [The wizard GUI](../howto/gui.md),
[Saving and loading](../howto/saving_and_loading.md).

## Conventions

SI units in the core numerical interfaces (session display units are named
explicitly); a centred, baseband angular-frequency grid; the physics
Fourier convention ($+i\omega t$ forward, $1/2\pi$ on the inverse); traces stored
as `(Nomega, Ndelay)`. The trace error is written $R$. The known ambiguities —
absolute phase, time origin, and for SHG the direction of time — are documented
rather than hidden.

Further reading: [Conventions](conventions.md),
[FFT convention](fft_convention.md), [Glossary](glossary.md).
