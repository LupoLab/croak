# Documentation layout review — phase 1

This review record is excluded from the published manual. It records the
navigation prototype and the remaining editorial work, not a completed
scientific or API review.

## Review checkpoint

Approve or revise the five-section layout, the introduction sequence, the
user-guide chapter order, and the gallery grouping before the full editing pass.
Existing detailed pages and tutorial calculations are retained at this checkpoint.

## Navigation

```text
croak
  Introduction
    Get and install croak
      Installing the GUI
      Installation
    Functionality and capabilities
    GUI or script interface
    Quickstart
    Your first retrieval
    Dispersive retrieval
    The experimental workflow
    Where to go next
  User guide
    Measurement fundamentals and conventions
      The PNPS framework
      Interactions
      Conventions
      FFT convention
      Grids, wavelength ranges and the SHG carrier
    Preparing and checking data
      Cleaning and regridding measured data
      Loading numerically simulated traces
      Marginal consistency checks
      FROG trace marginals as consistency checks
    Modelling the measurement
      The forward model
      Modelling geometric time smearing
      Modelling the collection aperture
    Retrieval and numerical methods
      Choosing a solver
      Retrieval as nonlinear least squares
      Retrieval algorithms
      Gradients: automatic differentiation and the Wirtinger adjoint
      Regularisation
      Fitting the medium thickness and delay-zero
    Interpreting results
      Post-processing and plotting results
      Validation
    Propagation and dispersion tuning
      Materials and mirrors
    Uncertainty
      Estimating FWHM uncertainty
      Uncertainty of the retrieved pulse duration
    Saving, sessions, and reproducible workflows
      The wizard GUI
      Saving and loading
    Glossary and references
      Glossary and notation
      References
  Examples
    Comparing algorithms
    Dispersion tuning
  Developer's guide
    Development environment and checks
    Package architecture
    Extending interactions and solvers
    GUI and session integration
    Numerical validation
    Building and maintaining the documentation
    Project name and history
  API
    Core physics
    Retrieval
    Workflow
    Plotting
    JAX backend
```

## Existing pages and their destinations

All existing page URLs are retained. Each published page has one navigation
parent; links and gallery cards provide other routes to the same content.

| Existing source | Navigation destination | Editorial treatment after layout approval |
|---|---|---|
| `docs/explanation/algorithms.md` | User guide → Retrieval and numerical methods → Retrieval algorithms | Retain; review scientific accuracy, clarity, terminology, cross-references, and reader progression. |
| `docs/explanation/forward_model.md` | User guide → Modelling the measurement → The forward model | Retain; review scientific accuracy, clarity, terminology, cross-references, and reader progression. |
| `docs/explanation/gradients.md` | User guide → Retrieval and numerical methods → Gradients: automatic differentiation and the Wirtinger adjoint | Keep mathematical derivations in the user guide; extract contributor-only implementation rationale. |
| `docs/explanation/interactions.md` | User guide → Measurement fundamentals and conventions → Interactions | Keep physical kernels and scientific derivative conventions; extract extension implementation details into the developer guide. |
| `docs/explanation/marginals.md` | User guide → Preparing and checking data → FROG trace marginals as consistency checks | Retain; review scientific accuracy, clarity, terminology, cross-references, and reader progression. |
| `docs/explanation/pnps_framework.md` | User guide → Measurement fundamentals and conventions → The PNPS framework | Retain; review scientific accuracy, clarity, terminology, cross-references, and reader progression. |
| `docs/explanation/retrieval_theory.md` | User guide → Retrieval and numerical methods → Retrieval as nonlinear least squares | Retain; review scientific accuracy, clarity, terminology, cross-references, and reader progression. |
| `docs/explanation/uncertainty_estimation.md` | User guide → Uncertainty → Uncertainty of the retrieved pulse duration | Retain statistical derivations and limitations; connect estimators to user choices and move internal assembly details to developer material. |
| `docs/explanation/validation.md` | User guide → Interpreting results → Validation | Reconcile protocols, field planes, figures, numerical claims, and limitations against the two revised drafts. |
| `docs/getting_started/gui.md` | Introduction → Get and install croak → Installing the GUI | Retain the route for users without Python experience; harmonise with the installation overview. |
| `docs/getting_started/installation.md` | Introduction → Get and install croak → Installation | Retain user installation; consolidate overlapping setup guidance and move contributor-only procedures to the developer guide. |
| `docs/getting_started/quickstart.md` | Introduction → Quickstart | Retain; review scientific accuracy, clarity, terminology, cross-references, and reader progression. |
| `docs/howto/collection_aperture.md` | User guide → Modelling the measurement → Modelling the collection aperture | Retain; review scientific accuracy, clarity, terminology, cross-references, and reader progression. |
| `docs/howto/fitting_thickness_tau0.md` | User guide → Retrieval and numerical methods → Fitting the medium thickness and delay-zero | Retain; review scientific accuracy, clarity, terminology, cross-references, and reader progression. |
| `docs/howto/geometric_smearing.md` | User guide → Modelling the measurement → Modelling geometric time smearing | Retain; review scientific accuracy, clarity, terminology, cross-references, and reader progression. |
| `docs/howto/grids_and_filtering.md` | User guide → Measurement fundamentals and conventions → Grids, wavelength ranges and the SHG carrier | Retain; review scientific accuracy, clarity, terminology, cross-references, and reader progression. |
| `docs/howto/gui.md` | User guide → Saving, sessions, and reproducible workflows → The wizard GUI | Retain; review scientific accuracy, clarity, terminology, cross-references, and reader progression. |
| `docs/howto/loading_simulated.md` | User guide → Preparing and checking data → Loading numerically simulated traces | Retain; review scientific accuracy, clarity, terminology, cross-references, and reader progression. |
| `docs/howto/marginal_checks.md` | User guide → Preparing and checking data → Marginal consistency checks | Retain; review scientific accuracy, clarity, terminology, cross-references, and reader progression. |
| `docs/howto/materials_and_mirrors.md` | User guide → Propagation and dispersion tuning → Materials and mirrors | Retain; review scientific accuracy, clarity, terminology, cross-references, and reader progression. |
| `docs/howto/postprocessing.md` | User guide → Interpreting results → Post-processing and plotting results | Retain; review scientific accuracy, clarity, terminology, cross-references, and reader progression. |
| `docs/howto/preprocessing.md` | User guide → Preparing and checking data → Cleaning and regridding measured data | Retain; review scientific accuracy, clarity, terminology, cross-references, and reader progression. |
| `docs/howto/regularisation.md` | User guide → Retrieval and numerical methods → Regularisation | Retain; review scientific accuracy, clarity, terminology, cross-references, and reader progression. |
| `docs/howto/saving_and_loading.md` | User guide → Saving, sessions, and reproducible workflows → Saving and loading | Retain; review scientific accuracy, clarity, terminology, cross-references, and reader progression. |
| `docs/howto/solver_selection.md` | User guide → Retrieval and numerical methods → Choosing a solver | Retain; review scientific accuracy, clarity, terminology, cross-references, and reader progression. |
| `docs/howto/uncertainty.md` | User guide → Uncertainty → Estimating FWHM uncertainty | Retain; review scientific accuracy, clarity, terminology, cross-references, and reader progression. |
| `docs/index.md` | croak | Rewritten as the five-section home page; former capability details remain in the Features page and related chapters; quantitative claims are in the scientific audit. |
| `docs/reference/api/core.md` | API → Core physics | Retain API grouping; edit rendered prose in the source docstrings without changing interfaces or behaviour. |
| `docs/reference/api/index.md` | API | Retain API groupings; review all rendered public docstrings in phase 4. |
| `docs/reference/api/jax.md` | API → JAX backend | Retain API grouping; edit rendered prose in the source docstrings without changing interfaces or behaviour. |
| `docs/reference/api/plotting.md` | API → Plotting | Retain API grouping; edit rendered prose in the source docstrings without changing interfaces or behaviour. |
| `docs/reference/api/retrieval.md` | API → Retrieval | Retain API grouping; edit rendered prose in the source docstrings without changing interfaces or behaviour. |
| `docs/reference/api/workflow.md` | API → Workflow | Retain API grouping; edit rendered prose in the source docstrings without changing interfaces or behaviour. |
| `docs/reference/bibliography.md` | User guide → Glossary and references → References | Retain; review scientific accuracy, clarity, terminology, cross-references, and reader progression. |
| `docs/reference/conventions.md` | User guide → Measurement fundamentals and conventions → Conventions | Retain; review scientific accuracy, clarity, terminology, cross-references, and reader progression. |
| `docs/reference/features.md` | Introduction → Functionality and capabilities | Restructure as the detailed capabilities overview; reconcile model/solver support and link to worked examples. |
| `docs/reference/fft_convention.md` | User guide → Measurement fundamentals and conventions → FFT convention | Retain; review scientific accuracy, clarity, terminology, cross-references, and reader progression. |
| `docs/reference/glossary.md` | User guide → Glossary and references → Glossary and notation | Retain; review scientific accuracy, clarity, terminology, cross-references, and reader progression. |
| `docs/reference/naming.md` | Developer's guide → Project name and history | Move ownership to developer history; the acronym is already prominent in reader-facing entry pages. |
| `docs/tutorials/01_first_retrieval.md` | Introduction → Your first retrieval | Retain one canonical tutorial; edit pacing and interpretation, execute, and refresh the committed cache when changed. |
| `docs/tutorials/02_choosing_an_algorithm.md` | Examples → Comparing algorithms | Retain one canonical tutorial; edit pacing and interpretation, execute, and refresh the committed cache when changed. |
| `docs/tutorials/03_dispersive_retrieval.md` | Introduction → Dispersive retrieval | Retain one canonical tutorial; edit pacing and interpretation, execute, and refresh the committed cache when changed. |
| `docs/tutorials/04_experimental_workflow.md` | Introduction → The experimental workflow | Retain one canonical tutorial; edit pacing and interpretation, execute, and refresh the committed cache when changed. |
| `docs/tutorials/05_dispersion_tuning.md` | Examples → Dispersion tuning | Retain one canonical tutorial; edit pacing and interpretation, execute, and refresh the committed cache when changed. |

Coverage: **44 existing Markdown pages**, all assigned; **66 total published pages** in this prototype.

## Examples and datasets

The gallery is a linked catalogue at this checkpoint. Phase 4 adds worked
pages and executed outputs using the existing calculation sources.

| Source | Gallery group / destination | Treatment |
|---|---|---|
| `examples/example_retrieval.py` | Getting started | Worked comparison of shapes, interactions, noise, and dispersion; link the basic case to the first tutorial. |
| `examples/example_workflow.py` | Getting started / experimental workflow | Use the introductory tutorial as the canonical walkthrough; retain the runnable script. |
| `examples/example_defringe.py` | Preparing data | Worked fringe-removal example with before/after plots. |
| `examples/example_arpls_baseline.py` | Preparing data | Worked fringe and baseline correction with checks for pulse distortion. |
| `examples/synthetic_duv_trace.py` | Preparing data | Shared synthetic input generator; retain as helper, not a duplicate tutorial. |
| `examples/example_geometric_smearing.py` | Instrument effects and validation | Worked geometry example; preserve model assumptions and diagnostics. |
| `examples/example_paper_thickness.py` | Instrument effects and validation | Describe actual settings; reconcile its relation to the revised paper rather than claiming exact reproduction. |
| `examples/example_paper_rdw.py` | Instrument effects and validation | Make simulated provenance, aperture, field convention, and actual protocol explicit. |
| `examples/example_thickness_uncertainty.py` | Analysing and transforming pulses | Worked calibration-uncertainty example; state independence assumptions when combining estimates. |
| `examples/data/README.md` | Instrument effects and validation | Review dataset provenance, field plane, stored observables, and claimed connection to the paper. |
| `examples/data/tgfrog_sim_1fs_uvfs.h5` | Substrate-thickness example | Preserve bundled numerical data; verify metadata and field conventions before interpreting output. |
| `examples/data/tgfrog_sim_rdw_duv.h5` | Simulated RDW example | Preserve bundled numerical data; verify aperture and field conventions. |

The remaining two tutorials are owned by Examples: algorithm comparison and
dispersion tuning. The three introductory tutorials are linked by gallery cards.

## Rendered API sections

Every current API section is retained. The modules below define the source
scope for the public-docstring editing pass; the layout phase does not claim
that their prose has already been reviewed.

| API page | Section | Source modules |
|---|---|---|
| `docs/reference/api/core.md` | Grid and transforms | `croak.grid` |
| `docs/reference/api/core.md` | Pulses | `croak.pulses` |
| `docs/reference/api/core.md` | Interactions | `croak.interactions` |
| `docs/reference/api/core.md` | Forward model | `croak.forward` |
| `docs/reference/api/core.md` | Geometric smearing | `croak.smearing` |
| `docs/reference/api/core.md` | Chromatic focal mixture | `croak.focal` |
| `docs/reference/api/core.md` | Collection aperture | `croak.collection` |
| `docs/reference/api/core.md` | Materials | `croak.materials` |
| `docs/reference/api/core.md` | Gases | `croak.gases` |
| `docs/reference/api/core.md` | Maths utilities | `croak.maths` |
| `docs/reference/api/core.md` | Constants | `croak.constants` |
| `docs/reference/api/jax.md` | JAX forward model | `croak.forward_jax` |
| `docs/reference/api/jax.md` | JAX metrics | `croak.metrics_jax` |
| `docs/reference/api/jax.md` | JAX pulse parameterisation | `croak._jax_pulse` |
| `docs/reference/api/plotting.md` | Plotting | `croak.plotting` |
| `docs/reference/api/retrieval.md` | The `retrieve` wrapper | `croak.retrieve` |
| `docs/reference/api/retrieval.md` | Solver base | `croak.solver` |
| `docs/reference/api/retrieval.md` | COPRA | `croak.copra` |
| `docs/reference/api/retrieval.md` | COPRA (JAX) | `croak.copra_jax` |
| `docs/reference/api/retrieval.md` | L-BFGS (analytic gradients) | `croak.lbfgs` |
| `docs/reference/api/retrieval.md` | L-BFGS (hand-gradient JAX) | `croak.lbfgs_hand` |
| `docs/reference/api/retrieval.md` | L-BFGS (autodiff) | `croak.lbfgs_ad` |
| `docs/reference/api/retrieval.md` | L-BFGS (Optimistix) | `croak.optimistix_lbfgs` |
| `docs/reference/api/retrieval.md` | Levenberg–Marquardt | `croak.lm` |
| `docs/reference/api/retrieval.md` | Levenberg–Marquardt (Optimistix) | `croak.optimistix_lm` |
| `docs/reference/api/retrieval.md` | CMA-ES (global) | `croak.cmaes` |
| `docs/reference/api/retrieval.md` | Warm-started L-BFGS | `croak.warm_lbfgs` |
| `docs/reference/api/retrieval.md` | Result | `croak.result` |
| `docs/reference/api/retrieval.md` | Progress reporting | `croak.progress` |
| `docs/reference/api/retrieval.md` | Metrics | `croak.metrics` |
| `docs/reference/api/workflow.md` | I/O | `croak.io` |
| `docs/reference/api/workflow.md` | Preprocessing | `croak.preprocess` |
| `docs/reference/api/workflow.md` | Pipeline | `croak.pipeline` |
| `docs/reference/api/workflow.md` | Processing | `croak.processing` |
| `docs/reference/api/workflow.md` | Known-truth errors | `croak.truth_metrics` |
| `docs/reference/api/workflow.md` | Marginal consistency checks | `croak.marginal_checks` |
| `docs/reference/api/workflow.md` | Dispersion tuning | `croak.dispersion` |
| `docs/reference/api/workflow.md` | Mirrors | `croak.mirrors` |
| `docs/reference/api/workflow.md` | Refractive-index database | `croak.refractive_db` |
| `docs/reference/api/workflow.md` | Saving and loading | `croak.save` |
| `docs/reference/api/workflow.md` | Uncertainty analysis | `croak.uncertainty`, `croak.covariance` |
| `docs/reference/api/workflow.md` | Sessions (headless replay) | `croak.session.params`, `croak.session.options`, `croak.session.pipeline`, `croak.session.dispersion`, `croak.session.engine`, `croak.session.retarget` |
| `docs/reference/api/workflow.md` | Command line and script generation | `croak.cli`, `croak.scripting` |

## Other documentation surfaces

| Surface | Treatment |
|---|---|
| README | Acronym and navigation updated now; full scientific and editorial pass follows. |
| CHANGELOG | Record the restructuring under Unreleased; preserve historical release entries. |
| Existing paper figures and hero/GUI images | Inventory below; check provenance, captions, scientific consistency, and legibility in later phases. |
| Sphinx configuration, CSS, templates, requirements, Read the Docs | Retain the existing stack; verify navigation, builds, and cache behaviour. |
| Tutorial cache | Preserve calculations in phase 1; refresh and commit when execution changes. |

## Figure inventory

- `docs/_static/gui_retrieve.png`
- `docs/_static/hero.png`
- `docs/_static/paper/figure_aperture.png`
- `docs/_static/paper/figure_boxcars3d.png`
- `docs/_static/paper/figure_dispersion_failure.png`
- `docs/_static/paper/figure_loop.png`
- `docs/_static/paper/figure_rdw.png`
- `docs/_static/paper/figure_smearing_dd.png`
- `docs/_static/paper/figure_thickness1fs.png`
