# Documentation content map and review record

This record is excluded from the published manual. It accounts for all original
pages, rendered API domains, examples and data, and records their final ownership.

## Review checkpoint and completed treatment

The five-section layout was approved by the user before the full editing pass.
The introduction, nine guide chapters, grouped gallery, developer guide and API
now follow that structure. The table below records the retained paths and the
editorial/scientific treatment applied. No original page was deleted or moved.
Contributor kernel-extension code and live-view architecture were extracted
into the developer guide; their original pages retain onward links. Installation
routes share one detailed installation reference. The statistical explanation
and practical estimator instructions retain separate roles and cross-reference.

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
    Retrieval across pulse shapes and interactions
    Scripted measurement workflow
    Removing carrier fringes
    Removing an asymmetric baseline
    Comparing algorithms
    Geometric smearing
    Retrieval through a thickness series
    Retrieving a simulated RDW pulse
    Propagating thickness uncertainty
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

| Existing source | Navigation destination | Treatment applied |
|---|---|---|
| `docs/explanation/algorithms.md` | User guide → Retrieval and numerical methods → Retrieval algorithms | Retained and edited for scientific accuracy, terminology, cross-references and reading flow. |
| `docs/explanation/forward_model.md` | User guide → Modelling the measurement → The forward model | Retained and edited for scientific accuracy, terminology, cross-references and reading flow. |
| `docs/explanation/gradients.md` | User guide → Retrieval and numerical methods → Gradients: automatic differentiation and the Wirtinger adjoint | Retained mathematical derivations in the guide; linked contributor-only implementation guidance. |
| `docs/explanation/interactions.md` | User guide → Measurement fundamentals and conventions → Interactions | Retained physical kernels and derivative conventions; moved extension examples to the developer guide. |
| `docs/explanation/marginals.md` | User guide → Preparing and checking data → FROG trace marginals as consistency checks | Retained and edited for scientific accuracy, terminology, cross-references and reading flow. |
| `docs/explanation/pnps_framework.md` | User guide → Measurement fundamentals and conventions → The PNPS framework | Retained and edited for scientific accuracy, terminology, cross-references and reading flow. |
| `docs/explanation/retrieval_theory.md` | User guide → Retrieval and numerical methods → Retrieval as nonlinear least squares | Retained and edited for scientific accuracy, terminology, cross-references and reading flow. |
| `docs/explanation/uncertainty_estimation.md` | User guide → Uncertainty → Uncertainty of the retrieved pulse duration | Retained statistical derivations and limitations; connected estimators to user choices and qualified covariance/bootstrap assumptions. |
| `docs/explanation/validation.md` | User guide → Interpreting results → Validation | Reconciled protocols, field planes, figures, numerical claims and limitations against the two revised drafts. |
| `docs/getting_started/gui.md` | Introduction → Get and install croak → Installing the GUI | Retained the route for users without Python experience; linked the canonical installation instructions. |
| `docs/getting_started/installation.md` | Introduction → Get and install croak → Installation | Consolidated user installation guidance; linked contributor environment instructions. |
| `docs/getting_started/quickstart.md` | Introduction → Quickstart | Retained and edited for scientific accuracy, terminology, cross-references and reading flow. |
| `docs/howto/collection_aperture.md` | User guide → Modelling the measurement → Modelling the collection aperture | Retained and edited for scientific accuracy, terminology, cross-references and reading flow. |
| `docs/howto/fitting_thickness_tau0.md` | User guide → Retrieval and numerical methods → Fitting the medium thickness and delay-zero | Retained and edited for scientific accuracy, terminology, cross-references and reading flow. |
| `docs/howto/geometric_smearing.md` | User guide → Modelling the measurement → Modelling geometric time smearing | Retained and edited for scientific accuracy, terminology, cross-references and reading flow. |
| `docs/howto/grids_and_filtering.md` | User guide → Measurement fundamentals and conventions → Grids, wavelength ranges and the SHG carrier | Retained and edited for scientific accuracy, terminology, cross-references and reading flow. |
| `docs/howto/gui.md` | User guide → Saving, sessions, and reproducible workflows → The wizard GUI | Retained and edited for scientific accuracy, terminology, cross-references and reading flow. |
| `docs/howto/loading_simulated.md` | User guide → Preparing and checking data → Loading numerically simulated traces | Retained and edited for scientific accuracy, terminology, cross-references and reading flow. |
| `docs/howto/marginal_checks.md` | User guide → Preparing and checking data → Marginal consistency checks | Retained and edited for scientific accuracy, terminology, cross-references and reading flow. |
| `docs/howto/materials_and_mirrors.md` | User guide → Propagation and dispersion tuning → Materials and mirrors | Retained and edited for scientific accuracy, terminology, cross-references and reading flow. |
| `docs/howto/postprocessing.md` | User guide → Interpreting results → Post-processing and plotting results | Retained and edited for scientific accuracy, terminology, cross-references and reading flow. |
| `docs/howto/preprocessing.md` | User guide → Preparing and checking data → Cleaning and regridding measured data | Retained and edited for scientific accuracy, terminology, cross-references and reading flow. |
| `docs/howto/regularisation.md` | User guide → Retrieval and numerical methods → Regularisation | Retained and edited for scientific accuracy, terminology, cross-references and reading flow. |
| `docs/howto/saving_and_loading.md` | User guide → Saving, sessions, and reproducible workflows → Saving and loading | Retained and edited for scientific accuracy, terminology, cross-references and reading flow. |
| `docs/howto/solver_selection.md` | User guide → Retrieval and numerical methods → Choosing a solver | Retained and edited for scientific accuracy, terminology, cross-references and reading flow. |
| `docs/howto/uncertainty.md` | User guide → Uncertainty → Estimating FWHM uncertainty | Retained and edited for scientific accuracy, terminology, cross-references and reading flow. |
| `docs/index.md` | croak | Rewritten as the five-section home page; former capability details remain in the Features page and related chapters; quantitative claims are in the scientific audit. |
| `docs/reference/api/core.md` | API → Core physics | Retain API grouping; edit rendered prose in the source docstrings without changing interfaces or behaviour. |
| `docs/reference/api/index.md` | API | Retained groupings; reviewed source docstrings for all 50 original rendered modules, plus delay-origin conventions (see `api_review.md`). |
| `docs/reference/api/jax.md` | API → JAX backend | Retain API grouping; edit rendered prose in the source docstrings without changing interfaces or behaviour. |
| `docs/reference/api/plotting.md` | API → Plotting | Retain API grouping; edit rendered prose in the source docstrings without changing interfaces or behaviour. |
| `docs/reference/api/retrieval.md` | API → Retrieval | Retain API grouping; edit rendered prose in the source docstrings without changing interfaces or behaviour. |
| `docs/reference/api/workflow.md` | API → Workflow | Retain API grouping; edit rendered prose in the source docstrings without changing interfaces or behaviour. |
| `docs/reference/bibliography.md` | User guide → Glossary and references → References | Retained and edited for scientific accuracy, terminology, cross-references and reading flow. |
| `docs/reference/conventions.md` | User guide → Measurement fundamentals and conventions → Conventions | Retained and edited for scientific accuracy, terminology, cross-references and reading flow. |
| `docs/reference/features.md` | Introduction → Functionality and capabilities | Restructure as the detailed capabilities overview; reconcile model/solver support and link to worked examples. |
| `docs/reference/fft_convention.md` | User guide → Measurement fundamentals and conventions → FFT convention | Retained and edited for scientific accuracy, terminology, cross-references and reading flow. |
| `docs/reference/glossary.md` | User guide → Glossary and references → Glossary and notation | Retained and edited for scientific accuracy, terminology, cross-references and reading flow. |
| `docs/reference/naming.md` | Developer's guide → Project name and history | Move ownership to developer history; the acronym is already prominent in reader-facing entry pages. |
| `docs/tutorials/01_first_retrieval.md` | Introduction → Your first retrieval | Retain one canonical tutorial; edit pacing and interpretation, execute, and refresh the committed cache when changed. |
| `docs/tutorials/02_choosing_an_algorithm.md` | Examples → Comparing algorithms | Retain one canonical tutorial; edit pacing and interpretation, execute, and refresh the committed cache when changed. |
| `docs/tutorials/03_dispersive_retrieval.md` | Introduction → Dispersive retrieval | Retain one canonical tutorial; edit pacing and interpretation, execute, and refresh the committed cache when changed. |
| `docs/tutorials/04_experimental_workflow.md` | Introduction → The experimental workflow | Retain one canonical tutorial; edit pacing and interpretation, execute, and refresh the committed cache when changed. |
| `docs/tutorials/05_dispersion_tuning.md` | Examples → Dispersion tuning | Retain one canonical tutorial; edit pacing and interpretation, execute, and refresh the committed cache when changed. |

Coverage: **44 existing Markdown pages**, all assigned; **74 total published pages** after adding eight worked gallery pages.

## Examples and datasets

The gallery now contains eight executed script pages and the two gallery-owned
tutorials. Script calculations are loaded with MyST-NB `:load:`; there is one
runnable source for each calculation and its cache key includes that source.

| Source | Gallery group / destination | Treatment |
|---|---|---|
| `examples/example_retrieval.py` | Getting started | Worked comparison of shapes, interactions, noise, and dispersion; link the basic case to the first tutorial. |
| `examples/example_workflow.py` | Getting started / experimental workflow | Retain the introductory tutorial as the paced walkthrough; execute the script in `docs/examples/workflow.md` with its own explanation. |
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

Every original API section is retained. The 51 modules below were reviewed in
source, including rendered attribute descriptions; see `api_review.md` for the
docstring inventory. Delay-origin functions and previously unrendered public
collection helpers are now directly addressable.

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
| `docs/reference/api/retrieval.md` | Delay-origin convention (added) | `croak.delay_origin` |
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
| README | Acronym, navigation, scope, conventions, examples and draft citations updated. |
| CHANGELOG | Recorded the restructuring under Unreleased; retained historical release entries. |
| Existing paper figures and hero/GUI images | Checked provenance and captions; six revised manuscript figures added. Original figure URLs retained. |
| Sphinx configuration, CSS, templates, requirements, Read the Docs | Retained the existing stack; verified navigation, builds and cache behaviour. |
| Tutorial cache | Refreshed for revised executable content; original unchanged calculations remain cache hits. |

## Original figure inventory

- `docs/_static/gui_retrieve.png`
- `docs/_static/hero.png`
- `docs/_static/paper/figure_aperture.png`
- `docs/_static/paper/figure_boxcars3d.png`
- `docs/_static/paper/figure_dispersion_failure.png`
- `docs/_static/paper/figure_loop.png`
- `docs/_static/paper/figure_rdw.png`
- `docs/_static/paper/figure_smearing_dd.png`
- `docs/_static/paper/figure_thickness1fs.png`

## Added worked pages

| Script | Canonical worked page |
|---|---|
| `example_retrieval.py` | `docs/examples/retrieval.md` |
| `example_workflow.py` | `docs/examples/workflow.md` |
| `example_defringe.py` | `docs/examples/defringe.md` |
| `example_arpls_baseline.py` | `docs/examples/baseline.md` |
| `example_geometric_smearing.py` | `docs/examples/geometric_smearing.md` |
| `example_paper_thickness.py` | `docs/examples/thickness_series.md` |
| `example_paper_rdw.py` | `docs/examples/rdw.md` |
| `example_thickness_uncertainty.py` | `docs/examples/thickness_uncertainty.md` |

Each page includes inputs, method, execution, output interpretation, limitations,
runnable source, and contextual guide links. `synthetic_duv_trace.py` is the
shared input generator for both preparation examples, included in their cache keys.

The six added draft figures and their provenance are listed in
`docs/_static/paper/README.md`. The final manual contains 74 published content
pages; generated search and API indexes are additional build output.
