# Scientific consistency audit

This record is excluded from the published manual. The initial discrepancies
below were used to guide the completed editorial and scientific pass. Their
resolution is recorded at the end. Notebook examples were executed, but the
manuscript's full research campaign was not rerun. Package behaviour is unchanged.

## Evidence and precedence

The agreed manuscript sources are the unpublished drafts in
`/Users/jt52/Documents/Papers/FROG/newpaper/`:

- `prx/main.tex`: *Differentiable physical models for ultrashort-pulse retrieval*.
- `prresearch/main.tex`: *A dispersive and spatial model of transient-grating
  frequency-resolved optical gating*.
- `NOTATION.md`, `SCIENTIFIC_AUDIT.md`, and `CLAIMS_AND_FIGURES.md`: shared
  conventions, corrected claims, limitations, and figure/data provenance.

The current package determines supported options and defaults. The papers
describe particular models and numerical protocols; their chosen settings must
not silently replace package defaults. Manuscript corrections establish what
the revised drafts claim, not a fresh independent validation by this task.
Private source paths must not become dependencies of documentation builds.

## Initial discrepancies

| ID | Affected material | Evidence and required correction | Planned phase |
|---|---|---|---|
| S01 | `reference/bibliography.md`, `explanation/validation.md`, README, example scripts and dataset README | These cite one older paper title. The supplied sources are two unpublished drafts with distinct inverse-problem and forward-model scopes. Update titles and describe their roles without implying publication. | 3–4 |
| S02 | `explanation/validation.md`, “A real single-cycle pulse”; dataset and example descriptions | The RDW pulse and instrument trace are simulated. PRX explicitly excludes the provisional experimental section. Replace ambiguous “real”/“measurement” language with accurate simulated provenance. | 3–4 |
| S03 | `explanation/validation.md`, thickness and RDW results; README; existing paper figures | The manual quotes older durations (including 1.03/1.06 fs references and full-model 1.12–1.21 fs) without the revised common-plane convention. The claim register C01/C02 gives the revised Gaussian 1.046531–1.079009 fs range against 1.0529 fs, and RDW standard-model 2.433400 fs after complex-field conversion. Reconcile cases, planes, and sampling before updating numbers; do not substitute numbers across datasets. | 3–4 |
| S04 | `explanation/validation.md`, “The virtual instrument” and retrieval protocol; `example_paper_*` descriptions | The manual describes fringe/baseline processing and fitted offsets as the paper protocol. Audit A13 describes the adopted unfiltered, consistently centred protocol, no free tau0, and spectral weight 0.03. Label the scripts by their actual settings; distinguish examples from reproduction of the revised study. | 3–4 |
| S05 | `explanation/validation.md`, structured-pulse and medium-thickness claims | The revised drafts retain a failed negative-GDD cell at 2 µm and do not establish universal model validity, a Raman ceiling, or a general thickness limit. Remove unsupported generalisations while retaining the underlying physical discussion and explicitly bounded evidence. | 3 |
| S06 | `explanation/retrieval_theory.md`, `explanation/algorithms.md`, `howto/solver_selection.md` | Claims that all solvers minimise the same objective and converge to the same minimum conflict with `croak/solver.py`: gradient solvers use R plus penalties, whereas LM appends penalty residuals and uses an R-squared objective. Non-convex optimisation also does not guarantee a shared endpoint. State the actual objective and comparison conditions. | 3–4 |
| S07 | `explanation/forward_model.md`, `explanation/gradients.md`, `explanation/algorithms.md`, README | Statements that projection methods cannot represent spatial mixtures are broader than the implementation restriction. Audit A06 distinguishes the ordinary single-field update from vector/mixed-state projections. Describe croak's solver support without an impossibility claim. | 3–4 |
| S08 | `explanation/interactions.md`, “The adjoint convention” | The page says croak computes gradients by hand rather than AD, while the package exposes both routes. Retain the adjoint mathematics; make the supported derivative routes consistent with the gradient chapter and API. | 3–4 |
| S09 | `howto/loading_simulated.md`, third-order exponent discussion | A universal or additive power-law explanation is too broad for the revised broadband model. PRResearch retains the paraxial omega/n field-generation factor, separates detector densities and Jacobians, and does not infer a universal effective-field exponent from single-frequency products (audit A02/A07). Keep calibration examples as conditional empirical examples. | 3–4 |
| S10 | Validation reporting recommendations and uncertainty discussion | Agreement with an auxiliary spectrum used in the objective is a consistency check, not independent validation. Solver-start spread is not measurement or calibration uncertainty (audit A17/A18). Harmonise the guide, examples, figures, and API prose. | 3–4 |
| S11 | `explanation/forward_model.md`, solver-support note; home-page legacy highlights | The note's “autodiff solvers only” list omits `warm-lbfgs`, although the wrapper/default routes through the AD solver. The former home page also described all three interactions as thin or dispersive; `ForwardModel` rejects dispersive SHG. Reconcile support tables against the current interfaces. The new home page avoids these overbroad summaries. | 3–4 |
| S12 | Package-default prose versus paper settings | `croak.solver.REG_DEFAULTS` uses gradient weights (0.01, 0.03), while the revised main paper protocol uses spectral weight 0.03. Preserve current defaults and name the paper's settings explicitly. No runtime default change is part of this documentation task. | 3–4 |

## Scientific checks used during the pass

These checks were prompted by the manuscript audit and applied to the corresponding
guide and API descriptions.

| ID | Topic and material | Required check |
|---|---|---|
| C01 | Field definitions: conventions, simulated loading, post-processing, truth metrics, examples | Distinguish source, transverse-integrated beamlet, effective field, and on-axis slab-entrance field. Convert the complex spectrum before comparing temporal profiles; state amplitude, global-phase, time-origin, and time-direction ambiguities. |
| C02 | Smearing equations and diagrams | Reconcile p/theta definitions, arm order, delay signs, Gaussian field/intensity conventions, and the scope of the “attenuation only” result. Keep the non-Gaussian limitation and temporal-phase versus spectral GDD/TOD/FOD distinction (audit A04/A05/A21). |
| C03 | Interaction and gradient equations | Check conjugations under bar(z) = dJ/dz*, including split-gate kernels (audit A01). Do not generalise PG/TG and SD equivalence or TG validation to arbitrary complex fields and interactions (A11). |
| C04 | Collection aperture and focal model | Check coherent sums before intensity, phase restoration, absolute versus envelope frequency, Fourier normalisation, hard-hole quadrature versus physical transmission, and the geometric clearance factor. Separate exact Parseval statements from model assumptions (A03/A14/A15). |
| C05 | Broadband non-collinear propagation | Do not infer exact broadband phase matching from the degenerate fourth-corner construction; state the approximations and limits (A08). |
| C06 | Weighted objective, gains, and centring | Distinguish residual weights from inverse variances, retain squared weights in the general row-gain formula, and state when row-constant weights cancel. `croak.delay_origin.recentring` differentiates gain-dependent centring and requires uniform delays. Explain this separately from fitting a free delay offset (A09/A10). |
| C07 | Fixed-field validation and convergence | Separate complete-trace/global-gain tests from fitted row-response tests, rms/peak from relative L2, and forward-model validation from inverse retrieval. Include the actual support, reference accuracy, and quadrature refinement (C06/C09/C10 in the claim register). |
| C08 | Reporting uncertainty | Preserve estimator assumptions, identifiability, nuisance parameters, calibration uncertainty, coverage, and correlations when propagating to another plane. Do not describe every bootstrap spread as a calibrated confidence interval or automatic lower bound. |
| C09 | Algorithm comparisons and timing | Verify that compared objectives, constraints, initial fields, preprocessing, stopping rules, and hardware match. Keep implementation performance separate from accuracy or scientific validity. |
| C10 | Bibliography and public availability | Update manuscript citations as drafts; check cited primary sources and existing links during the final pass. Do not invent publication status or public availability of private data. |

## Existing figures requiring reconciliation

| Current asset | Check against revised material |
|---|---|
| `figure_boxcars3d.png` | Geometry, arm labels, and the field reference plane versus PRResearch's instrument section. |
| `figure_loop.png` | Objective normalisation, spectral constraints, row gains, and centring versus revised PRX Figure 2. |
| `figure_dispersion_failure.png` | Dataset, duration convention, and residual definition versus the common-plane PRX failure exhibit. |
| `figure_thickness1fs.png` | Actual protocol, reference duration, plotted series, and sampling versus PRX Figure 3. |
| `figure_rdw.png` | Aperture, field conversion, simulated provenance, and whether the panel corresponds to the revised PRX Figure 5 case. |
| `figure_smearing_dd.png` | Geometry definitions, Gaussian assumptions, axes, and the evidence behind quantitative comparisons. |
| `figure_aperture.png` | Delay-dependent coherent collection and the scope of any spectral-efficiency curve versus revised PRResearch Figure 6. |

Do not overwrite a figure with superficially similar paper artwork. Match its
data and convention first, then regenerate or replace it with a caption that
states what it establishes. Retain uncertain substantive material for review.

## Resolution

- **S01–S05:** citations, simulated provenance, protocol, field planes and all
  validation captions updated against the revised drafts. Six revised figures
  replace the older quantitative evidence. Negative-GDD failures, the distinct
  TOD protocol and the case where the extended model has a lower profile error
  remain visible. Bundled scripts are explicitly illustrative legacy-data cases.
- **S06–S08, S11–S12:** objective families, solver support, adjoints, AD and
  package-versus-paper defaults reconciled in guide, examples and source
  docstrings. SciPy's `LM` class defaults to TRF; Optimistix options are not
  identical to the NLopt interface. Warm-stage approximations are explicit.
- **S09–S10:** detector factors, density Jacobians, field-plane conversion and
  auxiliary-spectrum interpretation separated; solver-start spread is identified
  as a convergence diagnostic.
- **C01–C06:** checked Fourier signs, conjugations, split-gate adjoints, Gaussian
  temporal-phase versus spectral GDD, timing symbols, coherent collection,
  phase restoration, absolute-frequency aperture mapping, quadrature versus
  physical transmission, finite-beam assumptions, weighted gains and centring.
- **C07–C09:** distinguished fixed-field checks from inverse retrievals, global
  versus row gains, rms/peak versus relative L2, solver objectives, non-convex
  endpoints, conditional precision and empirical coverage. Documented unsupported
  covariance models and delay-bootstrap combinations; see `runtime_findings.md`.
- **C10:** checked primary publication records, corrected the Wang et al. page
  range to 2400–2405 and added DOI links. Retained the supplied software DOI;
  the external resolver could not be verified in this session. No publication
  metadata was invented for the private drafts.

## Retained questions and evidence limits

The old fixed 10 µm collection crossover, edge-energy/regularisation benchmark
numbers and historical timing claims are not substantiated by the revised
protocol. They are retained in `legacy_collection_aperture.md`,
`legacy_regularisation.md`, `legacy_algorithms.md` and `api_legacy_claims.md` for
provenance review, rather than republished as current evidence. Original figure
files and old page URLs remain available. These values should only be reinstated
with their exact generating fields, model settings, metrics and archived runs.

The known-field examples establish results for their stated cases, not universal
identifiability, a Raman ceiling, an experimental validation or a maximum valid
slab thickness. The thickness-uncertainty example is explicitly biased in its
central estimate (about 2.30 fs in the final cache for a 3.00 fs generator); its narrow conditional
statistical interval does not certify accuracy. Finite bootstrap populations
illustrate interfaces rather than establish coverage.

Repeated local executions of that example with the same explicit seeds and
calculation settings gave approximately 2.30 and 2.64 fs, with similar residuals
near 0.308 %. Only source wording changed between these runs. The cause of the
variation was not isolated; the gallery records it as a numerical reproducibility
and convergence limitation, rather than promising a stable benchmark.

The copied manuscript figures were inspected against their source descriptions
and captions. This edit did not independently reproduce the manuscript campaign.
All six assets are committed locally; the build does not read private paths.

## Editorial archive

The `legacy_*.md` records and `superseded_validation.md` preserve substantive
older material where provenance or interpretation changed. They are excluded
from Sphinx and are not alternative user documentation. Current explanations
have one authoritative home, with contextual links from the introduction,
chapter landings, gallery and API. `api_review.md` records source-docstring scope.

## Execution checks added during review

The dispersive introduction's original 38.4 fs periodic window was too short for
its 50 µm propagation example. The revised 153.6 fs window and ±40 fs scan avoid
that wraparound in the demonstrated case. A fixed-field 20-versus-40 depth-node
comparison gives a relative trace L2 difference around 1.33e-5, with delay-edge
signal around 4.68e-5 of peak. The notebook prints both values; this is numerical
convergence evidence, not an independent physical validation.

The README and Quickstart now use an explicit initial Gaussian and fixed seed,
matching the paced first tutorial. Their original random start could stop at a
poor solution. The guide continues to explain non-convexity and multiple starts.
