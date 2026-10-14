# Scientific consistency audit — phase 1

This is an editorial working record, excluded from the published manual.
It records discrepancies and checks required before the scientific editing pass
can be considered complete. No new retrieval campaign has been run and no
package behaviour has been changed at this checkpoint.

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

## Confirmed discrepancies

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

## Checks required during the scientific pass

These are targeted checks prompted by the manuscript audit, not yet confirmed
errors in every corresponding croak page.

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

## Editorial work kept separate from scientific findings

The page map records navigation ownership and planned splits. The later prose
pass will replace implementation chronology with reader-oriented explanations,
define terms before use, make notation and British spelling consistent, remove
duplicate explanations, and supply context and next steps. Mathematical and
physical detail remains in the relevant user-guide chapter.

The layout checkpoint introduces conservative summaries without rewriting the
legacy scientific pages. Those pages and their existing figures still need the
checks above; the prototype must not be represented as the completed manual.
