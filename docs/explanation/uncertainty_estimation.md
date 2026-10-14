# Uncertainty of the retrieved pulse duration

An uncertainty analysis asks how a retrieved quantity changes when the uncertain
inputs change. For FWHM, those inputs may include detector noise, sampled data,
substrate thickness and other calibrations. This chapter explains the statistical
assumptions and limitations; the [practical guide](../howto/uncertainty.md) and
[worked thickness example](../examples/thickness_uncertainty.md) show the calls.

Single-trace resampling gives a **conditional estimate**. It does not automatically
include calibration or model error, guarantee coverage, or provide a mathematical
lower bound on total uncertainty. An incorrect noise model or failed replicate
optimisation can make the spread too large or too small.

```{note}
This is the *why*; for the *how* see the how-to,
[Estimating FWHM uncertainty](../howto/uncertainty.md). Methods A and B below ship in
{mod}`croak.uncertainty` ({func}`~croak.uncertainty.estimate_fwhm_uncertainty`); Method C
(the Laplace covariance) ships in {mod}`croak.covariance`
({func}`~croak.covariance.covariance_uncertainty` /
{func}`~croak.covariance.parameter_covariance`). One *systematic* — the substrate thickness
of a dispersive retrieval — is also propagated, by
{func}`~croak.uncertainty.thickness_bootstrap` (Method D below). Any of these can report the
error bar **at a different beamline point** by propagating each replicate through the
intervening dispersion (see [Propagating the uncertainty to a different beamline
point](#propagation) below).
```

(the-short-answer-and-its-four-conditions)=
## Conditions for interpretation

A single-trace bootstrap gives a defensible error bar **only** when:

1. **It is a statistical (precision) interval, conditional on the forward model and the
   noise model being correct.** It does *not* include systematic error (calibration,
   geometry, model mismatch), which often dominates real FROG.
2. **Measurement-noise variance is separated from algorithmic variance** — controlled by
   the warm-start policy (see [below](#separating-measurement-from-algorithmic-variance)).
3. **The interval is coverage-calibrated**, not merely a spread, before it is called
   "68 %".
4. **The FWHM is a meaningful summary** — i.e. the pulse is single-peaked. For structured
   pulses it is not, and the bootstrap distribution reveals that.

Granted these, one may write **"1.4 ± 0.2 fs (1σ statistical; systematic uncertainty not
included)"**. Repeated acquisitions test measurement variability, but shared calibration
biases can persist across all of them. A total budget also needs calibration and
model sensitivity assessments.

## Why this is even possible — the statistical structure of FROG

**The trace contains redundant measurements.** Its number of pixels can exceed
the number of field parameters, but pixel count alone does not establish
identifiability. Correlated noise, spectral support, fitted row gains and nearly
null parameter directions can reduce the information. Resampling probes
sensitivity within the chosen data and model assumptions.

**FWHM is gauge-invariant.** FROG has trivial ambiguities the
trace cannot fix. For SHG-FROG these are the absolute time origin, the carrier–envelope
phase (CEP), *and* the direction of time $E(t)\leftrightarrow E^*(-t)$. **TG-FROG — the PG
kernel $E(t)\,|E(t-\tau)|^2$, `interaction="pg"` in croak — breaks the time-reversal
ambiguity** for general fields, although special cases and incomplete data
can remain ambiguous. CEP does not change $|E(t)|^2$; translation moves the profile without
changing its width. A time flip also preserves the FWHM. So the FWHM of the
intensity envelope is a **well-defined scalar statistic with no gauge alignment
required** — unlike per-time or per-frequency *field* error bands, which would need each
replicate aligned (remove linear spectral phase → time shift; remove constant phase →
CEP; for SHG also resolve the flip) before they could be averaged. This is precisely what
makes the FWHM a clean target for resampling.

**No closed-form propagation in general.** Retrieval is a nonlinear inverse map; there is
no exact analytic propagation from trace noise to FWHM. Resampling / Monte-Carlo is the
practical route. A local linearisation is also possible (the [covariance route](#method-c)),
but is often more stable in a justified low-dimensional parameterisation.

## What an error bar can and cannot mean

Decompose the total uncertainty of the quoted FWHM into four parts:

- **Statistical / precision** — photon (shot) noise and detector (read) noise. Resampling
  **can** estimate this. *It is the only component a single-trace bootstrap addresses.*
- **Algorithmic** — non-convexity, stagnation, local minima, regularisation bias.
  Partially controllable; **must be separated** from the statistical part via the
  warm-start policy.
- **Systematic** — delay-axis calibration, wavelength calibration, detection spectral
  response, the TG third-order efficiency scaling
  ({func}`~croak.preprocess.third_order_scale`), marginal / phase-matching-bandwidth
  corrections, spatio-temporal coupling, finite beam geometry. A bootstrap **cannot see
  any of these** — every replicate reuses the same (mis)calibrated axes and the same
  forward model. In real FROG these often **dominate**.
- **Model / parameterisation bias** — a dispersive basis (B-spline phase, or a GDD/TOD/FOD
  Taylor model) cannot represent structure it does not include; resampling *within* the
  model is blind to the missing degrees of freedom and **underestimates**.

**Consequence:** label the resampling interval by what was varied. It omits
unvaried error sources, but is not guaranteed to be a lower bound on total uncertainty. A credible duration quote pairs it with a separate,
explicit **systematic budget** — or states plainly that systematics are excluded.

Three resampling estimators are available. They answer slightly different questions and
make different assumptions, so the strongest practice is to run more than one and compare.

## Method A — parametric Monte-Carlo bootstrap

**Procedure.**

1. Retrieve from the full trace → best field $\hat{\tilde E}$ and its scaled clean model
   trace $\hat T = \mu\,T_\text{sim}(\hat{\tilde E})$ — exactly
   {attr}`RetrievalResult.trace <croak.result.RetrievalResult.trace>` with its scale
   factor `mu`.
2. Adopt a noise model $\sigma(\cdot)$ (see [The noise model](#the-noise-model)).
3. For $b = 1\dots B$: draw a synthetic trace $T_b = \hat T + n_b$, with $n_b$ from the
   noise model, clipped at zero.
4. Re-retrieve each $T_b$, warm-started from $\hat{\tilde E}$.
5. Compute $\text{FWHM}_b$ for each. The distribution $\{\text{FWHM}_b\}$ gives the
   interval.

**Interpretation.** This is Efron's *parametric bootstrap*: it treats the fitted model as
truth and propagates the measurement noise through the full nonlinear retrieval. Under
Gaussian noise the FROG least-squares fit is the maximum-likelihood estimate (see
[Retrieval as nonlinear least squares](retrieval_theory.md)), so this is Monte-Carlo
sampling of the MLE's sampling distribution.

```{list-table}
:header-rows: 1
:widths: 50 50

* - Advantages
  - Disadvantages
* - Most physically transparent — noise enters where it does in the experiment.
  - **Conditions on the best fit being truth**: the interval is *centred* on a possibly
    biased point. It is a *width*, not a *bias*, estimate.
* - Handles arbitrary, **signal-dependent** noise (per-pixel Poisson + read) naturally.
  - **Depends on the noise model** and its calibration.
* - Works identically for **COPRA and LBFGS** (any solver), thin or dispersive.
  - Cost: $B$ full retrievals (mitigated by warm-starting and JAX batching).
* - Reuses `RetrievalResult.trace`, an `_add_noise`-style step and warm-start `guess=`.
  - Preprocessing colours the noise (see caveat below).
```

```{warning}
croak's preprocessing (DC/fringe NUFFT filters, the $\lambda\to\omega$ regridding in
{func}`~croak.preprocess.load_and_clean`) **correlates and colours** the noise. Adding
white noise to the *post-processed* $\hat T$ therefore models the wrong statistics. The
rigorous variant injects noise in the **raw measurement domain** and pushes each
replicate through the *same* preprocessing pipeline — at higher cost per replicate.
```

## Method B — weighted / leave-out data bootstrap

Data resampling estimates sensitivity to the observed measurement units without
specifying a parametric detector-noise distribution. It still assumes that the
chosen units and resampling scheme represent the variability of interest.

croak implements **delay-column** and **frequency-row** resampling, with
replacement or leave-out variants. Delay resampling rebuilds the selected delay
columns; frequency resampling uses multiplicities through residual weights
$\sqrt{k_n}$, so a row sampled $k_n$ times contributes $k_n r_n^2$ to the loss.
The standard retrieval interface accepts per-frequency weights, not an arbitrary
per-pixel weight matrix. These details matter when translating general bootstrap
formulae into a supported calculation.

Whole-column resampling retains within-exposure correlations when a column is a
measurement unit. Adjacent exposures may also be correlated, and scanning drift
is not automatically represented. Filtering and interpolation introduce further
correlation. A leave-out fraction changes the experiment being approximated and
must be reported; a raw leave-out spread is not automatically a calibrated
standard error. Resampled delays can be nonuniform or repeated, which is
incompatible with models requiring a uniform delay convolution. Use a supported
parametric route for such models and retain the original settings.

The method can be compared with parametric resampling, but agreement does not
validate assumptions shared by both. See Wang *et al.* in
[references](../reference/bibliography.md) for the FROG bootstrap background.

(method-c)=
## Method C — Laplace / Jacobian covariance (for dispersive retrieval)

The current covariance interface supports the core model and reduced smearing,
but does not expose chromatic focal/collection models, depth weighting or
`delay_origin="marginal_peak"`. Do not use a covariance built from a different
model as the uncertainty of the original retrieval. The Jacobian omits
regularisation curvature; penalised fits need additional care. With an explicit
noise model, residual whitening currently replaces the residual weights,
although those weights still enter the fitted gain. See the
{func}`~croak.covariance.parameter_covariance` notes.


When the phase is retrieved in a **low-dimensional basis** — croak's cubic **B-spline
phase** (`phase_basis="bspline"`, ~20 control points) or an effective GDD/TOD/FOD Taylor
model — retrieval becomes a genuine nonlinear least-squares fit of a few parameters
$\theta$, $\ \hat\theta = \arg\min_\theta \lVert r(\theta)\rVert^2$, and a local quadratic
approximation gives the parameter covariance directly:

```{math}
:label: laplace
\operatorname{Cov}(\hat\theta)\;\approx\;\hat\sigma^2\,\bigl(J^\top J\bigr)^{-1},
\qquad
\hat\sigma^2=\frac{\lVert r(\hat\theta)\rVert^2}{\,n-p\,},
```

where $J=\partial r/\partial\theta$ at the optimum, $n$ is the number of (weighted) data
points and $p=\dim\theta$. For an identifiable Gaussian-noise model, the Fisher information supplies a
local precision bound under regularity conditions. The estimated inverse Hessian
is an approximation to sampling covariance, not a guarantee that an estimator
attains that bound. croak *already computes $J$*: {class}`~croak.lm.LM` (and
{class}`~croak.optimistix_lm.OptxLM`) build `jax.jacfwd` of
{func}`croak.metrics_jax.residual_vector`. {func}`~croak.covariance.parameter_covariance`
rebuilds that residual at the retrieved spectrum and forms the covariance; because it is
reconstructed from the spectrum and the JAX forward model it can analyse a non-LM result whose model and parameterisation are supported — the least-squares algorithm only decides whether $J$ is exposed *natively*.

**Whitening and scale.** With the residual divided pixel-by-pixel by a
{class}`~croak.uncertainty.NoiseModel` σ, the scale is absolute, $\hat\sigma^2=1$ and
$\operatorname{Cov}=(J^\top J)^{+}$ — consistent with the parametric bootstrap, which
treats the same noise model as truth. Without a noise model croak falls back to the internal
reduced-χ² estimate {eq}`laplace`.

**Gauge degeneracy.** FROG's continuous trivial ambiguities — the absolute phase (CEP) and
the absolute timing (a linear spectral phase) — are *flat directions* of the objective, so
$J^\top J$ is **rank-deficient**. croak uses the pseudo-inverse (eigenvalues below `rcond` ×
the largest are dropped); known gauge directions do not change gauge-invariant quantities such as FWHM
and spectral intensity. A temporal profile must be aligned before comparison.
Other discarded directions may represent weakly identifiable physical changes;
setting their covariance to zero can understate uncertainty. Inspect rank and
sensitivity to the pseudoinverse cutoff. {attr}`CovarianceResult.null_dim <croak.covariance.CovarianceResult>`
reports its dimension.

**Propagate to FWHM by local Gaussian sampling.** croak draws $\theta^{(k)}\sim\mathcal
N(\hat\theta,\operatorname{Cov})$ and evaluates $\text{FWHM}(\theta^{(k)})$ for each — *no
re-retrieval*, just forward field evaluations through the same B-spline → field → FWHM path
Discarded parameter directions carry zero covariance in this local approximation;
align profiles before comparing them.
This sidesteps the FWHM's non-differentiability (interpolated half-max crossings, which
make a first-order delta method fragile), captures the curvature of the field → FWHM map,
and returns a full {class}`~croak.uncertainty.UncertaintyResult` (`method="covariance"`) —
drop-in compatible with the bootstrap output, so the same intervals, temporal band and
plotting apply.

```{list-table}
:header-rows: 1
:widths: 50 50

* - Advantages
  - Disadvantages
* - **No repeated retrieval** — one Jacobian and parameter sampling; cost
    still depends on grid and parameter dimension.
  - **Local (Laplace) approximation** — under-represents skew/non-Gaussianity for
    multimodal/skewed posteriors (low SNR, near an ambiguity, near the transform limit).
    Resampling can probe departures, but also needs a suitable noise model and coverage checks.
* - Gives a local approximation for an identifiable model under the stated
    noise assumptions; see [retrieval theory](retrieval_theory.md).
  - **Inherits the parameterisation's model bias**: it answers "how well is *this model*
    determined," not "how well is the true pulse known."
* - Gives the **full parameter covariance** at once — error bars on every derived quantity
    (spectral phase/intensity, GDD/TOD/FOD, the propagated pulse) fall out of one matrix.
  - $J^\top J$ can be **ill-conditioned**. The current covariance omits penalty
    curvature; regularisation in the original fit can also bias $\hat\theta$.
* - A **low-dimensional basis** (`phase_basis="bspline"`) can improve conditioning;
    compare with a suitable bootstrap rather than assuming agreement.
  - For a full **pointwise** retrieval ($p\sim 2N$) the many soft (poorly constrained)
    modes dominate and the estimate is unstable; degenerate draws are dropped, but prefer A,
    B, or the B-spline basis there.
```

## Method D — substrate thickness (a measured systematic, propagated)

Methods A–C estimate the **statistical** error and are, by construction, blind to
systematics — every replicate reuses the same forward model. For a **dispersive**
retrieval one systematic can be turned into a number, because it is itself a *measured*
input: the **substrate thickness** $L$, through which the model propagates the field
(`material`/`thickness`; see [The dispersive forward model](forward_model.md)). When $L$
is measured independently with an error bar, $L\sim\mathcal N(L_0,\sigma_L)$ — e.g.
$9.952\pm0.539$ µm — the retrieved field, and hence its FWHM, inherits a systematic
uncertainty. For short, deep-UV pulses (≈1 fs, ≈230 nm) the dispersion of even a few
hundred nm of silica is significant, and this term can exceed the
statistical one of Methods A/B.

**Procedure** ({func}`~croak.uncertainty.thickness_bootstrap`). A brute-force Monte-Carlo
over the *measured prior*: for $b=1\dots B$ draw $L_b\sim\mathcal N(L_0,\sigma_L)$ (clipped
at 0), rebuild the dispersive forward model at $L_b$, **re-retrieve the same measured
trace** (warm-started from the $L_0$ solution), and take the spread of
$\{\text{FWHM}_b\}$. It is the systematic complement to Method A: A perturbs the *data*
and holds the model fixed; D perturbs the *model* and holds the data fixed. If their effects are independent and approximately Gaussian, their variances add, $\sigma_\text{tot}^2=\sigma_\text{stat}^2+\sigma_L\text{-induced}^2$.

```{list-table}
:header-rows: 1
:widths: 50 50

* - Advantages
  - Disadvantages
* - Promotes the specified calibration contribution into a defensible, quotable number from an
    existing measurement (the substrate metrology).
  - Covers **only** the thickness systematic — calibration, geometry and model mismatch
    remain outside the budget.
* - Assumption-light: just the measured Gaussian prior pushed through the *exact* forward
    model and retrieval (no linearisation).
  - Cost: $B$ full retrievals, each rebuilding the dispersion model (thickness is a
    *constructor* parameter), so potentially more expensive per replicate than Method A.
* - Bonus diagnostic: the **FROG error vs thickness** curve reveals whether the data
    itself constrains $L$ (a clear minimum) or not (flat ⇒ the full prior propagates).
  - Higher residuals in the prior tails may be physical misfit or failed optimisation.
    This method retains them rather than truncating the prior; inspect convergence.
```

```{note}
The per-draw trace error $R(L)$ can change with thickness, but can also reflect
convergence and compensation by the fitted field. Inspect it alongside the pulse. If $R(L)$ has a clear minimum inside the prior,
the trace *partially constrains* the thickness but this alone establishes no conservative coverage guarantee;
a combined data-plus-prior treatment (a thickness posterior from $R(L)$) could tighten it.
croak propagates the external measured prior as specified. Joint inference from
data and prior is a separate analysis.

The **data-driven alternative** is to *fit* the thickness as a retrieval parameter
(`fit_thickness=True` on `lbfgs-ad`/`lm`; see [Fitting the medium thickness and
delay-zero](../howto/fitting_thickness_tau0.md)) and read its standard error from the
Method-C covariance (`sigma_thickness`). That replaces the external prior with the trace's
own constraint — appropriate when $R(L)$ has a clear minimum, and the natural complement to
Method D's prior-only propagation. Use one or the other for the thickness, not both.
```

(propagation)=
## Propagating the uncertainty to a different beamline point

The retrieved pulse lives at the **measurement plane** (the FROG apparatus). Often the
quantity of interest is the pulse — and its error bar — at a *different* point in the
beamline: back-propagated through the optics between the FROG and a target plane, or
compressed to a transform-limited point. Propagation between two points is a
**deterministic, linear** spectral-transfer operation,

```{math}
\tilde E'(\omega) = H(\omega)\,\tilde E(\omega),
\qquad
H(\omega) = a(\omega)\,e^{\,i\varphi_\text{prop}(\omega)},
```

a phase multiply (Taylor / material / gas / free-space dispersion) and, optionally, a
mirror-reflectivity amplitude $a(\omega)$ — exactly what
{func}`croak.dispersion.apply_dispersion` and the dispersion stage already apply.
{func}`croak.session.dispersion.transfer_function` resolves a saved
{class}`~croak.session.params.DispersionParams` into $H(\omega)$.

The key point is that this is **not** a matter of naively widening the error bars. The
temporal pulse at the new point depends on the *interplay* of the retrieved spectral
amplitude and phase, and $H(\omega)$ mixes phase across the whole spectrum into the time
domain; the errors are strongly correlated across frequency and between amplitude and
phase. The correct procedure propagates the **full joint distribution**. The bootstrap
already produces an ensemble of complete spectra carrying those correlations, so applying
the *same* $H(\omega)$ to each replicate (or, for Method C, to each posterior draw) and
re-taking the FWHM/band preserves correlations within that ensemble — pass `propagation=H` to any estimator
({func}`~croak.uncertainty.parametric_bootstrap`,
{func}`~croak.uncertainty.resampling_bootstrap`,
{func}`~croak.uncertainty.thickness_bootstrap`,
{func}`~croak.covariance.covariance_uncertainty`).

What this does and does not capture:

- **Pure-phase propagation is unitary**: the spectral intensity and its uncertainty are
  unchanged; only the temporal shape and its uncertainty move. Nothing is lost in the
  operation itself.
- Near a transform-limited target the pulse is short and the FWHM becomes very sensitive to
  residual phase error, so the relative FWHM uncertainty can change substantially; evaluate it from the
  propagated ensemble rather than rescaling a single error bar.
- Removing a mirror divides by its reflectivity $a(\omega)$ (clamped near reflectivity
  dips); back-propagating through deep notches amplifies noise there.
- $H(\omega)$ cannot create energy where the FROG constrained none — frequencies outside
  the retrieved support stay unconstrained.
- The intervening dispersion is taken as **exactly known**: the propagated bar excludes the
  optics' own uncertainty. (A `thickness_bootstrap`-style systematic on the propagation
  optics is the natural extension.)

In the GUI (stage 5) tick *"at dispersion-stage point"* to evaluate the bar at the beamline
point configured in the Dispersion stage.

## The noise model

The "conditional on the noise model" caveat makes this the crux for Method A (and it sets
$\hat\sigma$ in Method C). Three options:

**(a) From the post-fit residual** — estimate noise from
$r=T_\text{meas}-\mu\,T_\text{sim}$ after a good retrieval.

- *Advantages.* Needs **no extra data**; uses the actual noise realisation; can recover
  *signal dependence* by binning the residual variance against the model level
  $\mu T_\text{sim}$ (read-noise intercept + shot-noise slope).
- *Disadvantages.* The residual is **noise + model misfit**, so a wrong forward model
  *can overestimate* noise without necessarily giving conservative coverage. The fit also absorbs
  a little noise, biasing $\hat\sigma^2$ down by approximately $1-p/n$ for the variance in a simple linear fit. Effective rank,
  fitted gains and regularisation complicate this correction in a retrieval.

**(b) From a raw-trace background region** — read-noise $\sigma$ from a signal-free corner
of the *unfiltered* trace.

- *Advantages.* A clean, fit-independent estimate of **detector read noise**.
- *Disadvantages.* Captures **only additive read noise**, not the shot noise on the signal
  where the pulse information lives. Needs the raw trace and a signal-free
  region. And because filtering/regridding **alters the noise statistics**, a $\sigma$
  measured pre-filter does not describe the post-filter trace — estimate noise in the
  domain where it will be injected (the raw-domain caveat of Method A again).

**(c) User-supplied $\sigma$ / camera gain** — the full model $\sigma^2(\text{pixel}) =
g\cdot\text{counts} + \sigma_\text{read}^2$.

- *Advantages.* **Most rigorous** when the camera is characterised; correctly captures the
  signal-dependent **Poisson shot noise** that dominates at high signal.
- *Disadvantages.* Needs calibration (gain $g$, read $\sigma_\text{read}$) and a map from
  the normalised trace back to counts; least convenient.

**Recommendation.** Prefer **(c)** when the camera is characterised; otherwise use **(b)**
for the read-noise floor *combined with* a shot-noise term inferred from the
signal-dependent binning of **(a)**. For fidelity, inject noise in the **raw domain and
push it through the same preprocessing** as the measurement.

## From a distribution to a quoted number

- **Point estimate** — the full-data best-fit FWHM (preferred) or the bootstrap median.
- **Interval** — percentiles of $\{\text{FWHM}_b\}$: 16th–84th for a "±" 1σ-equivalent,
  2.5th/97.5th for 95 %. *State which.* The FWHM distribution is usually skewed (FWHM is
  bounded below and the map is nonlinear), so report the actual interval method. croak offers percentile and bias-corrected
  percentile intervals; its bias correction does not estimate the acceleration
  term of a full BCa interval.
- **Replicate count $B$** — ~200–300 for a 1σ / standard error; ~1000–2000 for stable tail
  quantiles. Method C avoids re-retrieval but field sampling and profile storage still have a cost.
- **Estimator hygiene** — oversample before measuring (as {func}`~croak.processing.process_result`
  does, 8×, via {func}`croak.maths.fwhm`) to suppress jitter in the half-max crossings.
- **Report alongside the number** — the method (A/B/C), the resampling unit (pixel vs
  delay-column block), the warm-start policy, $B$, the fraction of replicates converged
  below the noise floor, the trace error $R$ and the noise floor $R_0$, the SNR, and the
  **"statistical-only"** caveat.

(separating-measurement-from-algorithmic-variance)=
### Separating measurement from algorithmic variance

Each replicate retrieval can stagnate or fall into a different basin, and *that* variance
is not measurement uncertainty. The warm-start policy decides which question you answer:

- **Warm-start every replicate from the full-data solution** → estimates local data-induced movement around the selected basin. *Recommended for the FWHM ±.*
- **Random-restart every replicate** → mixes in basin-hopping → can mix optimisation and data variability; this
  answers a different question ("how reproducible is the whole pipeline, optimiser
  included").

First examine sensitivity of the full-data fit to starts and budget; no finite
multi-start or CMA-ES run proves that it found a global basin. Inspect replicate
residuals and convergence diagnostics. A high residual can indicate stagnation,
model mismatch or an unusual noise realisation; automatic rejection can truncate
the uncertainty distribution. Report any filtering rule and its effect.

## Coverage calibration — turning a spread into a confidence interval

A spread is not a 68 % interval until its **coverage** is verified. The harness is offline
and run once per instrument configuration, on croak's existing synthetic machinery:

1. Generate many **known** pulses spanning the regime of interest (vary FWHM, chirp/GDD,
   and shape — single-peak vs structured), on the experiment's grid, with
   `make_synthetic_experiment`.
2. For each: synthesise its trace, add **realistic** noise at the measured SNR, run the
   **entire** retrieve-then-bootstrap pipeline, and form the nominal 68 % interval.
3. Record whether the interval contains the **true** FWHM; aggregate empirical coverage
   over trials.
4. If the nominal 68 % covers ≈ 68 %, the interval is calibrated; otherwise report the
   empirical coverage or a correction factor.

This step also exposes **bias** (bootstrap-median vs truth — which resampling does *not*
correct) and reveals where the FWHM summary breaks down: multi-peaked pulses give
multimodal bootstrap distributions, and there an RMS / second-moment width (or the full
distribution) is the honest report rather than a single "±".

## Practical recommendation

Choose the uncertainty source first. Use a calibrated detector model for a
parametric study, empirical resampling with justified measurement units for a
cross-check, and a covariance calculation to examine local parameter correlations.
Propagate independently measured calibration distributions separately. Retain the
same physical model and constraints used for the original retrieval.

Check convergence, increase replicate count until the quantities you report are
stable, and test coverage in representative simulations. Disagreement between
estimators is diagnostic but does not uniquely identify its cause: noise,
parameterisation, optimisation and nonlinear statistics can all contribute.

## Verdict on "1.4 ± 0.2 fs"

Report interval endpoints when the distribution is asymmetric. A useful report
states the field plane, estimator, nominal level, replicate count, noise or prior
source, warm-start policy and excluded uncertainties. A symmetric half-width
should not conceal that the interval is displaced from the point estimate.
Repeated measurements complement this analysis but do not remove shared systematic
bias. See the [practical guide](../howto/uncertainty.md) for saving the samples and
interval metadata.

## How it maps onto the code

The [uncertainty API](../reference/api/workflow.md#uncertainty-analysis) documents
the resampling functions, covariance calculation, coverage experiments and
combination of independent contributions. Contributor workflow integration is
covered in the [developer's guide](../developer_guide/gui_sessions.md).

## Next

- [Retrieval as nonlinear least squares](retrieval_theory.md) — the MLE framing the
  Laplace covariance rests on.
- [Validation](validation.md) — how croak's retrievals are checked against known pulses.
- [Post-processing and plotting results](../howto/postprocessing.md) — where the FWHM and
  dispersion coefficients come from.
- [References](../reference/bibliography.md) — the bootstrap and FROG-error literature.
