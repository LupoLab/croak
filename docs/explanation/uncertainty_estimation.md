# Uncertainty of the retrieved pulse duration

From a single TG-FROG measurement croak retrieves a complex field $\tilde E(\omega)$, and
from it the FWHM of the temporal intensity $|E(t)|^2$. A natural question follows: *what
is the error bar on that number?* Can we honestly write **"1.4 ± 0.2 fs"** for the FWHM,
and is there a rigorous statistical procedure — a bootstrap — that earns the "± 0.2 fs"?

This page analyses what is and is not possible from a **single** trace. The short answer
is *yes, a statistically meaningful random-error bar is obtainable* — and the FWHM is an
unusually well-behaved quantity to bootstrap — *but* the number it produces is a
**lower bound** on the true uncertainty, and quoting it responsibly takes some care.

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

## The short answer, and its four conditions

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
included)"**. The gold standard for *total* uncertainty is still repeated independent
measurements; single-trace resampling yields the statistical component alone.

## Why this is even possible — the statistical structure of FROG

**The problem is heavily overdetermined.** A trace carries $N_\omega \times N_\tau$ points (typically
$10^3$–$10^5$) constraining $\sim 2N$ unknowns for a pointwise field, or only $\sim 10$–$30$
for a dispersive/B-spline phase. The data vastly outnumber the unknowns, noise on
individual pixels averages down, and *that redundancy is what makes resampling
informative* — dropping or perturbing a fraction of the data still leaves the solution
well-determined, and how far it then moves measures the information content.

**FWHM is gauge-invariant.** FROG has trivial ambiguities the
trace cannot fix. For SHG-FROG these are the absolute time origin, the carrier–envelope
phase (CEP), *and* the direction of time $E(t)\leftrightarrow E(-t)$. **TG-FROG — the PG
kernel $E(t)\,|E(t-\tau)|^2$, `interaction="pg"` in croak — breaks the time-reversal
ambiguity**, a genuine advantage over SHG. The remaining freedoms (translation, CEP) do
not change $|E(t)|^2$, and even a time flip preserves the FWHM. So the FWHM of the
intensity envelope is a **well-defined scalar statistic with no gauge alignment
required** — unlike per-time or per-frequency *field* error bands, which would need each
replicate aligned (remove linear spectral phase → time shift; remove constant phase →
CEP; for SHG also resolve the flip) before they could be averaged. This is precisely what
makes the FWHM a clean target for resampling.

**No closed-form propagation in general.** Retrieval is a nonlinear inverse map; there is
no exact analytic propagation from trace noise to FWHM. Resampling / Monte-Carlo is the
practical route. Only for a *low-dimensional* parameterisation does a local linearisation
(the [Laplace / Cramér–Rao route](#method-c))
become available.

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

**Consequence:** the resampling interval is a *lower bound* on the total uncertainty and
must be labelled "statistical." A credible duration quote pairs it with a separate,
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
  - **Only as good as the noise model** — garbage in, garbage out.
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

This is the classic FROG bootstrap introduced by Wang, Zeek, Trebino & Kvam (2003):
resample the *measured* data and re-retrieve, with **no explicit noise model**. Both forms
below ride on the per-pixel `weights` argument every croak solver already accepts (built by
{func}`croak.solver.build_weights` and honoured by the COPRA / LBFGS / LM residuals):

- **Leave-out (jackknife-like).** Zero the weight of a random fraction $f$ of pixels each
  replicate; the retrieval simply ignores them. The spread over draws is the error
  estimate.
- **Multinomial / Poisson weights (true nonparametric bootstrap).** Draw integer pixel
  multiplicities $w_i \sim \text{Multinomial}(N, 1/N)$ (or $\text{Poisson}(1)$) and run
  the *weighted* retrieval — exactly Efron's resample-with-replacement, expressed through
  the weight vector.

**Interpretation.** A nonparametric bootstrap over measurement units: it estimates how
sensitive the solution is to *which* data you happen to have, assumes nothing about the
noise distribution, and does **not** assume the best fit is truth.

```{list-table}
:header-rows: 1
:widths: 50 50

* - Advantages
  - Disadvantages
* - **No noise model required** — assumption-light; a strong cross-check on Method A.
  - **Pixels are not i.i.d.**: neighbours are correlated through the FFT structure and the
    marginals, so the reported variance is only *approximately* the noise-induced one.
* - Reuses the existing `weights` argument; no new forward/noise code.
  - The **resampling unit is ambiguous** (see block-bootstrap note below).
* - Robust; does not condition on the fitted model.
  - Does not model **signal-dependent** noise (background and peak pixels resampled alike).
* -
  - The leave-out fraction $f$ is a free knob; results drift with it.
```

```{tip}
For a **delay-scanned** TG-FROG, each delay column is one laser exposure, so shot-to-shot
energy/pointing fluctuations are correlated *within a column*. A **block bootstrap over
whole delay columns** (resample columns, not pixels) respects that correlation and is more
appropriate for scanned data; per-pixel resampling underestimates it.
```

(method-c)=
## Method C — Laplace / Jacobian covariance (for dispersive retrieval)

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
points and $p=\dim\theta$. For Gaussian noise this is the inverse **Fisher information** —
the **Cramér–Rao bound**, the best precision any unbiased estimator of $\theta$ can
achieve. croak *already computes $J$*: {class}`~croak.lm.LM` (and
{class}`~croak.optimistix_lm.OptxLM`) build `jax.jacfwd` of
{func}`croak.metrics_jax.residual_vector`. {func}`~croak.covariance.parameter_covariance`
rebuilds that residual at the retrieved spectrum and forms the covariance; because it is
reconstructed from the spectrum and the JAX forward model it works for *any* result, not
only an LM one — the least-squares algorithm only decides whether $J$ is exposed *natively*.

**Whitening and scale.** With the residual divided pixel-by-pixel by a
{class}`~croak.uncertainty.NoiseModel` σ, the scale is absolute, $\hat\sigma^2=1$ and
$\operatorname{Cov}=(J^\top J)^{+}$ — consistent with the parametric bootstrap, which
treats the same noise model as truth. Without a noise model croak falls back to the internal
reduced-χ² estimate {eq}`laplace`.

**Gauge degeneracy.** FROG's continuous trivial ambiguities — the absolute phase (CEP) and
the absolute timing (a linear spectral phase) — are *flat directions* of the objective, so
$J^\top J$ is **rank-deficient**. croak uses the pseudo-inverse (eigenvalues below `rcond` ×
the largest are dropped); the null space carries no variance for any **gauge-invariant**
quantity (FWHM, $|\tilde E(\omega)|^2$, the temporal intensity), whose gradient is
orthogonal to it. {attr}`CovarianceResult.null_dim <croak.covariance.CovarianceResult>`
reports its dimension.

**Propagate to FWHM by posterior sampling.** croak draws $\theta^{(k)}\sim\mathcal
N(\hat\theta,\operatorname{Cov})$ and evaluates $\text{FWHM}(\theta^{(k)})$ for each — *no
re-retrieval*, just forward field evaluations through the same B-spline → field → FWHM path
(the gauge null modes carry zero variance, so the draws never jitter the CEP or timing).
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
* - **Near-instant** — one Jacobian + cheap sampling, no re-retrieval; ideal for
    interactive use and as the analytic cross-check on A and B.
  - **Local (Laplace) approximation** — under-represents skew/non-Gaussianity for
    multimodal/skewed posteriors (low SNR, near an ambiguity, near the transform limit).
    There the bootstrap is the reference.
* - Rigorous for the parameterised model under the local-Gaussian assumption — exactly the
    NLLS/MLE regime of [Retrieval theory](retrieval_theory.md).
  - **Inherits the parameterisation's model bias**: it answers "how well is *this model*
    determined," not "how well is the true pulse known."
* - Gives the **full parameter covariance** at once — error bars on every derived quantity
    (spectral phase/intensity, GDD/TOD/FOD, the propagated pulse) fall out of one matrix.
  - $J^\top J$ can be **ill-conditioned**; the P-spline penalty (`reg_phase`) regularises
    it but then biases $\hat\theta$ — an explicit bias/variance trade-off.
* - Best in a **low-dimensional basis** (`phase_basis="bspline"`), where it agrees with the
    bootstrap to within a small factor.
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
hundred nm of silica is significant, and this term routinely **dominates** the
statistical one of Methods A/B.

**Procedure** ({func}`~croak.uncertainty.thickness_bootstrap`). A brute-force Monte-Carlo
over the *measured prior*: for $b=1\dots B$ draw $L_b\sim\mathcal N(L_0,\sigma_L)$ (clipped
at 0), rebuild the dispersive forward model at $L_b$, **re-retrieve the same measured
trace** (warm-started from the $L_0$ solution), and take the spread of
$\{\text{FWHM}_b\}$. It is the systematic complement to Method A: A perturbs the *data*
and holds the model fixed; D perturbs the *model* and holds the data fixed. The two are
independent and add in quadrature, $\sigma_\text{tot}^2=\sigma_\text{stat}^2+\sigma_L\text{-induced}^2$.

```{list-table}
:header-rows: 1
:widths: 50 50

* - Advantages
  - Disadvantages
* - Promotes the usually-*dominant* systematic into a defensible, quotable number from an
    existing measurement (the substrate metrology).
  - Covers **only** the thickness systematic — calibration, geometry and model mismatch
    remain outside the budget.
* - Assumption-light: just the measured Gaussian prior pushed through the *exact* forward
    model and retrieval (no linearisation).
  - Cost: $B$ full retrievals, each rebuilding the dispersion model (thickness is a
    *constructor* parameter), so dearer per replicate than Method A.
* - Bonus diagnostic: the **FROG error vs thickness** curve reveals whether the data
    itself constrains $L$ (a clear minimum) or not (flat ⇒ the full prior propagates).
  - A raised error in the prior's tails is *physical misfit*, not stagnation — so the
    usual convergence filter must be **disabled** (it would clip the distribution).
```

```{note}
Because an off-nominal thickness genuinely fits the trace worse, the per-draw FROG error
$R(L)$ is informative in its own right. If $R(L)$ has a clear minimum inside the prior,
the trace *partially constrains* the thickness and the propagated error is conservative;
a combined data-plus-prior treatment (a thickness posterior from $R(L)$) could tighten it.
croak propagates the external measured prior as-is — the honest, conservative choice when
the thickness was measured independently.

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
re-taking the FWHM/band is exact — pass `propagation=H` to any estimator
({func}`~croak.uncertainty.parametric_bootstrap`,
{func}`~croak.uncertainty.resampling_bootstrap`,
{func}`~croak.uncertainty.thickness_bootstrap`,
{func}`~croak.covariance.covariance_uncertainty`).

What this does and does not capture:

- **Pure-phase propagation is unitary**: the spectral intensity and its uncertainty are
  unchanged; only the temporal shape and its uncertainty move. Nothing is lost in the
  operation itself.
- Near a transform-limited target the pulse is short and the FWHM becomes very sensitive to
  residual phase error, so the *relative* FWHM uncertainty **grows** — real physics the
  ensemble captures correctly.
- Removing a mirror divides by its reflectivity $a(\omega)$ (clamped near reflectivity
  dips); back-propagating through deep notches genuinely amplifies noise there.
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
  *overestimates* noise (conservative — not the worst failure mode). The fit also absorbs
  a little noise, biasing $\hat\sigma^2$ down by $\sqrt{1-p/n}$ — negligible when $p\ll n$
  (pointwise) but not for a tiny dispersive $p$, where the $n-p$ denominator in
  {eq}`laplace` matters.

**(b) From a raw-trace background region** — read-noise $\sigma$ from a signal-free corner
of the *unfiltered* trace.

- *Advantages.* A clean, fit-independent estimate of **detector read noise**.
- *Disadvantages.* Captures **only additive read noise**, not the shot noise on the signal
  where the pulse information lives. Needs the raw trace and a genuinely signal-free
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
  bounded below and the map is nonlinear), so prefer **BCa** (bias-corrected and
  accelerated) intervals over naive percentiles.
- **Replicate count $B$** — ~200–300 for a 1σ / standard error; ~1000–2000 for stable tail
  quantiles. Method C needs no retrievals, so $B$ can be $10^4$ for free.
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

- **Warm-start every replicate from the full-data solution** → isolates the data-induced
  movement → a clean **precision** estimate. *Recommended for the FWHM ±.*
- **Random-restart every replicate** → mixes in basin-hopping → *over*-estimates; this
  answers a different question ("how reproducible is the whole pipeline, optimiser
  included").

Either way, first confirm the full-data solution sits in the global basin (a
{class}`~croak.cmaes.CMAES` or multi-start check, once), and **discard or flag** replicates
that fail to reach $R\approx R_0$. By Geib et al. (2019) a correctly converged retrieval
bottoms out at the **noise floor** $R_0 = $ {func}`frog_error <croak.metrics.frog_error>`
of the noisy trace against the clean one; replicates above it are stagnation, not signal.

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

- **Default rigorous answer** — Method A (parametric MC bootstrap), noise from option (c),
  or (b)+(a); warm-started; $B \sim 300$–$1000$; BCa percentiles; validated by the
  coverage harness.
- **Assumption-light cross-check** — Method B (weighted bootstrap, *block over delay
  columns* for scanned data); needs no noise model.
- **Instant cross-check & for dispersive fits** — Method C (Laplace via the
  {class}`~croak.lm.LM` Jacobian), giving FWHM *and* GDD/TOD/FOD error bars in milliseconds.
- **Triangulate.** Agreement of A, B and C is strong evidence. Disagreement is diagnostic:
  A vs B → the noise model is wrong; C vs A/B → the posterior is non-Gaussian/multimodal or
  the parameterisation is too stiff.
- **COPRA vs LBFGS for replicates.** Both warm-start cleanly. LBFGS/LM are smoother and
  cheaper for low-dimensional dispersive fits (and LM uniquely gives Method C for free);
  COPRA is robust for full pointwise retrieval. The JAX twins
  ({class}`~croak.copra_jax.COPRAJax`, {class}`~croak.lbfgs_ad.LBFGSAD`) batch replicates for
  throughput.

## Verdict on "1.4 ± 0.2 fs"

**Quotable — with discipline.** The "± 0.2 fs" is a legitimate *statistical* (precision)
interval if it is coverage-calibrated (or honestly labelled as bootstrap percentiles),
the warm-start policy is documented, the noise model is justified, and a *separate*
systematic budget is given (or systematics are explicitly excluded). Because those
systematics — delay/wavelength calibration, spectral response, geometry, model mismatch —
frequently exceed the statistical part, the honest full statement is typically:

> **FWHM = 1.4 ± 0.2 fs** (1σ statistical, parametric bootstrap, $B=500$, warm-started),
> plus systematic uncertainty from calibration not captured by resampling.

A single trace can never deliver the *total* uncertainty that repeated independent
measurements would; resampling delivers the statistical lower bound on it — which is
exactly, and only, what a careful "±" should claim.

## How it maps onto the code

{mod}`croak.uncertainty` implements Methods A and B on top of existing pieces:

- {attr}`RetrievalResult.trace <croak.result.RetrievalResult.trace>` and its `mu` — the
  clean model trace simulated from, for Method A.
- delay-column / frequency-row resampling of the measured trace (the latter via the solver
  `weights` argument) — Method B; no per-pixel weights are needed.
- {func}`croak.maths.fwhm` and the oversampling of
  {func}`~croak.processing.process_result` — the gauge-invariant statistic.
- warm-start `guess=` on every solver — to isolate measurement variance.
- {func}`~croak.uncertainty.synthetic_experiment` +
  {func}`~croak.uncertainty.coverage_calibration` — ground truth and coverage validation.
- {func}`~croak.uncertainty.thickness_bootstrap` — Method D: the per-draw thickness is
  injected into the dispersive {class}`~croak.forward.ForwardModel` constructor (rebuilt
  each draw), the same trace re-retrieved, and the spread reported via the shared
  assembly/interval machinery; {func}`~croak.plotting.plot_thickness_sensitivity` adds the
  FWHM- and error-vs-thickness diagnostics.

Method C (the Laplace covariance) reuses the same `jax.jacfwd` of
{func}`croak.metrics_jax.residual_vector` as {class}`~croak.lm.LM`, rebuilt at the retrieved
spectrum: {func}`~croak.covariance.parameter_covariance` forms
$\hat\sigma^2(J^\top J)^{+}$ (gauge null space dropped by the pseudo-inverse), and
{func}`~croak.covariance.covariance_uncertainty` propagates it to the FWHM/band by posterior
sampling, returning the shared {class}`~croak.uncertainty.UncertaintyResult`. Any estimator
takes a `propagation=H` transfer function to report at a different beamline point. See the
how-to, [Estimating FWHM uncertainty](../howto/uncertainty.md), for usage.

## Next

- [Retrieval as nonlinear least squares](retrieval_theory.md) — the MLE framing the
  Laplace covariance rests on.
- [Validation](validation.md) — how croak's retrievals are checked against known pulses.
- [Post-processing and plotting results](../howto/postprocessing.md) — where the FWHM and
  dispersion coefficients come from.
- [References](../reference/bibliography.md) — the bootstrap and FROG-error literature.
