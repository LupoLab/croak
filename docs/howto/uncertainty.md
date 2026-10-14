# Estimating FWHM uncertainty

A retrieved duration depends on noisy data and uncertain calibration. Choose
which contribution to propagate, then repeat the retrieval or evaluate its local
covariance. The result is conditional on those assumptions, not an automatic
total experimental error bar. See [the theory](../explanation/uncertainty_estimation.md)
and the [thickness example](../examples/thickness_uncertainty.md).

## Quick start

Start from a completed retrieval and its measured trace:

```python
import numpy as np, croak

# ... obtain `td` (TraceData) and `result` (RetrievalResult) as usual ...
u = croak.estimate_fwhm_uncertainty(
    result, td.trace,
    method="parametric",          # Method A: Monte-Carlo over detector noise
    n_resamples=300,
    maxiters=result_maxiters,     # MUST match the original retrieval settings
    rng=np.random.default_rng(0),
)
print(u.summary())
# FWHM = 6.01 ± 0.28 fs (1σ statistical, parametric, B=300, warm-started; …)
```

`u` is an {class}`~croak.uncertainty.UncertaintyResult`: `point_estimate` (the full-data
FWHM, s), `interval_68` / `interval_95` (s), the raw `samples`, and diagnostics
(`n_converged`, `frog_errors`, `noise_floor`).

```{important}
The `**retrieve_kwargs` you pass (``maxiters``, ``material``/``thickness``, ``reg_*``,
``phase_basis`` …) **must reproduce the original retrieval** — each replicate re-runs the
solver with them, warm-started from the solution. Mismatched settings bias the interval.
```

## Choosing a method

```{list-table}
:header-rows: 1
:widths: 22 78

* - `method`
  - When to use
* - `"parametric"`
  - Default. Simulate from the best-fit trace + a noise model, re-retrieve. Needs a
    {class}`~croak.uncertainty.NoiseModel` (or estimates one from the residual).
* - `"delay"`
  - Resample whole delay columns (block bootstrap) — the natural unit for a delay-scanned
    trace. No noise model needed.
* - `"frequency"`
  - Resample frequency rows. No noise model needed.
* - `"thickness"`
  - **Systematic, not statistical.** Propagate a measured substrate-thickness prior
    through a *dispersive* retrieval (see [below](#substrate-thickness-systematic)).
* - `covariance_uncertainty(...)`
  - **Fast analytic.** The linearised Gauss–Newton covariance —
    {func}`~croak.covariance.covariance_uncertainty`. No re-retrieval; the natural
    cross-check on the bootstraps (see [below](#analytic-covariance)).
```

The first three are *statistical* estimators. Running more than one and comparing is the
strongest check: agreement builds confidence; disagreement points at the noise model
(A vs B) or a non-Gaussian posterior. The covariance function is the local analytic
counterpart.

```{note}
The statistical methods drop stagnated replicates by default: any replicate whose
trace error exceeds `convergence_tol` × the median replicate error
(`convergence_tol=3.0`) is discarded, as is any non-finite result. Compare
`u.n_converged` with `u.n_resamples` to see how many were kept. The thickness method
disables this filter (`convergence_tol=1e9`), because there a raised error in the
prior's tails is physical misfit.
```

## The noise model (parametric)

For `method="parametric"` you supply how noisy the trace is. If you pass `measured` but no
`noise`, it is estimated from the post-fit residual ({func}`~croak.uncertainty.noise_from_residual`).
Otherwise:

```python
from croak.uncertainty import NoiseModel, noise_from_background

NoiseModel(sigma=0.01)                 # 1 % of peak, additive
NoiseModel(sigma=0.01, gain=2e-4)      # + signal-dependent (shot) term
noise_from_background(raw_trace, (slice(0, 8), slice(0, 8)))  # read noise from a corner
```

`sigma` as a scalar is a **fraction of the trace peak**; as an array it is the absolute
per-pixel standard deviation. See the explanation page for the trade-offs of each source.

(substrate-thickness-systematic)=
## Substrate-thickness systematic (dispersive retrieval)

A dispersive retrieval propagates the field through a substrate of an *assumed*
thickness (`material`/`thickness`). When that thickness is an independently **measured**
input with its own error bar — e.g. `9.952 ± 0.539` µm — the retrieved FWHM inherits a
*systematic* uncertainty that the statistical bootstraps above cannot see: every one of
their replicates reuses the *same* fixed thickness. For short, deep-UV pulses this term
can dominate: in the [thickness example](../examples/thickness_uncertainty.md) it is about
six times larger than the parametric bootstrap.

{func}`~croak.uncertainty.thickness_bootstrap` (method `"thickness"`) propagates the
measured thickness prior $L\sim\mathcal N(L_0,\sigma_L)$ by brute-force Monte-Carlo: it
draws thicknesses, rebuilds the dispersive forward model at each, **re-retrieves the same
measured trace**, and takes the spread of the per-draw FWHM.

```python
u = croak.thickness_bootstrap(
    result, td.trace,
    central_thickness=9.952e-6,   # measured L0 (m)
    thickness_sigma=0.539e-6,     # measured σ_L (m)
    material="SiO2-Franta",
    npoints=20,                   # reproduce the retrieval's dispersive settings
    maxiters=...,                 # …and the rest of its solver settings
    n_resamples=300, collect_profiles=True,
)
print(u.summary())
# FWHM = 1.42 ± 0.21 fs (1σ from substrate thickness 9.952 ± 0.539 µm, systematic, B=300)

croak.plot_thickness_sensitivity(u)   # FWHM-vs-thickness, FROG-error-vs-thickness, band
```

If the statistical and calibration contributions are independent and approximately
Gaussian, a quadrature approximation is:

```python
from math import hypot
total_pm = hypot(stat.plus_minus, u.plus_minus)   # ± half-widths (s)
```

```{tip}
The `plot_thickness_sensitivity` figure also shows **FROG error vs thickness** — a
consistency check. A flat curve means the data does not constrain the thickness (the full
prior propagates); a clear minimum near $L_0$ means the trace itself prefers a thickness,
partially constraining it. A higher error in the tails is mostly physical misfit, so
this method does not convergence-filter it; check for failed optimisation yourself.
```

A complete, runnable script is in `examples/example_thickness_uncertainty.py`.

(analytic-covariance)=
## Analytic covariance (no re-retrieval)

The current covariance interface supports the core model and reduced smearing,
but does not expose chromatic focal/collection models, depth weighting or
`delay_origin="marginal_peak"`. Do not use a covariance built from a different
model as the uncertainty of the original retrieval. The Jacobian omits
regularisation curvature; penalised fits need additional care. With an explicit
noise model, residual whitening currently replaces the residual weights,
although those weights still enter the fitted gain. See the
{func}`~croak.covariance.parameter_covariance` notes.


{func}`~croak.covariance.covariance_uncertainty` (a separate function, not a dispatcher method) is the fast
analytic alternative: it rebuilds the least-squares residual Jacobian $J$ at the retrieved
spectrum, forms the Gauss–Newton covariance $\hat\sigma^2 (J^\top J)^{+}$, and propagates it
to the FWHM by sampling the Gaussian parameter posterior — one Jacobian and cheap forward
evaluations instead of hundreds of re-retrievals. It returns the same
{class}`~croak.uncertainty.UncertaintyResult` as the bootstraps.

```python
u = croak.covariance_uncertainty(
    result, td.trace,
    noise=0.02,                   # whitens the residual (absolute, like the bootstrap)
    phase_basis="bspline", n_nodes=8,   # reproduce the retrieval's parameterisation
    n_samples=400, collect_profiles=True,
)
print(u.summary())
# FWHM = 36.25 ± 0.24 fs (1σ linearised covariance, B=400 Gaussian draws; …)

cov = croak.parameter_covariance(result, td.trace, noise=0.02,
                                phase_basis="bspline", n_nodes=8)
print(cov.rank, cov.null_dim, cov.reduced_chisq)   # full covariance + diagnostics
```

```{important}
Use it in a **low-dimensional basis** (`phase_basis="bspline"`), where it agrees with the
bootstrap to within a small factor. For a full *pointwise* retrieval the many soft
(poorly constrained) modes make it unstable. It is a first-order (Laplace) approximation —
near the transform limit or at low SNR, examine resampling and coverage as well. The forward-model and
parameterisation keywords must reproduce the original retrieval.
```

(propagate-to-a-beamline-point)=
## Reporting the bar at a different beamline point

Every estimator accepts a `propagation` transfer function $H(\omega)$ that propagates each
replicate to a different point in the beamline before the FWHM/band is taken — exact, because
propagation is a deterministic linear map and the ensemble carries the full
amplitude/phase/frequency correlations. Build $H(\omega)$ from a
{class}`~croak.session.params.DispersionParams` with
{func}`~croak.session.dispersion.transfer_function`:

```python
from croak.session.dispersion import transfer_function
from croak.session.params import DispersionParams

# e.g. back out 1 mm of silica the pulse passed through before the FROG
p = DispersionParams(material_thickness_mm={"SiO2": -1.0})
H = transfer_function(p, result.grid, result.omega0)

u = croak.parametric_bootstrap(result, noise=0.02, propagation=H,
                              propagation_label="−1 mm SiO2", maxiters=...)
print(u.summary())   # FWHM = … ± … fs at −1 mm SiO2 (…)
```

The same `propagation=H` works for {func}`~croak.covariance.covariance_uncertainty` and the
other estimators. The intervening dispersion is assumed exactly known. In the GUI, tick
*"at dispersion-stage point"* to reuse the Dispersion stage's settings as $H$.

## Reading and plotting the result

```python
u.point_estimate, u.interval_68, u.plus_minus   # value (s), (lo, hi) (s), ± half-width (s)
fig = croak.plot_uncertainty(u)                  # FWHM histogram (+ confidence band)
```

For the temporal confidence band, run with `collect_profiles=True`:

```python
u = croak.estimate_fwhm_uncertainty(result, td.trace, n_resamples=300,
                                   collect_profiles=True, maxiters=..., rng=...)
croak.plot_uncertainty(u)   # two panels: FWHM histogram + 68 %/95 % intensity band
```

The `plus_minus` is the half-width of the **68 % interval** (robust to outliers), not the
sample standard deviation — for a skewed distribution report the actual interval endpoints as well as the
point estimate; a symmetric half-width can conceal an offset interval.

## Combining estimates and saving

Independent contributions — *one* statistical estimate and each systematic — combine into
a total with {func}`~croak.uncertainty.combine_uncertainties`, which convolves their sample
distributions (preserving skew; reducing to quadrature
$\sigma_\text{tot}=\sqrt{\sum_i\sigma_i^2}$ in the Gaussian limit) and combines the
temporal bands too:

```python
total = croak.combine_uncertainties([stat, thickness])   # method="combined"
print(total.summary())   # FWHM = … ± … fs (1σ combined: parametric ⊕ thickness, …)
```

```{warning}
Combine only sources that estimate *different* contributions. ``parametric``/``delay``/
``frequency`` all estimate the **same** statistical precision — combining them
double-counts it. Pick one statistical estimate and combine it with the systematics.
```

Save the intervals (and the 68 %/95 % temporal bands) into a result file with
{func}`~croak.save.save_uncertainty`, which appends an ``uncertainty/<method>`` group to an
HDF5 file (creating it, or extending a ``result.h5`` from
{func}`~croak.save.save_result`):

```python
croak.save_uncertainty({"parametric": stat, "thickness": thickness, "combined": total},
                      "result.h5", force=True)
```

In the GUI's **Uncertainty** stage each run is retained in a list; tick the independent
ones, click **Combine ticked**, then **Save…** to write them all into ``result.h5``.

## Calibrating the interval (is it really 68 %?)

A spread is only a *confidence* interval once its coverage is verified.
{func}`~croak.uncertainty.coverage_calibration` runs the whole pipeline on synthetic pulses
with known FWHM and reports how often the nominal interval contains the truth:

```python
cov = croak.uncertainty.coverage_calibration(
    noise=NoiseModel(sigma=0.01), n_trials=50, n_resamples=200,
    level=0.68, method="parametric", rng=np.random.default_rng(0),
)
print(cov.summary())   # coverage 66 % of nominal 68 % (…); median-bias +0.05 fs
```

If the nominal 68 % covers ≈ 68 %, the interval is calibrated for that regime; otherwise
report the empirical coverage. This is an offline check per instrument configuration (many
retrievals — keep `n_trials`/`n_resamples` modest while exploring).

## In the GUI

The wizard's **Uncertainty** stage (after Dispersion) runs the bootstrap in the background
on the current retrieval, using the same solver settings, and plots the result. Pick the
method, the replicate count and (for the parametric method) the noise source, then click
**Estimate**; **Stop** cancels mid-run. For a dispersive retrieval, choose method
**thickness** and set **Thickness σ (µm)** to your measured substrate-thickness
uncertainty (the central value is the retrieval's own thickness) to get the systematic
FWHM error bar.

## Performance

Each replicate is a full warm-started retrieval, so cost ≈ `n_resamples` × one retrieval.
Use a solver supporting the original model and a budget that converges the
replicates; `n_resamples ≈ 200`–`300` suffices for a 1σ interval, more for the
95 % tails.

## See also

- [Uncertainty of the retrieved pulse duration](../explanation/uncertainty_estimation.md) — the theory and caveats.
- [Post-processing and plotting results](postprocessing.md) — where the FWHM comes from.
- [References](../reference/bibliography.md) — the bootstrap and FROG-error literature.
