# Retrieval as nonlinear least squares

Pulse retrieval estimates a complex spectrum by matching a predicted intensity
trace to data. This chapter defines the residual, the fitted detector scale and
the role of additional constraints. Read it alongside the
[algorithm comparison](../tutorials/02_choosing_an_algorithm.md) and
[solver selection guide](../howto/solver_selection.md).

Weighted least squares corresponds to a Gaussian likelihood when the residual
weights are inverse noise standard deviations and the noise assumptions hold.
Uniform or heuristic weights need not have that statistical interpretation.
The optimisation is non-convex: different starts and algorithms can reach
different stationary points. Regularisation also changes the objective.

## The objective

Let $T_\text{meas}(\omega,\tau)$ be the (normalised) measured trace and
$T_\text{sim}(\omega,\tau;\tilde E)$ the modelled trace from the
[forward model](forward_model.md). The residual to minimise is

```{math}
:label: residual
r(\tilde E) \;=\; \sum_{\omega,\tau}
   \Bigl[\bigl(T_\text{meas} - \mu\, T_\text{sim}\bigr)\, w_\omega\Bigr]^2,
```

where $w_\omega$ are per-frequency weights (uniform by default) and $\mu$ is a
global intensity **scale factor**. Because retrieval is scale-invariant — the
trace is an intensity and the overall amplitude of $\tilde E$ is not observable —
$\mu$ is solved in closed form at every evaluation:

```{math}
:label: mu
\mu \;=\;
   \frac{\sum_{\omega,\tau} T_\text{meas}\,T_\text{sim}\,w_\omega^2}
        {\sum_{\omega,\tau} T_\text{sim}^2\,w_\omega^2}.
```

This is {func}`croak.metrics.compute_mu`, and {eq}`residual` is
{func}`croak.metrics.compute_r`.

### Per-frequency scaling (`R_omega`)

Every solver offers an adaptive variant in which $\mu$ becomes a
*per-frequency* vector $\mu_\omega$ ({func}`~croak.metrics.compute_mu_per_freq`),
enabled with `R_omega=True`. This absorbs a frequency-dependent detection
efficiency that a single global scale cannot. It is most useful on experimental
data whose response is uncalibrated.

One gain per frequency removes the absolute amplitude information in the trace's
frequency marginal. The remaining delay structure still constrains the field,
but its spectral amplitude can become poorly determined, especially near the
band edges. A lower residual is expected with more fitted gains and does not
establish a better spectrum. Use an appropriate spectral constraint and inspect
[edge energy](../howto/postprocessing.md) and the
[regularisation diagnostics](../howto/regularisation.md#per-frequency-scaling).

For general residual weights $w_{nm}$, a row gain is

```{math}
\mu_n=\frac{\sum_m w_{nm}^2 M_{nm}T_{nm}}
                 {\sum_m w_{nm}^2 T_{nm}^2}.
```

Only weights constant along a row cancel from this ratio. The standard solver
interface takes per-frequency weights; resampling and masked objectives may
need more general weights. A residual weight of $1/\sigma$ corresponds to an
inverse variance of $1/\sigma^2$ in the squared loss.

### Delay origin and gain-dependent centring

The ordinary field time-origin ambiguity is separate from calibrating the
relative delay of the replicas. With `delay_origin="marginal_peak"`, croak
recentres the model using its response-weighted delay marginal. The gains and
centre depend on the trial field, so their derivatives belong to the objective;
{mod}`croak.delay_origin` includes those dependencies. This option requires
uniformly spaced delays. The default `"coincidence"` retains the model's replica
coincidence origin. A free `fit_tau0` is a different operation: it adds a fitted
instrument parameter. See [fitting model parameters](../howto/fitting_thickness_tau0.md).

## The trace error R

The convergence metric is the normalised, dimensionless **trace error** $R$ —
the standard "FROG error", generalised to any PNPS trace
({func}`croak.metrics.compute_R`):

```{math}
:label: R
R \;=\; \sqrt{\frac{r}{MN\,\max(T_\text{meas}\,w)^2}}.
```

$M$ and $N$ are the delay and frequency counts. The normalisation removes an
overall intensity scale, but the reported value still depends on the retained
rectangle, weights, preprocessing and fitted gains. Including empty pixels can
lower an rms/peak error. Compare values only with these choices stated.

There is no universal boundary between a good and a poor retrieval. Compare the
residual with noise and calibration evidence, inspect its structure, and test
model and sampling sensitivity. A low $R$ can accompany a substantially wrong
field, as the [validation examples](validation.md) demonstrate.

```{admonition} R, and the "G-error"
:class: note
The FROG literature usually writes this quantity $G$ (the "G-error"). croak calls
it **$R$** everywhere — in the equations above, in {func}`croak.metrics.compute_R`,
and as `result.error` / `result.errors` on every
{class}`~croak.result.RetrievalResult`. They are the same normalised RMS trace
error; only the symbol differs.
```

{func}`croak.metrics.frog_error` (aliased {func}`croak.metrics.trace_error`) is the
convenience that chains {eq}`mu`, {eq}`residual` and {eq}`R` for a measured and a
simulated trace.

## Why least squares (and not just projection)?

COPRA combines a projection-style local stage with a gradient stage on the
full trace residual. Gradient solvers instead optimise a scalar loss directly;
Levenberg–Marquardt operates on a vector of residuals. These routes can improve
least-squares agreement, but none guarantees a global minimum.

Without regularisation, $R$ and $R^2$ have the same minimisers. With penalties,
the distinction matters. Gradient solvers use

```{math}
J_{\rm grad}=R+\sum_j\lambda_j P_j,
\qquad
J_{\rm LM}=R^2+\sum_j\lambda'_j P_j.
```

LM appends scaled penalty residuals to its residual vector. For a given nonzero
$R$, matching stationary conditions suggests $\lambda'_j=2R\lambda_j$ locally.
This is not a fixed conversion valid along an entire optimisation. croak's
[default weights](../howto/regularisation.md) differ between these solver families;
use matched constraints when comparing algorithms, and record the actual values.

## Gradients

Descending {eq}`residual` needs the gradient $\partial r/\partial\tilde E^*$.
croak has two routes to it:

- **JAX automatic differentiation** of the forward model (used by
  {class}`~croak.lbfgs_ad.LBFGSAD`, {class}`~croak.optimistix_lbfgs.OptxLBFGS` and
  the Jacobians of {class}`~croak.lm.LM` / {class}`~croak.optimistix_lm.OptxLM`).
  This is the general route: it extends to model features the closed form does not
  cover, and to fitted parameters beyond the field itself;
- a **hand-derived Wirtinger adjoint** through the same map (used by
  {class}`~croak.copra.COPRA` and {class}`~croak.lbfgs.LBFGS`) — exact, costing one
  extra forward pass, and free of any JAX dependency on the hot path.

Where both apply they agree to numerical precision, which is how each is held
against the other. The [Gradients](gradients.md) page derives the adjoint, sets
out what each route covers, and analyses the cost.

## Amplitude-and-phase vs phase-only retrieval

By default croak retrieves the full complex spectrum — amplitude *and* phase. When
the spectral amplitude is independently and reliably measured, you can fix it and
retrieve only the phase (`phase_only=True`), which halves the unknowns and
stabilises the fit on difficult data. Phase-only retrieval pairs naturally with
the phase smoothness penalty; see [Regularisation](../howto/regularisation.md).

## Next

- [Retrieval algorithms](algorithms.md) — how COPRA, L-BFGS, L-BFGS-AD and LM work.
- [Solver selection](../howto/solver_selection.md) — which to use when.
- [Gradients](gradients.md) — automatic differentiation and the Wirtinger adjoint.
