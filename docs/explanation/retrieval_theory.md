# Retrieval as nonlinear least squares

Pulse retrieval is an inverse problem: find the spectrum $\tilde E(\omega)$ whose
modelled trace best matches the measurement. croak treats this as a **weighted
nonlinear least-squares** fit, which under Gaussian measurement noise is the
maximum-likelihood estimate. Every solver — step-based or quasi-Newton — minimises
the same objective and converges to the same minimum; they differ only in their
trajectory there.

```{image} ../_static/paper/figure_loop.png
:alt: The retrieval loop — a candidate spectrum enters the differentiable forward model, the weighted residual plus regularisation forms the objective, and its AD gradient drives a standard optimiser.
:width: 95%
:align: center
```

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

The cost is not incidental: one free scale per row is precisely the frequency
marginal, so $\mu_\omega$ removes the trace's grip on the retrieved **spectral
amplitude** altogether, leaving it constrained only through each row's delay
structure. That grip is weakest at the edges of the band, and since
$|E(\omega)|^2\ge 0$ the freed bins can only grow. Rω also *lowers* the reported
$R$, because it adds one parameter per row — so a smaller error is not evidence
the spectrum improved. Always pair it with `phase_only` or `reg_spectrum`; see
[Regularisation](../howto/regularisation.md#per-frequency-scaling) for the
measurements, and {func}`croak.processing.edge_energy_fraction` for the diagnostic.

## The trace error R

The convergence metric is the normalised, dimensionless **trace error** $R$ —
the standard "FROG error", generalised to any PNPS trace
({func}`croak.metrics.compute_R`):

```{math}
:label: R
R \;=\; \sqrt{\frac{r}{MN\,\max(T_\text{meas}\,w)^2}}.
```

$M$ and $N$ are the delay and frequency counts. Dividing by the trace size and
peak makes $R$ comparable across trace dimensions and intensity scales. As a rule
of thumb:

| $R$ | quality |
|-----|---------|
| $\lesssim 10^{-3}$ | excellent (clean synthetic data, converged) |
| $10^{-3}\text{–}10^{-2}$ | good experimental retrieval |
| $\gtrsim 10^{-2}$ | poor — suspect the guess, preprocessing or model |

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

Classic FROG algorithms (GPA, PIE, …) iterate *projections* and do not, in
general, converge to the least-squares minimum of {eq}`residual` — they can stall
at a worse fit even on noise-free data. croak's solvers are different:

- {doc}`COPRA <algorithms>` retains a fast projection-style *local* stage but adds
  a **global** gradient-descent stage on the full residual {eq}`residual`, so it
  reaches the least-squares minimum;
- {doc}`L-BFGS, L-BFGS-AD and LM <algorithms>` minimise {eq}`residual` (or its
  square-root, the trace error) directly.

Both COPRA and L-BFGS converge to the *same* minimum; see N. C. Geib *et al.*,
*Optica* **6**, 495 (2019), and the [algorithms page](algorithms.md).

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
