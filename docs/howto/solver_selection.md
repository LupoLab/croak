# Choosing a solver

croak has ten solvers in four families: COPRA, L-BFGS, Levenberg–Marquardt and
the global CMA-ES, several with on-device JAX or Optimistix twins. Choose first by
the model and constraints a solver supports, then by its speed and final
accuracy on your data. The local solvers minimise the same
[least-squares objective](../explanation/retrieval_theory.md) but differ in
penalty scaling and stopping rules. This page is the short version; the
[algorithm tutorial](../tutorials/02_choosing_an_algorithm.md) runs an executed
comparison.

croak's default, `warm-lbfgs` with `reg_spectrum=0.01` and `reg_amp=0.03`, is a
general-purpose starting point. The validation retrievals in the
[papers](../reference/bibliography.md) instead use `lbfgs-ad` from a
transform-limited start for 400 iterations with `reg_spectrum=0.03` and
`reg_amp=0.03`; use those settings to reproduce them.

## At a glance

Start here; these six cover almost every case:

| Solver | Gradient | Best for | Notes |
|--------|----------|----------|-------|
| {class}`~croak.warm_lbfgs.WarmLBFGS` | JAX autodiff | the **default**; when you do not want to think about the initial guess | `lbfgs-ad` seeded from a COPRA local sweep: same model, far less sensitive to where it starts |
| {class}`~croak.copra.COPRA` | analytic adjoint | full amplitude + phase in the core model | fast initial descent, robust from poor guesses |
| {class}`~croak.lbfgs.LBFGS` | analytic adjoint | phase-only; custom **regularisation** | steady descent; analytic gradients for the core model |
| {class}`~croak.lbfgs_ad.LBFGSAD` | JAX autodiff | anything the core model does not cover | the solver with the **widest model coverage**: smearing, fitted thickness/τ₀/smearing, B-spline basis, temporal penalty |
| {class}`~croak.lm.LM` | JAX Jacobian | **polishing** to the lowest error | superlinear; use after COPRA/L-BFGS; its penalty weights are scaled differently from L-BFGS |
| {class}`~croak.cmaes.CMAES` | none (population search) | escaping a poor local minimum with no good guess | population search; pair with `phase_basis="bspline"`; needs `croak[evo]` |

Reach for `lbfgs-ad` (or `lm`/`lm-optx`) whenever you need something the analytic
adjoint does not carry — see [Gradients](../explanation/gradients.md) for exactly
where the boundary lies.

The remaining four are on-device reimplementations of the same algorithms:

| Solver | Twin of | Why reach for it |
|--------|---------|------------------|
| {class}`~croak.copra_jax.COPRAJax` | `copra` | same COPRA, on-device JAX numerics |
| {class}`~croak.lbfgs_hand.LBFGSHand` | `lbfgs` | analytic gradient, jitted/vectorised in JAX (no AD); typically the fastest L-BFGS |
| {class}`~croak.optimistix_lbfgs.OptxLBFGS` | `lbfgs-ad` | the L-BFGS loop runs entirely on-device |
| {class}`~croak.optimistix_lm.OptxLM` | `lm` | an on-device JAX-Jacobian LM |

## By task

- **General amplitude-and-phase retrieval, clean to moderately noisy data** →
  `copra`. It converges fast and tolerates a rough initial guess.
- **Phase-only retrieval** (amplitude fixed to a measured spectrum) → `lbfgs`
  with `phase_only=True`, optionally `reg_phase`.
- **Noisy experimental data** → `lbfgs` with mild
  [smoothness regularisation](regularisation.md) (`reg_amp`/`reg_phase`).
- **Uncalibrated spectral response** → `R_omega=True` fits a gain for each
  frequency row, but never on its own: add `phase_only=True` or `reg_spectrum`
  with a measured spectrum, or the retrieved spectrum grows spurious energy at
  the band edges while $R$ falls. See
  [per-frequency scaling](regularisation.md#per-frequency-scaling).
- **Squeezing out the last factor in $R$** → run `copra` or `lbfgs`, then a short
  `lm` polish (20–50 iterations).
- **Stuck in a poor local minimum / no usable initial guess** → a `cma-es` global
  search, usually with `phase_basis="bspline"` (≈20–30 spline nodes), then refine
  the result with `lbfgs-ad`. See [global retrieval](../explanation/algorithms.md#global-retrieval-cma-es).
- **Spurious satellite pulses or a temporal pedestal** → `lbfgs-ad` (or `cma-es`)
  with the temporal penalty `reg_time`/`time_window` (see
  [Regularisation](regularisation.md#temporal-regularisation)).
- **Keeping the loop on a JAX device** → the twins `copra-jax`, `lbfgs-hand`,
  `lbfgs-optx` and `lm-optx`. Their options and stopping rules follow their own
  optimisers; time the first (compiling) call separately from repeated runs.
- **Checking a hand-derived gradient or a new interaction** → compare the
  derivative with finite differences and autodiff on complex test fields; see
  [numerical validation](../developer_guide/validation.md).
- **`lm-optx` iterations feel slow on a large grid** → it already defaults to the
  faster normal-equations solve (`linear_solver="normal"`); the remaining cost is
  the Jacobian and linear solve, which grow with the parameter count, so
  `phase_basis="bspline"` is the big lever. Fall back to `linear_solver="qr"` only
  if the solve misbehaves (see
  [Retrieval algorithms](../explanation/algorithms.md)).

## Run from several random restarts

The objective is non-convex, so a single run can land in a poor local minimum.
The standard practice is to retrieve from **5–10 random initial guesses** and
keep the result with the lowest `error`:

```python
import numpy as np, croak

rng = np.random.default_rng(0)
best = None
for _ in range(8):
    res = croak.retrieve(trace, g.omega, delays, "pg",
                        algorithm="copra", guess=None, rng=rng, maxiters=300)
    if best is None or res.error < best.error:
        best = res
print(best.error)
```

`guess=None` draws a random Gaussian-spectrum start. Reusing one seeded
generator draws a different, reproducible start for each run; resetting it to
the same seed every time would repeat the same start.

For a penalised fit, `error` is only the trace term, so also compare the
penalised objective and the plausibility of the pulse. Start-to-start variation
is a convergence diagnostic, not a measurement uncertainty.

## A two-stage recipe

For the lowest final error, converge cheaply and then polish:

```python
# 1. converge with COPRA (or LBFGS)
res = croak.retrieve(trace, g.omega, delays, "pg", algorithm="copra", maxiters=300)

# 2. polish with Levenberg–Marquardt, seeded from the COPRA spectrum
res = croak.retrieve(trace, g.omega, delays, "pg", algorithm="lm",
                    guess=res.spectrum, maxiters=50)
```

## Dispersive problems

Solver choice is independent of whether you model a slab: pass
`material`/`thickness`/`npoints`/`omega0` to *any* solver to retrieve through a
dispersive medium (PG/SD only — see [Interactions](../explanation/interactions.md)).
"D-COPRA" in the literature is exactly `COPRA` with a `material` set.

Geometric smearing is the exception: pass `smearing` to one of the **autodiff**
solvers (`warm-lbfgs`, `lbfgs-ad`, `lm`, `lm-optx`, `lbfgs-optx`, `cma-es`) only.
The incoherent sum over quadrature nodes breaks COPRA's magnitude-replacement
projection and is not covered by the analytic adjoints, so those solvers raise
instead of silently ignoring the kernel. See [Geometric smearing](geometric_smearing.md).

## Convergence and stopping

- **COPRA**: `maxiters`, plus optional `reltol` (stop on a plateau of the best
  error) and `abstol` (stop at a target error). By default it runs all `maxiters`.
- **LBFGS / LBFGSAD / LM**: `maxiters` caps the evaluations; `reltol`/`abstol` are
  the optimiser's function/step tolerances (defaults `1e-4`/`1e-8`). For
  {class}`~croak.lm.LM` these become SciPy's `ftol` and `xtol`/`gtol`.

```{tip}
**Polishing with LM: tighten `abstol`, not `reltol`.** For the SciPy route,
`abstol` sets both the step (`xtol`) and gradient (`gtol`) tolerances, and its
`1e-8` default is a step-size floor: it is what stops an LM polish, usually while
the error is still falling steeply. On the clean PG trace of the
[algorithm tutorial](../tutorials/02_choosing_an_algorithm.md#why-lm-stops-early),
tightening it to `1e-12` lowers the final error by about three orders of
magnitude for about 1.4× the evaluations, while tightening `reltol` changes
nothing.
```

By default {func}`croak.retrieve` shows a **live progress bar** and prints a short
end-of-run summary (`progress="auto"`); pass `progress=False` to silence it or
`progress=True` to force it (e.g. when redirecting output). For programmatic
monitoring, supply your own `callback(iteration, R, best_R)` (which replaces the
bar), or inspect `res.errors` after the fact with
{func}`croak.plotting.plot_convergence`. See {mod}`croak.progress`.

## See also

- [Retrieval algorithms](../explanation/algorithms.md) — how each one works.
- [Choosing an algorithm](../tutorials/02_choosing_an_algorithm.md) — an executed comparison.
- [Regularisation](regularisation.md) — stabilising L-BFGS/LM on noisy data.
