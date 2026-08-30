# Choosing a solver

croak has ten solvers ([algorithms](../explanation/algorithms.md)), but they are
really four algorithm families — COPRA, L-BFGS, Levenberg–Marquardt and the global
CMA-ES — with several numerically-equivalent JAX/Optimistix twins. The local
solvers reach the same [least-squares minimum](../explanation/retrieval_theory.md)
but differ in speed, flexibility and final accuracy. This guide is the short
version: which to use, and how to run them robustly.

## At a glance

Start here — these six cover almost every case:

| Solver | Gradient | Best for | Notes |
|--------|----------|----------|-------|
| {class}`~croak.warm_lbfgs.WarmLBFGS` | JAX autodiff | the **default**; when you do not want to think about the initial guess | `lbfgs-ad` seeded from a COPRA local sweep: same model and minimum, far less sensitive to where it starts |
| {class}`~croak.copra.COPRA` | analytic adjoint | full amplitude + phase, on its own | fast initial descent, robust from poor guesses |
| {class}`~croak.lbfgs.LBFGS` | analytic adjoint | phase-only; custom **regularisation** | steady descent; most flexible objective |
| {class}`~croak.lbfgs_ad.LBFGSAD` | JAX autodiff | anything the core model does not cover | the solver with the **widest model coverage**: smearing, fitted thickness/τ₀/smearing, B-spline basis, temporal penalty |
| {class}`~croak.lm.LM` | JAX Jacobian | **polishing** to lowest error | superlinear; use after COPRA/L-BFGS |
| {class}`~croak.cmaes.CMAES` | none (global) | escaping bad local minima with no good guess | population search; pair with `phase_basis="bspline"`; needs `croak[evo]` |

Reach for `lbfgs-ad` (or `lm`/`lm-optx`) whenever you need something the analytic
adjoint does not carry — see [Gradients](../explanation/gradients.md) for exactly
where the boundary lies. Running it against `lbfgs` on the same trace is also how
the two gradient routes are cross-checked.

The remaining four are on-device reimplementations — same results, different
execution:

| Solver | Twin of | Why reach for it |
|--------|---------|------------------|
| {class}`~croak.copra_jax.COPRAJax` | `copra` | same COPRA, on-device JAX numerics |
| {class}`~croak.lbfgs_hand.LBFGSHand` | `lbfgs` | analytic gradient, jitted/vectorised in JAX (no AD) — typically the fastest L-BFGS |
| {class}`~croak.optimistix_lbfgs.OptxLBFGS` | `lbfgs-ad` | the L-BFGS loop runs entirely on-device |
| {class}`~croak.optimistix_lm.OptxLM` | `lm` | a genuine on-device JAX-Jacobian LM |

## By task

- **General amplitude-and-phase retrieval, clean-to-moderate data** → `copra`. It
  converges fast and tolerates a rough initial guess.
- **Phase-only retrieval** (amplitude fixed to a measured spectrum) → `lbfgs`
  with `phase_only=True`, optionally `reg_phase`. It empirically beats COPRA here.
- **Noisy experimental data** → `lbfgs` with mild
  [smoothness regularisation](regularisation.md) (`reg_amp`/`reg_phase`).
- **Uncalibrated spectral response** → `R_omega=True`, but **never on its own**:
  add `phase_only=True` or a light `reg_spectrum`, or the retrieved spectrum grows
  spurious energy at the band edges while $R$ *falls*. See
  [per-frequency scaling](regularisation.md#per-frequency-scaling).
- **Squeezing out the last factor in $R$** → run `copra` or `lbfgs`, then a short
  `lm` polish (20–50 iterations).
- **Stuck in a poor local minimum / no usable initial guess** → a `cma-es` global
  search, usually with `phase_basis="bspline"` (≈20–30 spline nodes), then refine
  the result with `lbfgs-ad`. See [global retrieval](../explanation/algorithms.md#global-retrieval-cma-es).
- **Spurious satellite pulses or a temporal pedestal** → `lbfgs-ad` (or `cma-es`)
  with the temporal penalty `reg_time`/`time_window` (see
  [Regularisation](regularisation.md#temporal-regularisation)).
- **Want pure on-device JAX speed** → swap in the equivalent twin
  (`copra-jax`, `lbfgs-hand`, `lbfgs-optx`, `lm-optx`); they give the same answer
  as their NumPy counterpart with the loop kept on the accelerator.
- **Checking a hand-derived gradient or a new interaction** → compare `lbfgs`
  against `lbfgs-ad` on the same trace; they should track to numerical precision.
- **`lm-optx` iterations feel slow on a large grid** → it already defaults to the
  faster normal-equations solve (`linear_solver="normal"`); the remaining cost is
  the Jacobian, which shrinks quadratically with the parameter count, so
  `phase_basis="bspline"` is the big lever. Fall back to `linear_solver="qr"` only
  if the solve misbehaves (see
  [Retrieval algorithms](../explanation/algorithms.md)).

## Run from several random restarts

The objective is non-convex, so a single run can land in a poor local minimum.
The standard practice is to retrieve from **5–10 random initial guesses** and keep
the result with the lowest `error`:

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

`guess=None` draws a random Gaussian-spectrum start; passing a fresh `rng` makes
each restart independent (and reproducible from the seed).

## A two-stage recipe

For the lowest final error, converge cheaply then polish:

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
The paper's "D-COPRA" is exactly `COPRA` with a `material` set.

Geometric smearing is the exception: pass `smearing` to one of the **autodiff**
solvers (`lbfgs-ad`, `lm`, `lm-optx`, `lbfgs-optx`, `cma-es`) only. The incoherent
sum over quadrature nodes breaks COPRA's magnitude-replacement projection and is not
covered by the hand-written adjoints, so those solvers raise instead of silently
ignoring the kernel. See [Geometric smearing](geometric_smearing.md).

## Convergence and stopping

- **COPRA**: `maxiters`, plus optional `reltol` (stop on a plateau of the best
  error) and `abstol` (stop at a target error). By default it runs all `maxiters`.
- **LBFGS / LBFGSAD / LM**: `maxiters` caps the evaluations; `reltol`/`abstol` are
  the optimiser's function/step tolerances (defaults `1e-4`/`1e-8`). For
  {class}`~croak.lm.LM` these become SciPy's `ftol` and `xtol`/`gtol`.

```{tip}
**Polishing with LM: loosen `abstol`.** The `1e-8` default is a *step-size* floor,
and it is what stops an LM polish — usually while the error is still falling
steeply. On a clean trace, dropping it to `1e-14` takes the final error six orders
of magnitude lower for about twice the work, whereas changing `reltol` does
nothing at all. See
[Choosing an algorithm](../tutorials/02_choosing_an_algorithm.md#why-lm-stops-early).
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
