# Choosing a solver

Choose a solver first by the model and constraints it supports, then by its
convergence and cost on your data. The ten named solvers belong to four families.
Their [objectives](../explanation/retrieval_theory.md) and stopping rules can differ;
non-convex optimisation does not guarantee the same answer. This page gives
practical starting choices; the [gallery](../tutorials/02_choosing_an_algorithm.md)
shows an executed comparison.

## At a glance

Start with the models and objectives supported by these six solvers:

| Solver | Gradient | Best for | Notes |
|--------|----------|----------|-------|
| {class}`~croak.warm_lbfgs.WarmLBFGS` | JAX autodiff | the **default**; a general starting workflow | `lbfgs-ad` seeded from a COPRA local sweep: refines a projected initial estimate under the chosen model |
| {class}`~croak.copra.COPRA` | analytic adjoint | full amplitude + phase in the core model | local projection followed by global descent; useful for initialisation |
| {class}`~croak.lbfgs.LBFGS` | analytic adjoint | phase-only; smoothness **regularisation** | analytic gradients for the core model |
| {class}`~croak.lbfgs_ad.LBFGSAD` | JAX autodiff | anything the core model does not cover | the solver with the **widest model coverage**: smearing, fitted thickness/τ₀/smearing, B-spline basis, temporal penalty |
| {class}`~croak.lm.LM` | JAX Jacobian | local least-squares refinement | can refine an existing estimate; uses a different penalty scaling from L-BFGS |
| {class}`~croak.cmaes.CMAES` | none (population search) | exploring beyond one local initialisation | pair with `phase_basis="bspline"`; needs `croak[evo]`; no guarantee of a global minimum |

Reach for `lbfgs-ad` (or `lm`/`lm-optx`) whenever you need something the analytic
adjoint does not carry — see [Gradients](../explanation/gradients.md) for exactly
where the boundary lies. Gradient validation compares derivatives directly;
agreement between final solver results is a separate consistency check.

The remaining four are on-device reimplementations — different execution and optimisation details:

| Solver | Twin of | Why reach for it |
|--------|---------|------------------|
| {class}`~croak.copra_jax.COPRAJax` | `copra` | same COPRA, on-device JAX numerics |
| {class}`~croak.lbfgs_hand.LBFGSHand` | `lbfgs` | analytic gradient, jitted/vectorised in JAX (no AD) |
| {class}`~croak.optimistix_lbfgs.OptxLBFGS` | `lbfgs-ad` | the L-BFGS loop runs entirely on-device |
| {class}`~croak.optimistix_lm.OptxLM` | `lm` | an on-device JAX-Jacobian LM |

## By task

- **General amplitude-and-phase retrieval in the core model** → try `copra` as
  an initial retrieval, then inspect the fit and compare several starts.
- **Phase-only retrieval** (amplitude fixed to a measured spectrum) → `lbfgs`
  with `phase_only=True`, optionally `reg_phase`. Check sensitivity to the phase constraint.
- **Noisy experimental data** → `lbfgs` with mild
  [smoothness regularisation](regularisation.md) (`reg_amp`/`reg_phase`).
- **Uncalibrated spectral response** → `R_omega=True` fits a gain for each
  frequency row. Use an appropriate amplitude constraint such as
  `phase_only=True` or `reg_spectrum` when a suitable independent spectrum is
  available: otherwise spurious band-edge energy can grow while $R$ falls. See
  [per-frequency scaling](regularisation.md#per-frequency-scaling).
- **Squeezing out the last factor in $R$** → run `copra` or `lbfgs`, then a short
  `lm` polish (20–50 iterations).
- **Stuck in a poor local minimum / no usable initial guess** → a `cma-es` global
  search, usually with `phase_basis="bspline"` (≈20–30 spline nodes), then refine
  the result with `lbfgs-ad`. See [global retrieval](../explanation/algorithms.md#global-retrieval-cma-es).
- **Spurious satellite pulses or a temporal pedestal** → `lbfgs-ad` (or `cma-es`)
  with the temporal penalty `reg_time`/`time_window` (see
  [Regularisation](regularisation.md#temporal-regularisation)).
- **Keeping calculations on a JAX device** → consider `copra-jax`, `lbfgs-hand`,
  `lbfgs-optx` or `lm-optx`. Check the solver's supported options and stopping
  rules; benchmark compilation and repeated evaluation separately.
- **Checking a hand-derived gradient or a new interaction** → compare the
  derivative with finite differences and autodiff on complex test fields. See
  [numerical validation](../developer_guide/validation.md). Solver trajectories
  need not coincide and do not establish derivative correctness.
- **`lm-optx` iterations feel slow on a large grid** → it already defaults to the
  faster normal-equations solve (`linear_solver="normal"`); the remaining cost is
  the Jacobian and linear solve, whose costs depend on parameter and residual counts, so
  `phase_basis="bspline"` is the big lever. Fall back to `linear_solver="qr"` only
  if the solve misbehaves (see
  [Retrieval algorithms](../explanation/algorithms.md)).

## Run from several random restarts

The objective is non-convex, so a single run can land in a poor local minimum.
The example below compares **eight random initial guesses** using COPRA, whose
objective here is the trace error. Choose a restart budget appropriate to your
problem; eight starts do not establish uniqueness or guarantee the best solution.

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

`guess=None` draws a random Gaussian-spectrum start; reusing a seeded generator draws successive starts reproducibly. Resetting it
to the same seed for every run repeats the same draw.

For a penalised fit, compare the complete objective as well as the trace error
and the physical plausibility of the pulse. The lowest unpenalised `error` need
not identify the best penalised solution. Start-to-start variation is a
convergence diagnostic, not a measurement-uncertainty estimate.

## A two-stage recipe

An existing estimate can seed a local least-squares refinement:

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
The introductory dispersive tutorial uses COPRA with a material specified.

Geometric smearing is the exception: pass `smearing` to one of the **autodiff**
solvers (`warm-lbfgs`, `lbfgs-ad`, `lm`, `lm-optx`, `lbfgs-optx`, `cma-es`) only. The spatial mixture is not implemented by croak's ordinary single-field COPRA
update or its current analytic adjoints, so those solvers raise instead of silently
ignoring the kernel. See [Geometric smearing](geometric_smearing.md).

## Convergence and stopping

- **COPRA**: `maxiters`, plus optional `reltol` (stop on a plateau of the best
  error) and `abstol` (stop at a target error). By default it runs all `maxiters`.
- **LBFGS / LBFGSAD / LM**: `maxiters` caps the evaluations; `reltol`/`abstol` are
  the optimiser's function/step tolerances (defaults `1e-4`/`1e-8`). For
  {class}`~croak.lm.LM` these become SciPy's `ftol` and `xtol`/`gtol`.

```{tip}
**Refining with LM: inspect the stopping reason.** For the SciPy route, `abstol`
sets both step and gradient tolerances. Reducing it can improve a locally converged fit,
at additional cost; the gain
is problem-dependent and does not establish better field accuracy. See
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
