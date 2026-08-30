---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  name: python3
---

# Choosing an algorithm

croak has ten [solvers](../explanation/algorithms.md) behind a single API, grouped
into four algorithm families (COPRA, L-BFGS, Levenberg–Marquardt and the global
CMA-ES). The local solvers reach the same least-squares minimum but differ in their
path and final accuracy. This tutorial runs three of them on one trace and compares
convergence.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import croak

g = croak.Grid(128, dt=0.3e-15)
ew = croak.gaussian_pulse(g, 3.0e-15, phases=[6e-30, 30e-45])   # GDD + TOD
delays = np.linspace(-24e-15, 24e-15, 90)
trace = croak.maketrace(g.omega, delays, ew, "pg")              # PG-FROG
```

## Run several algorithms

We seed every solver from the *same* initial guess so the comparison is fair, and
record the convergence history `res.errors`.

```{code-cell} ipython3
guess = croak.gaussian_pulse(g, 2.0e-15)     # a deliberately rough start

results = {}
for algo in ("copra", "lbfgs", "lm"):
    results[algo] = croak.retrieve(trace, g.omega, delays, "pg",
                                  algorithm=algo, guess=guess, maxiters=200,
                                  rng=np.random.default_rng(0))

for algo, res in results.items():
    print(f"{algo:6s}  R = {res.error:.2e}  ({len(res.errors)} evaluations)")
```

That spread is startling — COPRA reaches machine precision, L-BFGS stalls around
$10^{-5}$, and LM lands in between after only a handful of evaluations. It is
worth understanding, because **almost all of it is an artefact of noise-free
synthetic data** and disappears on a real measurement. The rest of this tutorial
explains why.

## Convergence trajectories

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(5.5, 4))
for algo, res in results.items():
    ax.semilogy(np.arange(1, len(res.errors) + 1), res.errors, label=algo)
ax.set_xlabel("iteration / evaluation")
ax.set_ylabel("trace error R")
ax.set_title("convergence")
ax.legend()
fig.tight_layout()
```

(The x-axis counts iterations for COPRA and objective evaluations for the others,
so compare shapes rather than absolute counts.)

## Why LM stops early

LM did not run out of iterations — it stopped itself after ~19 evaluations, on a
convergence test. {class}`~croak.lm.LM` passes `reltol` to SciPy as `ftol` (the
relative *cost* reduction) and `abstol` as both `xtol` and `gtol` (the *step* and
*gradient* floors). It is worth knowing which one is binding, so vary them one at
a time:

```{code-cell} ipython3
def lm_run(**tol):
    r = croak.retrieve(trace, g.omega, delays, "pg", algorithm="lm",
                      guess=guess, maxiters=400, **tol)
    return f"R = {r.error:.2e}  ({len(r.errors)} evaluations)"

print("vary reltol (ftol), abstol at its 1e-8 default:")
for reltol in (1e-4, 1e-8, 1e-12):
    print(f"   reltol={reltol:.0e}  {lm_run(reltol=reltol, abstol=1e-8)}")

print("vary abstol (xtol and gtol), reltol at its 1e-4 default:")
for abstol in (1e-8, 1e-12, 1e-15):
    print(f"   abstol={abstol:.0e}  {lm_run(reltol=1e-4, abstol=abstol)}")
```

`reltol` makes no difference at all — the cost is still falling fast when LM
stops, so `ftol` never triggers. It is **`abstol`** that binds: the Gauss–Newton
*step* has become smaller than $10^{-8}$, and SciPy calls that converged. Loosen
that floor and LM delivers the superlinear convergence it is famous for, six
orders of magnitude deeper for roughly twice the work.

Plotting the three runs together makes the point better than the numbers do. They
are not three different descents — they are the *same* descent, cut short at three
different places:

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(6, 4))
for abstol, lw, color in ((1e-15, 1.2, "C2"), (1e-12, 2.2, "C1"), (1e-8, 3.5, "C0")):
    r = croak.retrieve(trace, g.omega, delays, "pg", algorithm="lm", guess=guess,
                      maxiters=400, reltol=1e-4, abstol=abstol)
    n = np.arange(1, len(r.errors) + 1)
    ax.semilogy(n, r.errors, lw=lw, color=color, alpha=0.85,
                label=f"abstol={abstol:.0e}  →  R={r.error:.1e}")
    ax.plot(n[-1], r.errors[-1], "*", ms=16, color=color, mec="k", mew=0.5, zorder=5)
ax.set_xlabel("residual evaluation")
ax.set_ylabel("trace error R")
ax.set_title("LM follows one path — abstol only decides where it stops (★)")
ax.legend(fontsize=8, loc="lower left")
ax.grid(alpha=0.3)
fig.tight_layout()
```

The default (thick blue) gives up at the first star, on the steepest part of the
curve, with more than six orders of magnitude still available.

**If you use LM as a polishing stage, set `abstol` yourself** — the default is
tuned for stopping sensibly on noisy data, not for chasing a synthetic trace to
machine precision.

## Why L-BFGS stalls, and why it does not matter

Tightening the tolerance does *not* rescue L-BFGS — it burns the whole budget and
stays near $10^{-5}$. The cause is the shape of the objective, and it is worth
seeing directly. Perturb the exact solution by a small amount $\epsilon$ and watch
the trace error:

```{code-cell} ipython3
from croak.metrics import trace_error

rng = np.random.default_rng(0)
direction = rng.standard_normal(ew.shape) + 1j * rng.standard_normal(ew.shape)
direction /= np.linalg.norm(direction)
scale = np.linalg.norm(ew)

print(f"{'eps':>8} {'R':>11} {'R/eps':>8} {'R^2/eps^2':>10}")
for eps in (1e-1, 1e-2, 1e-3, 1e-4):
    perturbed = ew + eps * scale * direction
    R = trace_error(trace, croak.maketrace(g.omega, delays, perturbed, "pg"))
    print(f"{eps:8.0e} {R:11.3e} {R / eps:8.4f} {R**2 / eps**2:10.6f}")
```

`R/eps` is **constant**: the trace error grows *linearly* with the distance from
the solution. On log axes that is a straight line of slope 1, against slope 2 for
$R^2$:

```{code-cell} ipython3
eps = np.logspace(-5, -0.5, 25)
R = np.array([trace_error(trace, croak.maketrace(g.omega, delays,
                                                 ew + e * scale * direction, "pg"))
              for e in eps])

fig, ax = plt.subplots(figsize=(5.5, 4))
ax.loglog(eps, R, "o-", ms=3, label=r"$R$  — what L-BFGS minimises")
ax.loglog(eps, R**2, "s-", ms=3, label=r"$R^2$  — what LM minimises")
ax.loglog(eps, 0.012 * eps, "k--", lw=1, label="slope 1  (a cone)")
ax.loglog(eps, 1.45e-4 * eps**2, "k:", lw=1, label="slope 2  (a paraboloid)")
ax.set_xlabel(r"distance from the solution, $\epsilon$")
ax.set_ylabel("objective")
ax.set_title("the shape of the two objectives at the minimum")
ax.legend(fontsize=8)
fig.tight_layout()

print(f"fitted slopes:  R -> {np.polyfit(np.log(eps), np.log(R), 1)[0]:.3f}"
      f"   R^2 -> {np.polyfit(np.log(eps), np.log(R**2), 1)[0]:.3f}")
```

$R$ is a norm, $\sqrt{\sum r^2}$, so at a perfect fit it has a $|x|$-shaped cusp —
a **cone**, not a paraboloid — and its gradient does **not** vanish at the
minimum. A quasi-Newton method like L-BFGS builds a *quadratic* model of the
objective and line-searches along it; against a cone that model is wrong at every
scale, the steps collapse, and it parks at a finite $R$.

The other two never meet the cusp. LM works on the residual *vector* — effectively
$R^2$, whose `R^2/eps^2` column is likewise constant, i.e. genuinely quadratic and
smooth at the solution — so Gauss–Newton converges superlinearly. COPRA does not
line-search $R$ at all; its projection step simply has the true pulse as a fixed
point.

This is not a defect in the gradient. `lbfgs` and `lbfgs-ad` take completely
different routes to it and stall at the same place:

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(5.5, 4))
for algo, style in (("lbfgs", dict(lw=3.0, alpha=0.45)),
                    ("lbfgs-ad", dict(lw=1.2, ls="--"))):
    r = croak.retrieve(trace, g.omega, delays, "pg", algorithm=algo, guess=guess,
                      maxiters=400, reltol=1e-12, abstol=1e-15)
    ax.semilogy(np.arange(1, len(r.errors) + 1), r.errors,
                label=f"{algo}  (R = {r.error:.2e})", **style)
ax.set_xlabel("objective evaluation")
ax.set_ylabel("trace error R")
ax.set_title("analytic vs autodiff gradient: same descent, same stall")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
```

The two curves lie on top of each other for all 400 evaluations. Whatever is
stopping L-BFGS, it is not the gradient.

## On real data they agree

A perfect fit is only reachable when the data is noise-free and generated by the
same forward model. On any measurement the best achievable $R$ is set by the
noise, the minimum sits at $R>0$, and the cusp is nowhere near. Add 1 % noise and
the ranking evaporates:

```{code-cell} ipython3
noise = np.random.default_rng(1).standard_normal(trace.shape)
noisy = np.clip(trace + 0.01 * trace.max() * noise, 0.0, None)

def centre(t, intensity):
    i = int(np.argmax(intensity))
    return (t - t[i]) * 1e15, intensity / intensity.max()

fig, (ax_conv, ax_pulse) = plt.subplots(1, 2, figsize=(10, 4))
ax_pulse.plot(*centre(g.t, np.abs(g.ifft(ew)) ** 2),
              lw=4, alpha=0.35, color="k", label="true")

for algo in ("copra", "lbfgs", "lm"):
    kw = dict(reltol=1e-12, abstol=1e-15) if algo != "copra" else {}
    r = croak.retrieve(noisy, g.omega, delays, "pg", algorithm=algo, guess=guess,
                      maxiters=400, rng=np.random.default_rng(0), **kw)
    ax_conv.semilogy(np.arange(1, len(r.errors) + 1), r.errors,
                     label=f"{algo}  (R = {r.error:.3e})")
    ax_pulse.plot(*centre(r.t, r.intensity_t), lw=1.4, label=algo)

ax_conv.set_xlabel("iteration / evaluation")
ax_conv.set_ylabel("trace error R")
ax_conv.set_title("1 % noise: all three reach the same floor")
ax_conv.legend(fontsize=8)
ax_conv.grid(alpha=0.3)
ax_pulse.set_xlim(-15, 15)
ax_pulse.set_xlabel("time (fs)")
ax_pulse.set_ylabel("intensity (norm.)")
ax_pulse.set_title("…and all three find the pulse")
ax_pulse.legend(fontsize=8)
fig.tight_layout()
```

The left panel is the point of this tutorial: the twelve-order-of-magnitude spread
from the clean trace has collapsed to about 1 %, and all three solvers sit on the
same noise floor. **Choose a solver for its robustness, its features and its
speed, not for the last digit of $R$ on synthetic data.**

(COPRA's saw-tooth is normal — its local sweep accepts steps that raise the error
before recovering, and `res.errors` records every one. The result it returns is
the best iterate, not the last.)

The right panel carries the caveat. All three recover the pulse, but each carries
its own noise-induced wiggle, and a FWHM read off the half-maximum is sensitive to
exactly that. On noisy data the minimum is broad and shallow, so several visibly
different pulses fit about equally well — a property of the *data*, not of any
solver. It is why real work needs [random
restarts](../howto/solver_selection.md) and an [uncertainty
estimate](../howto/uncertainty.md) rather than a single number.

## The two gradient routes agree

`lbfgs-ad` takes its gradient from [automatic
differentiation](../explanation/gradients.md) of the forward model — the general
route, and the one that reaches smearing, the fitted extras and the B-spline
basis. `lbfgs` is the same optimiser on the hand-derived adjoint, which covers the
core model. On a problem both can do they should track each other, which is how
each is held against the other.

```{code-cell} ipython3
hand = croak.retrieve(trace, g.omega, delays, "pg",
                     algorithm="lbfgs", guess=guess, maxiters=120)
auto = croak.retrieve(trace, g.omega, delays, "pg",
                     algorithm="lbfgs-ad", guess=guess, maxiters=120)
print(f"lbfgs     R = {hand.error:.3e}")
print(f"lbfgs-ad  R = {auto.error:.3e}")
```

## Takeaways

- **COPRA** is fast and robust for amplitude-and-phase retrieval; the
  production default `warm-lbfgs` uses it as the warm-up stage.
- Use **L-BFGS** when you need [regularisation](../howto/regularisation.md) or
  phase-only retrieval. Its stall on noise-free data is a property of minimising
  a norm, not a weakness on real measurements.
- Add a short **LM** polish for the lowest final error — and **tighten `abstol`**,
  or its step-size floor will stop it while there is still a lot left to gain.
- Judge a solver on a *noisy* trace. On clean synthetic data the final $R$ mostly
  measures how each solver behaves in a limit that no measurement reaches.
- Run from several [random restarts](../howto/solver_selection.md) on real data —
  and when those still get stuck, reach for a **CMA-ES**
  [global search](../explanation/algorithms.md#global-retrieval-cma-es) followed by
  a local refine.
- Each of `copra`/`lbfgs`/`lm` also has a numerically-equivalent on-device JAX twin
  (`copra-jax`, `lbfgs-hand`/`lbfgs-optx`, `lm-optx`) if you want the loop to run on
  an accelerator.
