# Retrieval algorithms

croak ships ten solvers. All share the same [forward model](forward_model.md),
the same [least-squares objective](retrieval_theory.md), the uniform
{meth}`~croak.solver.Retriever.run` entry point and the same
{class}`~croak.result.RetrievalResult`. They are registered in
{data}`croak.retrieve.ALGORITHMS` and selected by name through
{func}`croak.retrieve`:

| `algorithm` | Class | Gradient source | Family |
|-------------|-------|-----------------|--------|
| `"copra"` | {class}`~croak.copra.COPRA` | analytic Wirtinger adjoint | step-based |
| `"copra-jax"` | {class}`~croak.copra_jax.COPRAJax` | analytic Wirtinger adjoint (JAX) | step-based |
| `"lbfgs"` | {class}`~croak.lbfgs.LBFGS` | analytic Wirtinger adjoint | quasi-Newton |
| `"lbfgs-hand"` | {class}`~croak.lbfgs_hand.LBFGSHand` | analytic Wirtinger adjoint (JAX) | quasi-Newton |
| `"lbfgs-ad"` | {class}`~croak.lbfgs_ad.LBFGSAD` | JAX autodiff | quasi-Newton |
| `"lbfgs-optx"` | {class}`~croak.optimistix_lbfgs.OptxLBFGS` | JAX autodiff | quasi-Newton |
| `"lm"` | {class}`~croak.lm.LM` | JAX Jacobian | Levenberg–Marquardt |
| `"lm-optx"` | {class}`~croak.optimistix_lm.OptxLM` | JAX Jacobian | Levenberg–Marquardt |
| `"cma-es"` | {class}`~croak.cmaes.CMAES` | none (derivative-free) | global / population |
| `"warm-lbfgs"` | {class}`~croak.warm_lbfgs.WarmLBFGS` | JAX autodiff | quasi-Newton (COPRA-seeded) |

The last row is a *composition* rather than a fifth family: `lbfgs-ad` started
from a COPRA local-projection sweep instead of from the raw initial guess. It is
the default because the initial guess is the largest source of run-to-run
variation and the warm-up removes most of it; see
["The warm start"](#the-warm-start) below.

The first six rows are really **three algorithms with twin implementations**:
`copra`/`copra-jax`, `lbfgs`/`lbfgs-hand`/`lbfgs-ad`/`lbfgs-optx` and `lm`/`lm-optx`
each minimise the *same* objective and converge to the *same* answer — they differ
only in where the numerics run (host NumPy vs on-device JAX) and how the gradient
is obtained. The genuinely distinct families are **COPRA** (step-based), **L-BFGS**
(quasi-Newton), **Levenberg–Marquardt** and **CMA-ES** (global). The four sections
below describe each family, and ["JAX and Optimistix twins"](#jax-and-optimistix-twins)
explains the variants.

Dispersive propagation is a *parameter* of each solver
(`material`/`thickness`/`npoints`/`omega0`), not a separate algorithm — the
"D-COPRA" of the companion paper is simply {class}`~croak.copra.COPRA` with a
`material` set.

## COPRA

The **Common Pulse Retrieval Algorithm** ({class}`~croak.copra.COPRA`,
implemented in {func}`croak.copra.run_copra`) is fast and robust from poor
guesses. It
works directly with the frequency-domain signal $\psi(\omega)$ that the forward
model produces and alternates two modes.

**Local mode.** Sweep the delays in *random order*. For each delay $\tau_j$,
form the **amplitude projection** $\psi' = \sqrt{T_\text{meas}/\mu}\,
e^{i\arg\psi}$ — the signal with its magnitude replaced by the measured one — and
take a gradient step on $\tilde E$ toward it. The per-delay distance is
$Z_j = \sum_\omega \lvert\psi' - \psi\rvert^2$ and the step uses a heuristic size

```{math}
\gamma_j = \frac{Z_j}{\max\bigl(g^{\,j}_\text{max},\, g^{\,j-1}_\text{max}\bigr)},
```

where $g_\text{max}$ is a running maximum gradient norm. No line search is needed.
Random ordering acts as an implicit regulariser, giving smoother intermediate
spectra.

**Global mode.** Once the local sweep stalls (the best error stops improving for
a few iterations), COPRA takes one *joint* gradient-descent step over all delays
at once: first a descent of the residual {eq}`residual` in the signal
$\psi_\text{all}$, then a descent in $\tilde E$ toward the updated signal, with
step size scaled by `alpha` (default `0.25`). This global stage is what drives
COPRA to the true [least-squares minimum](retrieval_theory.md) of the
[objective $r$](retrieval_theory.md#the-objective).

**Adaptive scaling (`Rω`).** By default the optimal scale $\mu$ in the projection
above is a single global factor. With `R_omega=True` it is computed *per frequency*
(one factor per spectral row), matching the gradient solvers' $R_\omega$ mode. On a
frequency row whose **measured** trace is identically zero across every delay — an
out-of-band row left after spectral cropping — that per-frequency $\mu$ is zero, so
the rescale $T_\text{meas}/\mu$ is a $0/0$; COPRA guards those degenerate rows (the
target is simply zero there, driving the model amplitude down to match the data)
rather than letting a `NaN` poison the adjoint.

That guard keeps COPRA numerically sound, but the main cost of `Rω` is shared by
every solver: giving each row its own scale removes the frequency marginal, and
with it the trace's grip on the retrieved spectral amplitude. Use it only together
with `phase_only` or `reg_spectrum` —
see [Regularisation](../howto/regularisation.md#per-frequency-scaling).

COPRA keeps the best-error spectrum and returns it. Convergence can be controlled
with `maxiters`, `reltol` (stop when the best error plateaus over a 10-iteration
window) and `abstol` (stop when the best error reaches a target).

```python
res = croak.retrieve(trace, g.omega, delays, "pg",
                    algorithm="copra", maxiters=300, reltol=1e-4)
```

## L-BFGS (analytic gradients)

{class}`~croak.lbfgs.LBFGS` minimises the trace error $R$ over a pulse
parameterisation ({class}`~croak.pulses.ArrayPulse`, amplitude + phase) with
NLopt's `LD_LBFGS` quasi-Newton optimiser. The gradient is analytic: the
$R$-cotangent on the trace is propagated through
{meth}`~croak.forward.ForwardModel.adjoint_single` and then through the pulse's
{meth}`~croak.pulses.Pulse.ew_vjp` — no autodiff.

L-BFGS is the **flexible** solver. It shares **phase-only** retrieval
(`phase_only=True`) and **per-frequency adaptive scaling** (`R_omega=True`) with
COPRA, but because it optimises a generic objective it also accommodates the
regularisation options COPRA does not:

- second-difference **smoothness regularisation** of amplitude (`reg_amp`) and
  phase (`reg_phase`), with closed-form gradients added to the objective;
- a **spectral-match** penalty (`reg_spectrum`, `spectrum_target`) pulling the
  retrieved amplitude toward an independently measured spectrum.

The amplitude and spectral penalties are **on by default** at the settled
weights (0.03 and 0.01; the spectral term binds only when a target spectrum is
supplied) — pass `0` to disable them.

See [Regularisation](../howto/regularisation.md). Its initial descent is more
gradual than COPRA's but it converges steadily to the same minimum, and for
phase-only problems it typically *outperforms* COPRA.

## L-BFGS-AD (autodiff)

{class}`~croak.lbfgs_ad.LBFGSAD` is the same NLopt `LD_LBFGS` optimiser over the
same parameterisation, but its gradient comes from **JAX automatic
differentiation** of the JAX forward model ({mod}`croak.forward_jax`) rather than
the hand-derived adjoint.

Because AD differentiates whatever the model and parameterisation produce, this
is the most feature-complete solver in the package. Everything the closed-form
adjoint does not cover lives here:

- [geometric time smearing](../howto/geometric_smearing.md), whose incoherent sum
  over quadrature nodes breaks both COPRA's projection and the hand-written
  adjoints;
- the [fitted extras](../howto/fitting_thickness_tau0.md) `fit_thickness`,
  `fit_tau0` and `fit_smearing`;
- the [B-spline phase basis](#phase-parameterisation-the-b-spline-basis), which
  makes it the natural local refiner after a `cma-es` global search;
- the [temporal penalty](#temporal-regularisation).

It is also the quick path to a new [interaction](interactions.md) for which no
adjoint has been derived, and — since the optimiser and objective otherwise match
`lbfgs` exactly — the way the two gradient routes are cross-checked: run both on
the same problem and they should track to numerical precision.

```{admonition} Robustness to NLopt failures
:class: note
Both NLopt-backed solvers (`lbfgs`, `lbfgs-ad`) carry two guards. A non-finite
objective or gradient is replaced by a large finite penalty with a zeroed
gradient ({func}`croak.solver.guard_nonfinite`), so the line search backtracks
instead of poisoning the optimiser state. And if NLopt itself fails internally
(a line-search breakdown or roundoff limit — observed for roughly one in six
random starts on large dispersive grids), the solver catches the exception and
returns the **best iterate found** rather than aborting the retrieval. Each
guard warns once per run, so a salvaged result is visible in the log; treat the
warning as a hint to try more random starts or a better initial guess.
```

## Levenberg–Marquardt

{class}`~croak.lm.LM` drives the *flattened* FROG residual
$(T_\text{meas}-\mu T_\text{sim})\,w/\sqrt{\text{denom}}$ — whose Euclidean norm
is exactly $R$ — to zero with {func}`scipy.optimize.least_squares`. Its **Jacobian
is supplied by JAX** ({func}`jax.jacfwd`, forward mode, since there are far fewer
parameters than residuals). Levenberg–Marquardt has superlinear local convergence
and reaches the **lowest final trace error** of the local solvers, which makes it
a good *polishing* stage after COPRA or L-BFGS.

```{admonition} Why the default is method="trf", not "lm"
:class: warning
{class}`~croak.lm.LM` defaults to SciPy's trust-region `method="trf"` with the
analytic JAX Jacobian — **not** MINPACK's classic `method="lm"`. MINPACK's
Fortran driver calls the user Jacobian back from compiled code, and evaluating the
JAX/XLA Jacobian program from inside that callback crashes the interpreter. `trf`
is pure-Python, accepts the analytic Jacobian and is numerically equivalent here.
If you explicitly request `method="lm"`, croak honours it but switches to a numeric
(finite-difference) Jacobian so the process cannot crash (with a warning).
```

LM supports the same forward-model options, phase-only mode, `R_omega` and
regularisation as L-BFGS (the penalties are appended as extra residual rows, so
the regularisation weight scales the penalty relative to $R^2$ rather than $R$).

## JAX and Optimistix twins

Seven of the ten algorithms run their numerics in JAX. They divide into two
groups, and the distinction matters:

- `copra-jax`, `lbfgs-hand` and `lbfgs-optx` are **on-device reimplementations**
  of COPRA and L-BFGS. They produce numerically equivalent results — the same
  objective, the same parameterisation, the same convergence — and exist for
  speed and platform reach, not new physics. Use them interchangeably with their
  NumPy counterparts.
- `lbfgs-ad`, `lm`, `lm-optx` and `cma-es` take their gradients from automatic
  differentiation, and that **does** buy new physics and new estimands:
  geometric smearing, fitted medium thickness, delay-zero and smearing width,
  the B-spline phase basis and the temporal penalty are reachable only here (see
  [Gradients](gradients.md)).

| `algorithm` | Twin of | What changes | Why |
|-------------|---------|--------------|-----|
| `"copra-jax"` | `copra` | global step is `vmap`/`jit`, the local sweep a `lax.scan` | a generation runs as one batched device call |
| `"lbfgs-hand"` | `lbfgs` | the *same* analytic Wirtinger gradient, but jitted and `vmap`-ed in JAX (still **no** autodiff) | host→device transfers removed; the trace evaluation is the bottleneck, not the gradient maths |
| `"lbfgs-ad"` | `lbfgs` | gradient from `jax.value_and_grad` of the JAX forward model | the most feature-complete solver: smearing, fitted extras, B-spline basis, temporal penalty. Also differentiates a new interaction with no hand-written adjoint, and cross-checks the analytic one |
| `"lbfgs-optx"` | `lbfgs-ad` | NLopt's host-side loop → Optimistix's on-device {class}`optimistix.LBFGS` | the whole BFGS loop stays on-device |
| `"lm-optx"` | `lm` | SciPy `least_squares` → on-device {class}`optimistix.LevenbergMarquardt` (Jacobian via `lineax`) | a *genuine* JAX-Jacobian LM, sidestepping the MINPACK reentrancy crash that forces {class}`~croak.lm.LM` to `method="trf"` |

`lbfgs-ad` has the widest model coverage of these. It is also the cross-check: running
`lbfgs` and `lbfgs-ad` on the same trace should track to numerical precision,
which is how the two gradient routes are held against each other on real
retrievals (`tests/test_ad_vs_analytic.py`; see [Gradients](gradients.md)).
The `copra-jax`/`lbfgs-optx` twins do not expose anything their originals lack;
`lm-optx` adds only `linear_solver` (below), which selects how its on-device
damped Gauss--Newton system is solved and has no SciPy counterpart. `lbfgs-hand` carries the same regularisation penalties as
`lbfgs` (`reg_amp`, `reg_phase`, `reg_spectrum`); `copra-jax`, like `copra`,
takes none of them. The B-spline basis is available on `lbfgs-ad`,
`lbfgs-optx`, `lm`, `lm-optx` and `cma-es`, and the temporal penalty
`reg_time` (below) only on `lbfgs-ad` and `cma-es`.

```{note}
**`linear_solver` on `lm-optx`.** Every Levenberg--Marquardt iteration solves the
linear least-squares problem

$$
\begin{bmatrix} J \\ \sqrt{\lambda}\,I \end{bmatrix} d
= \begin{bmatrix} r \\ 0 \end{bmatrix},
$$

whose normal equations are $(J^\mathsf{T}J + \lambda I)\,d = J^\mathsf{T}r$.
croak offers two routes:

- `linear_solver="normal"` (the default) forms the Gram matrix
  $J^\mathsf{T}J + \lambda I$ and takes its Cholesky factor.
- `linear_solver="qr"` factorises the stacked $(m+n)\times n$ operator directly.
  This is Optimistix's own default and never forms $J^\mathsf{T}J$.

`"normal"` is the default because it is roughly **2x faster** on a full-size
retrieval: the Gram matrix is one BLAS `gemm`, which a tuned BLAS runs at close to
peak throughput, whereas LAPACK's blocked Householder QR of a tall-skinny matrix is
panel-bound and does not (on an Apple M5 Max: 0.26 s versus 3.0 s for a
$10^5\times10^3$ Jacobian, taking a 512-point PG retrieval from 2.9 s to 1.2 s per
iteration at an identical final error).

Squaring the condition number is the trade-off. It costs less than it might appear
to, because the FROG Jacobian is *already* strongly rank-deficient — typically only
a few hundred of its singular values sit above the float64 epsilon — so both routes
depend on $\lambda > 0$ to regularise rather than on $J$ having full rank. Switch to
`"qr"` if a badly scaled problem stalls or returns a non-finite result.
```

```{note}
**`reltol` on the Optimistix twins.** Optimistix's built-in termination scales its
bar as `atol + rtol·|f|`, which is `atol`-dominated near convergence, so on its own
it effectively ignores `rtol`. croak therefore gives `lbfgs-optx`/`lm-optx` the same
relative-improvement **plateau** stop as COPRA (stop once the best FROG error
improves by less than `reltol` over a 10-iteration window), so `reltol` behaves
consistently across all solvers; `abstol` is still Optimistix's `atol`.
```

## Global retrieval (CMA-ES)

Every solver above is a *local* optimiser: COPRA's projections and the
gradient/Jacobian methods all descend from the initial guess, so a poor start can
strand them in a bad local minimum (the usual remedy is
[several random restarts](../howto/solver_selection.md)). {class}`~croak.cmaes.CMAES`
(`algorithm="cma-es"`) instead runs a **global**, derivative-free population search
— CMA-ES (default), Separable CMA-ES or Differential Evolution, via the optional
[evosax](https://github.com/RobertTLange/evosax) backend (`pip install croak[evo]`).

Because the forward model is JAX, the whole population is evaluated in a single
jitted `vmap`, so a generation costs roughly one batched forward pass. CMA-ES has
no gradient to follow, so its cost scales with the number of parameters; it is most
effective on the **reduced B-spline phase basis** (≈20–30 parameters) rather than
the full per-frequency grid. The intended workflow is therefore *global then local*:

```python
# 1. global search over a low-dimensional spline-phase basis
g0 = croak.retrieve(trace, g.omega, delays, "pg", algorithm="cma-es",
                   phase_basis="bspline", n_nodes=24, maxiters=300)

# 2. polish locally from that result
res = croak.retrieve(trace, g.omega, delays, "pg", algorithm="lbfgs-ad",
                    guess=g0.spectrum, maxiters=300)
```

It accepts the same smoothness / spectral-match / temporal regularisation as the
gradient solvers — the penalties are added to the FROG error to form the *total*
the strategy minimises, while selection and reporting stay on the bare FROG error,
so results remain directly comparable to the other retrievers.

## The warm start

`warm-lbfgs` ({class}`~croak.warm_lbfgs.WarmLBFGS`) runs COPRA's *local* stage
alone and hands the result to `lbfgs-ad` as its initial guess. Nothing else
changes: same forward model, same objective, same minimum being sought.

The reason it is the default is that the two stages carry different amounts of
measured information per iteration. COPRA's local sweep replaces each delay's
signal modulus with the measured one — a projection, not a descent step — which
moves a long way toward the data in very few iterations. Its global stage is an
ordinary gradient descent and is not needed here, so it is disabled by holding
`stall_patience` at a value the run can never reach. What arrives at L-BFGS is
therefore a cheap estimate that already agrees with the trace, from which the
gradient solver refines under the *full* model — including the parts COPRA's
projection structurally cannot represent, above all geometrical smearing, where
the measurement constrains an incoherent mixture with no single signal to
project.

Both stages report progress, so the convergence curve covers the whole run and
the GUI's live preview is active from the first COPRA iteration rather than only
after the handover. The join is drawn as a dashed rule
({attr}`~croak.result.RetrievalResult.stage_boundaries`), because the two halves
do not count the same unit of work: one COPRA "iteration" is a full local sweep
over every delay, while one L-BFGS step is a single function evaluation. Compare
each stage against itself, not across the rule.

Measured across a Gaussian-beam control and an apertured mask, at transform
limit, ±2 fs² and the mask ±0.625 fs² pair, at the 9.5 µm experimental
substrate: mean complex error 38% lower (median 63%), mean duration error 52%
lower, worst-case duration error 69.3% → 20.8%. Four of the eight cases improved
by more than 20% in complex error and none degraded by more than 20%; the
duration improved in seven of eight. It does little where the answer was already
good and a great deal where it was not: it improves poor starts and leaves good
ones roughly unchanged.

Over a 1–40 µm ladder the complex-error ranges improve on most arms, clearly so
on both chirped Gaussian arms and on mask −2 fs². (The retrieved-*duration*
ranges over depth are mixed and slightly wider on the mask arms; that is one
statistic, not the whole picture.)

The largest effect is on **variance**. The spread in retrieved duration over
starting points collapses on every chirped arm — five-fold on Gaussian +2 and
mask −2, twelve-fold on Gaussian −2 (±36.9 → ±3.1 points) and mask +2 (±6.0 →
±0.5) — while the transform-limited arms, never start-sensitive, are unchanged.
A ±37% spread over initial guesses means the mean value is not informative on
its own — and on real data you never learn which draw you got. Multi-restart is
then also a diagnostic: agreement between warm and cold starts indicates a
well-behaved landscape, while disagreement gives a spread you can report.

One regression is known — the Gaussian transform-limited arm at depths away from
9.5 µm, where the warm complex error reaches 0.63 against a cold maximum of
0.18. That is the case where the cold start was already reliable. It costs one
extra COPRA run; use `lbfgs-ad` directly if you are supplying a starting point
you already trust.

## Phase parameterisation: the B-spline basis

By default a solver optimises one phase value per frequency bin (`grid.n`
parameters, `phase_basis="pointwise"`). Setting `phase_basis="bspline"` (with
`n_nodes`, default 20) instead represents the spectral phase as a **cubic B-spline**
over the non-zero-spectrum support, optimising only the `n_nodes` control points and
holding the amplitude at the guess. This is a structural regulariser: it cuts
the dimensionality, guarantees a $C^2$-smooth phase by construction, and
makes the problem tractable for the derivative-free CMA-ES search. It implies
phase-only retrieval. Supported by `lbfgs-ad`, `lbfgs-optx`, `lm`, `lm-optx` and
`cma-es`; see [Regularisation](../howto/regularisation.md#the-b-spline-phase-basis)
and {func}`croak._jax_pulse.make_spline_phase_basis`.

## Temporal regularisation

`lbfgs-ad` and `cma-es` additionally accept a **temporal** penalty (`reg_time`, with
a `time_window` in seconds) that suppresses the fraction of pulse energy falling
outside a time window — useful for removing spurious satellites and pedestals far
from the main pulse while keeping the spectrum and field an exact Fourier pair. See
[Regularisation](../howto/regularisation.md#temporal-regularisation).

## How they compare

Both COPRA and L-BFGS converge to the same least-squares minimum, but their
*paths* differ: COPRA descends rapidly at first then slows; L-BFGS descends
gradually but steadily. A practical recipe is **L-BFGS (or COPRA) to converge,
then a short LM polish** to reduce the final $R$, run from several
[random restarts](../howto/solver_selection.md). When even restarts fail to escape
a bad minimum, a **`cma-es` global search** (usually on the B-spline basis)
followed by a local refine is the heavier-duty alternative. The
[solver-selection guide](../howto/solver_selection.md) gives concrete
recommendations by problem type.

The companion paper's solver study (see [Validation](validation.md)) puts
numbers on this recipe, on retrievals of first-principles simulated traces:

- Levenberg–Marquardt reaches the same solution as L-BFGS in about a twelfth
  of the iterations, but each iteration assembles a full Jacobian — roughly
  23× the wall clock per step — which is why it is used as a
  warm-started polish rather than a cold solver. From random starts its trust
  region is not a global search and it becomes the least reliable choice.
- Multi-start with lowest-$R$ selection is part of the protocol on hard
  traces, not a precaution: over 219 random starts on strongly chirped
  single-cycle traces, a single start landed in the correct basin only
  45–80% of the time, while the lowest-trace-error member of each ensemble
  landed there in every configuration tested.
- COPRA is best run as designed, with its single global scale $\mu$: the
  retained frequency marginal acts as a built-in regulariser. Handing it
  per-frequency factors instead opens the weakly constrained amplitude
  directions with no penalty to close them — half of the tested restarts
  failed outright, and on a chirped trace it returned 0.9 fs for a 5.5 fs
  pulse. The gradient solvers pair $R_\omega$ with a spectral constraint for
  the same reason.

## Next

- [Solver selection](../howto/solver_selection.md) — decision tables and restart strategy.
- [Gradients](gradients.md) — the analytic adjoint vs the JAX twin.
- [Choosing an algorithm](../tutorials/02_choosing_an_algorithm.md) — an executed comparison.
