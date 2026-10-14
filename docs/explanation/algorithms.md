# Retrieval algorithms

A retrieval algorithm searches the field parameters under a chosen measurement
model and objective. Use [solver selection](../howto/solver_selection.md) for a
practical starting choice and [the executed comparison](../tutorials/02_choosing_an_algorithm.md)
for an example. This chapter explains the numerical methods and their limits.

Ten named solvers share the {meth}`~croak.solver.Retriever.run` interface and
{class}`~croak.result.RetrievalResult`. Supported models, constraints, objectives
and stopping rules differ. The names accepted by {func}`croak.retrieve` are:

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

Warm L-BFGS composes a COPRA local sweep with L-BFGS-AD refinement. The remaining
names represent four families: COPRA, L-BFGS, Levenberg–Marquardt and CMA-ES.
Implementations of the same family can differ in line searches, stopping and
floating-point trajectories. Matching objectives does not guarantee matching
endpoints in a non-convex problem. See [objective normalisation](retrieval_theory.md).

The dispersive slab is a model option, available for PG/SD across solvers;
spatial mixtures require a supported JAX route. Numerical method and physical
model are separate choices.

## COPRA

The **Common Pulse Retrieval Algorithm** ({class}`~croak.copra.COPRA`,
implemented in {func}`croak.copra.run_copra`) works directly with the frequency-domain signal $\psi(\omega)$ that the forward
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
step size scaled by `alpha` (default `0.25`). The global stage descends the full [least-squares residual](retrieval_theory.md),
without guaranteeing a global minimum.

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
with it the trace's grip on the retrieved spectral amplitude. Use a justified phase-only constraint in COPRA, or choose a gradient solver when
a soft `reg_spectrum` constraint is needed; COPRA does not accept that penalty —
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

The amplitude and spectral penalties are **on by default** at the package
weights (0.03 and 0.01; the spectral term binds only when a target spectrum is
supplied) — pass `0` to disable them.

See [Regularisation](../howto/regularisation.md).
Compare runs under matched settings before attributing a difference to the
algorithm alone.

## L-BFGS-AD (autodiff)

{class}`~croak.lbfgs_ad.LBFGSAD` is the same NLopt `LD_LBFGS` optimiser over the
same parameterisation, but its gradient comes from **JAX automatic
differentiation** of the JAX forward model ({mod}`croak.forward_jax`) rather than
the hand-derived adjoint.

Because AD differentiates whatever the model and parameterisation produce, this
is the most feature-complete solver in the package. Everything the closed-form
adjoint does not cover lives here:

- [geometric time smearing](../howto/geometric_smearing.md), which is not implemented by croak's single-field COPRA projection or current
  analytic-adjoint solvers;
- the [fitted extras](../howto/fitting_thickness_tau0.md) `fit_thickness`,
  `fit_tau0` and `fit_smearing`;
- the [B-spline phase basis](#phase-parameterisation-the-b-spline-basis), which
  makes it the natural local refiner after a `cma-es` global search;
- the [temporal penalty](#temporal-regularisation).

A new interaction must first be implemented in the JAX forward model. Where
the analytic and AD paths overlap, compare their directional derivatives at the
same field; agreement of optimised endpoints alone is a weaker derivative check.

```{note}
NLopt-backed solvers warn if a non-finite evaluation is guarded or optimisation
ends early after an internal failure. They can return the best recorded iterate.
Treat the warning as a reason to inspect convergence and rerun with justified
settings; a returned result does not establish successful termination.
```

## Levenberg–Marquardt

{class}`~croak.lm.LM` drives the *flattened* FROG residual
$(T_\text{meas}-\mu T_\text{sim})\,w/\sqrt{\text{denom}}$ — whose Euclidean norm
is exactly $R$ — to zero with {external+scipy:func}`scipy.optimize.least_squares`. Its **Jacobian
is supplied by JAX** ({external+jax:func}`jax.jacfwd`, forward mode, since there are far fewer
parameters than residuals). Jacobian-based least-squares methods can converge rapidly near a well-resolved
solution, making them useful for refinement after another solver. Their cost and
achievable residual depend on conditioning, tolerances and constraints.

```{note}
The class named `LM` defaults to SciPy's trust-region reflective method
(`method="trf"`) with a JAX Jacobian. Explicit `method="lm"` uses a numerical
Jacobian because of a compiled-callback compatibility limitation, and emits a
warning. `lm-optx` implements a JAX-based Levenberg–Marquardt route.
```

LM supports the same forward-model options, phase-only mode, `R_omega` and
regularisation as L-BFGS (the penalties are appended as extra residual rows, so
the regularisation weight scales the penalty relative to $R^2$ rather than $R$).

## JAX and Optimistix twins

The JAX variants use compiled forward calculations and differ in where the
optimisation loop runs. This can improve repeated evaluations, while initial
compilation can dominate a small job. Their physical support is not identical:

| Solver | Numerical route |
|---|---|
| `copra-jax` | COPRA with compiled analytic operations |
| `lbfgs-hand` | L-BFGS with the compiled analytic gradient |
| `lbfgs-ad` | NLopt L-BFGS with JAX AD |
| `lbfgs-optx` | Optimistix L-BFGS with JAX AD |
| `lm` | SciPy trust-region least squares with JAX Jacobian by default |
| `lm-optx` | Optimistix Levenberg–Marquardt |

Check accepted options through {func}`croak.retrieve.algorithm_params`.
The [regularisation table](../howto/regularisation.md#summary) and
[spatial-model guides](../user_guide/models.md) state model and parameter support.

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
[evosax](https://github.com/RobertTLange/evosax) backend (`uv add "croak[evo]"`).

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

It accepts spectral-amplitude, phase, spectrum-match and temporal penalties.
Selection minimises their sum with the trace error; the reported trace error
excludes those penalties. Compare the same total objective when assessing a
change of solver. A population search does not certify a global minimum.

## The warm start

`warm-lbfgs` ({class}`~croak.warm_lbfgs.WarmLBFGS`) runs COPRA's local
projection sweeps, then passes their best field to `lbfgs-ad`. The final stage
uses the full configured AD model and objective. The warm stage uses a simpler
model: it can score the core dispersive model with reduced smearing, but its
local projection does not invert that mixture. It does not include the focal or
collection model, joint scalar fits, regularisation or marginal centring.

It is the default because the initial guess is the largest source of run-to-run
variation, and the two stages carry different amounts of measured information per
iteration. COPRA's local sweep replaces each delay's signal modulus with the
measured one — a projection, not a descent step — which moves a long way toward the
data in very few iterations. What arrives at L-BFGS is a cheap estimate that already
agrees with the trace, from which the gradient solver refines under the full model.
See the [solver API](../reference/api/retrieval.md#warm-started-l-bfgs) for stage
budgets and supported settings.

Both stages report progress, so the convergence curve covers the whole run and
the GUI's live preview is active from the first COPRA iteration rather than only
after the handover. The join is drawn as a dashed rule
({attr}`~croak.result.RetrievalResult.stage_boundaries`), because the two halves
do not count the same unit of work: one COPRA "iteration" is a full local sweep
over every delay, while one L-BFGS step is a single function evaluation. Compare
each stage against itself, not across the rule.

croak's own solver benchmarks, on simulated deep-UV TG-FROG traces (a Gaussian-beam control and the apertured mask, at transform
limit, ±2 fs² and the mask ±0.625 fs² pair, at the 9.5 µm substrate), compared
`warm-lbfgs` with cold-started `lbfgs-ad`: mean complex error 38 % lower (median
63 %), mean duration error 52 % lower, worst-case duration error 69.3 % → 20.8 %.
Four of the eight cases improved by more than 20 % in complex error and none
degraded by more than 20 %; the duration improved in seven of eight. The warm start
improves poor starts and leaves good ones roughly unchanged. Over a 1–40 µm ladder
the complex-error ranges improve on most arms, clearly so on both chirped Gaussian
arms and on mask −2 fs²; the retrieved-duration ranges over depth are mixed and
slightly wider on the mask arms.

The largest effect is on **variance**. The spread in retrieved duration over
starting points collapses on every chirped arm — five-fold on Gaussian +2 fs² and
mask −2 fs², twelve-fold on Gaussian −2 fs² (±36.9 → ±3.1 %) and mask +2 fs²
(±6.0 → ±0.5 %) — while the transform-limited arms, never start-sensitive, are
unchanged. A ±37 % spread over initial guesses means a single cold-start value is
not informative on its own. Multi-restart is then also a diagnostic: agreement
between warm and cold starts indicates a well-behaved landscape, and disagreement
gives a spread you can report (an optimisation diagnostic, not an experimental
error bar).

One regression is known: the Gaussian transform-limited arm at depths away from
9.5 µm, where the warm complex error reaches 0.63 against a cold maximum of 0.18 —
the case where the cold start was already reliable. Use `lbfgs-ad` directly if you
are supplying a starting point you already trust.

```{note}
croak's defaults (`warm-lbfgs`, `reg_spectrum=0.01`, `reg_amp=0.03`) differ from
the protocol used in the companion papers (`lbfgs-ad` from a transform-limited
start, 400 iterations, spectral and amplitude weights both 0.03); see
[validation](validation.md#the-retrieval-protocol).
```

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

COPRA descends rapidly at first, then slows; L-BFGS descends gradually but
steadily. A practical recipe is **`warm-lbfgs` (or COPRA, or L-BFGS) to converge,
then a short LM polish** to reduce the final $R$, run from several
[random restarts](../howto/solver_selection.md) on hard traces, keeping the
lowest-$R$ result. When restarts fail to escape a bad minimum, a **`cma-es` global
search** (usually on the B-spline basis) followed by a local refine is the
heavier-duty alternative. Compare algorithms on the same data, model, initial field,
constraints and objective normalisation, and record JAX compilation separately from
repeated evaluation time.

- **Levenberg–Marquardt is a polish, not a cold solver.** It needs far fewer
  iterations than L-BFGS, but each assembles a full Jacobian and costs much more
  wall clock; from random starts its trust region is not a global search.
- **COPRA is best run with its single global scale $\mu$.** The retained frequency
  marginal acts as a built-in constraint on the spectral amplitude; per-frequency
  factors (`R_omega`) open the weakly constrained amplitude directions with no
  penalty to close them. The gradient solvers pair `R_omega` with a spectral
  constraint for the same reason (see [regularisation](../howto/regularisation.md#per-frequency-scaling)).

The [algorithm tutorial](../tutorials/02_choosing_an_algorithm.md) is an executed
comparison, and the [solver-selection guide](../howto/solver_selection.md) gives
recommendations by problem type.

## Next

- [Solver selection](../howto/solver_selection.md) — decision tables and restart strategy.
- [Gradients](gradients.md) — the analytic adjoint vs the JAX twin.
- [Choosing an algorithm](../tutorials/02_choosing_an_algorithm.md) — an executed comparison.
