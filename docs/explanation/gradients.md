# Gradients: automatic differentiation and the Wirtinger adjoint

A gradient tells the optimiser how a change in spectral amplitude or phase
changes the fit. This chapter develops the complex derivative used for that
calculation. It is useful when interpreting solver support, verifying a model,
or understanding why nuisance parameters can be fitted alongside the field.
Start with [the objective](retrieval_theory.md) and [forward model](forward_model.md).

croak provides both analytic adjoints and automatic differentiation (AD). The
analytic path covers the local thin-medium kernels and dispersive PG/SD; the
JAX path also supports spatial models and additional fitted parameters. The two
are checked where their domains overlap. Contributor implementation and testing
belong in [numerical validation](../developer_guide/validation.md).

## Wirtinger calculus in one paragraph

The objective is a *real* function of a *complex* spectrum, so it is not
holomorphic and ordinary complex differentiation does not apply. The right tool is
Wirtinger calculus: treat $v$ and $v^*$ as independent, and define the cotangent
of a complex variable $v$ as

```{math}
\bar v \;\equiv\; \frac{\partial L}{\partial v^*}.
```

For a real loss $L$, $\bar v$ is (half) the steepest-ascent direction, and
$\tilde E \leftarrow \tilde E - \gamma\,\bar{\tilde E}$ is gradient descent. Every
analytic adjoint uses this conjugate cotangent, with
$dL=2\operatorname{Re}(\bar v^*\,dv)$. For a vector, sum this pairing over its
components. This identity fixes the factors of two and conjugations below.

## The reverse pass

The forward model {eq}`dispersive` is a chain of linear maps (FFTs, propagation
phases, the delay ramp, the quadrature sum) around one pointwise nonlinear step
(the interaction signal). Reverse-mode differentiation walks the chain backwards,
applying the adjoint of each link. The elementary rules are:

- **raw FFT** $\;\to\;$ its adjoint is $N$ times the inverse DFT;
- **raw inverse FFT** $\;\to\;$ its adjoint is $1/N$ times the forward DFT;
- **pointwise multiply by $a$** $\;\to\;$ multiply the cotangent by $a^*$;
- **the interaction** $\;\to\;$ `signal_adjoint` (below).

{meth}`croak.forward.ForwardModel.adjoint_single` implements exactly this for the
dispersive model: it back-propagates the trace cotangent through the signal
propagation phase, the inverse DFT, the interaction, the forward DFT and the input
propagation phase, summing over the quadrature nodes, to return
$\partial L/\partial\tilde E^*$ on the centred grid. It reuses the time-domain
fields recorded by the matching forward pass (`signal_single(..., record=True)`),
so the reverse calculation has a cost comparable to a forward evaluation,
as in reverse-mode AD. Actual cost depends on the model and array sizes.

## Per-interaction signal adjoints

The only geometry-specific piece is the adjoint of the instantaneous signal
$s(E, G)$. Given the output cotangent $\bar s$, each
[interaction](interactions.md) returns $(\bar E, \bar G)$:

| Geometry | $s$ | $\bar E$ | $\bar G$ |
|----------|-----|----------|----------|
| SHG | $E G$ | $G^*\,\bar s$ | $E^*\,\bar s$ |
| SD | $E^2 G^*$ | $2 E^*\,G\,\bar s$ | $E^2\,\bar s^*$ |
| PG | $E\lvert G\rvert^2$ | $\lvert G\rvert^2\,\bar s$ | $2 G\,\operatorname{Re}(E^*\bar s)$ |

These are the closed forms in {meth}`SHG.signal_adjoint
<croak.interactions.SHG.signal_adjoint>`,
{meth}`SD.signal_adjoint <croak.interactions.SD.signal_adjoint>` and
{meth}`PG.signal_adjoint <croak.interactions.PG.signal_adjoint>`. SHG is
holomorphic in both arguments; SD is holomorphic in $E$ and anti-holomorphic in
$G$; PG is real in the gate intensity — the mixed cases are exactly where naïve
"complex derivative" intuition fails and Wirtinger bookkeeping is essential.

## Split-gate adjoint

For TG with distinct local replicas, write $s=a b_+ b_-^*$. Its three cotangents
are

```{math}
\bar a=b_+^*b_-\bar s,\qquad
\bar b_+=a^*b_-\bar s,\qquad
\bar b_-=a b_+\bar s^*.
```

The last term is conjugate-linear. When the gates coincide, their contributions
add to the PG gate adjoint above. The timing shifts and depth propagators then
carry each cotangent back to the common unknown spectrum. A spatially collected
model must differentiate its coherent collection sum before propagating these
local derivatives; an intensity mixture instead sums intensity derivatives.

## From spectrum to parameters

COPRA descends directly on $\tilde E$. The quasi-Newton solvers instead optimise a
real **parameter vector** $u$ via a {class}`~croak.pulses.Pulse`
({class}`~croak.pulses.ArrayPulse`: amplitude then phase). The last link of the
chain is therefore the pulse's own vector–Jacobian product,
{meth}`~croak.pulses.Pulse.ew_vjp`, which maps the spectrum cotangent
$\bar{\tilde E}$ to the parameter gradient $\partial L/\partial u$. Regularisation
penalties contribute closed-form gradients added at this stage.

## Automatic differentiation — the general route

{mod}`croak.forward_jax` expresses the *same* forward map in JAX, and
{mod}`croak.metrics_jax` / {mod}`croak._jax_pulse` mirror the metrics and
parameterisation. From these JAX produces gradients automatically, used by
{class}`~croak.lbfgs_ad.LBFGSAD` and {class}`~croak.optimistix_lbfgs.OptxLBFGS`
(reverse-mode gradients), {class}`~croak.lm.LM` and
{class}`~croak.optimistix_lm.OptxLM` (forward-mode Jacobians,
{external+jax:func}`jax.jacfwd`), and {class}`~croak.cmaes.CMAES` (which needs no derivatives
but relies on the same jitted model to evaluate a population at once).

A substantial part of what croak can do depends on the model being
differentiable:

- **Geometric time smearing.** Averaging over a distribution of internal delays
  means the measurement is no longer the intensity of one local signal field.
  This breaks COPRA's magnitude-replacement projection and the hand-written
  adjoint, so croak implements smearing through the JAX model. See [Geometric smearing](../howto/geometric_smearing.md).
- **Fitting parameters beyond the field**: the medium thickness (`fit_thickness`),
  the delay-zero offset (`fit_tau0`) and the smearing width (`fit_smearing`) all
  become just more coordinates to differentiate with respect to. See
  [Fitting thickness and τ₀](../howto/fitting_thickness_tau0.md).
- **Alternative parameterisations** — the B-spline phase basis and the temporal
  penalty (`reg_time`) — which are defined in the JAX pulse module.
- **Levenberg–Marquardt**, which needs a full residual Jacobian, not a gradient.
- **Laplace/covariance uncertainty**, which needs that same Jacobian. It works on
  results produced by NumPy solvers when their model and parameterisation are
  supported by the reconstructed JAX residual. See [Uncertainty](../howto/uncertainty.md).
- **New interactions.** Once a kernel has been implemented in the JAX forward
  model, AD supplies its derivatives. Registration in the NumPy interaction
  registry alone does not implement this path; see
  [extending interactions](../developer_guide/extending.md).

## The hand-derived adjoint — a fast specialisation

For the core model croak also carries the closed-form Wirtinger adjoint derived
above. It covers **all three thin-medium interactions and dispersive PG/SD**, with the
pointwise parameterisation, phase-only mode, per-frequency `R_omega` scaling and
the amplitude/phase/spectral penalties. It does *not* cover the smearing kernel or
the extra fitted scalars — the specialisation is in model features and
parameterisation, not in optical physics.

The analytic route has a predictable cost without JAX compilation in the NumPy
solvers. Compiled variants can reduce repeated evaluation costs, but relative
speed depends on grid size, model, hardware and compilation reuse. Derivative
choice alone does not establish which solver is fastest.

## Why keep both

Independent derivative calculations provide useful numerical checks. Agreement
between AD, analytic adjoints and finite differences checks the implemented map;
it does not validate the physical assumptions of that map. Finite differences
also have step-size and round-off limits. The
[developer validation guide](../developer_guide/validation.md) describes the
repository checks; [scientific validation](validation.md) distinguishes them
from comparisons with independently generated instrument traces.

## Next

- [The forward model](forward_model.md) — the map being differentiated.
- [Interactions](interactions.md) — where `signal_adjoint` lives.
- [Retrieval as least squares](retrieval_theory.md) — what the gradient descends.
