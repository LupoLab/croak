# Gradients: automatic differentiation and the Wirtinger adjoint

Gradient-based retrieval needs the derivative of the trace error with respect to
the complex spectrum. croak's forward model is written to be **differentiable end
to end in JAX**, so automatic differentiation is the general route: it is what
lets the package fit parameters beyond the field itself, carry an instrument
response through the model, and take on a new interaction before anyone has
derived its adjoint by hand.

For the core model croak *also* carries a **hand-derived Wirtinger adjoint**,
used by the pure-NumPy {class}`~croak.copra.COPRA` and {class}`~croak.lbfgs.LBFGS`
solvers. The two routes are cross-checked against each other to ~1e-6 on every
interaction, thin and dispersive (`tests/test_ad_vs_analytic.py`).

This page explains both, what each covers, and why the package keeps both.

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
adjoint in croak returns this conjugate cotangent.

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
so a gradient costs **one extra forward pass** — the same asymptotic cost as
reverse-mode AD.

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
{func}`jax.jacfwd`), and {class}`~croak.cmaes.CMAES` (which needs no derivatives
but relies on the same jitted model to evaluate a population at once).

A substantial part of what croak can do depends on the model being
differentiable:

- **Geometric time smearing.** Averaging over a distribution of internal delays
  means the measurement is no longer $|\psi|^2$ of a *single* signal field, which
  breaks both COPRA's magnitude-replacement projection and the hand-written
  adjoints. See [Geometric smearing](../howto/geometric_smearing.md).
- **Fitting parameters beyond the field**: the medium thickness (`fit_thickness`),
  the delay-zero offset (`fit_tau0`) and the smearing width (`fit_smearing`) all
  become just more coordinates to differentiate with respect to. See
  [Fitting thickness and τ₀](../howto/fitting_thickness_tau0.md).
- **Alternative parameterisations** — the B-spline phase basis and the temporal
  penalty (`reg_time`) — which are defined in the JAX pulse module.
- **Levenberg–Marquardt**, which needs a full residual Jacobian, not a gradient.
- **Laplace/covariance uncertainty**, which needs that same Jacobian. It works on
  *any* result, including one produced by a NumPy solver, because the Jacobian is
  rebuilt post hoc from the JAX model. See [Uncertainty](../howto/uncertainty.md).
- **New interactions.** A new geometry needs only its `signal`; JAX differentiates
  it, so you can prototype and retrieve *before* deriving a closed-form adjoint —
  or without ever deriving one.

## The hand-derived adjoint — a fast specialisation

For the core model croak also carries the closed-form Wirtinger adjoint derived
above. It covers **all three interactions, thin or dispersive**, with the
pointwise parameterisation, phase-only mode, per-frequency `R_omega` scaling and
the amplitude/phase/spectral penalties. It does *not* cover the smearing kernel or
the extra fitted scalars — the specialisation is in model features and
parameterisation, not in optical physics.

What it buys: exact gradients for the cost of one extra forward pass, with a
predictable, JIT-free cost — and, because it is pure NumPy, `copra` and `lbfgs`
run without JAX touching the hot path at all.

What it does *not* buy is raw speed. A well-written reverse pass and a good AD
backend land on the same operation count, and in practice the trace evaluation
dominates the gradient maths: {class}`~croak.lbfgs_hand.LBFGSHand`, which uses the
*analytic* gradient jitted and `vmap`-ed in JAX, is faster than either the NumPy
`lbfgs` or the autodiff `lbfgs-ad`. The speed-up comes from JIT and vectorisation,
not from the choice of gradient.

## Why keep both

Because they check each other. The analytic adjoint is easy to get subtly wrong,
and AD is an independent implementation of the same derivative: running `lbfgs`
against `lbfgs-ad` on the same trace is a continuous regression test on the
hand-derived maths. `tests/test_ad_vs_analytic.py` asserts agreement to ~1e-6
across all three interactions, thin and dispersive, phase-only and regularised —
AD has no step-size truncation error, so the comparison is far sharper than a
finite-difference check. The companion paper (see [Validation](validation.md))
derives the complete reverse pass at tutorial density, from the definition of
the Wirtinger derivatives to the per-interaction adjoint rules above; both
routes are also checked there against central finite differences at the
5×10⁻¹⁰ level.

## Next

- [The forward model](forward_model.md) — the map being differentiated.
- [Interactions](interactions.md) — where `signal_adjoint` lives.
- [Retrieval as least squares](retrieval_theory.md) — what the gradient descends.
