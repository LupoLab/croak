# JAX backend

The differentiable forward model, metrics and pulse parameterisation. These mirror
the NumPy {class}`~croak.forward.ForwardModel`, {mod}`croak.metrics` and
{class}`~croak.pulses.ArrayPulse`, and back the JAX-native solvers —
{class}`~croak.lbfgs_ad.LBFGSAD`, {class}`~croak.lm.LM`,
{class}`~croak.lbfgs_hand.LBFGSHand`, {class}`~croak.copra_jax.COPRAJax`,
{class}`~croak.optimistix_lbfgs.OptxLBFGS`, {class}`~croak.optimistix_lm.OptxLM` and
{class}`~croak.cmaes.CMAES`.

This is the general gradient route: it supplies the derivatives for geometric
smearing and the fitted extras that the closed-form adjoint does not cover, the
Jacobians Levenberg–Marquardt and the Laplace covariance need, derivatives for
new interactions with no hand-written adjoint, and on-device execution of a whole
loop or population. It also serves as the independent oracle the analytic
adjoints are tested against. See [Gradients](../../explanation/gradients.md).

## JAX forward model

```{eval-rst}
.. automodule:: croak.forward_jax
   :members:
```

## JAX metrics

```{eval-rst}
.. automodule:: croak.metrics_jax
   :members:
```

## JAX pulse parameterisation

The differentiable `u → spectrum` map shared by the JAX solvers, the reduced
cubic-B-spline phase basis, and the smoothness / spectral / temporal penalties.

```{eval-rst}
.. automodule:: croak._jax_pulse
   :members:
```
