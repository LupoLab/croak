# Retrieval

The {func}`~croak.retrieve.retrieve` entry point, the shared solver base, the ten
algorithms ({data}`croak.retrieve.ALGORITHMS`), the result object, the live
progress reporter and the trace-error metrics. See
[Retrieval algorithms](../../explanation/algorithms.md) for how each solver works
and [Choosing a solver](../../howto/solver_selection.md) for when to use which.

## The `retrieve` wrapper

```{eval-rst}
.. automodule:: croak.retrieve
   :members:
```

## Solver base

```{eval-rst}
.. automodule:: croak.solver
   :members:
```

## COPRA

```{eval-rst}
.. automodule:: croak.copra
   :members:
```

## COPRA (JAX)

The `copra-jax` twin: identical algorithm, per-iteration numerics vectorised with
`vmap`/`jit` and `lax.scan`.

```{eval-rst}
.. automodule:: croak.copra_jax
   :members:
```

## L-BFGS (analytic gradients)

```{eval-rst}
.. automodule:: croak.lbfgs
   :members:
```

## L-BFGS (hand-gradient JAX)

The `lbfgs-hand` twin: the same analytic Wirtinger gradient as
{class}`~croak.lbfgs.LBFGS`, but jitted and `vmap`-ed in JAX (no autodiff).

```{eval-rst}
.. automodule:: croak.lbfgs_hand
   :members:
```

## L-BFGS (autodiff)

```{eval-rst}
.. automodule:: croak.lbfgs_ad
   :members:
```

## L-BFGS (Optimistix)

The `lbfgs-optx` twin: the same JAX objective as {class}`~croak.lbfgs_ad.LBFGSAD`,
driven by Optimistix's on-device L-BFGS instead of NLopt.

```{eval-rst}
.. automodule:: croak.optimistix_lbfgs
   :members:
```

## Levenberg–Marquardt

```{eval-rst}
.. automodule:: croak.lm
   :members:
```

## Levenberg–Marquardt (Optimistix)

The `lm-optx` twin: a genuine JAX-Jacobian Levenberg–Marquardt running entirely
on-device, sidestepping the MINPACK reentrancy issue of {class}`~croak.lm.LM`.

```{eval-rst}
.. automodule:: croak.optimistix_lm
   :members:
```

## CMA-ES (global)

The `cma-es` evolution-strategy retriever (CMA-ES / sep-CMA / DE via
[evosax](https://github.com/RobertTLange/evosax)) for global search from poor
initial guesses, typically paired with the B-spline phase basis. Requires the
optional `croak[evo]` extra.

```{eval-rst}
.. automodule:: croak.cmaes
   :members:
```

## Warm-started L-BFGS

The default `warm-lbfgs` retriever: {class}`~croak.lbfgs_ad.LBFGSAD` on the full
extended model, started from a COPRA local-projection sweep rather than from the
raw initial guess. Same forward model and same minimum as `lbfgs-ad`; what
changes is how much the answer depends on where it started.

```{eval-rst}
.. automodule:: croak.warm_lbfgs
   :members:
```

## Result

```{eval-rst}
.. automodule:: croak.result
   :members:
```

## Progress reporting

The live progress bar and end-of-retrieval summary, installed automatically by
{func}`~croak.retrieve.retrieve` (controlled by its `progress` argument) unless you
pass your own `callback`.

```{eval-rst}
.. automodule:: croak.progress
   :members:
```

## Metrics

```{eval-rst}
.. automodule:: croak.metrics
   :members:
```
