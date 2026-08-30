# API reference

Complete reference for every public module in croak, generated from the
NumPy-style docstrings. The pages group the package by role; the
[quickstart](../../getting_started/quickstart.md) and
[how-to guides](../../howto/preprocessing.md) show these APIs in context.

The headline API is importable from the top-level `croak` namespace
(e.g. `croak.Grid`, `croak.retrieve`, `croak.load_and_clean`); the submodules
(`croak.maths`, `croak.materials`, `croak.metrics`, …) are also exposed directly. The
individual solver classes are normally selected by name through
{func}`croak.retrieve` rather than imported — the registry
{data}`croak.retrieve.ALGORITHMS` maps each `algorithm` string to its class.

```{toctree}
:maxdepth: 2

core
retrieval
workflow
plotting
jax
```

## Map of the package

| Area | Modules |
|------|---------|
| [Core physics](core.md) | {mod}`~croak.grid`, {mod}`~croak.pulses`, {mod}`~croak.interactions`, {mod}`~croak.forward`, {mod}`~croak.smearing`, {mod}`~croak.focal`, {mod}`~croak.collection`, {mod}`~croak.materials`, {mod}`~croak.gases`, {mod}`~croak.maths`, {mod}`~croak.constants` |
| [Retrieval](retrieval.md) | {mod}`~croak.retrieve`, {mod}`~croak.solver`, {mod}`~croak.copra`, {mod}`~croak.copra_jax`, {mod}`~croak.lbfgs`, {mod}`~croak.lbfgs_hand`, {mod}`~croak.lbfgs_ad`, {mod}`~croak.optimistix_lbfgs`, {mod}`~croak.lm`, {mod}`~croak.optimistix_lm`, {mod}`~croak.cmaes`, {mod}`~croak.warm_lbfgs`, {mod}`~croak.result`, {mod}`~croak.progress`, {mod}`~croak.metrics` |
| [Workflow](workflow.md) | {mod}`~croak.io`, {mod}`~croak.preprocess`, {mod}`~croak.pipeline`, {mod}`~croak.processing`, {mod}`~croak.dispersion`, {mod}`~croak.mirrors`, {mod}`~croak.refractive_db`, {mod}`~croak.save`, {mod}`~croak.uncertainty`, {mod}`~croak.covariance` |
| [Plotting](plotting.md) | {mod}`~croak.plotting` |
| [JAX backend](jax.md) | {mod}`~croak.forward_jax`, {mod}`~croak.metrics_jax`, {mod}`~croak._jax_pulse` |
