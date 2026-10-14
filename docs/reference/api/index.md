# API reference

Reference to croak's physics, retrieval and workflow interfaces, generated from
NumPy-style docstrings. The pages group the package by role; the
[quickstart](../../getting_started/quickstart.md) and
[user guide](../../user_guide/index.md) show these APIs in context. The
[examples](../../examples/index.md) provide runnable calculations, and the
[developer's guide](../../developer_guide/index.md) explains how to contribute
to the package.

The headline API is importable from the top-level `croak` namespace
(e.g. `croak.Grid`, `croak.retrieve`, `croak.load_and_clean`); the submodules
(`croak.maths`, `croak.materials`, `croak.metrics`, …) are also exposed directly. The
individual solver classes are normally selected by name through
{func}`croak.retrieve` rather than imported — the registry
{data}`croak.retrieve.ALGORITHMS` maps each `algorithm` string to its class.

Terms used throughout the manual are defined in the
[glossary](../glossary.md), and the papers and published methods behind croak are
listed in the [references](../bibliography.md). Units, sign conventions and the
FFT convention are collected under [conventions](../conventions.md).

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
| [Retrieval](retrieval.md) | {mod}`~croak.retrieve`, {mod}`~croak.solver`, {mod}`~croak.copra`, {mod}`~croak.copra_jax`, {mod}`~croak.lbfgs`, {mod}`~croak.lbfgs_hand`, {mod}`~croak.lbfgs_ad`, {mod}`~croak.optimistix_lbfgs`, {mod}`~croak.lm`, {mod}`~croak.optimistix_lm`, {mod}`~croak.cmaes`, {mod}`~croak.warm_lbfgs`, {mod}`~croak.result`, {mod}`~croak.progress`, {mod}`~croak.metrics`, {mod}`~croak.delay_origin` |
| [Workflow](workflow.md) | {mod}`~croak.io`, {mod}`~croak.preprocess`, {mod}`~croak.pipeline`, {mod}`~croak.processing`, {mod}`~croak.truth_metrics`, {mod}`~croak.marginal_checks`, {mod}`~croak.dispersion`, {mod}`~croak.mirrors`, {mod}`~croak.refractive_db`, {mod}`~croak.save`, {mod}`~croak.uncertainty`, {mod}`~croak.covariance`, {mod}`~croak.session.engine`, {mod}`~croak.cli`, {mod}`~croak.scripting` |
| [Plotting](plotting.md) | {mod}`~croak.plotting` |
| [JAX backend](jax.md) | {mod}`~croak.forward_jax`, {mod}`~croak.metrics_jax`, {mod}`~croak._jax_pulse` |
