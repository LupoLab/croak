# The name

**croak** stands for *Complete Retrieval by Optimisation and Adjoint Kernels*.

*Complete* is the substantive claim. Retrieval returns the full complex field —
amplitude *and* phase — and, in the dispersive geometries, the field as it
propagates inside the nonlinear medium rather than a thin-medium approximation to
it. *Adjoint kernels* names the machinery: each interaction (SHG, SD, PG)
contributes a kernel to the forward map, and each carries a matching hand-derived
[Wirtinger adjoint](../explanation/gradients.md) that yields exact gradients for
the cost of one extra forward pass.

## Why it used to be called toad

For as long as it was an internal tool the package was called **toad** — *Trace
Optimisation through Automatic Differentiation*. That acronym described what was
then the headline feature: the forward model is differentiable end to end, so a
JAX twin of every solver can be built and used to check the analytic gradients
against automatic differentiation.

The name could not be kept for the public release, because `toad` was already
taken on the Python Package Index. Rather than decorate it — `toad-frog`,
`pytoad`, and similar — the package was renamed outright.

Automatic differentiation is, if anything, more central now than when it named
the package: geometric smearing, the fitted medium thickness and delay-zero,
the B-spline phase basis and the Laplace covariance all depend on it.
What changed is that it is no longer the only thing the name needs to cover. The package
grew a hand-derived Wirtinger adjoint for the core model, a complete
measurement-to-result workflow, and uncertainty quantification — and "trace
optimisation through automatic differentiation" had come to describe one component
rather than the whole.

*Adjoint kernels* covers both routes: each interaction contributes a kernel to the
forward map, and its derivative is available either in closed form or through AD.

## The mascot

The mascot was not renamed. The wizard's icon is still a toad, which — given what
toads are known for — remains approximately correct.

```{note}
Nothing in the public API ever carried the old name, so there is no compatibility
shim to be aware of: the package is imported as `croak`, the console scripts are
`croak` and `croak-gui`, and saved result files record a `croak_version` header.
Files written by the pre-release `toad` package are not read by this one.
```
