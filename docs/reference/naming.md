# The name

**croak** stands for *Complete Retrieval by Optimisation and Adjoint Kernels*.

The name describes the aim of recovering spectral amplitude and phase using a
physical measurement model and optimisation. It is not a guarantee of unique
retrieval or complete knowledge of the instrument. The field reference,
unobservable phase and time origin, and any interaction-specific ambiguities
still need to be stated.

The [adjoint derivation](../explanation/gradients.md) gives analytic derivatives
for the core kernels; automatic differentiation supplies derivatives for the
implemented JAX models, including spatial and fitted-parameter extensions.

## Why it used to be called toad

For as long as it was an internal tool the package was called **toad** — *Trace
Optimisation through Automatic Differentiation*. That acronym described what was
then the headline feature: the forward model is differentiable end to end, so a
JAX twin of every solver can be built and used to check the analytic gradients
against automatic differentiation.

The name could not be kept for the public release, because `toad` was already
taken on the Python Package Index. Rather than decorate it — `toad-frog`,
`pytoad`, and similar — the package was renamed outright.

The public package uses both analytic adjoints and automatic differentiation.
These are complementary ways to evaluate derivatives and check numerical
implementations.

## The mascot

The mascot was not renamed. The wizard's icon is still a toad, which — given what
toads are known for — remains approximately correct.

```{note}
Nothing in the public API ever carried the old name, so there is no compatibility
shim to be aware of: the package is imported as `croak`, the console scripts are
`croak` and `croak-gui`, and saved result files record a `croak_version` header.
Files written by the pre-release `toad` package are not read by this one.
```
