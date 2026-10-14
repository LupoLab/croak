# Modelling the measurement

The forward model predicts the recorded trace for a trial pulse. Choose its
physical assumptions before interpreting a retrieval: a small residual measures
agreement with that model, and does not by itself establish that the model
represents your instrument.

Start with the thin-medium and dispersive-slab models, then consider crossed-beam
arrival times, chromatic focusing, and the collection aperture. The detailed
pages include the equations, approximations, and quadrature choices.

```{toctree}
:maxdepth: 1

../explanation/forward_model
../howto/geometric_smearing
../howto/collection_aperture
```

The [dispersive retrieval tutorial](../tutorials/03_dispersive_retrieval.md)
introduces propagation through a slab. Explore further
[instrument examples](../examples/index.md#instrument-effects-and-validation),
or consult [materials and mirrors](../howto/materials_and_mirrors.md) for optical
data. Continue with [retrieval and numerical methods](retrieval.md).
