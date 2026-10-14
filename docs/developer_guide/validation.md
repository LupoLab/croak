# Numerical validation

Check a numerical change at the level where its assumptions enter. A successful
retrieval alone cannot establish that a derivative, physical model, or data
convention is correct.

- **Transforms and grids:** check normalisation, axis ordering, signs, and
  round trips against the [Fourier convention](../reference/fft_convention.md).
- **Interactions and gradients:** compare finite differences, analytic
  adjoints, and automatic differentiation where each route is available.
  Use explicit tolerances and test non-trivial complex fields.
- **Forward models:** test limiting cases and quadrature convergence while
  holding the input field and instrument parameters fixed.
- **Retrievals:** use synthetic known-field cases and distinguish trace error
  from field error. Account for the measurement's unobservable phase and time
  origin consistently.
- **Workflows:** check units, coordinate transformations, persistence, and
  agreement between the GUI and headless paths.

Keep tests deterministic and independent of network access. Physical-validation
claims also need the data convention, metric, and tested parameter range. See
[interpreting results](../user_guide/results.md) for the scientific context and
[the environment guide](environment.md) for the required repository checks.
