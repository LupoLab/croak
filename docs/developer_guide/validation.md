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

(collection-reference-tests)=
## Collection reference tests

Two properties are worth asserting on your own geometry, and both are in
`tests/test_collection.py`:

* **Parseval.** On a Cartesian focal grid with an unbounded aperture the model must
  return the incoherent sum exactly. Getting the $1/(2\pi)^2$ or the area weights wrong
  shows up here and nowhere else.
* **A dense 2-D FFT.** Building the focal field from the arm tilts directly, taking a
  plain transverse FFT and windowing it in k reproduces the quadrature model with no
  fitted scale (asserted at 2e-3 of peak at the default test resolution; the 3e-7
  figure quoted elsewhere is the mixture-quadrature convergence at `n_azimuth=16`,
  a different check). The per-depth variant of the same reference — exact
  $k_z(\omega,k_\perp)$ propagation of every depth node's signal to the slab exit —
  validates `depth_transverse` to 1.4–5.7e-5 of peak and pins its sign.

## Depth-dependent focal phases

The low-level {func}`croak.forward_jax.make_trace_fn` interface accepts
`depth_node_phase`, complex phase factors with shape `(focal.nodes, npoints)`.
This is a diagnostic hook for testing a prescribed depth-dependent focal phase;
it is not an automatically inferred material or geometry correction. Keep the
field, quadratures and other settings fixed when comparing it with a reference
calculation. For a supported physical propagation option, see
[depth-dependent transverse propagation](../howto/collection_aperture.md).
