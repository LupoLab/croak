# Runtime findings outside the documentation change

No runtime fixes are made here. These observations require separate changes
and regression tests if addressed.

- `parameter_covariance` with an explicit noise model divides residuals by
  the noise standard deviation but does not apply the supplied residual
  weights, although those weights enter the intensity-gain fit. A zero weight
  therefore does not exclude that frequency row from the whitened covariance.
  The API and guide now state this limitation.
- Covariance cannot reproduce focal/collection, depth-weight or marginal-peak
  centred retrievals because those options are absent from its interface.
- Delay-column resampling generates repeated/nonuniform delays and cannot be
  used with a forward model requiring a uniform-delay convolution or centring.
  This is an estimator compatibility limitation, not a new implementation.
- `post_filter` replaces the spectrum while retaining the original fitted
  trace, gain and error. The documentation now explicitly identifies these as
  diagnostics of the unfiltered fit.
- The initial repository test run emitted 47 numerical warnings and
  Qt/Matplotlib deleted-canvas exceptions, detailed in `verification.md`.

- `marginal_peak_delay_jax` clips its peak index to an interior sample before
  handling an edge maximum. It therefore returns the adjacent interior delay
  for an edge peak, rather than the actual boundary sample stated by its former
  docstring. The API now describes the behaviour and recommends complete delay
  support. Fixing the boundary rule requires a separate regression-tested change.
