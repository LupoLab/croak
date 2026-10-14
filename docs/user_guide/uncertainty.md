# Uncertainty

An error bar depends on what is allowed to vary: measurement noise, the
sampling of the trace, or a calibrated instrument parameter such as the
substrate thickness. croak provides estimators for each and reports the result
on the retrieved FWHM, optionally at a different beamline point. Variation
between solver starts is a convergence diagnostic, not a measurement
uncertainty.

## Which page do I need?

| I want to… | Read |
|---|---|
| run a bootstrap, choose a method and report the interval | [Estimating FWHM uncertainty](../howto/uncertainty.md) |
| propagate a substrate-thickness calibration through a dispersive retrieval | {ref}`Substrate-thickness systematic <substrate-thickness-systematic>` |
| get a fast analytic estimate without re-retrieving | {ref}`Analytic covariance <analytic-covariance>` |
| understand the statistics and their assumptions | [Uncertainty of the retrieved pulse duration](../explanation/uncertainty_estimation.md) |

```{toctree}
:maxdepth: 1

../howto/uncertainty
../explanation/uncertainty_estimation
```

## Key concepts

The **parametric bootstrap** simulates new traces from the best fit plus a
noise model and re-retrieves each; the **resampling bootstraps** resample delay
columns or frequency rows. The **thickness** method is a systematic: it
propagates a measured thickness prior rather than noise. The **analytic
covariance** linearises the fit around its minimum. Independent contributions
combine by convolving their sampling distributions, which reduces to addition in
quadrature in the Gaussian limit. Coverage calibration on synthetic data
tests whether a nominal 68 % interval really covers the truth 68 % of the time.

## Main entry points

- {func}`~croak.uncertainty.estimate_fwhm_uncertainty`: one call for every
  bootstrap method.
- {class}`~croak.uncertainty.NoiseModel`, {func}`~croak.uncertainty.noise_from_residual`:
  the noise model for the parametric bootstrap.
- {func}`~croak.covariance.covariance_uncertainty`: the linearised covariance.
- {func}`~croak.uncertainty.combine_uncertainties`,
  {func}`~croak.uncertainty.coverage_calibration`: combining and calibrating.

The [thickness-uncertainty example](../examples/index.md#thickness-uncertainty)
propagates a measured calibration uncertainty. Then
[save the analysis](workflows.md) with its settings.
