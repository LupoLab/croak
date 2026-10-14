# Uncertainty

An uncertainty estimate depends on what is allowed to vary: measurement noise,
sampling, or a calibrated instrument parameter such as substrate thickness.
Choose an estimator that represents the uncertainty you want to quantify and
state its assumptions when reporting the result. Variation between solver
starts is a convergence diagnostic and is not a measurement uncertainty.

```{toctree}
:maxdepth: 1

../howto/uncertainty
../explanation/uncertainty_estimation
```

The [thickness-uncertainty example](../examples/index.md#thickness-uncertainty)
illustrates propagation of a measured calibration uncertainty. Review
[interpreting results](results.md) alongside any error bar, then
[save the analysis](workflows.md) with its settings.
