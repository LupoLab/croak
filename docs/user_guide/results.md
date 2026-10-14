# Interpreting results

A retrieval returns a complex field, a predicted trace and a convergence
history. Interpreting it means inspecting the measured and predicted traces and
their residual together with the temporal and spectral profiles, settling the
trivial ambiguities, and checking that summary numbers such as the FWHM
describe the pulse you think they do. This chapter also explains how croak's
models were validated against simulated instruments whose pulse is known.

## Which page do I need?

| I want to… | Read |
|---|---|
| plot a retrieval, compute durations, phases and peak power | [Post-processing and plotting results](../howto/postprocessing.md) |
| resolve the SHG direction-of-time ambiguity | [SHG: settle the direction of time first](../howto/postprocessing.md#shg-settle-the-direction-of-time-first) |
| check whether retrieved spectral wings are real | [Is the spectrum real](../howto/postprocessing.md#is-the-spectrum-real) |
| know how well each model recovers a known pulse | [Validation](../explanation/validation.md) |

```{toctree}
:maxdepth: 1

../howto/postprocessing
../explanation/validation
```

## Key concepts

The trace error $R$ measures agreement with the model, not with the true
pulse; on simulated data the known-truth errors measure the latter. A FWHM is
one summary of a pulse: it says nothing about satellites or phase. The
**edge-energy fraction** (spectral energy near the edges of the retrieval band)
flags spectral wings that are artefacts of the retrieval rather than of the
pulse. Durations are only comparable between fields defined at the same plane.

## Main entry points

- {func}`~croak.processing.process_result`: durations, phases, Taylor
  coefficients and residuals in one {class}`~croak.processing.ProcessedResult`.
- {func}`~croak.plotting.plot_retrieval`: the twelve-panel summary figure.
- {func}`~croak.processing.resolve_time_direction`: settle the SHG time ambiguity.
- {func}`~croak.processing.edge_energy_fraction`, {func}`~croak.processing.post_filter`:
  spectral-edge diagnostics and windowing.
- {func}`~croak.truth_metrics.truth_errors`: errors against a known pulse.

The [first retrieval](../tutorials/01_first_retrieval.md) introduces the summary
figure, and the [simulation examples](../examples/index.md#instrument-effects-and-validation)
compare retrievals with a known field. Continue with
[propagation and dispersion tuning](dispersion.md) or [uncertainty](uncertainty.md).
