# Preparing and checking data

Before retrieving a pulse, check what your file records: the signal, its spectral
coordinate and units, the delay direction, and any independent pulse spectrum.
Inspect the trace before and after cleaning. Regridding and spectral-response
corrections change the representation of the data and need to match the model.

```{toctree}
:maxdepth: 1

../howto/preprocessing
../howto/loading_simulated
../howto/marginal_checks
../explanation/marginals
```

Follow the [experimental workflow](../tutorials/04_experimental_workflow.md) for
an end-to-end introduction. The [data preparation examples](../examples/index.md#preparing-data)
illustrate fringe and baseline removal. Next, choose how to
[model the measurement](models.md).
