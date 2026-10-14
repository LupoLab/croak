# Preparing and checking data

Before retrieving, establish what the file records: the signal, its spectral
coordinate and units, the delay axis and its sign, and any independently
measured spectrum. Then remove what the model does not describe (fringes,
background, out-of-band noise) and put the trace on the retrieval grid. This
chapter covers measured files, numerically simulated scans, and the marginal
checks that test a trace for consistency before any retrieval is run.

## Which page do I need?

| I want to… | Read |
|---|---|
| load a measured trace and clean, window and regrid it | [Cleaning and regridding measured data](../howto/preprocessing.md) |
| remove interferometric fringes or an asymmetric baseline | [Cleaning and regridding measured data](../howto/preprocessing.md) |
| load a simulated scan and score a retrieval against its known pulse | [Loading numerically simulated traces](../howto/loading_simulated.md) |
| test a trace against an independently measured spectrum | [Marginal consistency checks](../howto/marginal_checks.md) |
| understand why the marginals constrain the trace | [FROG trace marginals](../explanation/marginals.md) |

```{toctree}
:maxdepth: 1

../howto/preprocessing
../howto/loading_simulated
../howto/marginal_checks
../explanation/marginals
```

## Key concepts

Preprocessing produces a {class}`~croak.preprocess.TraceData`: a normalised
trace on a uniform frequency and delay grid, with the carrier and the delay
origin recorded. Conversion from a wavelength axis applies the
$\lambda^2$ density Jacobian. The **frequency marginal** of a trace is predicted
by the pulse spectrum (exactly for SHG, as a transform-limited estimate for
PG/TG/SD), so a mismatch flags a calibration or spectral-response problem before
it biases the retrieval. A **third-order efficiency exponent** corrects a
wavelength-dependent detection efficiency.

## Main entry points

- {func}`~croak.io.list_datasets`, {func}`~croak.io.unit_to_si`: pick datasets
  from an HDF5 or NPZ file.
- {func}`~croak.preprocess.load_and_clean`: the whole cleaning pipeline in one call.
- {func}`~croak.preprocess.defringe_carrier`, {func}`~croak.preprocess.arpls_baseline`:
  the individual cleaning steps.
- {func}`~croak.io.read_simulated_scan`: read a simulated scan with its truth fields.
- {func}`~croak.marginal_checks.predicted_marginal`,
  {func}`~croak.marginal_checks.auto_third_order_exponent`: marginal checks.

The [experimental workflow](../tutorials/04_experimental_workflow.md) runs the
sequence end to end, and the [data preparation examples](../examples/index.md#preparing-data)
show fringe and baseline removal. Next, [model the measurement](models.md).
