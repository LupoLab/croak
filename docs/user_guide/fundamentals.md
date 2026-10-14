# Measurement fundamentals and conventions

A FROG measurement records the spectrum of a nonlinear signal as the delay
between two copies of the pulse is scanned. croak treats this as one member of
the PNPS (parameterised nonlinear process spectra) family: a pulse, a scan
parameter and a signal operator define the trace, and retrieval inverts that
map. This chapter sets out the framework, the three supported interactions,
and the units, grids and transform conventions every later calculation uses.

## Which page do I need?

| I want to… | Read |
|---|---|
| understand what the trace encodes and its ambiguities | [The PNPS framework](../explanation/pnps_framework.md) |
| choose between SHG, PG/TG and SD, or see their signal operators | [Interactions](../explanation/interactions.md) |
| check units, delay sign or phase sign | [Conventions](../reference/conventions.md) |
| understand the discrete Fourier transform and its normalisation | [FFT convention](../reference/fft_convention.md) |
| build a grid, choose a wavelength band or handle the SHG carrier | [Grids, wavelength ranges and the SHG carrier](../howto/grids_and_filtering.md) |

```{toctree}
:maxdepth: 1

../explanation/pnps_framework
../explanation/interactions
../reference/conventions
../reference/fft_convention
../howto/grids_and_filtering
```

## Key concepts

All quantities passed to croak are in SI units: seconds, radians per second and
metres. Spectra live on a **baseband** angular-frequency grid, offset from a
carrier `omega0`; the absolute frequency is `grid.omega + omega0`. The
**interaction** (`"shg"`, `"pg"` or `"sd"`) fixes the signal operator and hence
the trivial ambiguities: SHG cannot tell the direction of time, while PG and SD
can. A trace is an array of shape `(Nomega, Ndelay)`.

## Main entry points

- {class}`~croak.grid.Grid`, {func}`~croak.grid.gridparams_lambda`: matched time
  and frequency axes.
- {func}`~croak.pulses.gaussian_pulse`, {func}`~croak.pulses.sech_pulse`: test
  pulses with a stated intensity FWHM and spectral phase.
- {func}`~croak.forward.maketrace`, {func}`~croak.interactions.get_interaction`:
  predict a trace for a pulse and interaction.

For a calculation you can follow step by step, see
[your first retrieval](../tutorials/01_first_retrieval.md). Continue with
[preparing and checking data](data.md).
