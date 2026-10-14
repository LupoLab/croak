# Propagation and dispersion tuning

A retrieved field refers to a particular plane in the optical system. Applying
known material and mirror dispersion moves it to another plane: back to the
source through the beam-path optics, or forward through a compressor to find the
glass thickness or chirped-mirror bounce count that shortens the pulse. This is
post-processing of a retrieved field, distinct from including the nonlinear
medium in the retrieval's [forward model](models.md).

## Which page do I need?

| I want to… | Read |
|---|---|
| add or remove glass, gas or a custom material | [Materials and mirrors](../howto/materials_and_mirrors.md) |
| model chirped-mirror bounces or back-propagate coating mirrors | [Chirped mirrors](../howto/materials_and_mirrors.md#chirped-mirrors) |
| auto-tune a compressor for the shortest pulse | [Dispersion tuning tutorial](../tutorials/05_dispersion_tuning.md) |

```{toctree}
:maxdepth: 1

../howto/materials_and_mirrors
```

## Key concepts

Dispersion is a spectral phase: a material of length $L$ adds $\beta(\omega)L$,
its propagation constant with the group-delay term removed so the pulse stays
centred, and a mirror adds its tabulated phase per bounce (and, for a coating,
its reflectivity). Removing an element applies the negative phase. Taylor
coefficients (GDD, TOD) describe the residual phase near the carrier; a
broadband pulse may need the full material phase instead.

## Main entry points

- {func}`~croak.dispersion.apply_dispersion`: apply materials, Taylor terms and
  mirrors to a spectrum.
- {func}`~croak.dispersion.auto_material`, {func}`~croak.dispersion.auto_taylor`:
  optimise a compressor for peak power.
- {func}`~croak.materials.register_material`, {func}`~croak.dispersion.register_mirror`:
  add custom optical data.
- {func}`~croak.gases.gas_refractive_index`: pressure- and temperature-scaled gases.

The [workflow API](../reference/api/workflow.md#dispersion-tuning) lists the
remaining functions. Continue with [uncertainty](uncertainty.md), which can
report an error bar at another beamline point.
