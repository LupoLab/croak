# Modelling the measurement

The forward model predicts the recorded trace for a trial pulse, and the
retrieval returns the pulse that best explains the data *under that model*. A
small residual shows that the model reproduces the trace; it does not show that
the model describes the instrument. For few-femtosecond and deep-ultraviolet
pulses the difference is large: on simulated deep-ultraviolet TG-FROG scans the
thin-medium model fits the trace to a residual below 0.54 % yet returns
1.35–2.41 fs for a 1.05 fs pulse ([validation](../explanation/validation.md)). This chapter describes the
hierarchy of models croak provides and when each is needed.

## Which page do I need?

| My instrument has… | Read |
|---|---|
| a nonlinear medium thick enough to disperse the pulse (PG, SD) | [The forward model](../explanation/forward_model.md) |
| crossed beams (BOXCARS) whose arrival times vary across the focus | [Modelling geometric time smearing](../howto/geometric_smearing.md) |
| a collection aperture that clips the signal beam chromatically | [Modelling the collection aperture](../howto/collection_aperture.md) |
| an uncertain thickness or delay zero to fit | [Fitting the medium thickness and delay-zero](../howto/fitting_thickness_tau0.md) |

```{toctree}
:maxdepth: 1

../explanation/forward_model
../howto/geometric_smearing
../howto/collection_aperture
```

## Key concepts

The models form a ladder. The **standard** model is the thin, local
interaction. The **dispersive slab** propagates the input and signal fields
through a medium of given material and thickness (PG and SD only). The
**extended** model adds geometric delay smearing as a kernel set by the mask
geometry. The **full** model replaces the kernel with an explicit chromatic
focal quadrature and a modelled collection aperture. Each step costs more, and
the full model benefits from a GPU. The retrieved field refers to the plane the
model defines; compare fields only at the same plane.

## Main entry points

- {class}`~croak.forward.ForwardModel`: the forward model, with material,
  thickness and depth quadrature.
- {func}`~croak.smearing.square_boxcars_kernel`, {class}`~croak.smearing.SmearingKernel`:
  geometric smearing from the mask geometry.
- {func}`~croak.focal.focal_mixture`: the chromatic focal quadrature.
- {func}`~croak.collection.mask_hole_aperture`, {func}`~croak.collection.aperture_from_scan`:
  the collection aperture.

The [dispersive retrieval tutorial](../tutorials/03_dispersive_retrieval.md)
introduces the slab model, and the [instrument examples](../examples/index.md#instrument-effects-and-validation)
compare the models on simulated scans. Optical data for the slab are in
[materials and mirrors](../howto/materials_and_mirrors.md). Continue with
[retrieval and numerical methods](retrieval.md).
