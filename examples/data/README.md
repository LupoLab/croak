# Example datasets

These two reduced HDF5 datasets contain numerical TG-FROG instrument
simulations generated with ModelPNPS.jl. They are bundled with the repository
for runnable examples; they are not experimental measurements.
[`tools/reduce_scansave.py`](../../tools/reduce_scansave.py) reduces the original
`scansave` output to selected collection windows and substrate thicknesses,
retaining the trace axes, geometry and stored complex beamlet spectrum.

The virtual instrument uses a folded-BOXCARS mask with 1.0 mm holes and a
1.0 mm edge gap, a 100 mm focal length, fused silica and a physical collection
aperture. Propagation, spatial overlap and collection arise from the instrument
simulation rather than from croak's retrieval forward model. The simulation
still has physical and numerical approximations; comparison is a model test,
not an experimental accuracy guarantee.

## `tgfrog_sim_1fs_uvfs.h5`

A nominally 1 fs transform-limited input at 260 nm, recorded through the 0.5 mm
collection aperture at 4, 9.5 and 40 µm thickness. The stored integrated beamlet
has a duration near 1.03 fs. `example_paper_thickness.py` compares three retrieval
models on this reduced dataset.

## `tgfrog_sim_rdw_duv.h5`

A resonant-dispersive-wave pulse from a numerical hollow-capillary-fibre source,
with a structured spectrum and trailing satellites. The stored integrated
beamlet has a duration near 1.06 fs. The instrument traces use a 2.0 mm collection
aperture and 9.5 and 20 µm fused-silica thicknesses.
`example_paper_rdw.py` retrieves the 9.5 µm case.

## Conventions and reproducibility

- The delay axes have a uniform core with 0.25 fs spacing and coarser wings.
  Select `uniform_delay_core=True` or suitable delay limits before using a
  smearing kernel that requires uniform delays.
- Traces are angular-frequency densities on the native simulation grid.
  The simulated loader applies the coordinate Jacobian when converting to
  wavelength-domain preprocessing arrays.
- `grid/Eω_beamlet_re` and `grid/Eω_beamlet_im` store the complex spectrum of
  the transverse-integrated mask beamlet. It is not automatically the same
  field as a full model's on-axis slab-entrance field. Convert the complex
  spectrum to a common reference before comparing temporal profiles or phases.
- The examples retain legacy preprocessing and fit choices for teaching. They
  do not reproduce the adopted unfiltered, consistently centred protocol in
  the revised unpublished drafts. The revised on-axis reference durations and
  results must not be substituted into these examples without matching their
  data, field conversion, settings and sampling.

See the manual's [simulated-data guide](../../docs/howto/loading_simulated.md),
[worked gallery](../../docs/examples/index.md) and
[validation chapter](../../docs/explanation/validation.md) for interpretation.
