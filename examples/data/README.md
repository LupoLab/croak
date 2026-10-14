# Example datasets

Two reduced HDF5 datasets of numerically simulated TG-FROG measurements,
generated with [ModelPNPS.jl](https://github.com/LupoLab/ModelPNPS.jl). They are simulated measurements, not
experimental data. Each is a `scansave`-format file that
`croak.io.read_simulated_scan` reads directly, cut down from the full simulation
output with [`tools/reduce_scansave.py`](../../tools/reduce_scansave.py): selected
collection windows, a few substrate thicknesses and no transverse-grid arrays.
Everything a retrieval needs is kept, including the complex spectrum of the
beamlet that gates the interaction.

Both simulate the same virtual instrument: a folded-BOXCARS mask (1.0 mm holes,
1.0 mm edge gap), an f = 100 mm focus into fused silica, and signal collection
through a physical aperture. The propagation is a full three-dimensional
simulation, so the dispersion, phase matching, geometric smearing and chromatic
collection in the traces come from the instrument physics, not from croak's
retrieval forward model.

## `tgfrog_sim_1fs_uvfs.h5` (1.3 MB)

A transform-limited 1 fs pulse at 260 nm (about 1.03 fs for the integrated
beamlet after the mask), recorded through the 0.5 mm collection hole at three
substrate thicknesses: 4, 9.5 and 40 µm. Used by
`examples/example_paper_thickness.py`, which compares the extended and standard models.

## `tgfrog_sim_rdw_duv.h5` (2.1 MB)

A simulated resonant-dispersive-wave (RDW) pulse from a hollow-capillary-fibre
source: about 1.06 fs, single-cycle, with a structured spectrum and trailing
satellites, measured by the same virtual instrument through the 2.0 mm
collection hole at 9.5 and 20 µm of fused silica. Used by
`examples/example_paper_rdw.py`, which retrieves the 9.5 µm case.

## Notes

- The delay axes have a uniform core (0.25 fs steps) and coarser wing points.
  Crop to the core (`uniform_delay_core=True` on `SimulatedLoadParams`, or
  `tau_min_fs`/`tau_max_fs` on `PreprocParams`) before using the smearing
  kernel or `delay_origin="marginal_peak"`, which need uniform delays.
- The traces are angular-frequency densities on the simulation's native grid;
  `croak.session.pipeline.assemble_simulated_load_data` applies the Jacobian
  when converting to the wavelength-domain preprocessing arrays.
- `grid/Eω_beamlet_re` and `grid/Eω_beamlet_im` store the complex spectrum of
  the mask-vignetted gate beamlet, integrated over the transverse plane, so a
  retrieval can be scored against the known field with `croak.truth_metrics`.
  The full focal model's field is the on-axis field at the slab entrance;
  convert both to the same reference before comparing temporal profiles or
  phases.
- The example scripts use a general workflow (`warm-lbfgs`, with de-fringing
  on), not the protocol of the papers' validation retrievals, so their
  durations differ from the published numbers.

See the manual's [simulated-data guide](../../docs/howto/loading_simulated.md),
[example gallery](../../docs/examples/index.md) and
[validation chapter](../../docs/explanation/validation.md).
