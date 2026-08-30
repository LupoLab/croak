# Example datasets

Two reduced datasets from the first-principles 3D instrument simulations used
to validate croak in the companion paper (J. C. Travers and C. Brahms,
*Extreme ultrashort pulse retrieval with differentiable physical forward
models*, to be published). Each is a `scansave`-format HDF5 file that
`croak.io.read_simulated_scan` reads directly, cut down from the full
simulation output with [`tools/reduce_scansave.py`](../../tools/reduce_scansave.py):
one collection window, a few substrate thicknesses, and no transverse-grid
arrays. Everything the retrieval needs is kept, including the exact complex
ground-truth field of the pulse that gates the interaction.

Both simulate the same virtual TG-FROG instrument: a folded-BOXCARS mask
(1.0 mm apertures, 1.0 mm edge gap), an f = 100 mm focus into fused silica,
and signal collection through a physical aperture. The propagation is a full
3D nonlinear simulation (ModelPNPS.jl), so the dispersion, phase matching,
geometric smearing and chromatic collection in the traces are real, not
imposed by any retrieval model.

## `tgfrog_sim_1fs_uvfs.h5` (1.3 MB)

A transform-limited 1 fs pulse at 260 nm (1.03 fs after the mask), recorded
through the 0.5 mm collection hole at three substrate thicknesses: 4, 9.5 and
40 µm. This is the trace behind the paper's central thickness-series result.
Used by `examples/example_paper_thickness.py` and the documentation's
validation example.

## `tgfrog_sim_rdw_duv.h5` (2.1 MB)

A simulated resonant-dispersive-wave (RDW) pulse from a hollow-capillary-fibre
source — 1.06 fs, single-cycle, with a structured spectrum and trailing
satellites — measured by the same virtual instrument through the 2.0 mm
collection hole, at 9.5 and 20 µm of fused silica. This is the paper's
final validation case. Used by `examples/example_paper_rdw.py`.

## Notes

- The delay axes have a uniform core (0.25 fs steps) and coarser wing points;
  crop to the core (`tau_min_fs`/`tau_max_fs`, or
  `uniform_delay_core=True`) before using the smearing kernel, which needs
  uniform delays.
- The traces are angular-frequency densities on the simulation's native grid;
  `croak.session.pipeline.assemble_simulated_load_data` applies the correct
  Jacobian when regridding. See the how-to guide *Loading numerically
  simulated traces* in the documentation.
- The stored `grid/Eω_beamlet_re/_im` datasets are the complex spectrum of the
  mask-vignetted gate beam — the retrievable ground truth — so retrievals can
  be scored against the known field with `croak.truth_metrics`.
