# References

(companion-paper)=
## Companion papers

- J. C. Travers and C. Brahms, "Differentiable physical models for
  ultrashort-pulse retrieval," in preparation. — the inverse problem: the
  retrieval protocol and what each forward model recovers from simulated
  TG-FROG measurements.
- J. C. Travers and C. Brahms, "A dispersive and spatial model of
  transient-grating frequency-resolved optical gating," in preparation. — the
  coherent slab, the timing reduction, chromatic focal fields, finite collection
  and the fixed-field validation of the forward model.

The [validation chapter](../explanation/validation.md) summarises their results.

If you use croak in published work, please cite the first paper, together with
the software version you used: the all-version software archive is
[doi:10.5281/zenodo.22182305](https://doi.org/10.5281/zenodo.22182305), and the
repository's `CITATION.cff` gives both citations.

## Foundational work

- N. C. Geib, M. Zilk, T. Pertsch, and F. Eilenberger, "Common pulse retrieval
  algorithm: a fast and universal method to retrieve ultrashort pulses,"
  *Optica* **6**, 495–505 (2019), [doi:10.1364/OPTICA.6.000495](https://doi.org/10.1364/OPTICA.6.000495). — the COPRA algorithm and the least-squares /
  projection analysis that croak's solvers build on.
- R. Trebino, *Frequency-Resolved Optical Gating: The Measurement of Ultrashort
  Laser Pulses* (Springer, 2000). — the standard reference on FROG, its
  geometries and the trace (FROG) error.
- D. J. Kane and R. Trebino, "Characterization of arbitrary femtosecond pulses
  using frequency-resolved optical gating," *IEEE J. Quantum Electron.* **29**,
  571–579 (1993). — the original FROG.

## Uncertainty & statistics

- Z. Wang, E. Zeek, R. Trebino, and P. Kvam, "Determining error bars in
  measurements of ultrashort laser pulses," *J. Opt. Soc. Am. B* **20**,
  2400–2405 (2003), [doi:10.1364/JOSAB.20.002400](https://doi.org/10.1364/JOSAB.20.002400). — the bootstrap (data-resampling) method for FROG error bars.
- B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap* (Chapman &
  Hall, 1993). — the parametric, nonparametric and residual bootstrap.
- T. J. DiCiccio and B. Efron, "Bootstrap confidence intervals," *Statist. Sci.*
  **11**, 189–228 (1996). — BCa (bias-corrected and accelerated) intervals for
  skewed estimators. croak implements bias correction without the acceleration
  term (BC), not BCa; see [uncertainty theory](../explanation/uncertainty_estimation.md).
- G. A. F. Seber and C. J. Wild, *Nonlinear Regression* (Wiley, 1989). — the
  $\hat\sigma^2 (J^\top J)^{-1}$ local covariance approximation of a nonlinear
  least-squares fit, with its regularity and noise assumptions.

## Related techniques

- M. Miranda *et al.*, "Simultaneous compression and characterization of
  ultrashort laser pulses using chirped mirrors and glass wedges" (d-scan),
  *Opt. Express* **20**, 688–697 (2012), [doi:10.1364/OE.20.000688](https://doi.org/10.1364/OE.20.000688).
- D. Spangenberg, E. Rohwer, M. H. Brügmann, and T. Feurer, "Ptychographic
  ultrafast pulse reconstruction" (time-domain ptychography),
  *Opt. Lett.* **40**, 1002–1005 (2015), [doi:10.1364/OL.40.001002](https://doi.org/10.1364/OL.40.001002).

## Software and data sources

- [ModelPNPS.jl](https://github.com/LupoLab/ModelPNPS.jl) — the companion
  papers' instrument simulator: an open-source Julia package (built on Luna.jl)
  that simulates PNPS measurements in three dimensions from the underlying
  propagation physics. It produced the validation datasets summarised in
  [Validation](../explanation/validation.md), including the reduced scans
  shipped in `examples/data/`.
- [Luna.jl](https://github.com/LupoLab/Luna.jl) — an open-source (MIT) Julia
  package for nonlinear optical pulse propagation. croak's Sellmeier
  coefficients, gas-density scaling and bundled chirped-mirror tables are taken
  from Luna's `PhysData` module, and {func}`~croak.io.read_simulated_scan` reads
  the HDF5 files written by Luna's `scansave`.
