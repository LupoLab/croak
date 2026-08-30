# References

## Companion paper

The theory, dispersive forward model, solvers and validation in this
documentation are described in:

> J. C. Travers and C. Brahms, *Extreme ultrashort pulse retrieval with
> differentiable physical forward models* (to be published).

Please cite it if you use croak in published work. The software itself is
archived on Zenodo: [doi:10.5281/zenodo.22182305](https://doi.org/10.5281/zenodo.22182305)
(all versions; see the repository's `CITATION.cff`). See the project README for
the paper's BibTeX entry and DOI once published.

## Foundational work

- N. C. Geib, M. Zilk, T. Pertsch, and F. Eilenberger, "Common pulse retrieval
  algorithm: a fast and universal method to retrieve ultrashort pulses,"
  *Optica* **6**, 495–505 (2019). — the COPRA algorithm and the least-squares /
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
  2400–2407 (2003). — the bootstrap (data-resampling) method for FROG error bars.
- B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap* (Chapman &
  Hall, 1993). — the parametric, nonparametric and residual bootstrap.
- T. J. DiCiccio and B. Efron, "Bootstrap confidence intervals," *Statist. Sci.*
  **11**, 189–228 (1996). — BCa (bias-corrected and accelerated) intervals for
  skewed estimators such as the FWHM.
- G. A. F. Seber and C. J. Wild, *Nonlinear Regression* (Wiley, 1989). — the
  $\hat\sigma^2 (J^\top J)^{-1}$ covariance (Laplace / Cramér–Rao) of a nonlinear
  least-squares fit.

## Related techniques

- M. Miranda *et al.*, "Simultaneous compression and characterization of
  ultrashort laser pulses using chirped mirrors and glass wedges" (d-scan),
  *Opt. Express* **20**, 688–697 (2012).
- D. Spangenberg, E. Rohwer, M. H. Brügmann, and T. Feurer, "Ptychographic
  ultrafast pulse reconstruction" (time-domain ptychography),
  *Opt. Lett.* **40**, 1002–1005 (2015).

## Software and data sources

- [Luna.jl](https://github.com/LupoLab/Luna.jl) — an open-source (MIT) Julia
  package for nonlinear optical pulse propagation. croak's Sellmeier
  coefficients, gas-density scaling and bundled chirped-mirror tables are taken
  from Luna's `PhysData` module, and {func}`~croak.io.read_simulated_scan` reads
  the HDF5 files written by Luna's `scansave`.
- [refractiveindex.info](https://refractiveindex.info) — the tabulated
  refractive-index database reachable from {mod}`croak.refractive_db` (optional
  `ridb` extra).
