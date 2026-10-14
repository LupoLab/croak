# User guide

This guide follows a retrieval from the measurement to a reported pulse: what a
FROG trace records, how to prepare it, which physical model to fit, how to run
and judge a retrieval, how to interpret and propagate the result, how to put an
error bar on it, and how to keep the analysis reproducible. Each chapter
collects the practical how-to pages and the physics and numerical background
for one step.

If you are new to croak, begin with the [introduction](../introduction/index.md)
and [your first retrieval](../tutorials/01_first_retrieval.md). The
[examples](../examples/index.md) provide runnable applications of every chapter.

| Step | Chapter | Covers |
|---|---|---|
| 1 | [Measurement fundamentals](fundamentals.md) | the PNPS framework, interactions, units, grids and Fourier conventions |
| 2 | [Preparing and checking data](data.md) | loading files, cleaning and regridding, simulated scans, marginal checks |
| 3 | [Modelling the measurement](models.md) | thin and dispersive media, geometric smearing, focal and collection models |
| 4 | [Retrieval and numerical methods](retrieval.md) | solver choice, the objective, gradients, regularisation, fitted parameters |
| 5 | [Interpreting results](results.md) | plots, ambiguities, durations, spectral checks, validation |
| 6 | [Propagation and dispersion tuning](dispersion.md) | materials, gases, chirped and coating mirrors, compression |
| 7 | [Uncertainty](uncertainty.md) | bootstraps, the thickness systematic, analytic covariance |
| 8 | [Saving and reproducible workflows](workflows.md) | the GUI, HDF5 results, TOML sessions, replay and scripts |
| — | [Glossary and references](reference.md) | terminology, notation and the literature |

```{toctree}
:maxdepth: 1

fundamentals
data
models
retrieval
results
dispersion
uncertainty
workflows
reference
```
