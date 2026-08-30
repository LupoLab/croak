---
sd_hide_title: true
---

# croak

```{rubric} Complete retrieval of ultrashort pulses from FROG traces
```

**croak** retrieves the complex electric field of an ultrashort laser pulse — full
spectral amplitude *and* phase — from a delay-scanned nonlinear-process spectrum.
It ships the three FROG geometries (SHG, SD and PG/transient-grating), ten
solvers over a single physical model, and the whole workflow around them: loading,
cleaning, retrieval, post-processing, uncertainty, plotting and saving. The core
is a headless Python library built on NumPy, SciPy and JAX; the GUI is optional.

The difference from a classic FROG code is the forward model, which can
contain the measurement physics that standard retrieval idealises away: the
**dispersive propagation** of the input fields and the generated signal inside
the nonlinear medium, the **geometric delay smearing** of a non-collinear
crossed-beam geometry, and the **chromatic collection aperture**. All three are
parameters of the ordinary solvers; with them switched off, the same code
retrieves a conventional thin-medium trace. This is what quantitative
retrieval of few-femtosecond deep-ultraviolet pulses requires — validated
against first-principles 3D instrument simulations in the
[companion paper](explanation/validation.md), whose retrievals were all done
with croak.

```{image} _static/hero.png
:alt: A retrieved pulse — measured vs retrieved trace, residual, temporal and spectral profiles.
:align: center
:width: 90%
```

```{code-block} python
import numpy as np, croak

g = croak.Grid(128, dt=0.3e-15)
ew = croak.gaussian_pulse(g, 2.5e-15, phases=[4e-30])     # 2.5 fs, chirped
delays = np.linspace(-18e-15, 18e-15, 87)
trace = croak.maketrace(g.omega, delays, ew, "shg")       # synthesise an SHG-FROG trace

res = croak.retrieve(trace, g.omega, delays, "shg", algorithm="warm-lbfgs")
print(res.error)        # final trace error R
field = res.field       # retrieved E(t)
```

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`rocket` Getting started
:link: getting_started/quickstart
:link-type: doc

Install croak and run your first retrieval in ten lines.
:::

:::{grid-item-card} {octicon}`mortar-board` Tutorials
:link: tutorials/01_first_retrieval
:link-type: doc

Executed, end-to-end walk-throughs from a synthetic trace to a retrieved pulse.
:::

:::{grid-item-card} {octicon}`book` Explanation
:link: explanation/pnps_framework
:link-type: doc

The PNPS formalism, the dispersive forward model and how the solvers work.
:::

:::{grid-item-card} {octicon}`tools` How-to guides
:link: howto/preprocessing
:link-type: doc

Task-focused recipes for cleaning data, choosing a solver and tuning dispersion.
:::

:::{grid-item-card} {octicon}`code` API reference
:link: reference/api/index
:link-type: doc

Every public module, class and function, generated from the docstrings.
:::

:::{grid-item-card} {octicon}`device-desktop` GUI
:link: howto/gui
:link-type: doc

A PyQt6 wizard that drives the whole library — load, retrieve, tune, save.
:::

::::

## Highlights

- **Ten solvers, one forward model.** Four families —
  {class}`~croak.copra.COPRA`, L-BFGS, Levenberg–Marquardt and a global
  {class}`~croak.cmaes.CMAES` search — over a single physical model, behind one
  `algorithm=` argument. Swap solvers without changing anything else. The
  default, {class}`~croak.warm_lbfgs.WarmLBFGS`, composes two of them.
- **Differentiable end to end.** The model is written in JAX, so automatic
  differentiation supplies gradients for anything added to it — geometric
  smearing, fitted medium thickness and delay-zero, the B-spline phase basis,
  Laplace covariance, and any new interaction, with no adjoint derivation
  needed.
- **A hand-derived adjoint where it applies.** For the core model —
  all three interactions, thin or dispersive — croak also carries an exact
  Wirtinger adjoint, so {class}`~croak.copra.COPRA` and {class}`~croak.lbfgs.LBFGS`
  run in pure NumPy. The two routes are cross-checked to ~1e-6.
- **On-device execution.** {class}`~croak.copra_jax.COPRAJax`,
  {class}`~croak.lbfgs_hand.LBFGSHand`, {class}`~croak.optimistix_lbfgs.OptxLBFGS`
  and {class}`~croak.optimistix_lm.OptxLM` keep the whole loop on the accelerator.
- **The instrument in the model.** Geometric time smearing of a non-collinear
  BOXCARS geometry, from a closed-form kernel set by the mask geometry (or
  fitted from the data), and the collection aperture as a coherent chromatic
  filter — the effects that otherwise inflate the retrieved duration and bias
  the retrieved chirp.
- **Dispersive media.** Propagate through a material slab (fused silica, CaF₂, …)
  with a Gauss–Legendre depth integral — needed for accurate DUV/VUV retrieval —
  with materials from the built-in Sellmeier set, your own `n(λ)`, the
  refractiveindex.info database, or **chirped mirrors**.
- **Three geometries.** SHG, SD and PG (the PG kernel is the transient-grating /
  TG-FROG signal).
- **The whole workflow as a library.** Loading, cleaning, regridding,
  post-processing, dispersion tuning, plotting and saving — all callable without
  the GUI, with a live progress bar on every retrieval.
- **SI units throughout**, a centred angular-frequency axis, and the physics
  Fourier convention.

```{toctree}
:hidden:
:caption: Getting started

getting_started/installation
getting_started/gui
getting_started/quickstart
```

```{toctree}
:hidden:
:caption: Tutorials

tutorials/01_first_retrieval
tutorials/02_choosing_an_algorithm
tutorials/03_dispersive_retrieval
tutorials/04_experimental_workflow
tutorials/05_dispersion_tuning
```

```{toctree}
:hidden:
:caption: Explanation

explanation/pnps_framework
explanation/interactions
explanation/forward_model
explanation/retrieval_theory
explanation/marginals
explanation/algorithms
explanation/gradients
explanation/validation
explanation/uncertainty_estimation
```

```{toctree}
:hidden:
:caption: How-to guides

howto/preprocessing
howto/marginal_checks
howto/loading_simulated
howto/grids_and_filtering
howto/solver_selection
howto/regularisation
howto/fitting_thickness_tau0
howto/geometric_smearing
howto/collection_aperture
howto/materials_and_mirrors
howto/postprocessing
howto/uncertainty
howto/saving_and_loading
howto/gui
```

```{toctree}
:hidden:
:caption: Reference

reference/features
reference/conventions
reference/fft_convention
reference/glossary
reference/bibliography
reference/naming
reference/api/index
```
