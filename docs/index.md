---
sd_hide_title: true
---

# croak

```{rubric} Complete Retrieval by Optimisation and Adjoint Kernels
```

**croak** reconstructs the spectral amplitude and phase of an ultrashort laser
pulse from a frequency-resolved optical gating (FROG) trace. Use the graphical
interface or a Python script to prepare data, retrieve a pulse, interpret the
result, and estimate uncertainty.

(highlights)=
## What croak can do

The measurement model can include propagation through a dispersive nonlinear
medium, the different arrival times of crossing beams, and the collection of the
generated signal. These effects can matter when the pulse is short enough for
the instrument to change the trace appreciably. Start with the
[capabilities overview](reference/features.md) to see which models and
interactions are supported.

```{image} _static/hero.png
:alt: A retrieval summary showing the data and model traces, residual, and temporal and spectral pulse profiles.
:align: center
:width: 90%
```

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Introduction
:link: introduction/index
:link-type: doc

Install croak, choose an interface, and follow three guided retrieval tutorials.
:::

:::{grid-item-card} User guide
:link: user_guide/index
:link-type: doc

Work through a measurement, from data preparation to interpretation. Find the
physics, equations, and numerical methods alongside each topic.
:::

:::{grid-item-card} Examples
:link: examples/index
:link-type: doc

Explore runnable examples of data cleaning, retrieval, instrument effects,
dispersion tuning, and uncertainty.
:::

:::{grid-item-card} Developer's guide
:link: developer_guide/index
:link-type: doc

Set up a development environment, understand the package, and contribute code
or documentation.
:::

:::{grid-item-card} API
:link: reference/api/index
:link-type: doc

Look up Python functions, classes, parameters, return values, and conventions.
:::

::::

New to pulse retrieval? Begin with the [introduction](introduction/index.md).
If you already have a trace, follow [preparing and checking
data](user_guide/data.md). To understand a result, see [interpreting
results](user_guide/results.md) and [uncertainty](user_guide/uncertainty.md).

```{toctree}
:hidden:
:maxdepth: 3

Introduction <introduction/index>
User guide <user_guide/index>
Examples <examples/index>
Developer's guide <developer_guide/index>
API <reference/api/index>
```
