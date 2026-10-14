# Examples

Choose an example by the question you want to explore. The introductory
tutorials explain a complete calculation step by step; the focused examples
explore data preparation, retrieval choices, instrument effects, and analysis.
Use the [user guide](../user_guide/index.md) for the underlying methods and
assumptions.

The Python scripts and example datasets are supplied in the repository. Follow
[installation from a clone](../getting_started/installation.md#from-a-clone-development-and-examples),
then run the commands below from the repository root. Download links provide the
source for inspection; some scripts also need the neighbouring helper or data
files from the checkout.

## Getting started

::::{grid} 1 1 2 3
:gutter: 2

:::{grid-item-card} Your first retrieval
:link: ../tutorials/01_first_retrieval
:link-type: doc

Generate a synthetic SHG-FROG trace, retrieve the pulse, and interpret the plots.
:::

:::{grid-item-card} Dispersive retrieval
:link: ../tutorials/03_dispersive_retrieval
:link-type: doc

Explore the effect of propagation through a nonlinear medium.
:::

:::{grid-item-card} Experimental workflow
:link: ../tutorials/04_experimental_workflow
:link-type: doc

Load, clean, retrieve, plot, and save using a synthetic measurement file.
:::

::::

[Retrieval across shapes and interactions](retrieval.md)
explores pulse shapes, interactions, noise, and dispersion.
[A standalone workflow](workflow.md) provides a runnable load-to-save script
corresponding to the introductory tutorial.

```bash
uv run python examples/example_retrieval.py
uv run python examples/example_workflow.py
```

## Preparing data

- **Fringe removal:** [worked fringe removal](defringe.md)
  applies delay-domain filtering to a synthetic DUV trace and plots the result.
- **Baseline removal:** [worked baseline removal](baseline.md)
  combines fringe removal with an asymmetric baseline estimate.

Both use {download}`synthetic_duv_trace.py <../../examples/synthetic_duv_trace.py>`
to generate their input. See [preparing and checking data](../user_guide/data.md)
for the physical meaning of the corrections and the checks to make afterwards.

```bash
uv run python examples/example_defringe.py
uv run python examples/example_arpls_baseline.py
```

## Comparing retrieval approaches

The [algorithm comparison](../tutorials/02_choosing_an_algorithm.md) explores
convergence using several solvers. Read it alongside
[retrieval and numerical methods](../user_guide/retrieval.md): solver comparisons
depend on the model, objective, initialisation, and stopping criteria as well
as the optimisation method.

## Instrument effects and validation

| Example | What it explores | Background |
|---|---|---|
| [Geometric smearing](geometric_smearing.md) | Arrival-time differences in a crossed-beam geometry and their effect on retrieval | [Geometric smearing](../howto/geometric_smearing.md) |
| [Substrate thickness](thickness_series.md) | Retrievals from reduced 3D instrument simulations at several substrate thicknesses | [The forward model](../explanation/forward_model.md) |
| [Simulated RDW pulse](rdw.md) | A structured pulse from a simulated resonant-dispersive-wave source | [Loading simulated traces](../howto/loading_simulated.md) |

The two simulation scripts use datasets in `examples/data/`; their
{download}`dataset descriptions <../../examples/data/README.md>` explain the
stored quantities. These are numerical examples, not experimental measurements.
Their settings should be inspected when comparing their results with a paper.

```bash
uv run python examples/example_geometric_smearing.py
uv run python examples/example_paper_thickness.py
uv run python examples/example_paper_rdw.py
```

See [modelling the measurement](../user_guide/models.md) for model selection and
[interpreting results](../user_guide/results.md) for comparisons with known fields.

## Analysing and transforming pulses

The [dispersion tuning example](../tutorials/05_dispersion_tuning.md) shows how
to explore compression and calculate peak power. Read
[propagation and dispersion tuning](../user_guide/dispersion.md) before applying
these operations to a retrieved field at a particular reference plane.

### Thickness uncertainty

[Thickness uncertainty](thickness_uncertainty.md)
propagates a measured substrate-thickness uncertainty to the retrieved duration
and compares it with a statistical estimate. The example generates its own data.
The [uncertainty chapter](../user_guide/uncertainty.md) explains how to choose and
interpret the estimators.

```bash
uv run python examples/example_thickness_uncertainty.py
```

```{toctree}
:hidden:
:maxdepth: 1

retrieval
workflow
defringe
baseline
Comparing algorithms <../tutorials/02_choosing_an_algorithm>
geometric_smearing
thickness_series
rdw
thickness_uncertainty
Dispersion tuning <../tutorials/05_dispersion_tuning>
```
