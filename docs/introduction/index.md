# Introduction

**croak (Complete Retrieval by Optimisation and Adjoint Kernels)** recovers the
spectral amplitude and phase of an ultrashort pulse from a FROG measurement. A
retrieval finds the pulse whose predicted trace best matches the data; what that
pulse means depends on how faithfully the forward model describes the
instrument, which is why croak offers models of the measurement beyond the
usual thin, non-dispersive nonlinear medium.

This introduction is a learning sequence. Install croak, see what it can do,
choose between the graphical and the script interface, then work through the
Quickstart and three tutorials: a synthetic retrieval, propagation through a
dispersive medium, and a complete measurement workflow. The workflow tutorial
uses synthetic data, so you can follow it without a measurement of your own.

## Install croak

Both routes install the same library and need Python 3.14 or later; uv can
provide that Python for you.

- **Graphical interface only:** follow [Installing the GUI](../getting_started/gui.md).
  It installs a `croak-gui` command and needs no Python knowledge.
- **Python library:** follow [Installation](../getting_started/installation.md)
  for project environments, optional extras, supported platforms, GPUs and
  installation from a source checkout (needed for the bundled example scripts).

```{toctree}
:maxdepth: 1

Installing the GUI <../getting_started/gui>
Installation <../getting_started/installation>
Functionality and capabilities <../reference/features>
GUI or script interface <interfaces>
Quickstart <../getting_started/quickstart>
Your first retrieval <../tutorials/01_first_retrieval>
Dispersive retrieval <../tutorials/03_dispersive_retrieval>
The experimental workflow <../tutorials/04_experimental_workflow>
Where to go next <next_steps>
```

The [user guide](../user_guide/index.md) develops the practical and scientific
background for each step. Use the [examples](../examples/index.md) to explore a
particular problem, the [API](../reference/api/index.md) to look up a Python
argument, the [glossary](../reference/glossary.md) for terminology and the
[references](../reference/bibliography.md) for the papers behind the methods.
Contributors will find the development checks in the
[developer's guide](../developer_guide/index.md).
