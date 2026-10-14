# Installation

croak requires **Python 3.14 or later**. Use [uv](https://docs.astral.sh/uv/)
to manage Python and the package environment. If you want to use croak without
writing Python, follow [Installing the GUI](gui.md) first.

## From PyPI (recommended)

For a new analysis project, install uv using its
[installation instructions](https://docs.astral.sh/uv/getting-started/installation/),
then run:

```bash
uv init --python 3.14 pulse-retrieval
cd pulse-retrieval
uv add croak
```

In an existing uv project using Python 3.14 or later, run `uv add croak`.
Save the [Quickstart](quickstart.md) calculation in a file such as
`my_retrieval.py`, then run it inside the project environment:

```bash
uv run python my_retrieval.py
```

To include the graphical interface in this project, use `uv add "croak[gui]"`
and launch it with `uv run croak-gui`. The separate
[GUI installation](gui.md) uses a uv tool environment for a command available
outside a particular project.

## From a clone (development and examples)

The repository includes the example scripts and local simulation datasets used
in the [gallery](../examples/index.md). They are not installed with the Python
package. To run them:

```bash
git clone https://github.com/LupoLab/croak
cd croak
uv sync
uv run python examples/example_retrieval.py
```

`uv sync` creates the project environment and installs its declared dependencies.
For work on croak itself, continue with the
[development environment](../developer_guide/environment.md).

## Supported platforms

Installation depends on binary packages for JAX, NLopt, FINUFFT and, for the
GUI, Qt. CPU installations are intended for Apple Silicon macOS, Linux x86_64
and Windows x86_64. Exact operating-system requirements depend on the resolved
dependency versions. Intel macOS lacks a current standard JAX binary installation;
Linux aarch64 may require building some numerical dependencies from source.
See [JAX's platform table](https://docs.jax.dev/en/latest/installation.html)
when selecting a machine or diagnosing installation failures.

### Linux: Qt system libraries

The GUI also needs system libraries. On Debian/Ubuntu, the project's CI installs:

```bash
sudo apt-get install -y libegl1 libgl1 libxkbcommon-x11-0 libdbus-1-3 \
  libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
  libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 libxcb-cursor0
```

Other distributions use different package names. A headless GUI test can use
`xvfb-run -a` with Xvfb installed, or `QT_QPA_PLATFORM=offscreen`. Scripting
retrievals does not require Qt or a display server.

### JAX and GPUs

The default installation runs JAX on the CPU. A supported NVIDIA GPU can help
with large spatial forward models; compilation and transfer costs mean that a
GPU is not necessarily faster for a small retrieval. On a compatible Linux
system, declare the CUDA extra in the same project as croak, for example:

```bash
uv add "jax[cuda12]"
uv run python -c "import jax; print(jax.devices())"
```

For a GUI tool environment, include the dependency at installation:

```bash
uv tool install --python 3.14 "croak[gui]" --with "jax[cuda12]"
```

Choose a CUDA extra compatible with the installed driver and the resolved JAX
version. JAX also provides a CUDA 13 extra; consult its
[GPU installation instructions](https://docs.jax.dev/en/latest/installation.html)
for current driver and hardware requirements. The device check should list a
CUDA device. If it lists only a CPU, check the driver and the environment in which
the command ran. Adding the extra in a source checkout changes its dependency
files; keep machine-specific dependency changes separate from scientific work.

## Dependencies

| Purpose | Packages |
|---|---|
| Numerics | `numpy`, `scipy`, `jax`, `nlopt`, `optimistix` |
| Delay filtering | `finufft` |
| Data and sessions | `h5py`, `tomli-w` |
| Plotting and progress | `matplotlib`, `tqdm` |
| GUI (`gui` extra) | `pyqt6`, `superqt` |

Importing `croak` does not import Qt. The library and session replay work without
the graphical interface.

## Optional features

In an analysis project, choose the extras you need:

```bash
uv add "croak[gui]"          # graphical interface
uv add "croak[ridb]"         # refractiveindex.info database access
uv add "croak[evo]"          # CMA-ES retrieval
```

Extras can be combined, for example `uv add "croak[gui,ridb]"`. In a source
checkout, use `uv sync --extra gui --extra ridb` instead.

The `ridb` extra downloads the refractiveindex.info database on first use;
this needs an internet connection and local storage. Built-in materials need
no download. See [materials and mirrors](../howto/materials_and_mirrors.md).
The `evo` extra installs evosax and its dependencies for
[CMA-ES retrieval](../explanation/algorithms.md#global-retrieval-cma-es).
Neither extra is required for the introductory tutorials.

## Development and docs groups

Development checks and documentation builds use dependency groups in a source
checkout. Follow the [developer's guide](../developer_guide/environment.md) and
[documentation build instructions](../developer_guide/documentation.md).
A normal documentation build reuses cached notebook outputs; it does not prove
that unchanged examples were executed again.

## Verifying the install

```bash
uv run python -c "import croak; print(croak.__version__)"
```

Then follow the [Quickstart](quickstart.md), or run the bundled example above
from a checkout. Continue with [functionality and capabilities](../reference/features.md)
to decide which measurement model and interface fit your experiment.
