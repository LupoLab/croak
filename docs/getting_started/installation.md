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

Into an existing environment, use `uv pip install croak` or plain
`pip install croak` with a Python 3.14 interpreter (see the note below).
`uv tool upgrade croak` updates a [GUI tool installation](gui.md) later.

```{admonition} Python 3.14 is a hard requirement
:class: important
croak's source uses syntax introduced in Python 3.14 (for example unparenthesised
`except A, B:` clauses), so it does not even parse on 3.13. A plain `pip`
install with an older interpreter fails with an opaque dependency-resolution
error rather than a clear message, so check `python --version` first, or let uv
provide the interpreter (`uv python install 3.14`).
```

## From a clone (development and examples)

The repository includes the example scripts and the bundled reduced simulation
datasets used in the [gallery](../examples/index.md). They are not installed with
the Python package. To run them:

```bash
git clone https://github.com/LupoLab/croak
cd croak
uv sync
uv run python examples/example_retrieval.py
```

`uv sync` creates the project environment and installs its declared dependencies.
To use a plain virtual environment instead, run `python3.14 -m venv .venv`,
activate it, then `pip install -e .`. For work on croak itself, continue with the
[development environment](../developer_guide/environment.md).

## Supported platforms

croak's binary dependencies (`jax`, `nlopt`, `finufft` and, for the GUI, `pyqt6`)
publish wheels for a limited set of platforms. Where a wheel is missing, the
installer falls back to building from source, which needs CMake and a C++
toolchain.

| Platform | Status |
|---|---|
| Linux x86_64, glibc ≥ 2.34 (Ubuntu 22.04+, RHEL 9+) | supported |
| macOS on Apple Silicon, macOS ≥ 14 | supported |
| Windows x86_64 | supported |
| macOS on Intel | not supported: there is no current `jaxlib` wheel |
| Linux aarch64 | works; `nlopt` and `finufft` build from source |

### Linux: Qt system libraries

The GUI needs Qt's system libraries, which neither uv nor pip installs. On
Debian/Ubuntu, this is the list the project's CI installs:

```bash
sudo apt-get install -y libegl1 libgl1 libxkbcommon-x11-0 libdbus-1-3 \
  libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
  libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 libxcb-cursor0
```

Other distributions use different package names. On a headless machine, run
under `xvfb-run -a` with Xvfb installed, or set `QT_QPA_PLATFORM=offscreen`.
Scripting retrievals does not require Qt or a display server.

### JAX and GPUs

The default installation runs JAX on the CPU on every platform, which suits the
standard and extended forward models. The full focal model is the one that
benefits from a GPU: measured on an NVIDIA H200 it ran about 60× faster than an
Intel Xeon Gold 6240 (all cores), with identical results. For small retrievals,
compilation and transfer costs can make a GPU no faster than the CPU. croak needs
no code changes to use a GPU; installing a CUDA-enabled JAX build is the whole
job. On Linux x86_64 with an NVIDIA driver, the `jax[cuda12]` extra bundles the
CUDA libraries, so no system CUDA toolkit is needed:

- **In a uv project** that uses croak:

  ```bash
  uv add "jax[cuda12]"
  ```

- **In a croak checkout**, either add it to the project (`uv add "jax[cuda12]"`,
  which edits `pyproject.toml`; keep that on a machine-local branch) or inject
  it into the synced environment with `uv pip install "jax[cuda12]"`. The second
  form is undone by the next `uv sync`.

- **In the GUI tool installation**, add it at install time (with `--force` to
  convert an existing CPU-only installation):

  ```bash
  uv tool install --python 3.14 "croak[gui]" --with "jax[cuda12]"
  ```

- **With plain pip**: `pip install "jax[cuda12]"` into the same environment.

Check the result with:

```bash
uv run python -c "import jax; print(jax.devices())"
```

The output should list a CUDA device. If it lists only a CPU, check that
`nvidia-smi` works and that the command ran in the environment you installed
into. Choose a CUDA extra compatible with your driver; JAX also provides a CUDA 13
extra, and its [installation instructions](https://docs.jax.dev/en/latest/installation.html)
list current driver and hardware requirements.

## Dependencies

| Purpose | Packages |
|---|---|
| Numerics | `numpy`, `scipy`, `jax`, `nlopt`, `optimistix`, `lineax` |
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
The `evo` extra installs evosax for
[CMA-ES retrieval](../explanation/algorithms.md#global-retrieval-cma-es); evosax
brings in `flax`, `optax` and `orbax`, so it is not a small addition.
Neither extra is required for the introductory tutorials.

## Development and docs groups

Development checks and documentation builds use dependency groups in a source
checkout. Follow the [developer's guide](../developer_guide/environment.md) and
[documentation build instructions](../developer_guide/documentation.md).

## Verifying the install

```bash
uv run python -c "import croak; print(croak.__version__)"
```

Then follow the [Quickstart](quickstart.md), or run the bundled example above
from a checkout. Continue with [functionality and capabilities](../reference/features.md)
to decide which measurement model and interface fit your experiment.
