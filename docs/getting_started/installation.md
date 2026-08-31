# Installation

croak is [on PyPI](https://pypi.org/project/croak/) and needs **Python ≥
3.14**. The simplest route on macOS, Windows and Linux alike is
[uv](https://docs.astral.sh/uv/), which will fetch that Python for you.

## From PyPI (recommended)

Into a uv project:

```bash
uv add croak
```

Into any existing environment (`uv pip install croak`), or with plain pip
(`pip install croak`) — mind the Python-version admonition below. Add the GUI
with the `gui` extra: `uv add "croak[gui]"`.

To have the GUI as a standalone command without any project or environment of
your own, install it as a uv tool:

```bash
uv tool install --python 3.14 "croak[gui]"
croak-gui
```

(`uv tool upgrade croak` updates it later.) See
[Installing the GUI](gui.md) for the GUI-only walkthrough.

## From a clone (development and examples)

The bundled examples — including the companion paper's validation datasets in
`examples/data/` — ship with the repository, not the package, and developing
croak needs the test suite. For either, install from a clone:

```bash
git clone https://github.com/LupoLab/croak
cd croak
uv sync                  # create a virtualenv and install croak + dependencies
uv run pytest            # (optional) run the test suite
```

`uv run <command>` executes inside the managed environment, so a retrieval script
runs with:

```bash
uv run python my_retrieval.py
```

Prefer a plain virtual environment? `python3.14 -m venv .venv`, activate it,
then `pip install -e .`.

```{admonition} Python 3.14 is a hard requirement
:class: important
Not a recommendation: croak's source uses syntax introduced in 3.14 and will not
even parse on 3.13. On a plain `pip` install with an older interpreter you will
see an opaque resolution failure rather than a clear message, so check
`python --version` first — or let `uv` handle it, which is why the commands above
start with `uv python install`.
```

## Supported platforms

croak's numerical dependencies (`jax`, `nlopt`, `finufft`, `pyqt6`) ship binary
wheels for a limited set of platforms. Where a wheel is missing the installer
falls back to building from source, which needs CMake and a C++ toolchain.

| Platform | Status |
|---|---|
| Linux x86_64, glibc ≥ 2.34 (Ubuntu 22.04+, RHEL 9+) | fully supported |
| macOS on Apple Silicon, macOS ≥ 14 | fully supported |
| Windows x86_64 | fully supported |
| macOS on Intel | **not supported** — no `jaxlib` wheel is published |
| Linux aarch64 | works, but `nlopt` and `finufft` build from source |

### Linux: Qt system libraries

The GUI needs Qt's system libraries, which neither uv nor pip installs. This is
the same list the project's CI uses:

```bash
sudo apt-get install -y libegl1 libgl1 libxkbcommon-x11-0 libdbus-1-3 \
  libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
  libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 libxcb-cursor0
```

On a headless machine add `xvfb` and run under `xvfb-run -a`, or set
`QT_QPA_PLATFORM=offscreen`. The library itself needs none of this — only
`croak.gui` does.

### JAX and GPUs

The default install is **CPU-only JAX on every platform**, which is the right
configuration for the standard and extended forward models. The full focal
model is the one that benefits from a GPU (about 60× an Intel Xeon Gold 6240
on an NVIDIA H200, with identical results). croak needs no changes to use
one — installing a CUDA-enabled JAX build is the whole job. The extra is
`jax[cuda12]` (Linux x86_64 with an NVIDIA driver; it bundles the CUDA
libraries, so no system CUDA toolkit is needed):

- **In a uv project** that uses croak:

  ```bash
  uv add "jax[cuda12]"
  ```

- **In a croak checkout**: the environment is locked, so either make the
  extra part of the project (`uv add "jax[cuda12]"`, which edits
  `pyproject.toml` — fine on a machine-local branch) or inject it into the
  synced environment ad hoc:

  ```bash
  uv pip install "jax[cuda12]"
  ```

  The ad-hoc form is undone by the next `uv sync`, so re-run it after syncing.

- **In the standalone GUI tool install** (to run `croak-gui` retrievals on the
  GPU), add it with `--with` at install time:

  ```bash
  uv tool install --python 3.14 "croak[gui]" --with "jax[cuda12]"
  ```

  Re-running `uv tool install` with `--force` and the same `--with` also
  converts an existing CPU-only tool install.

- **Plain pip**: `pip install "jax[cuda12]"` into the same environment.

Verify with:

```bash
uv run python -c "import jax; print(jax.devices())"
```

which should list a `CudaDevice`. If it prints only `CpuDevice`, JAX fell back
— check `nvidia-smi` works and that the environment you ran in is the one you
installed into.

## Dependencies

The core install pulls in everything needed for retrieval and the workflow
pipeline; the GUI is an extra:

| Purpose | Packages |
|---------|----------|
| Numerics | `numpy`, `scipy`, `jax`, `nlopt`, `optimistix` |
| Signal processing | `finufft` (non-uniform FFT for delay filtering) |
| Data I/O | `h5py`, `tomli-w` |
| Plotting & progress | `matplotlib`, `tqdm` |
| GUI (`croak[gui]` extra) | `pyqt6`, `superqt` |

Importing `croak` does **not** import Qt — only `croak.gui` does. You can use the
entire library headless.

## Optional features

Three extras unlock optional functionality (none is needed for core retrieval):

```bash
uv add "croak[gui]"     # the PyQt6 wizard
uv add "croak[ridb]"    # refractiveindex.info material database
uv add "croak[evo]"     # evosax — the CMA-ES global retriever
# with pip:  pip install "croak[gui,ridb,evo]"
# in a clone: uv sync --extra gui --extra ridb --extra evo
```

- **`ridb`** ([`refractiveindex`](https://refractiveindex.info)) lets the
  dispersion tools and GUI browse arbitrary materials. The first lookup downloads
  the full database (~hundreds of MB) to `~/.refractiveindex.info-database`. See
  [Materials and chirped mirrors](../howto/materials_and_mirrors.md).
- **`evo`** ([`evosax`](https://github.com/RobertTLange/evosax)) enables the
  `cma-es` [global retriever](../explanation/algorithms.md#global-retrieval-cma-es).
  It runs on the JAX install croak already has, but `evosax` itself brings in
  `flax`, `optax` and `orbax`, so it is not a small addition.

## Development and docs groups

```bash
uv sync --group dev     # pytest + pytest-qt for the test suite
uv sync --group docs    # Sphinx toolchain to build this documentation
```

To build the documentation locally:

```bash
uv sync --group docs
uv run sphinx-build -b html docs docs/_build/html
# open docs/_build/html/index.html
```

The tutorials are executed as notebooks at build time, so a successful build also
confirms the examples still run against the installed package.

## Verifying the install

```bash
uv run python -c "import croak; print(croak.__version__)"
```

Then run the bundled end-to-end example:

```bash
uv run python examples/example_retrieval.py
```

It retrieves a range of pulse shapes, interaction geometries and noise levels
(including a dispersive case) and prints the trace error and recovered pulse
width for each.
