# Installation

croak needs **Python ≥ 3.14**. The simplest route on macOS, Windows and Linux
alike is [uv](https://docs.astral.sh/uv/), which will fetch that Python for you.

## From a clone (recommended for now)

```bash
git clone https://github.com/LupoLab/croak
cd croak
uv python install 3.14   # only if you do not already have it
uv sync                  # create a virtualenv and install croak + dependencies
uv run pytest            # (optional) run the test suite
```

`uv run <command>` executes inside the managed environment, so a retrieval script
runs with:

```bash
uv run python my_retrieval.py
```

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
on an NVIDIA H200, with identical results); install the matching JAX build
yourself (for example `uv pip install "jax[cuda12]"`) — croak needs no changes
to use it.

## With pip

If you prefer a plain virtual environment:

```bash
python3.14 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .
```

## Dependencies

The core install pulls in everything needed for retrieval, the workflow pipeline
**and** the GUI:

| Purpose | Packages |
|---------|----------|
| Numerics | `numpy`, `scipy`, `jax`, `nlopt`, `optimistix` |
| Signal processing | `finufft` (non-uniform FFT for delay filtering) |
| Data I/O | `h5py`, `tomli-w` |
| Plotting & progress | `matplotlib`, `tqdm` |
| GUI | `pyqt6`, `superqt` |

Importing `croak` does **not** import Qt — only `croak.gui` does. You can use the
entire library headless.

## Optional features

Two extras unlock optional functionality (neither is needed for core retrieval):

```bash
uv sync --extra ridb    # refractiveindex.info material database
uv sync --extra evo     # evosax — the CMA-ES global retriever
# with pip:  pip install "croak[ridb,evo]"
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
