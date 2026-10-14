# croak

**Complete Retrieval by Optimisation and Adjoint Kernels**

[![Tests](https://github.com/LupoLab/croak/actions/workflows/ci.yml/badge.svg)](https://github.com/LupoLab/croak/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/LupoLab/croak/graph/badge.svg)](https://codecov.io/gh/LupoLab/croak)
[![Docs](https://readthedocs.org/projects/croak/badge/?version=latest)](https://croak.readthedocs.io)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://github.com/microsoft/pyright)
[![PyPI](https://img.shields.io/pypi/v/croak)](https://pypi.org/project/croak/)
[![Python](https://img.shields.io/pypi/pyversions/croak)](https://pypi.org/project/croak/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![DOI](https://zenodo.org/badge/1351789227.svg)](https://doi.org/10.5281/zenodo.22182305)

**Retrieval of ultrashort laser pulses from FROG traces — spectral amplitude
and phase.**

croak retrieves the spectral amplitude and phase of an ultrashort laser pulse
from a frequency-resolved optical gating (FROG) trace. It supports SHG, SD and
PG/transient-grating local interactions, dispersive PG/SD models, geometric
delay smearing, and chromatic focal and collection models. The Python library
and optional graphical wizard cover loading, checking and preparing data,
retrieval, interpreting results, dispersion tuning, uncertainty and saving.

The model describes what the instrument records. For broadband pulses,
propagation and collection can change the trace enough that a thin-medium fit
has a small residual but an inaccurate field. croak provides a hierarchy of
models to investigate those effects. State the model, field reference and
assumptions when reporting a retrieval; trace agreement alone is not validation.
See the [capabilities overview](https://croak.readthedocs.io/en/latest/reference/features.html)
and the [scientific validation](https://croak.readthedocs.io/en/latest/explanation/validation.html),
which includes successful and failed cases from two unpublished drafts.

Start with the [Introduction](https://croak.readthedocs.io/en/latest/introduction/index.html).
For the graphical application, follow
[Installing the GUI](https://croak.readthedocs.io/en/latest/getting_started/gui.html);
no previous Python experience is needed.

![The wizard after a retrieval: measured and retrieved traces, residuals, the retrieved pulse and spectrum, and a numeric read-out.](docs/_static/gui_retrieve.png)

## Install

croak is [on PyPI](https://pypi.org/project/croak/) and needs **Python ≥ 3.14**.
The easiest route on every platform is [uv](https://docs.astral.sh/uv/), which
installs the right Python for you:

```bash
uv add croak                 # into a uv project
```

For the GUI as a standalone command, without a project:

```bash
uv tool install --python 3.14 "croak[gui]"
croak-gui
```

To work on croak itself — or to run the bundled examples, which ship with the
repository rather than the package — install from a clone:

```bash
git clone https://github.com/LupoLab/croak
cd croak
uv sync                  # create the environment and install croak
uv run python -c "import croak; print(croak.__version__)"
```

Then run anything inside that environment with `uv run`:

```bash
uv run python my_retrieval.py
uv run croak-gui
```

For supported platforms, system libraries and JAX CPU/GPU setup, see the
[installation guide](https://croak.readthedocs.io/en/latest/getting_started/installation.html).
The core models can run on a CPU; spatial quadrature increases cost and can
benefit from a supported GPU. Performance depends on the problem and hardware.

Optional extras: `croak[gui]` (the wizard), `croak[ridb]` (browse
refractiveindex.info materials — a few hundred MB downloaded on first use) and
`croak[evo]` (the cma-es global solver; pulls in `evosax` and its
`flax`/`optax` dependency tree). In a clone the same extras are
`uv sync --extra gui` etc.

## Quick start

```python
import numpy as np
import croak

# a time/frequency grid and a test pulse (2.5 fs Gaussian with some GDD)
g = croak.Grid(128, dt=0.3e-15)
ew = croak.gaussian_pulse(g, 2.5e-15, phases=[4e-30])   # phases = [GDD, TOD, ...]
delays = np.linspace(-18e-15, 18e-15, 87)

# generate a synthetic SHG-FROG trace
trace = croak.maketrace(g.omega, delays, ew, "shg")

# retrieve — the same call for every algorithm ("warm-lbfgs" is the default)
res = croak.retrieve(trace, g.omega, delays, "shg", algorithm="copra",
                     guess=croak.gaussian_pulse(g, 1.8e-15), maxiters=300,
                     rng=np.random.default_rng(0))
print(res.error)            # final trace error R
field = res.field           # retrieved E(t)
spectrum = res.spectrum     # retrieved E(omega)
```

On measured data the workflow is the same shape, with cleaning in front:

```python
fd = croak.io.list_datasets("frog.h5")          # generic HDF5/NPZ dataset picker
lam = fd.load("wavelength") * croak.io.unit_to_si("nm")
delays = fd.load("delay") * croak.io.unit_to_si("fs")

td = croak.load_and_clean(fd.load("trace"), lam, delays, "shg",
                          lam_min=730e-9, lam_max=900e-9, input_unit="delay")

result = croak.retrieve_from_tracedata(td, algorithm="lbfgs", maxiters=200)
pr = croak.process_result(result, measured=td.trace)
croak.plot_retrieval(result, measured=td.trace).savefig("retrieval.png")
croak.save_result(result, "retrieval.h5", processed=pr)
```

![A retrieved pulse: measured vs retrieved trace, residual, temporal and spectral profiles.](docs/_static/hero.png)

## Solvers

Select a solver with `algorithm=`. Supported model options and objectives
differ, so use the [solver guide](https://croak.readthedocs.io/en/latest/howto/solver_selection.html)
when changing algorithms.

| Family | Algorithms | Role |
|---|---|---|
| COPRA | `copra`, `copra-jax` | Local projection sweeps followed by global updates |
| L-BFGS | `lbfgs`, `lbfgs-hand`, `lbfgs-ad`, `lbfgs-optx` | Gradient optimisation with analytic or automatic derivatives |
| Warm-started L-BFGS | `warm-lbfgs` (default) | COPRA initialisation followed by the AD solver |
| Nonlinear least squares | `lm`, `lm-optx` | Residual/Jacobian methods; SciPy TRF by default for `lm`, Optimistix LM for `lm-optx` |
| Population search | `cma-es` | Evolution strategies, often followed by local refinement |

The core model has analytic Wirtinger adjoints. JAX supplies automatic
derivatives for implemented spatial models and fitted scalar parameters. The
manual retains [adjoint derivations](https://croak.readthedocs.io/en/latest/explanation/gradients.html),
[objective definitions](https://croak.readthedocs.io/en/latest/explanation/retrieval_theory.html)
and [model equations](https://croak.readthedocs.io/en/latest/explanation/forward_model.html).

## Performance

NumPy and JAX routes have different compilation and execution costs. Compare
matched models, grids, quadratures and stopping rules, and separate first-call
compilation from repeated runs. See the
[installation guide](https://croak.readthedocs.io/en/latest/getting_started/installation.html#jax-and-gpus)
for accelerator setup.

## Scope

The PNPS formalism also describes techniques such as d-scan and XFROG, but these
are not implemented retrieval workflows in croak. New measurement techniques
can require changes to the scan, forward model and derivative paths; adding a
local interaction alone is not sufficient.

The bundled instrument datasets were generated with
[ModelPNPS.jl](https://github.com/LupoLab/ModelPNPS.jl), a separate Julia
simulation package. They are simulated measurements, not experimental data.

## GUI

A PyQt6 wizard (Load → Marginal check → Preprocess → Retrieve → Dispersion →
Uncertainty) drives the same library functions, with embedded plots, threaded
retrieval and a live preview:

```bash
uv run croak-gui          # or: uv run python -m croak.gui
```

Sessions save to TOML, and `croak replay` reruns one headlessly — or
`croak script` turns it into a standalone Python file you can edit. Importing
`croak` does **not** require Qt; only `croak.gui` does. See
[Installing the GUI](https://croak.readthedocs.io/en/latest/getting_started/gui.html) and the
[GUI guide](https://croak.readthedocs.io/en/latest/howto/gui.html).

## Examples

```bash
uv run python examples/example_retrieval.py              # shapes, geometries, noise, dispersion
uv run python examples/example_workflow.py               # load → retrieve → save
uv run python examples/example_paper_thickness.py        # bundled DUV simulation, three models
uv run python examples/example_paper_rdw.py              # a simulated single-cycle RDW pulse
uv run python examples/example_geometric_smearing.py     # BOXCARS time smearing
uv run python examples/example_defringe.py               # interferometric fringe removal
uv run python examples/example_arpls_baseline.py         # + asymmetric baseline removal
uv run python examples/example_thickness_uncertainty.py  # fitted thickness & error bars
```

The first examples synthesise their own data. The two `example_paper_*`
scripts retrieve from reduced 3D instrument-simulation datasets, shipped in [`examples/data/`](examples/data/README.md). The
examples and their data live in the repository, not in the PyPI package, so
run them from a clone. The [worked gallery](https://croak.readthedocs.io/en/latest/examples/index.html)
explains their inputs, outputs and limitations. The bundled legacy examples
illustrate workflows; their settings do not reproduce the revised paper protocol.

## Documentation

Hosted at [croak.readthedocs.io](https://croak.readthedocs.io): tutorials, the
PNPS theory and the dispersive forward model, the solvers and gradient routes,
how-to guides, the validation results and the full API reference. The
[Features](https://croak.readthedocs.io/en/latest/reference/features.html) page lists everything the package does.
To build locally:

```bash
uv sync --group docs
uv run sphinx-build -b html docs docs/_build/html
```

Start with the [Introduction](https://croak.readthedocs.io/en/latest/introduction/index.html),
then follow the [User guide](https://croak.readthedocs.io/en/latest/user_guide/index.html)
or explore the [Examples](https://croak.readthedocs.io/en/latest/examples/index.html).
The [Developer's guide](https://croak.readthedocs.io/en/latest/developer_guide/index.html)
covers contributing, and the [API](https://croak.readthedocs.io/en/latest/reference/api/index.html)
documents the Python interfaces.

## The name

croak stands for *Complete Retrieval by Optimisation and Adjoint Kernels*. It
was called `toad` (*Trace Optimisation through Automatic Differentiation*)
while it was an internal tool, and was renamed for release because that name
was already taken on PyPI — hence the unchanged amphibian on the icon. See
[the docs](https://croak.readthedocs.io/en/latest/reference/naming.html).

## Citation

Cite the software via [doi:10.5281/zenodo.22182305](https://doi.org/10.5281/zenodo.22182305)
(all versions), or use [CITATION.cff](CITATION.cff). The scientific documentation
also draws on two **unpublished drafts** by J. C. Travers and C. Brahms:

- *Differentiable physical models for ultrashort-pulse retrieval*.
- *A dispersive and spatial model of transient-grating frequency-resolved optical gating*.

See the [references](https://croak.readthedocs.io/en/latest/reference/bibliography.html)
for the scope of each draft and the published sources underlying the methods.

## Authors and licence

croak is developed by John Travers and Christian Brahms in the
[LUPO research group](https://lupo-lab.com) at Heriot-Watt University.
Released under the MIT licence — see [LICENSE](LICENSE).
