---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  name: python3
---

# Propagating thickness uncertainty

Compare the effect of a specified substrate-thickness uncertainty with detector
noise in a synthetic dispersive retrieval.

## Inputs and method

The script generates a 3 fs pulse at 300 nm through 100 µm of fused silica,
adds seeded detector noise, and retrieves at the central thickness. It then
runs 40 thickness and 40 parametric replicates with fixed seeds. The time window
is wide enough to contain the scanned delays and dispersed pulse; convergence
still needs checking for a different pulse or slab.

Run from a [source checkout](../getting_started/installation.md#from-a-clone-development-and-examples):

```bash
uv run python examples/example_thickness_uncertainty.py
```

## Calculation and output

The cell below loads and executes the runnable script itself. Expand it to inspect
the source, or {download}`download example_thickness_uncertainty.py <../../examples/example_thickness_uncertainty.py>`.
Generated files are isolated in a temporary directory for this documentation run.

```{code-cell} ipython3
:tags: [hide-input]

import os
import sys
import tempfile
from pathlib import Path
from IPython.display import Image, display

# The notebook starts in docs/examples. Keep generated files outside the checkout.
repository = Path.cwd().parents[1]
workspace = tempfile.TemporaryDirectory(prefix="croak-gallery-")
(Path(workspace.name) / "examples").symlink_to(repository / "examples", target_is_directory=True)
sys.path.insert(0, str(repository / "examples"))
os.chdir(workspace.name)
```

```{code-cell} ipython3
:load: ../../examples/example_thickness_uncertainty.py
:tags: [hide-input]
```

```{code-cell} ipython3
:tags: [hide-input]

display(Image(filename="thickness_uncertainty.png"))
```

## Interpretation and limits

In the cached execution shown here the central estimate is about **2.30 fs**, although the
generating pulse is **3.00 fs**. The small trace residual therefore coexists with
a substantial duration bias. The noisy pedestal in the profile also shows why a
single FWHM and its resampling spread are insufficient to validate a pulse.
The figure marks the known duration for comparison. Treat this as a demonstration
of the uncertainty interfaces and their limitations: check regularisation,
initialisation, grid and quadrature convergence before reporting such an estimate.

Independent local executions with the same explicit seeds and numerical settings
gave central estimates of about **2.30–2.64 fs**, with trace errors near **0.308 %**.
The source of this numerical variation has not been isolated. The example is
therefore not a stable quantitative benchmark: fixed random seeds alone do not
establish convergence or reproducibility of this ill-conditioned fit. Compare
the printed output and profile on your system, not only the nominal settings.

The plot shows the duration's conditional sensitivity to the assumed thickness
prior. Forty replicates demonstrate the method but are too few for precise tail
probabilities. The printed quadrature combination approximates independent,
roughly Gaussian contributions; for skewed distributions use
{func}`croak.uncertainty.combine_uncertainties` and assess dependence explicitly.
Residual-based noise can include model error. Neither the statistical spread nor
the combined value is automatically a calibrated experimental interval. Read
[uncertainty methods](../howto/uncertainty.md) and
[their assumptions](../explanation/uncertainty_estimation.md).
