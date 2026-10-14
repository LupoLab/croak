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

# Retrieval with geometric smearing

Test the effect of a known BOXCARS timing kernel by modelling it, omitting it,
and fitting a common scale. Then vary mask spacing and slab thickness.

## Inputs and method

The script generates a transform-limited 1 fs PG trace at 260 nm with a reduced
Gaussian timing kernel. It uses a fixed initial pulse and seed, and disables
explicit regularisation for this matched synthetic comparison. The two sweeps
compare smeared and unsmeared traces at otherwise matched settings.

Run from a [source checkout](../getting_started/installation.md#from-a-clone-development-and-examples):

```bash
uv run python examples/example_geometric_smearing.py
```

## Calculation and output

The cell below loads and executes the runnable script itself. Expand it to inspect
the source, or {download}`download example_geometric_smearing.py <../../examples/example_geometric_smearing.py>`.
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
:load: ../../examples/example_geometric_smearing.py
:tags: [hide-input]
```

```{code-cell} ipython3
:tags: [hide-input]

display(Image(filename="geometric_smearing.png"))
```

## Interpretation and limits

A fitted kernel can reduce bias, but its agreement in a matched simulation
is not independent physical validation. For a transform-limited Gaussian the
split-gate channel is an amplitude factor; the fitted common scale is mainly
constrained through the delay channel. In a real instrument, changing mask spacing
also changes chromatic profiles and collection. The thickness sweep uses a finite
periodic grid; assess window and quadrature convergence before transferring any
numerical bias to an experiment. Read [geometric smearing](../howto/geometric_smearing.md)
and [collection](../howto/collection_aperture.md).
