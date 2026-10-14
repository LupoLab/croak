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

# A standalone load-to-save workflow

Run the full library workflow from a synthetic HDF5 file to a saved retrieval.
For a paced explanation, use the canonical
[experimental-workflow tutorial](../tutorials/04_experimental_workflow.md).

## Inputs and method

The script generates a chirped PG pulse and stores its trace as wavelength
density. It loads SI-converted axes, selects a band, regrids, retrieves with
seeded COPRA, and saves an HDF5 result and summary figure. The raw file is
temporary; output files are written to the working directory in a standalone run.

Run from a [source checkout](../getting_started/installation.md#from-a-clone-development-and-examples):

```bash
uv run python examples/example_workflow.py
```

## Calculation and output

The cell below loads and executes the runnable script itself. Expand it to inspect
the source, or {download}`download example_workflow.py <../../examples/example_workflow.py>`.
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
:load: ../../examples/example_workflow.py
:tags: [hide-input]
```

```{code-cell} ipython3
:tags: [hide-input]

display(Image(filename="retrieval.png"))
```

## Interpretation and limits

Inspect the residual, phase and edge energy alongside duration. Regridding
introduces interpolation error even though the generating model is known.
No separately measured fundamental spectrum is supplied here. Use
[preparing data](../user_guide/data.md) and
[saving and sessions](../howto/saving_and_loading.md) when adapting it to a measurement.
