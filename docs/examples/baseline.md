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

# Removing an asymmetric baseline

Estimate a smooth background beneath a synthetic FROG trace after fringe
removal, using an asymmetric least-squares baseline with a protected peak.

## Inputs and method

The shared DUV generator supplies the same type of known envelope, fringes and
background as the fringe example. The script first removes carrier fringes,
then estimates and subtracts the baseline for each wavelength row.

Run from a [source checkout](../getting_started/installation.md#from-a-clone-development-and-examples):

```bash
uv run python examples/example_arpls_baseline.py
```

## Calculation and output

The cell below loads and executes the runnable script itself. Expand it to inspect
the source, or {download}`download example_arpls_baseline.py <../../examples/example_arpls_baseline.py>`.
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
:load: ../../examples/synthetic_duv_trace.py
:tags: [hide-input]
```

```{code-cell} ipython3
:load: ../../examples/example_arpls_baseline.py
:tags: [hide-input]
```

```{code-cell} ipython3
:tags: [hide-input]

display(Image(filename="baseline_check.png"))
```

## Interpretation and limits

Compare the recovered trace with the known envelope and inspect the residual
background and any loss near the peak. A baseline is an inferred component,
not an independent background measurement. Broad physical wings can be removed
if smoothness and peak protection are inappropriate. Prefer measured background
information when available; see [preprocessing](../howto/preprocessing.md).
