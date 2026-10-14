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

# Removing carrier fringes

Remove synthetic carrier-frequency fringes while checking how much of the slow
FROG envelope remains. This example uses the shared synthetic DUV generator.

## Inputs and method

The helper constructs a known slow envelope, interference fringes, background
and noise. The script applies the carrier-aware delay low-pass and compares the
result with the known envelope. Inputs are generated locally with a fixed seed;
no research dataset is downloaded.

Run from a [source checkout](../getting_started/installation.md#from-a-clone-development-and-examples):

```bash
uv run python examples/example_defringe.py
```

## Calculation and output

The cell below loads and executes the runnable script itself. Expand it to inspect
the source, or {download}`download example_defringe.py <../../examples/example_defringe.py>`.
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
:load: ../../examples/example_defringe.py
:tags: [hide-input]
```

```{code-cell} ipython3
:tags: [hide-input]

display(Image(filename="defringe_check.png"))
```

## Interpretation and limits

Compare the filtered and target envelopes in both linear and logarithmic
views. Filtering can remove real delay structure as well as fringes; carrier
separation and delay sampling are essential. This demonstration does not justify
defringing a clean broadband trace. See [preprocessing](../howto/preprocessing.md)
and [grids and filtering](../howto/grids_and_filtering.md).
