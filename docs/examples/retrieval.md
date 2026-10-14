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

# Retrieval across pulse shapes and interactions

Compare SHG, PG and SD retrievals for Gaussian and sech pulses, then include a
dispersive slab. This is a compact survey after [your first retrieval](../tutorials/01_first_retrieval.md).

## Inputs and method

The script generates its own traces on a matched Fourier grid. It uses a fixed
noise seed and fixed retrieval seeds. Clean cases use COPRA; noisy cases use
L-BFGS with its package defaults, so this is not a controlled solver ranking.
The final case uses a 260 nm carrier and a 10 µm fused-silica slab.

Run from a [source checkout](../getting_started/installation.md#from-a-clone-development-and-examples):

```bash
uv run python examples/example_retrieval.py
```

## Calculation and output

The cell below loads and executes the runnable script itself. Expand it to inspect
the source, or {download}`download example_retrieval.py <../../examples/example_retrieval.py>`.
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
:load: ../../examples/example_retrieval.py
:tags: [hide-input]
```

## Interpretation and limits

Read the printed duration against the generating value for each pulse. In the
seeded noisy SHG case, the retrieved duration differs appreciably from the
generating Gaussian; retain that discrepancy when assessing the method. GDD
stretches a pulse beyond its transform-limited input width. Noise can change the
residual and field differently, and clipping negative noisy samples changes the
noise distribution. This script is an API survey, not a coverage or noise-model
validation. Follow [solver selection](../howto/solver_selection.md) and
[retrieval theory](../explanation/retrieval_theory.md) for controlled comparisons.
