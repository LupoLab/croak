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

# A simulated substrate-thickness series

Compare thin-medium and slab-plus-smearing fits at three substrate thicknesses
using the bundled reduced Gaussian instrument simulation.

## Inputs and method

The local file `examples/data/tgfrog_sim_1fs_uvfs.h5` contains slices at 4, 9.5
and 40 µm with a 0.5 mm collection window. The script uses defringing, an empirical
fourth-power spectral correction, the integrated beamlet spectrum, package
regularisation and fitted delay zero. Both models start from a transform-limited
1 fs estimate with a fixed seed.

Run from a [source checkout](../getting_started/installation.md#from-a-clone-development-and-examples):

```bash
uv run python examples/example_paper_thickness.py
```

## Calculation and output

The cell below loads and executes the runnable script itself. Expand it to inspect
the source, or {download}`download example_paper_thickness.py <../../examples/example_paper_thickness.py>`.
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
:load: ../../examples/example_paper_thickness.py
:tags: [hide-input]
```

```{code-cell} ipython3
:tags: [hide-input]

display(Image(filename="paper_thickness.png"))
```

## Interpretation and limits

The plot reports native retrieval fields against the stored integrated-beamlet
reference. It has no common-plane conversion and no full chromatic collection
model. It illustrates protocol sensitivity and model comparison; it is **not** a
reproduction of the revised draft thickness ladder. Its filtering, spectrum frame,
weights and fitted parameters differ. Use [validation](../explanation/validation.md)
for the revised common-plane results, and [simulated loading](../howto/loading_simulated.md)
for the file conventions. A low residual here must not be read as proof of a
correct entrance-plane pulse.
