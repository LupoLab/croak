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

# Retrieving a simulated RDW pulse

Explore a structured pulse from a simulated resonant-dispersive-wave source,
then a separately simulated TG-FROG instrument.

## Inputs and method

The bundled file `examples/data/tgfrog_sim_rdw_duv.h5` supplies a 2 mm collection
window; this script selects 9.5 µm. It applies defringing and a fourth-power
spectral correction. The extended fit frees smearing scale and delay zero;
the standard fit omits slab and smearing. Both use the same seeded workflow.

Run from a [source checkout](../getting_started/installation.md#from-a-clone-development-and-examples):

```bash
uv run python examples/example_paper_rdw.py
```

## Calculation and output

The cell below loads and executes the runnable script itself. Expand it to inspect
the source, or {download}`download example_paper_rdw.py <../../examples/example_paper_rdw.py>`.
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
:load: ../../examples/example_paper_rdw.py
:tags: [hide-input]
```

```{code-cell} ipython3
:tags: [hide-input]

display(Image(filename="paper_rdw.png"))
```

## Interpretation and limits

Both models fit the trace well, but only the extended model returns the pulse:
the standard thin-medium model gives a smooth pulse more than twice as long as
the reference, at a trace error of a fraction of a percent. A low trace error
alone does not certify a retrieval. Compare the satellites and the spectral
shape as well as the FWHM.

The grey fill is the reference pulse, the mask-vignetted gate beamlet stored with
the dataset, and the retrieved fields are shown in croak's native effective-field
convention. The measured spectrum constrains the retrieval, so agreement of the
spectrum is not independent evidence. The papers' validation uses a different
protocol and compares fields at the on-axis entrance plane; see
[a simulated single-cycle pulse](../explanation/validation.md#a-simulated-single-cycle-pulse)
and [field reference planes](../reference/conventions.md#field-reference-planes).
