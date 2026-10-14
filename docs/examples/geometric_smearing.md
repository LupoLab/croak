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
Gaussian timing kernel. Every retrieval uses `lbfgs-ad` without regularisation,
since this is a matched, noise-free comparison. It runs to tight stopping
tolerances from two Gaussian starts (1.0 and 1.6 fs) and keeps the lowest-error
fit. The two sweeps compare smeared and unsmeared traces at otherwise matched
settings. The script takes about two minutes.

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

With the kernel modelled, or with its common scale fitted (which returns 1.0000),
both starts recover the 1 fs pulse with the trace error at round-off. With the
kernel ignored, the model cannot reproduce the trace, and the result shows which
quantities carry the signature of the missing kernel:

- **The residual is robust.** The kernel-ignored trace error stays near
  $5 \times 10^{-3}$ for the nominal mask and grows steadily with the mask ratio,
  from $2.4 \times 10^{-3}$ at $d/D = 0.5$ to $1.1 \times 10^{-2}$ at
  $d/D = 1.25$. Both starts agree on it.
- **The FWHM is not a reliable measure.** Without a slab, the best fit absorbs
  the blur into broad, low-level temporal wings: the rms width grows from
  0.43 fs to 3.8 fs, while the FWHM comes out 0.065 fs *short*. Across the mask
  sweep the FWHM offset is negative and not monotonic in $d/D$. With a slab the
  offset turns positive (+0.22 fs at 10 µm, +0.11 fs at 20 µm, +0.08 fs at
  40 µm), and at 10 µm and 40 µm the two starts reach the same trace error with
  FWHMs about 0.06–0.08 fs apart. Both fits stop on the tolerances, well short
  of the iteration cap, so the spread reflects a nearly flat direction of the
  misspecified objective rather than a fit stopped early. A smeared-minus-clean
  difference cannot remove it.

Fitting the scale recovers the kernel in this matched simulation; on real data it
can also absorb other model errors. For a transform-limited Gaussian the
split-gate channel is an amplitude factor, so the scale is constrained mainly
through the delay channel. Read
[geometric smearing](../howto/geometric_smearing.md) and
[collection](../howto/collection_aperture.md).
