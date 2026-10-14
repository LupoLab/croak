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
adds seeded detector noise, and retrieves at the central thickness from a 4 fs
Gaussian start with the `lbfgs-ad` solver. The fundamental spectrum is supplied
as a spectral-match target, as an independently measured spectrum would be in an
experiment; here it is the exact spectrum. The script then runs 40 thickness and
40 parametric replicates with fixed seeds and the same solver settings. The 128 fs
time window contains the ±45 fs delay scan and the dispersed pulse.

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

The base retrieval returns 2.97 fs for the generating 3.00 fs pulse. The
thickness replicates lie on a smooth curve: the FWHM falls steadily as the
assumed thickness grows, and the trace error is smallest at the true 100 µm. The
thickness prior of 100 ± 5 µm adds ±0.008 fs to the FWHM, while the parametric
bootstrap gives 0.0013 fs. The substrate term dominates the error budget by about
a factor of six, the situation for thin deep-UV substrates that this example is
modelled on. The statistical bootstrap alone would understate the uncertainty.

### Why the spectrum and `lbfgs-ad`

The grid spans twice the carrier frequency, so most trace rows contain only
noise. Without a spectral target, both `lbfgs-ad` and the default `warm-lbfgs`
reproduce that noise by putting about 3 % of the spectral energy into the
outermost frequency bins, where the trace carries no signal. Their trace error
then falls to about 0.31 %, below the 0.354 % of the true pulse, and the FWHM
lands anywhere from about 1.9 to 3.2 fs depending on the start and assumed
thickness. The thickness replicates then scatter between about 1 and 3.5 fs.
Supplying the spectrum switches on croak's default spectral-match penalty
(`reg_spectrum`), which ties the retrieved amplitude to the measured band.
`warm-lbfgs` begins with a COPRA warm-up that has no spectral penalty; even with
the target it ends at 3.16 fs and a trace error of 0.350 %, slightly below that
of the true pulse. `lbfgs-ad` starts the penalised fit directly.

On your own retrievals, check {func}`croak.edge_energy_fraction`: a few per cent
indicates this failure. Supply the measured fundamental spectrum: pass
`lam_spec`/`Ilam_spec` to {func}`croak.load_and_clean`, and
{func}`croak.retrieve_from_tracedata` uses it as the target.

### Running this on your own data

A session saved from the GUI as `options.toml` already records the loader,
preprocessing and a dispersive retrieval, including the material and central
thickness. Set the uncertainty method to `"thickness"` with your measured 1σ and
replay it headlessly:

```python
from dataclasses import replace

import croak

options = croak.SessionOptions.from_toml("options.toml")
options = options.with_(
    uncertainty=replace(options.uncertainty, method="thickness", thickness_sigma_um=0.539)
)
session = croak.run_session(options, stages=("load", "preproc", "retrieve", "uncertainty"))
print(session.uncertainty.summary())
```

The central thickness is `options.retrieve.thickness_um`, and the retrieval must
be dispersive. To call {func}`croak.thickness_bootstrap` directly, pass it
`session.result` and `session.tracedata.trace`, with the material, quadrature
`npoints` and `algorithm` the retrieval used. In the GUI the same estimate is
stage 5 with method "thickness". See [saving and loading](../howto/saving_and_loading.md).

### Limits

The plot shows the duration's conditional sensitivity to the assumed thickness
prior. Forty replicates demonstrate the method but are too few for precise tail
probabilities. The printed quadrature combination approximates independent,
roughly Gaussian contributions; for skewed distributions use
{func}`croak.uncertainty.combine_uncertainties` and assess dependence explicitly.
Residual-based noise can include model error. Neither the statistical spread nor
the combined value is automatically a calibrated experimental interval. Read
[uncertainty methods](../howto/uncertainty.md) and
[their assumptions](../explanation/uncertainty_estimation.md).
