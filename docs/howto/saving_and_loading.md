# Saving and loading

{mod}`croak.save` persists retrievals to HDF5 and session options to TOML.

## Results → HDF5

{func}`~croak.save.save_result` writes a retrieval — and, optionally, its
[processed form](postprocessing.md) and the measured trace — to a single `.h5`
file:

```python
pr = croak.process_result(result, measured=td.trace)
croak.save_result(result, "retrieval.h5", processed=pr, measured=td.trace)
```

Pass `force=True` to overwrite an existing file. When the trace came from the
synthetic or simulated workflow, pass the `truth=` keyword (a
{class}`~croak.processing.TruthPulse`) to also store the known **ground-truth**
temporal intensity (`t_truth`, `It_truth`, `fwhm_truth_fs`) and spectrum
(`lam_truth`, `Ilam_truth`). A complex truth additionally stores its temporal and
spectral display phases (`phit_truth`, `phiw_truth`) and original complex spectrum
(`omega_truth`, `Ew_truth`), so a truth-seeded forward-model diagnostic remains
reproducible. Read it back with {func}`croak.save.load_result`:

```python
from croak.save import load_result
data = load_result("retrieval.h5")
```

HDF5 is portable and self-describing, so saved retrievals are easy to share or
re-plot later without re-running the solver.

### Several results in one file

A parameter study — a thickness series, a solver comparison, a noise sweep —
produces many retrievals that belong together. Pass `group=` to write the *same*
flat schema into a named group instead of the file root. The file is appended to
rather than truncated, so each call adds one run and leaves the others alone:

```python
for i, thickness_um in enumerate(thicknesses):
    result, pr = run_one(thickness_um)          # your sweep step
    croak.save_result(
        result, "sweep.h5", processed=pr, measured=td.trace,
        group=f"full/z{i:02d}", force=True,
    )
```

A slash-separated name nests, so `full/z00` and `naive/z00` keep two model
variants of the same point side by side. `force=True` replaces **only** the named
group; without it an existing group raises `FileExistsError`. Root-level datasets
and attributes you write yourself (the swept values, provenance) survive
untouched — useful for recording what the groups mean:

```python
import h5py
with h5py.File("sweep.h5", "a") as f:
    f["thickness_um"] = thicknesses
```

{func}`~croak.save.load_result` recurses into subgroups, so the whole sweep reads
back as a nested dict and each entry is still a complete, standard result:

```python
data = load_result("sweep.h5")
fwhm = [data["full"][k]["fwhm_retr_fs"] for k in sorted(data["full"])]
```

### Rebuilding a result without re-running the solver

{func}`~croak.save.result_from_saved` reconstructs a
{class}`~croak.result.RetrievalResult` straight from a {func}`~croak.save.load_result`
dict — the retrieved spectrum, trace, error history and fitted extras — so a saved
retrieval can be reloaded (and re-processed/plotted) without invoking the solver:

```python
from croak.save import load_result, result_from_saved
result = result_from_saved(load_result("retrieval.h5"))
```

The GUI's **Load previous retrieval** uses exactly this: it replays the cheap
load + preprocess steps and then rehydrates the saved `result.h5` onto the matching
grid, only re-running the solver if the file is absent or the grid no longer
matches.

## Session options → TOML

The GUI (and your own scripts) can persist the *parameters* of a session —
preprocessing choices, the chosen algorithm and its options — as a small TOML
file with {func}`croak.save.save_options` / {func}`croak.save.load_options`:

```python
from croak.save import save_options, load_options
save_options({"algorithm": "copra", "maxiters": 300, "lam_min": 730e-9}, "session.toml")
opts = load_options("session.toml")
```

This stores the *recipe*, not the data — handy for reproducing a retrieval or
seeding the [GUI](gui.md) with known-good settings.

### What the options file contains

A session written by the GUI (or by {meth}`~croak.session.options.SessionOptions.to_toml`)
starts with a small header and then one table per stage:

```toml
croak_version = "0.1.0"
options_version = 2
entry = "simulated"

[load]        # the measured-trace loader (files, datasets, units, backgrounds)
[simulated]   # the numerically simulated loader (scan file, window, band, …)
[preproc]
[retrieve]
[dispersion]
[uncertainty]
```

`entry` names the **stage-1 loader the trace came from** — `"experimental"`,
`"simulated"` or `"synthetic"` — and so which of `[load]` / `[simulated]` is the
live one; the other is written at its defaults. Without it a simulated session
would be indistinguishable from an experimental one with an empty file path, and
its provenance (which scansave file, which trace window, which thickness slice,
which loaded band) would be lost. {func}`croak.run_session`, `croak replay` and
`croak script` all dispatch on it, so a simulated session replays headlessly like
a measured one. Files written before `options_version = 2` have no `entry` and
read back as experimental.

The `"synthetic"` entry is recorded but cannot be replayed: the generator's
settings live only in its GUI controls and are not yet serialised, so reloading
such a session reopens the generator instead of pretending to restore it.

```{note}
TOML has no null, so options whose value is `None` (such as an unset
`z_thickness_um`) are **omitted** rather than written; reading them back restores
the `None` default.
```

## What to keep

- **The result `.h5`** is the authoritative output: it contains the retrieved
  spectrum, the carrier, the trace error and (if you passed them) the processed
  profiles and measured trace.
- **The options `.toml`** records *how* you got there, so the run is reproducible.

Together they make a retrieval fully portable.
