# Saving, sessions, and reproducible workflows

A result is reproducible when both the retrieved field and the settings that
produced it are kept. croak saves results to HDF5 and every setting of an
analysis to a TOML session file, which can be replayed without the GUI or turned
into an editable Python script. This chapter covers the graphical wizard, which
writes those sessions, and the saving and loading functions behind it.

## Which page do I need?

| I want to… | Read |
|---|---|
| use the wizard stage by stage | [The wizard GUI](../howto/gui.md) |
| save a result, or several, and load them back | [Saving and loading](../howto/saving_and_loading.md) |
| rerun a saved session headlessly or retarget it at new data | [Saving and loading](../howto/saving_and_loading.md) |
| decide between the GUI and a script | [GUI or script interface](../introduction/interfaces.md) |

```{toctree}
:maxdepth: 1

../howto/gui
../howto/saving_and_loading
```

## Key concepts

A **result file** (`result.h5`) holds the retrieved spectrum, grid, predicted
trace and convergence history, optionally with the measured trace, processed
quantities, a known truth pulse and uncertainty results. A
**session file** (`options.toml`) holds the per-stage parameters: loader,
preprocessing, retrieval, dispersion and uncertainty. `croak replay
options.toml` reruns a session; `croak script options.toml` writes the
equivalent Python script.

## Main entry points

- {func}`~croak.save.save_result`, {func}`~croak.save.load_result`,
  {func}`~croak.save.result_from_saved`: results in HDF5.
- {func}`~croak.save.save_options`, {func}`~croak.save.load_options`: sessions in TOML.
- {func}`~croak.session.engine.run_session`, {func}`~croak.session.retarget.retarget`:
  headless replay.
- {func}`~croak.scripting.generate_script`: export a session as a script.

The [experimental workflow](../tutorials/04_experimental_workflow.md) follows
the sequence from loading to saving, and the
[session API](../reference/api/workflow.md#sessions-headless-replay) documents
programmatic use.
