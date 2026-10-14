# GUI or script interface

Use the GUI when you want to inspect a trace interactively, adjust preparation
settings, and compare plots as you work. Use a Python script when you want an
explicit, repeatable sequence of operations or need to integrate retrieval into
your own analysis. You can move between these approaches using saved sessions.

## Work in the GUI

After [installing the GUI](../getting_started/gui.md), launch `croak-gui`.
The wizard guides you through loading data, preparing a trace, retrieving a
pulse, and analysing the result. The [GUI guide](../howto/gui.md) describes the
controls and stages; [preparing and checking data](../user_guide/data.md)
explains the choices behind them.

## Work in Python

The [Quickstart](../getting_started/quickstart.md) shows a small retrieval using
`croak.Grid`, `croak.maketrace`, and `croak.retrieve`. Continue with
[your first retrieval](../tutorials/01_first_retrieval.md) for an explanation of
the calculation and its plots. The [experimental workflow](../tutorials/04_experimental_workflow.md)
introduces loading, cleaning, regridding, and saving.

## Reuse a session

A saved session records the settings used for the workflow. The `croak replay`
command reruns a session without opening the GUI, and `croak script` generates
a Python script you can inspect and adapt. See [saving, sessions, and
reproducible workflows](../user_guide/workflows.md) for the commands and file
formats.

Whichever interface you choose, the [user guide](../user_guide/index.md)
explains the physics and interpretation, and the [examples](../examples/index.md)
show specific applications.
