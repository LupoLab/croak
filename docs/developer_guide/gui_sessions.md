# GUI and session integration

The GUI, saved-session replay, and generated scripts should express the same
workflow. Session parameters and the shared stage operations provide the link
between interactive controls and headless calculations.

When adding a setting, trace it from the session parameter definition through
the stage operation to the calculation it controls. Check loading and saving,
GUI state, replay, and script generation where the setting is exposed. Keep
Qt-specific behaviour at the interface boundary.

The [session API](../reference/api/workflow.md#sessions-headless-replay) documents
the serialisable settings and execution interfaces. The
[command-line and scripting API](../reference/api/workflow.md#command-line-and-script-generation)
covers headless use and generated scripts. User-facing instructions belong in
[saving, sessions, and reproducible workflows](../user_guide/workflows.md).
