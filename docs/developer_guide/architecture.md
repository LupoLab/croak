# Package architecture

The package separates the physical model, retrieval, and measurement workflow.
The GUI calls the workflow layer; scripts and saved sessions can use that layer
without opening a window.

| Area | Main responsibilities | Reference |
|---|---|---|
| Core physics | Grids, transforms, pulse construction, interactions, propagation, focusing, and collection | [Core physics API](../reference/api/core.md) |
| Retrieval | Solver selection, objectives, optimisation, progress, and result assembly | [Retrieval API](../reference/api/retrieval.md) |
| Workflow | Loading, preprocessing, post-processing, uncertainty, persistence, and session replay | [Workflow API](../reference/api/workflow.md) |
| Plotting | Plot data, matplotlib figures, and the Qt-free screen views the GUI draws | [Plotting API](../reference/api/plotting.md) |
| JAX calculations | Differentiable forward models, residuals, and pulse parameterisation | [JAX API](../reference/api/jax.md) |
| GUI | The PySide6 window: Home, the stages, worker threads, and pyqtgraph views | [GUI and session integration](gui_sessions.md) |

Keep physical calculations independent of GUI state and file I/O. Prefer the
shared workflow operations when adding a command or GUI control so that the
same settings retain the same meaning in scripts and saved sessions.

## The GUI

The GUI (`croak.gui`, the `gui` extra) sits on two Qt-free layers. The session
layer, `croak.session`, holds every setting as parameter dataclasses, maps each
stage onto the library, replays a saved session and writes a session's files;
the GUI, `croak replay` and `croak script` all go through it. The screen layer,
`croak.plotting.screen`, turns the data behind a figure into the arrays an
interactive view draws, using the same helpers as the matplotlib panels, so
the screen and an exported figure agree.

On top of them, `croak.gui.window.MainWindow` takes a session from Home through
Data, Prepare, Retrieve and Analyse. The route between the stages is a set of
pure rules (`croak.gui.workflow`); each stage is a page that binds its controls
to the session's parameters and hands long computations to worker threads; and
every on-screen plot is a pyqtgraph view of screen data. Numerics belong in the
library and the two layers beneath, not in `croak.gui`. [GUI and session integration](gui_sessions.md) describes the
modules and their contracts, and [GUI design](gui_design.md) the look and layout
rules.

The [forward-model chapter](../user_guide/models.md) explains the physics;
[retrieval and numerical methods](../user_guide/retrieval.md) explains the
objectives and derivatives.
