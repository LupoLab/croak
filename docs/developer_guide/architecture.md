# Package architecture

The package separates the physical model, retrieval, and measurement workflow.
The GUI calls the workflow layer; scripts and saved sessions can use that layer
without opening a window.

| Area | Main responsibilities | Reference |
|---|---|---|
| Core physics | Grids, transforms, pulse construction, interactions, propagation, focusing, and collection | [Core physics API](../reference/api/core.md) |
| Retrieval | Solver selection, objectives, optimisation, progress, and result assembly | [Retrieval API](../reference/api/retrieval.md) |
| Workflow | Loading, preprocessing, post-processing, uncertainty, persistence, and session replay | [Workflow API](../reference/api/workflow.md) |
| Plotting | Plot data and visualisation of traces, fields, and diagnostics | [Plotting API](../reference/api/plotting.md) |
| JAX calculations | Differentiable forward models, residuals, and pulse parameterisation | [JAX API](../reference/api/jax.md) |

Keep physical calculations independent of GUI state and file I/O. Prefer the
shared workflow operations when adding a command or GUI control so that the
same settings retain the same meaning in scripts and saved sessions.

The [forward-model chapter](../user_guide/models.md) explains the physics;
[retrieval and numerical methods](../user_guide/retrieval.md) explains the
objectives and derivatives.
