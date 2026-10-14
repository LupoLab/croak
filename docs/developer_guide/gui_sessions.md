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

## Live views (GUI support)

Two classes redraw nothing when new data arrive: they draw a figure once and then
push later data onto the existing artists, which is what keeps the wizard's live
previews cheap and steady. {class}`~croak.plotting.FrogFilterView` does this for
the six-panel filter figure (`attach(fig)` then `update(td)`), and
{class}`~croak.plotting.RetrievalPageView` for one page of retrieval panels
(`draw(fig, data)`, then `update(data)` whenever `can_update(data)` says the new
result differs only in values — same grid, same optional curves). Behind the page
view, every `draw_<key>` has an `update_<key>(artists, data)` twin, and the
invariant the tests hold them to is that `update(draw(d1), d2)` shows exactly
what `draw(d2)` shows. {func}`~croak.plotting.plot_convergence` returns its
artists for the same reason, and `update_convergence_curve` grows the error
history in place.
