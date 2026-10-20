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

An embedded `MplCanvas` keeps its owning Qt window alive for the duration of a
draw. A queued Matplotlib draw holds the Python canvas wrapper, but that alone
does not keep Qt from deleting the canvas when its parent is garbage-collected.
Retaining the window during rendering prevents a discarded stage from losing
its canvas midway through a draw; it does not retain the window after drawing.

When testing saved window geometry, expose both the original and restored
window with `qtbot.waitExposed` before comparing their displayed sizes. Qt
clamps saved geometry to the available screen before applying layout minimums
on show. Windows font metrics can make those minimums wider than the offscreen
test display, so comparing a shown window to a hidden one tests different
layout states.
