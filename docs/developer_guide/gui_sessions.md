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
[saving, sessions, and reproducible workflows](../user_guide/workflows.md) and
the [GUI guide](../howto/gui.md); the look and layout rules are in
[GUI design](gui_design.md).

## Three layers

The GUI is the thinnest of three layers, and only it imports Qt:

- **`croak.session`** — the per-stage parameter dataclasses
  (`croak.session.params`), the serialisable `SessionOptions`, the mapping of
  each stage onto the library (`croak.session.pipeline`), headless replay
  (`croak.session.engine`), retargeting (`croak.session.retarget`) and a saved
  session's files (`croak.session.files`: `save_session` writes `result.h5`,
  `options.toml` and `retrieval.pdf`; `result_mismatch` says whether a saved
  result was fitted to the trace a replay produced).
- **`croak.plotting.screen`** — what an on-screen view shows, as plain data:
  `MapView` for an image and `LineView` for curves, built by functions such as
  `trace_map`, `temporal_view`, `filter_views` and `uncertainty_views`. They use
  the same helpers as the matplotlib panels, so a curve on screen is the curve in
  an exported figure.
- **`croak.gui`** — Qt glue: widgets bound to the session's parameters, worker
  threads, and pyqtgraph views that draw the screen data.

Importing `croak` or `croak.gui` does not import Qt; {func}`croak.gui.run_gui`
does. A change that needs numerics belongs in the first two layers, where it is
tested without a window.

## The window

| Module | Holds |
|---|---|
| `window` | `MainWindow`: top bar, stage stack, drawer, menus, and the session actions (new, open, replay, retarget, save) |
| `workflow` | the route as pure rules: `Stage`, `Progress`, `blocked_reason`, `next_stage`, `primary_action`, `remember_recent` |
| `home` | Home: the entry tiles, Open a session…, About croak and the recent sessions |
| `about` | About: the version, links, authors and copyright, from the package metadata |
| `pages` | `default_pages` (the page factory), `DataPage` (the entry's loader, and a reopened session's restore) and `AnalysePage` (the rail) |
| `stage` | `PageBase`, the hooks the window reads, and `StagePage`, a page that is a sheet |
| `base` | `Stage`, the interface the stage modules are written against (`controls`, `set_plot_area`, `set_status`, `advance`), and `RunControls` |
| `sheet` | the sheet's parts: `Form`, `Section`, `ReadyLine`, `Segments`, `AdviceBox`, `DerivedGrid`, `ProgressStrip` and `Sheet` itself |
| `stage_load`, `stage_simulated`, `stage_synthetic` | the three loaders (Data) |
| `stage_marginal_check`, `stage_preprocess`, `stage_retrieve`, `stage_dispersion`, `stage_uncertainty` | the other stage pages |
| `view_panel` | `ViewPanel`, a titled plot of one screen view with its right-click menu, and the figure export of a plot or a page (`page_figure`, `export_figure`) |
| `retrieval_views` | the retrieval's panels on screen (`RetrievalPanels`) and Analyse's result pages (`ResultPage`, with Export and Pop out) |
| `plots` | the pyqtgraph views: `Plot`, `MapPlot`, `LinePlot`, the colour maps and the hover readers |
| `theme` | the palette, type scale, style sheet and the role helpers (`label`, `button`, `frame`, `set_flag`) |
| `controls`, `widgets`, `readout`, `mirror_stack` | form builders, slider/spin-box composites, Retrieve's numeric read-out, Dispersion's mirror stack |
| `help` | per-stage help: the dialog and the tooltips, from one `HELP` list |
| `state` | `WizardState`: the parameters and pipeline outputs, with cascading invalidation |
| `worker` | the `QThread` workers for preprocessing, retrieval and uncertainty |
| `settings` | remembered preferences in a `QSettings` store |
| `branding` | the application's name and icon |

### The route

Which stage can open and what the primary action does depend only on
`workflow.Progress`, a frozen snapshot of what the session holds. The rules are
pure functions with doctests and their own tests (`tests/gui/test_workflow.py`);
`MainWindow.refresh` re-applies them to the top bar whenever the state or a page
says something changed. Put a new navigation rule there, not in a widget's slot.

### Pages

A page tells the window what it can do through the `PageBase` hooks:
`on_enter`, `can_advance`, `busy`, `page_key` (which sub-page is shown, for Data
and Analyse), `act(kind)` for the primary action, `overview` (the retrieval
behind a saved session's PDF) and `show_help`. It emits `changed` when the
answers change and `message` for a status-bar notice.

`MainWindow` builds a session's four pages with a factory
(`pages.default_pages`), and rebuilds them for every new or reopened session,
because their controls bind to the parameter objects on the shared
`WizardState`. A test can pass its own factory to `MainWindow`.

Most stage pages subclass `base.Stage`, a `StagePage` that keeps the calls the
stage modules were first written against: sections go into `self.controls`
(from `controls.group(title, folded=…, settings_key=…)`), plots into
`set_plot_area(plot, *strips)`, and messages onto the ready line with
`set_status(text, "ok" | "warn" | "error")`. A loader implements `advance` and
emits `advanced("check")` or `advanced("retrieve")`; `DataPage` calls it on
Continue. A page that runs a worker mixes in `RunControls`, whose `busy` turns
the primary action into Stop.

Help is declared once per page as `HELP = [(section, [(label, text), ...])]`.
`apply_help` attaches each text as the tooltip of the row with that
`(section title, row label)`, the form shows the focused control's tooltip as
its one-line help, and `HelpDialog` lists them all. A row label spells a literal
ampersand `&&` (Qt's mnemonic escape); `HELP` keys are written as the user reads
them.

## Screen views (GUI support)

What a view shows is decided in `croak.plotting.screen`; how it is drawn is
decided in `croak.gui.plots`. The screen functions take the same inputs as the
matplotlib panels — for the retrieval, a `RetrievalPlotData` from
{func}`~croak.plotting.retrieval_plot_data` — and return arrays, labels, ranges
and curve roles. Two display choices differ from the figures: an image is
resampled onto evenly spaced axes at its finest spacing (an interactive image is
placed by its corners, a `pcolormesh` by every edge), and a logarithmic trace
shows only the positive part, leaving negative samples to the signed linear maps.

`retrieval_views.PANELS` maps each key of
{data}`~croak.plotting.RETRIEVAL_PANELS` to its screen function, so the GUI's
pages and the exported figures share the panel names and layouts
({data}`~croak.plotting.RETRIEVAL_PAGES`). A live retrieval calls
`RetrievalPanels.set_data` on the plots it already has, which replaces their data
without rebuilding the widgets. The figures of Analyse's result pages, the
pages' **Save…** buttons and a saved session's `retrieval.pdf` go through the
matplotlib `draw_*` and `plot_*` functions. Every other figure — a plot's
**Export this plot as a figure…**, or **Export this page as a figure…** on a page
without a figure of its own — is drawn from the screen views themselves by
`croak.plotting.export`, with the same role colours (`screen.ROLE_STYLE`), so a
new view can be exported without writing its figure.

The screen functions are Qt-free and tested in `tests/test_plotting_screen.py`;
a new view starts there.

## Threads

The workers in `croak.gui.worker` add only threading, progress signals and
cancellation to the stage operations in `croak.session.pipeline`, so the GUI and
the headless engine run identical computations. `RetrievalWorker` emits live
previews at most every `PREVIEW_MIN_INTERVAL` (0.25 s). Retrieve reports what
presenting each one cost, and the worker stretches the interval to
`cost / PREVIEW_DUTY` (a duty of 0.25), so presenting never takes more than about
a quarter of the wall-clock time on a slow machine.

## Remembered preferences

`croak.gui.settings` keeps the window geometry, each folding section's state
(sections with a `settings_key`), the read-out strip, Retrieve's spectrum axis
and the recent sessions in a `QSettings` INI store, opening a short-lived store
for every read or write. It always names the organisation and application
explicitly, which is what lets the test suite redirect the store to a per-test
directory with `QSettings.setPath` (`tests/conftest.py`).

## Testing the GUI

The GUI tests use pytest-qt with `qt_api = "pyside6"` and run offscreen
(`QT_QPA_PLATFORM=offscreen`, set in `tests/conftest.py` and the pytest
environment), driving the real widgets through `qtbot`. They count towards the
100 % coverage gate like everything else.

Many stage tests were written for the old six-stage wizard.
`tests/gui/gui_session.py` keeps them working: its `Driver` builds a
`MainWindow`, maps the old stage numbers onto the window (1 the loader on Data,
2 Check, 3 Prepare, 4 Retrieve, 5 Dispersion and 6 Uncertainty on
Analyse) and hands back the stage pages, which keep their behaviour. `goto`
bypasses the route rules, as the old stage setter did, so a test can look at a
page before its input exists. Tests of navigation itself use the window and the
route rules directly (`test_window.py`, `test_window_session.py`,
`test_workflow.py`).

When testing saved window geometry, expose both the original and restored
window with `qtbot.waitExposed` before comparing their displayed sizes. Qt
clamps saved geometry to the available screen before applying layout minimums
on show. Windows font metrics can make those minimums wider than the offscreen
test display, so comparing a shown window to a hidden one tests different
layout states.
