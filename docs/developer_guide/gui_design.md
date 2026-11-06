# GUI design

This page records how croak's window is meant to look and behave, so that a new
page or control fits the ones already there. It is a living page: change it in
the same pull request as a change to the GUI's look, layout or page set. The
code it describes is `croak/gui/theme.py` (colours, type and style sheet),
`croak/gui/sheet.py` (the stage layout) and `croak/gui/window.py` (the top bar);
[GUI and session integration](gui_sessions.md) covers the modules and their
contracts, and the [GUI guide](../howto/gui.md) what a user sees.

The window follows the look of LupoLab's solprop front end, so the two
applications read alike.

## Principles

- **One primary action.** Each place has one button that moves the session on,
  at the right of the top bar, and it is the only button drawn in the accent
  colour.
  Everything else is secondary: plain buttons on the form, the ⋯ menu, the menu
  bar. What the action is, and when a stage can open, are pure rules in
  `croak.gui.workflow`, not decisions scattered through slots. A stage that
  cannot open yet is disabled with the reason as its tooltip, never hidden.
- **Controls beside their plots.** A stage is one sheet: its settings on the
  left, the plots they change on the right, and changing a setting redraws the
  plots without a separate Apply where the computation is cheap enough.
- **No physics in the GUI.** `croak.gui` binds widgets to the parameters of
  `croak.session` and calls the library. A calculation the GUI needs is added to
  the library or to `croak.session`, where scripts and replay get it too and it
  is tested without a window.
- **Screen data shared with figures.** What a plot shows is decided in
  `croak.plotting.screen` from the same helpers as the matplotlib panels; the
  GUI only draws it. An exported figure is the matplotlib figure of the same
  data, so the screen and the saved file agree.
- **Help has one source.** A setting's help is written once, in its page's
  `HELP`, and appears as its tooltip, as the form's one-line help when it has
  focus, and in the page's help dialog.
- **Say it where it happens.** Whether a stage can go on is on its ready line,
  at the foot of the form; a passing notice (saved, copied, why a stage is
  closed) goes to the status bar, which hides when it is empty. Modal dialogs are
  for choosing files and for confirming that work will be discarded.
- **Remember arrangement, not results.** The window's size, folded sections,
  the read-out strip and display choices persist between launches; computed
  results persist only in saved sessions.

## Colour

Every colour is a token of `theme.Palette`, and the one style sheet is built
from the tokens; nothing else in the GUI calls `setStyleSheet`. Add a token
rather than a literal colour.

| Token | Value | Use |
|---|---|---|
| `ink` | `#1b2430` | text |
| `muted` | `#5a6675` | secondary text: help, units, plot tick labels |
| `heading` | `#3d4957` | small headings, unselected rail items |
| `disabled` | `#a4adb8` | disabled text |
| `axis` | `#b4bcc6` | plot axes |
| `accent` | `#0b6e99` | the primary action, focus, sliders and progress |
| `accent_hover` / `accent_off` | `#095d82` / `#b9c4cc` | the primary action under the pointer / disabled |
| `selected_bg` / `selected_ink` | `#e8f1f5` / `#0a5a7d` | the current stage, rail item or segment; links |
| `hover_bg` | `#f1f3f5` | a flat control under the pointer |
| `hairline` | `#e3e6ea` | lines between the window's regions |
| `border` | `#cdd3da` | inputs and plain buttons |
| `ok` / `warn` / `error` | `#1c7a4a` / `#8a5200` / `#b3261e` | the ready line's three states |
| `advice_bg` / `advice_border` | `#fdf8ee` / `#e6cfa8` | the amber advice box |
| `background` | `#ffffff` | everything else, plots included |

Curves take their colours in order from `plots.COLOURS` (blue, orange, green,
purple, red, grey, ochre, cyan), each line 1.7 px wide; a curve's role
(retrieved, measured, truth, phase, …) fixes its colour and dash, as in the
figures. Maps start white at zero: viridis for positive values and, on a signed
linear map, red for negative ones.

## Type

The window uses IBM Plex Sans (regular, medium and semibold), bundled in
`croak/gui/assets/fonts` under the SIL Open Font License. If it fails to load,
the system font is used and a warning is logged. The scale, in logical pixels:

| Role | Size | Weight |
|---|---|---|
| Home's heading (`h1`) | 28 | semibold |
| Page and brand headings (`h2`) | 15 | semibold |
| A derived value | 19 | medium |
| Body text, inputs, buttons | 13 | regular |
| Help, notes, plot titles, small labels | 12 | regular or medium |
| Plot ticks and axis labels | 11 | regular |

Widgets take their style from their *role*, which is their `objectName`
(`theme.label(text, "h3")`, `theme.button(text, "primary")`,
`theme.frame("form")`), and from boolean properties such as `current`, `warn`
and `error`, which `theme.set_flag` sets and re-applies.

## Layout

- **Target**: 1366×768 logical pixels and up, opened maximised.
- **Top bar**: 52 px high, with a hairline beneath. Brand and context at the
  left; Home, a hairline divider and the stage buttons in the middle; the
  primary action and ⋯ at the right.
- **Sheet**: the form is a fixed 340 px column (`sheet.FORM_WIDTH`) and the plots
  take the rest of the width. Below a window width of 1000 px the plots move
  under the form, which then spans the window. The form scrolls on its own; its
  help line and ready line stay at its foot, and its combo boxes shrink to the
  column rather than widen it.
- **Sections** fold in place. Settings most sessions leave alone start folded
  and remember their state (a `settings_key`), as Retrieve's advanced sections
  do.
- **Plots** are white, closed boxes (top and right axes drawn without numbers),
  with a crosshair and readout on hover, mouse zoom and pan, and a double-click
  back to the starting view. Plots that share an axis are linked. A plot is at
  least 140×100 px.
- **Strips** beneath the plots that have a natural height (Retrieve's numeric
  read-out) fold away and hand their height back to the plots.
- **Analyse** has a 170 px rail of pages on the left.
- **Export**: every plot has a right-click menu (figure, image, copy), and every
  page a figure (`PageBase.figure`), saved by Ctrl+E. Figures are drawn from the
  screen views by `croak.plotting.export`, with maps rasterized in a vector
  file.
- **Drawer**: 440 px on the right, for text the user may copy (the settings,
  the script).

## Page catalogue

| Place | Page | Primary action | Plots |
|---|---|---|---|
| Home | tiles, Open a session…, About croak, recent sessions | none | none |
| About | version, links (manual, GitHub, PyPI, issues), authors, copyright | none | none |
| Data | Load the trace / Load the scan / Generate the trace | Continue | the loaded trace; for a scan its maps and marginals; for a synthetic pulse the pulse, spectrum, spectrogram and trace |
| Check | — | Continue | the trace and the marginal comparison |
| Prepare | — | Continue | measured, filtered and regridded traces in dB and linear (2×3) |
| Retrieve | — | Retrieve / Stop / Retrieve again | the live overview, or convergence before the first preview; the numeric read-out |
| Analyse | Overview, Traces, Pulse, Diagnostics | Save | the retrieval's panels, two by two |
| Analyse | Dispersion | Save | dispersed spectrum, pulse, applied phase, spectrogram |
| Analyse | Uncertainty | Estimate / Stop | FWHM distribution (or FWHM and error against thickness), the temporal band |

## Left for later

These are known gaps, kept out of the redesign to keep it reviewable:

- **A dark theme.** The style sheet is built from one `Palette`, so a dark one
  would be a second instance, but the plots' white background and colour maps
  would need choosing too.
- **The unused sheet parts.** `sheet.DerivedGrid` (numbers derived from the
  settings) and `ProgressStrip` (state, bar and figures over the plots while
  work runs) are built but no stage uses them yet; a run's progress is on the
  ready line. `AdviceBox` so far carries one piece of advice: Prepare's, when
  the λ grid clips the trace.
- **The stage pages' interface.** The stage modules still build through
  `base.Stage`, which keeps the calls they were written against for the old
  wizard; moving them onto `StagePage` directly is a later clean-up.
- **Find a setting** is offered on Retrieve, the longest form, only.
- **Two ways to save a retrieval.** Retrieve's own **Save…** writes the same
  three files as **Save session…**, without the uncertainty estimates and
  without adding the session to Recent sessions.
