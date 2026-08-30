# Installing the GUI

This page is for people who want the graphical FROG retrieval program and
nothing else — no Python scripting, no library API. It gets you from zero to a
running wizard in four commands.

```{note}
Standalone installers (a downloadable app, no Python involved) are planned.
Until then the GUI installs through [uv](https://docs.astral.sh/uv/), which
handles Python and every dependency for you.
```

## 1. Install uv

macOS / Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or `brew install uv` on macOS. See the
[uv install docs](https://docs.astral.sh/uv/getting-started/installation/) for
other options.

## 2. Install croak

```bash
uv tool install --python 3.14 "croak[gui]"
```

uv downloads the right Python (3.14) if you do not have it, creates an
isolated environment, installs croak with the Qt interface from PyPI, and puts
the `croak-gui` command on your PATH. No system Python is touched.

On **Linux**, Qt also needs a set of system libraries that no Python installer
provides:

```bash
sudo apt-get install -y libegl1 libgl1 libxkbcommon-x11-0 libdbus-1-3 \
  libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
  libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 libxcb-cursor0
```

## 3. Run it

```bash
croak-gui
```

That is the whole install; the command works from anywhere. If the window
fails to appear on Windows, run
`uvx --python 3.14 --from "croak[gui]" python -m croak.gui` instead — the
`croak-gui` entry point opens no console, so that variant is the one that
shows a traceback.

```{image} ../_static/gui_retrieve.png
:alt: The wizard after a retrieval — measured and retrieved traces, residuals, the retrieved pulse and spectrum, and a numeric read-out.
:width: 95%
:align: center
```

## 4. Use it

The wizard walks through the retrieval in stages: **Load** (your measured
trace, in HDF5, `.npz` or delimited text — or a simulated scan), **Marginal
check**, **Preprocess** (filtering, background removal, regridding),
**Retrieve**, **Dispersion** (tuning with materials and chirped mirrors) and
**Uncertainty** (error bars on the retrieved duration). Every control has a
tooltip and every stage a help button.

The [GUI guide](../howto/gui.md) documents each stage in detail. Sessions save
to a TOML file that records every setting, so a retrieval can be re-run later
— including headlessly with `croak replay session.toml` — or exported as a
standalone Python script with `croak script session.toml` (the `croak` command
is installed alongside `croak-gui`).

## Updating

```bash
uv tool upgrade croak
```

If you would rather run from a checkout of the source — for example to try the
bundled examples too — clone the repository and use
`uv sync --extra gui` then `uv run croak-gui` from the `croak` directory; see
[Installation](installation.md).
