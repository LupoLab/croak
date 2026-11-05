"""croak's desktop GUI (PySide6, pyqtgraph plots).

The GUI is a thin layer over the croak library: it loads, cleans, retrieves,
post-processes, tunes dispersion, estimates uncertainty, plots and saves — all
through the library's functions and the Qt-free :mod:`croak.session`. Its window
takes a session from Home through Data, Prepare, Retrieve and Analyse.

Launch with the ``croak-gui`` command, :func:`run_gui`, or ``python -m croak.gui``.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Make the lazy __getattr__ re-exports visible to the type checker without
    # importing Qt at module import time (the runtime path is __getattr__ below).
    from .state import WizardState
    from .window import MainWindow

__all__ = ["MainWindow", "WizardState", "main", "run_gui"]


def run_gui(argv=None) -> int:
    """Create a ``QApplication`` (if needed) and run the window's event loop.

    The window opens maximised — the plots want the room. Un-maximising gives
    the size the user last left it (remembered between launches by
    :mod:`croak.gui.settings`, clamped to the current screen), or on a first run
    a default fitted to 90 % of the screen.

    ``argv[0]`` is replaced with the application name before the ``QApplication``
    is built: macOS takes its menu-bar title (and the "About …"/"Quit …" items)
    from it, so it would otherwise read ``croak-gui`` via the console script and
    ``__main__.py`` via ``python -m croak.gui``. Any further arguments are passed
    through untouched, so Qt's own flags (``-style``, ``-platform``, …) still work.
    """
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:  # the gui extra is optional — say how to get it
        raise ImportError(
            "croak's GUI needs the optional Qt dependencies, which are not "
            "installed. Install them with `pip install 'croak[gui]'` (or, in "
            "a uv checkout, `uv sync` — the dev group carries them). "
            "Everything except the GUI runs without Qt."
        ) from exc

    from .branding import APP_NAME, apply_identity
    from .window import MainWindow

    args = list(argv if argv is not None else sys.argv)
    app = QApplication.instance() or QApplication([APP_NAME, *args[1:]])
    apply_identity()
    window = MainWindow()
    window.showMaximized()
    return app.exec()


def main() -> int:
    """Console-script entry point (``croak-gui``); runs the GUI."""
    return run_gui()


def __getattr__(name):  # lazy re-exports (avoid importing Qt at module import)
    if name == "MainWindow":
        from .window import MainWindow

        return MainWindow
    if name == "WizardState":
        from .state import WizardState

        return WizardState
    raise AttributeError(name)
