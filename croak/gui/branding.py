"""Application icon and desktop identity for the wizard.

Keeps the window-system-facing identity — the name the desktop shows, the icon in
the macOS Dock / Windows taskbar / Linux dock — in one place, out of
:func:`~croak.gui.run_wizard`.

This module imports Qt, so it must only be imported from *inside* the functions
that already need Qt: importing :mod:`croak.gui` deliberately does not pull PyQt6
in, so the library stays usable headless.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

__all__ = ["APP_NAME", "ICON_PATH", "app_icon", "apply_identity"]

#: Name the desktop shows for the application (menu bar, taskbar, About/Quit).
APP_NAME = "croak"

#: Reverse-DNS-ish identifier Windows uses to group and pin taskbar entries.
_APP_USER_MODEL_ID = "croak.gui"

#: The icon master, located the way the package's other data files are (see
#: ``croak/materials.py`` and ``croak/mirrors.py``); hatchling's ``packages =
#: ["croak"]`` ships everything under ``croak/``, so it needs no build config.
ICON_PATH = Path(__file__).parent / "resources" / "croak.svg"

#: Sizes rendered from the SVG. A vector-backed QIcon reports no
#: ``availableSizes()``, and some platforms pick their raster from that list, so
#: the icon is built from explicit pixmaps rather than handed the file directly.
_ICON_SIZES = (16, 32, 64, 128, 256, 512)


def app_icon() -> QIcon:
    """Build the application icon from the packaged SVG.

    Requires a ``QGuiApplication`` to exist (rendering a pixmap does).

    Returns
    -------
    PyQt6.QtGui.QIcon
        The icon, carrying a pixmap at each of :data:`_ICON_SIZES`. Null only if
        the SVG is missing or Qt's SVG image plugin is unavailable.
    """
    base = QIcon(str(ICON_PATH))
    if base.isNull():
        return base
    icon = QIcon()
    for size in _ICON_SIZES:
        icon.addPixmap(base.pixmap(size, size))
    return icon


def apply_identity() -> None:
    """Give the running application its name, version and icon.

    The icon is what the macOS Dock, the Windows taskbar and the Linux dock show;
    Qt also uses the name for ``QSettings`` paths and the About box. Qt's setters
    are static and apply to the current application, so this takes no argument —
    but a ``QApplication`` must already exist.

    Note the macOS *menu bar* title is not set from here — Qt takes it from
    ``argv[0]``, which :func:`~croak.gui.run_wizard` overrides when it constructs
    the ``QApplication``.
    """
    from .. import __version__

    QApplication.setApplicationName(APP_NAME)
    QApplication.setApplicationDisplayName(APP_NAME)
    QApplication.setApplicationVersion(__version__)
    QApplication.setWindowIcon(app_icon())
    _set_windows_app_id()


def _set_windows_app_id() -> None:
    """Tell Windows this process is its own application, not the interpreter.

    Without an explicit AppUserModelID the taskbar groups the window under (and
    pins) ``python.exe``. Must run before any window is created. No-op elsewhere,
    and deliberately best-effort: a failure here is cosmetic.
    """
    if sys.platform != "win32":
        return
    import ctypes

    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            _APP_USER_MODEL_ID
        )
    except AttributeError, OSError:  # pragma: no cover - Windows-only path
        pass
