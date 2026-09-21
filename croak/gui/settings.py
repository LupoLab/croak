"""Persistent user preferences: window geometry, control-column width, folded strips.

The wizard remembers how its user last arranged it — the un-maximised window
size and position, the width of the control column, which auxiliary strips were
folded — so a small-screen user sets things up once. Values live in a
:class:`~PyQt6.QtCore.QSettings` store in INI format under the user scope, keyed
by the organisation and application names from :mod:`.branding`; passing those
explicitly means the store is the same whether or not
:func:`~croak.gui.branding.apply_identity` has run (the tests build a
:class:`~croak.gui.wizard.Wizard` without it).

Every function opens a short-lived ``QSettings``, reads or writes one value and
syncs, so the side effects stay at the edges and the module keeps no state. Only
the explicit-constructor form is used: the test suite redirects the store with
``QSettings.setPath(IniFormat, UserScope, directory)``, which that form honours,
whereas a default-constructed ``QSettings()`` would write to the developer's
real preferences.
"""

from __future__ import annotations

from PyQt6.QtCore import QByteArray, QSettings, QSize
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QWidget

from .branding import APP_NAME, ORGANISATION_NAME

__all__ = [
    "KEY_WINDOW_GEOMETRY",
    "KEY_CONTROLS_WIDTH",
    "DEFAULT_WINDOW_SIZE",
    "DEFAULT_CONTROLS_WIDTH",
    "SCREEN_FILL",
    "user_settings",
    "fit_to_screen",
    "restore_window",
    "save_window",
    "controls_width",
    "set_controls_width",
    "value",
    "set_value",
]

#: Settings key holding ``QWidget.saveGeometry()`` of the main window.
KEY_WINDOW_GEOMETRY = "window/geometry"
#: Settings key holding the control column's width in logical pixels.
KEY_CONTROLS_WIDTH = "stage/controls_width"
#: First-run window size (logical px) before clamping to the screen; large enough
#: for the plot pages at full text density on a 1920x1080 display.
DEFAULT_WINDOW_SIZE = QSize(1500, 950)
#: First-run width (logical px) of the control column on every stage.
DEFAULT_CONTROLS_WIDTH = 315
#: Fraction of the screen's available area a first-run window may fill.
SCREEN_FILL = 0.9


def user_settings() -> QSettings:
    """Open the wizard's settings store (INI, user scope, explicit names)."""
    return QSettings(
        QSettings.Format.IniFormat,
        QSettings.Scope.UserScope,
        ORGANISATION_NAME,
        APP_NAME,
    )


def fit_to_screen(size: QSize, available: QSize, *, fill: float = SCREEN_FILL) -> QSize:
    """Shrink ``size`` to at most ``fill`` of ``available``; never enlarge it.

    Parameters
    ----------
    size : QSize
        Requested window size (logical px).
    available : QSize
        The screen's available geometry (excluding docks and menu bars).
    fill : float
        Largest fraction of ``available`` to allow, in ``(0, 1]``.

    Examples
    --------
    >>> fit_to_screen(QSize(1500, 950), QSize(800, 800)).width()
    720
    >>> fit_to_screen(QSize(700, 600), QSize(800, 800)) == QSize(700, 600)
    True
    """
    return QSize(
        min(size.width(), int(available.width() * fill)),
        min(size.height(), int(available.height() * fill)),
    )


def _available_size(window: QWidget) -> QSize:
    """The available area of the screen ``window`` is (or will be) on."""
    screen = window.screen() or QGuiApplication.primaryScreen()
    if screen is None:  # no screen at all (never under a real QApplication)
        return DEFAULT_WINDOW_SIZE
    return screen.availableGeometry().size()


def restore_window(window: QWidget) -> None:
    """Size ``window`` from the saved geometry, or the default, to fit its screen.

    A saved geometry is restored as the user left it (Qt already keeps it on
    screen) and only clamped to the full available area, so a window deliberately
    sized to the whole display is honoured. With nothing saved the window gets
    :data:`DEFAULT_WINDOW_SIZE` capped at :data:`SCREEN_FILL` of the screen.
    """
    saved = user_settings().value(KEY_WINDOW_GEOMETRY)
    restored = isinstance(saved, QByteArray) and window.restoreGeometry(saved)
    if restored:
        fill = 1.0
    else:
        window.resize(DEFAULT_WINDOW_SIZE)
        fill = SCREEN_FILL
    fitted = fit_to_screen(window.size(), _available_size(window), fill=fill)
    if fitted != window.size():
        window.resize(fitted)


def save_window(window: QWidget) -> None:
    """Remember ``window``'s geometry (size, position, maximised state)."""
    store = user_settings()
    store.setValue(KEY_WINDOW_GEOMETRY, window.saveGeometry())
    store.sync()


def controls_width() -> int:
    """The remembered control-column width (logical px), else the default."""
    return int(user_settings().value(KEY_CONTROLS_WIDTH, DEFAULT_CONTROLS_WIDTH, int))


def set_controls_width(px: int) -> None:
    """Remember the control-column width (logical px) for every stage and launch."""
    store = user_settings()
    store.setValue(KEY_CONTROLS_WIDTH, int(px))
    store.sync()


def value(key: str, default: bool) -> bool:
    """Read a remembered flag (e.g. whether a strip is expanded)."""
    return bool(user_settings().value(key, default, bool))


def set_value(key: str, flag: bool) -> None:
    """Remember a flag."""
    store = user_settings()
    store.setValue(key, bool(flag))
    store.sync()
