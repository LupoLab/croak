"""Home: what to retrieve, a session to reopen, and the recent sessions.

Shown when no session is open, and from the brand button in the top bar. Each
tile, with a drawing of its own (``assets/tiles``), starts a new session from one
of the three entries, or opens a saved one; the recent sessions are
the ``options.toml`` files last saved or opened (:mod:`croak.gui.settings`),
each reopened with its data and, where it matches, its saved result.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QMouseEvent
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .theme import ASSETS, button, label

__all__ = ["ENTRY_TILES", "Home"]

TILE_ICON_SIZE = 44
"""The size of a tile's drawing, logical px."""

OPEN_TILE = (
    "open",
    "Open a session",
    "Reopen a saved session's options.toml, with its data and, where it still "
    "matches, its result.",
)
"""The tile that reopens a saved session: key, title, description."""

ENTRY_TILES = (
    (
        "experimental",
        "Measured trace",
        "Load a measured FROG trace from HDF5 or NPZ, with its independent "
        "spectrum, backgrounds and calibration.",
    ),
    (
        "simulated",
        "Simulated scan",
        "Load a numerically simulated scan (a Luna scansave file), with its "
        "known input pulse and instrument geometry.",
    ),
    (
        "synthetic",
        "Synthetic trace",
        "Generate a trace from a pulse you specify, with dispersion and noise, "
        "and test a retrieval against the known answer.",
    ),
)
"""The three ways to start a session: entry, title, description."""


class _Clickable(QFrame):
    """A frame that acts as a button: it lights up under the pointer and clicks."""

    clicked = Signal()

    def __init__(self, role: str) -> None:
        super().__init__()
        self.setObjectName(role)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 — Qt override
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(
            event.position().toPoint()
        ):
            self.clicked.emit()
        super().mouseReleaseEvent(event)


def _tile(key: str, title: str, text: str) -> _Clickable:
    """A tile: its drawing (``assets/tiles/<key>.svg``) beside its title and text."""
    tile = _Clickable("tile")
    row = QHBoxLayout(tile)
    row.setContentsMargins(16, 14, 16, 14)
    row.setSpacing(14)
    art = label("", "tileArt")
    icon = QIcon(str(ASSETS / "tiles" / f"{key}.svg"))
    art.setPixmap(icon.pixmap(QSize(TILE_ICON_SIZE, TILE_ICON_SIZE)))
    art.setFixedSize(TILE_ICON_SIZE, TILE_ICON_SIZE)
    row.addWidget(art, alignment=Qt.AlignmentFlag.AlignTop)
    words = QVBoxLayout()
    words.setSpacing(4)
    words.addWidget(label(title, "tileTitle"))
    words.addWidget(label(text, "tileText", wrap=True))
    words.addStretch(1)  # tiles in a row share a height; text stays at the top
    row.addLayout(words, stretch=1)
    return tile


def describe_session(path: Path) -> tuple[str, str]:
    """The title and detail line Home shows for a saved session at ``path``.

    The folder names the session (every session file is ``options.toml``); the
    detail is the folder's parent and the file's modification time.
    """
    when = datetime.fromtimestamp(path.stat().st_mtime).strftime("%d %b %Y, %H:%M")
    return path.parent.name, f"{path.parent.parent}  ·  {when}"


class Home(QWidget):
    """The start page.

    Attributes
    ----------
    start : Signal(str)
        A tile was chosen: the entry to start a session from.
    open_session : Signal(str)
        A session to reopen: an ``options.toml`` path, or ``""`` to ask for one.
    about : Signal
        About croak was asked for.
    """

    start = Signal(str)
    open_session = Signal(str)
    about = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("page")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        content.setObjectName("page")
        columns = QHBoxLayout(content)
        columns.setContentsMargins(40, 36, 40, 24)
        columns.setSpacing(40)

        main = QVBoxLayout()
        main.setSpacing(14)
        main.addWidget(label("What are you retrieving?", "h1"))
        grid = QGridLayout()
        grid.setSpacing(12)
        self.tiles: dict[str, _Clickable] = {}
        for index, (key, title, text) in enumerate((*ENTRY_TILES, OPEN_TILE)):
            tile = _tile(key, title, text)
            if key == OPEN_TILE[0]:
                tile.clicked.connect(lambda: self.open_session.emit(""))
            else:
                tile.clicked.connect(lambda entry=key: self.start.emit(entry))
            grid.addWidget(tile, index // 2, index % 2)
            self.tiles[key] = tile
        main.addLayout(grid)
        actions = QHBoxLayout()
        actions.addStretch(1)
        self.about_button = button("About croak", "link")
        self.about_button.clicked.connect(self.about)
        actions.addWidget(self.about_button)
        main.addLayout(actions)
        main.addStretch(1)
        columns.addLayout(main, stretch=3)

        side = QVBoxLayout()
        side.setSpacing(6)
        side.addWidget(label("Recent sessions", "h3"))
        self.recent = QVBoxLayout()
        self.recent.setSpacing(2)
        side.addLayout(self.recent)
        side.addStretch(1)
        columns.addLayout(side, stretch=2)

        scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        self.recent_items: list[_Clickable] = []
        self._recent_widgets: list[QWidget] = []
        self.show_recent([])

    def show_recent(self, paths: Sequence[str]) -> None:
        """List the sessions at ``paths`` that still exist, newest first."""
        for old in self._recent_widgets:
            self.recent.removeWidget(old)
            # Hidden at once: deleteLater alone leaves it drawn, loose in the
            # page, until the event loop next runs.
            old.hide()
            old.deleteLater()
        self._recent_widgets: list[QWidget] = []
        self.recent_items = []
        for raw in paths:
            path = Path(raw)
            if not path.is_file():
                continue
            title, detail = describe_session(path)
            item = _Clickable("item")
            item.setToolTip(str(path))
            layout = QVBoxLayout(item)
            layout.setContentsMargins(10, 7, 10, 7)
            layout.setSpacing(2)
            layout.addWidget(label(title, "itemTitle"))
            layout.addWidget(label(detail, "itemText", wrap=True))
            item.clicked.connect(lambda raw=raw: self.open_session.emit(raw))
            self.recent_items.append(item)
        hint = "Sessions you save or open appear here."
        shown = self.recent_items or [label(hint, "itemText", wrap=True)]
        for widget in shown:
            self.recent.addWidget(widget)
            self._recent_widgets.append(widget)
