"""About: what croak is, where it lives, who wrote it and under what licence.

Opened from Help ▸ About croak, the ⋯ menu and Home. The version, summary,
authors and links come from the installed package's metadata, so they cannot
drift from ``pyproject.toml``; only the copyright line and the PyPI address,
which the metadata does not carry, are written here.
"""

from __future__ import annotations

from collections.abc import Sequence
from importlib import metadata

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from .theme import PALETTE, label

__all__ = ["COPYRIGHT", "PYPI_URL", "About", "project_links"]

COPYRIGHT = "Copyright © 2026 John C. Travers and contributors"
"""The copyright line, as in the ``LICENSE`` file."""

PYPI_URL = "https://pypi.org/project/croak/"
"""The package's page on PyPI."""

ABOUT_WIDTH = 640
"""The About text's measure, logical px: a comfortable line to read."""

LICENCE_NOTE = (
    "croak is free software under the MIT licence. Its window is set in IBM "
    "Plex Sans, under the SIL Open Font License 1.1."
)


def project_links(entries: Sequence[str] | None = None) -> dict[str, str]:
    """Return the project's links by name: the manual, GitHub, PyPI and issues.

    Parameters
    ----------
    entries : sequence of str, optional
        ``Project-URL`` metadata entries, each ``"Name, url"``; the installed
        croak's by default. A link the entries lack is left out (PyPI's is
        always there).

    Examples
    --------
    >>> project_links(["Documentation, https://croak.readthedocs.io"])
    {'Manual': 'https://croak.readthedocs.io', 'Package on PyPI': 'https://pypi.org/project/croak/'}
    """  # noqa: E501 — a doctest's output must stay on one line
    if entries is None:
        entries = metadata.metadata("croak").get_all("Project-URL") or []
    urls = dict(entry.split(", ", 1) for entry in entries)
    links = {
        "Manual": urls.get("Documentation", ""),
        "Source on GitHub": urls.get("Repository", ""),
        "Package on PyPI": PYPI_URL,
        "Report an issue": urls.get("Issues", ""),
    }
    return {name: url for name, url in links.items() if url}


def _link(text: str, url: str) -> QLabel:
    # Qt draws links in the system's blue; the palette's accent matches the window.
    style = f"color: {PALETTE.accent}; text-decoration: none;"
    widget = label(f'<a href="{url}" style="{style}">{text}</a>', "aboutLink")
    widget.setTextFormat(Qt.TextFormat.RichText)
    widget.setOpenExternalLinks(True)
    widget.setToolTip(url)
    return widget


class About(QWidget):
    """The About page."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("page")
        meta = metadata.metadata("croak")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(40, 36, 40, 24)
        body = QWidget()
        body.setMaximumWidth(ABOUT_WIDTH)
        outer.addWidget(body)
        outer.addStretch(1)
        column = QVBoxLayout(body)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(10)
        column.addWidget(label("croak", "h1"))
        self.version = label(f"Version {meta['Version']}", "note")
        column.addWidget(self.version)
        column.addWidget(label(meta["Summary"], "", wrap=True))
        column.addSpacing(8)
        column.addWidget(label("Links", "h3"))
        self.links = project_links()
        for name, url in self.links.items():
            column.addWidget(_link(name, url))
        column.addSpacing(8)
        column.addWidget(label("Authors", "h3"))
        authors = meta["Author"] or ""
        self.authors = label(authors, "", wrap=True)
        column.addWidget(self.authors)
        column.addSpacing(8)
        column.addWidget(label(COPYRIGHT, "note"))
        column.addWidget(label(LICENCE_NOTE, "note", wrap=True))
