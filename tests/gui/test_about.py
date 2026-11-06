"""The About page: version, links, authors, copyright."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from croak.gui.about import COPYRIGHT, PYPI_URL, About, project_links

ROOT = Path(__file__).resolve().parents[2]


def test_about_shows_the_installed_version_authors_and_links(qtbot):
    about = About()
    qtbot.addWidget(about)
    meta = metadata.metadata("croak")
    assert meta["Version"] in about.version.text()
    assert about.authors.text() == meta["Author"]
    assert about.links["Package on PyPI"] == PYPI_URL
    assert set(about.links) == {
        "Manual",
        "Source on GitHub",
        "Package on PyPI",
        "Report an issue",
    }


def test_about_links_leave_out_what_the_metadata_lacks():
    assert project_links([]) == {"Package on PyPI": PYPI_URL}
    links = project_links(["Repository, https://example.org/croak"])
    assert links["Source on GitHub"] == "https://example.org/croak"


def test_about_copyright_matches_the_licence_file():
    licence = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert COPYRIGHT.replace("©", "(c)") in licence
