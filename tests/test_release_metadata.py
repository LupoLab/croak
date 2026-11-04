"""The release records name one version.

A release commit moves the version in several files by hand
(docs/developer_guide/releasing.md), and the rolling release PR takes its branch name
from ``CITATION.cff``, so a file left behind would misname the next release.
"""

import json
import re
import tomllib
from pathlib import Path

import croak

ROOT = Path(__file__).resolve().parents[1]


def _text(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def _version() -> str:
    return tomllib.loads(_text("pyproject.toml"))["project"]["version"]


def test_release_records_package_version_matches_pyproject():
    assert croak.__version__ == _version()


def test_release_records_citation_version_matches_pyproject():
    # Read as text: the citation's one top-level key needs no YAML parser.
    assert (
        re.search(r"^version: *(\S+) *$", _text("CITATION.cff"), re.M)[1] == _version()
    )


def test_release_records_lockfile_version_matches_pyproject():
    packages = tomllib.loads(_text("uv.lock"))["package"]
    assert [p["version"] for p in packages if p["name"] == "croak"] == [_version()]


def test_release_records_zenodo_names_no_version():
    # Zenodo prefers this file's version over the release tag, and it went stale here.
    assert "version" not in json.loads(_text(".zenodo.json"))


def test_release_records_changelog_ends_at_the_current_version():
    changelog = _text("CHANGELOG.md")
    released = re.search(
        r"^## \[(\d+\.\d+\.\d+)\] - \d{4}-\d{2}-\d{2}$", changelog, re.M
    )
    assert released[1] == _version()
    # The links at the foot of the file move with each release too.
    assert f"/compare/v{_version()}...HEAD" in changelog
    assert f"\n[{_version()}]: " in changelog
