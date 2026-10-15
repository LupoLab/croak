"""Tests for :mod:`croak.refractive_db` against a fake ``refractiveindex`` package.

The real optional dependency downloads a hundreds-of-MB database on first use, so
these tests inject a tiny stand-in module and catalogue instead. The live package
is exercised (when installed) by ``tests/test_mirrors.py::test_refractive_db_smoke``.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from croak import refractive_db

CATALOG_YAML = """\
- DIVIDER: "Not a shelf"
- SHELF: main
  name: "MAIN - simple inorganic materials"
  content:
    - DIVIDER: "Metals"
    - BOOK: Ag
      name: "Ag (Silver)"
      content:
        - PAGE: Johnson
          name: "Johnson and Christy 1972"
        - DIVIDER: "not a page"
        - PAGE: Rakic
    - BOOK: Empty
      name: "A book with no pages"
      content: []
- SHELF: glass
  content:
    - BOOK: BK7
      content:
        - PAGE: SCHOTT
- SHELF: hollow
  name: "A shelf with no usable books"
  content:
    - BOOK: NoPages
"""


class FakeMaterial:
    """Stand-in for ``refractiveindex.RefractiveIndexMaterial``.

    ``n(λ) = 1.5 + 0.01/λ²`` (λ in µm), valid on 0.4–1.0 µm unless the page is
    ``"NoRange"``. Records every query so tests can check the clipping.
    """

    created: list[tuple[str, str, str]] = []

    def __init__(self, shelf: str, book: str, page: str):
        self.page = page
        self.queries: list[np.ndarray] = []
        FakeMaterial.created.append((shelf, book, page))

    def get_wl_range(self, unit: str):
        assert unit == "m"
        return None if self.page == "NoRange" else (1.0e-6, 0.4e-6)

    def get_refractive_index(self, wavelength, unit: str):
        assert unit == "um"
        lam_um = np.asarray(wavelength, dtype=float)
        self.queries.append(lam_um)
        return 1.5 + 0.01 / lam_um**2


@pytest.fixture
def fake_ridb(monkeypatch, tmp_path):
    """Install the fake package and point the module at a temporary database."""
    module = types.ModuleType("refractiveindex")
    module.RefractiveIndexMaterial = FakeMaterial  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "refractiveindex", module)
    monkeypatch.setattr(refractive_db, "_DB_PATH", tmp_path)
    monkeypatch.setattr(refractive_db, "_CATALOG", None)
    FakeMaterial.created = []
    return tmp_path


def _write_catalog(db_path):
    (db_path / refractive_db._CATALOG_FILE).write_text(CATALOG_YAML, encoding="utf-8")


def test_available_reflects_whether_the_package_is_importable(monkeypatch):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert refractive_db.available() is False
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    assert refractive_db.available() is True


def test_catalog_lists_shelves_books_and_pages_in_order(fake_ridb):
    _write_catalog(fake_ridb)
    # dividers, page-less books and book-less shelves are skipped; a missing
    # ``name`` falls back to the identifier
    assert refractive_db.shelves() == [
        ("main", "MAIN - simple inorganic materials"),
        ("glass", "glass"),
    ]
    assert refractive_db.books("main") == [("Ag", "Ag (Silver)")]
    assert refractive_db.pages("main", "Ag") == [
        ("Johnson", "Johnson and Christy 1972"),
        ("Rakic", "Rakic"),
    ]
    assert refractive_db.pages("glass", "BK7") == [("SCHOTT", "SCHOTT")]
    # an existing catalogue never triggers the download
    assert FakeMaterial.created == []


def test_catalog_is_parsed_once_and_cached(fake_ridb, monkeypatch):
    _write_catalog(fake_ridb)
    calls = []
    real = refractive_db._parse_catalog
    monkeypatch.setattr(
        refractive_db, "_parse_catalog", lambda: calls.append(1) or real()
    )
    refractive_db.shelves()
    refractive_db.books("glass")
    assert calls == [1]


def test_catalog_of_an_empty_file_is_empty(fake_ridb):
    (fake_ridb / refractive_db._CATALOG_FILE).write_text("", encoding="utf-8")
    assert refractive_db.shelves() == []


def test_missing_catalog_triggers_the_download(fake_ridb, monkeypatch):
    # The download is a side effect of constructing any material; the fake writes
    # the catalogue the way the real package's first use does.
    def downloading_init(self, shelf, book, page):
        FakeMaterial.created.append((shelf, book, page))
        _write_catalog(fake_ridb)

    monkeypatch.setattr(FakeMaterial, "__init__", downloading_init)
    assert refractive_db.shelves()[0][0] == "main"
    assert FakeMaterial.created == [("main", "Ag", "Johnson")]


def test_missing_catalog_after_download_raises(fake_ridb):
    with pytest.raises(FileNotFoundError, match="catalogue not found"):
        refractive_db.shelves()
    assert FakeMaterial.created == [("main", "Ag", "Johnson")]


def test_make_material_masks_wavelengths_outside_the_valid_range(fake_ridb):
    n, wl_range = refractive_db.make_material("main", "Ag", "Johnson")
    # the declared range is normalised to (min, max) even if given reversed
    assert wl_range == (0.4e-6, 1.0e-6)
    lam = np.array([0.2e-6, 0.5e-6, 0.8e-6, 2.0e-6])
    out = n(lam)
    np.testing.assert_allclose(out[1:3], 1.5 + 0.01 / (lam[1:3] * 1e6) ** 2)
    assert np.isnan(out[0]) and np.isnan(out[3])


def test_make_material_clips_queries_into_the_valid_range(fake_ridb, monkeypatch):
    made = []
    real_init = FakeMaterial.__init__

    def recording_init(self, shelf, book, page):
        real_init(self, shelf, book, page)
        made.append(self)

    monkeypatch.setattr(FakeMaterial, "__init__", recording_init)
    n, _ = refractive_db.make_material("main", "Ag", "Johnson")
    n(np.array([0.1e-6, 5.0e-6]))
    # the package is never asked to extrapolate outside the page's range (µm)
    np.testing.assert_allclose(made[0].queries[-1], [0.4, 1.0])


def test_make_material_without_a_declared_range_is_unmasked(fake_ridb):
    n, wl_range = refractive_db.make_material("main", "Ag", "NoRange")
    assert wl_range is None
    out = n(np.array([0.1e-6, 10e-6]))
    assert np.isfinite(out).all()
