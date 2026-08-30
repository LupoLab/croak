r"""Optional bridge to the refractiveindex.info database.

Wraps the ``refractiveindex`` PyPI package to browse
`refractiveindex.info <https://refractiveindex.info>`_ materials.

The package is an **optional** dependency (``pip install croak[ridb]``); every
function here imports it lazily so importing :mod:`croak` never requires it. On
first use the package downloads the full polyanskiy database (hundreds of MB) to
``~/.refractiveindex.info-database`` and caches it.

The catalogue is exposed as ordered *shelf → book → page* lists of
``(identifier, label)`` for cascading GUI combos, and :func:`make_material`
returns an ``n(wavelength_m)`` callable (NaN outside the page's validity range,
so :func:`croak.materials.beta` masks it) plus that valid range.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["available", "shelves", "books", "pages", "make_material"]

_DB_PATH = Path.home() / ".refractiveindex.info-database"
_CATALOG_FILE = "catalog-nk.yml"

# Cached parsed catalogue:
#   {shelf_id: (shelf_label, {book_id: (book_label, {page_id: page_label})})}
_CATALOG: dict[str, tuple[str, dict[str, tuple[str, dict[str, str]]]]] | None = None


def available() -> bool:
    """Whether the optional ``refractiveindex`` package is importable."""
    import importlib.util

    return importlib.util.find_spec("refractiveindex") is not None


def _ensure_database() -> Path:
    """Return the catalogue path, triggering the one-time download if needed.

    The ``refractiveindex`` package downloads the database the first time a
    material is constructed, so we instantiate a stable, always-present page
    (``main/Ag/Johnson``) to force it, then locate ``catalog-nk.yml``.
    """
    catalog = _DB_PATH / _CATALOG_FILE
    if not catalog.exists():
        # Optional 'ridb' extra; not installed in the base/dev environment.
        from refractiveindex import (  # pyright: ignore[reportMissingImports]
            RefractiveIndexMaterial,
        )

        RefractiveIndexMaterial(shelf="main", book="Ag", page="Johnson")
    if not catalog.exists():
        raise FileNotFoundError(
            f"refractiveindex database catalogue not found at {catalog}"
        )
    return catalog


def _parse_catalog() -> dict[str, tuple[str, dict[str, tuple[str, dict[str, str]]]]]:
    """Parse ``catalog-nk.yml`` into nested ordered shelf/book/page dicts.

    The YAML is a list of shelves; each shelf's ``content`` lists books (and
    ``DIVIDER`` separators we skip); each book's ``content`` lists pages. The
    ``SHELF``/``BOOK``/``PAGE`` values are the identifiers the API expects, while
    ``name`` holds the human-readable label.
    """
    import yaml

    with open(_ensure_database(), encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    out: dict[str, tuple[str, dict[str, tuple[str, dict[str, str]]]]] = {}
    for shelf in raw or []:
        if "SHELF" not in shelf:
            continue
        books: dict[str, tuple[str, dict[str, str]]] = {}
        for book in shelf.get("content", []):
            if "BOOK" not in book:
                continue
            ps: dict[str, str] = {}
            for page in book.get("content", []):
                if "PAGE" in page:
                    ps[str(page["PAGE"])] = str(page.get("name", page["PAGE"]))
            if ps:
                books[str(book["BOOK"])] = (str(book.get("name", book["BOOK"])), ps)
        if books:
            out[str(shelf["SHELF"])] = (str(shelf.get("name", shelf["SHELF"])), books)
    return out


def _catalog() -> dict[str, tuple[str, dict[str, tuple[str, dict[str, str]]]]]:
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = _parse_catalog()
    return _CATALOG


def shelves() -> list[tuple[str, str]]:
    """Ordered ``(shelf_id, label)`` pairs in the database."""
    return [(sid, label) for sid, (label, _) in _catalog().items()]


def books(shelf_id: str) -> list[tuple[str, str]]:
    """Ordered ``(book_id, label)`` pairs in ``shelf_id``."""
    _, bks = _catalog()[shelf_id]
    return [(bid, label) for bid, (label, _) in bks.items()]


def pages(shelf_id: str, book_id: str) -> list[tuple[str, str]]:
    """Ordered ``(page_id, label)`` pairs in ``shelf_id``/``book_id``."""
    _, bks = _catalog()[shelf_id]
    _, ps = bks[book_id]
    return list(ps.items())


def make_material(
    shelf: str, book: str, page: str
) -> tuple[Callable[[ArrayLike], NDArray[np.float64]], tuple[float, float] | None]:
    """Build an ``n(wavelength_m)`` callable and valid range for a database page.

    Parameters
    ----------
    shelf, book, page : str
        refractiveindex.info identifiers (see :func:`shelves`/:func:`books`/
        :func:`pages`).

    Returns
    -------
    (n_of_lambda_m, wl_range_m)
        ``n_of_lambda_m(λ_m)`` returns the real refractive index (NaN outside the
        page's validity range so the propagation constant is masked).
        ``wl_range_m`` is ``(min_m, max_m)`` or ``None`` if the page declares no
        range.
    """
    # Optional 'ridb' extra; not installed in the base/dev environment.
    from refractiveindex import (  # pyright: ignore[reportMissingImports]
        RefractiveIndexMaterial,
    )

    mat = RefractiveIndexMaterial(shelf=shelf, book=book, page=page)
    rng = mat.get_wl_range(unit="m")
    if rng is not None:
        rng = (float(min(rng)), float(max(rng)))

    def n_of_lambda_m(lam_m: ArrayLike) -> NDArray[np.float64]:
        lam_m = np.asarray(lam_m, dtype=float)
        query = lam_m
        in_range = np.ones(lam_m.shape, dtype=bool)
        if rng is not None:
            lo, hi = rng
            in_range = (lam_m >= lo) & (lam_m <= hi)
            query = np.clip(lam_m, lo, hi)  # avoid out-of-range extrapolation errors
        n = np.asarray(mat.get_refractive_index(query * 1e6, unit="um"), dtype=float)
        return np.where(in_range, n, np.nan)

    return n_of_lambda_m, rng
