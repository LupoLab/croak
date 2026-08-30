"""Point an existing session at a new dataset folder.

Many acquisitions share a layout: the measured trace, independent spectrum and
their backgrounds live together in a per-dataset folder with stable filenames,
while the spectral-response calibration files sit in a fixed shared location.
:func:`retarget` rewrites only the dataset-folder paths
(:meth:`~croak.session.params.LoadParams.dataset_path_fields`) to a new base
folder, keeping their filenames, and leaves the calibration paths
(:meth:`~croak.session.params.LoadParams.fixed_path_fields`) untouched — so a
working retrieval can be re-pointed at the next dataset and rerun.
"""

from __future__ import annotations

import os
from dataclasses import replace

from .options import SessionOptions
from .params import LoadParams

__all__ = ["retarget"]


def _retarget_load(load: LoadParams, new_base_dir: str) -> LoadParams:
    """Return a copy of ``load`` with dataset paths moved into ``new_base_dir``."""
    changes: dict[str, str] = {}
    for field_name in load.dataset_path_fields():
        old = getattr(load, field_name)
        if old:  # empty string == unused; leave it empty
            changes[field_name] = os.path.join(new_base_dir, os.path.basename(old))
    return replace(load, **changes)


def retarget(options: SessionOptions, new_base_dir: str) -> SessionOptions:
    """Re-point a session's dataset files at ``new_base_dir``.

    Parameters
    ----------
    options : SessionOptions
        The session whose dataset folder should change.
    new_base_dir : str
        The new per-dataset folder. Each dataset file keeps its basename and is
        looked up here; calibration files are left as-is.

    Returns
    -------
    SessionOptions
        A copy with the load paths retargeted (all other stages unchanged).

    Raises
    ------
    ValueError
        If the session did not come from the experimental loader. Retargeting is
        a property of the measured per-dataset folder layout: a simulated session
        has one scansave file with no sibling spectrum/background convention, and
        a synthetic one has no file at all, so there is nothing to re-point and
        silently returning the session unchanged would be a lie.

    Notes
    -----
    No filesystem check is done here; a missing retargeted file surfaces loudly
    when the session is replayed (the loader raises on the absent path).
    """
    if options.entry != "experimental":
        raise ValueError(
            f"cannot retarget a {options.entry} session: only experimental "
            "sessions have a per-dataset folder to re-point"
        )
    return options.with_(load=_retarget_load(options.load, new_base_dir))
