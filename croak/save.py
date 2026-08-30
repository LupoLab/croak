"""Saving retrieval results (HDF5) and GUI session options (TOML).

:func:`save_result` writes a :class:`~croak.result.RetrievalResult` (optionally
enriched with a :class:`~croak.processing.ProcessedResult`) to an HDF5 file using
a flat key schema — one dataset per quantity, so the file is readable from any
HDF5 binding without knowing this package. The same schema can be written into a
named ``group`` instead of the root, which is how a series of retrievals (a
parameter sweep, say) share one file. :func:`save_options` / :func:`load_options`
persist the GUI's per-stage parameters as ``options.toml``.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import tomli_w

from .grid import Grid
from .processing import ProcessedResult, TruthPulse, process_result
from .result import RetrievalResult

if TYPE_CHECKING:
    from .uncertainty import UncertaintyResult

__all__ = [
    "save_result",
    "load_result",
    "result_from_saved",
    "save_uncertainty",
    "save_options",
    "load_options",
]


def _result_dict(
    result: RetrievalResult,
    processed: ProcessedResult | None,
    truth: TruthPulse | None = None,
) -> dict[str, Any]:
    """Flatten a result (+ optional processing/truth) into HDF5-friendly entries."""
    omega = result.omega
    d: dict[str, Any] = {
        "Ew_retr": result.spectrum,
        "Iw_retr": np.abs(result.spectrum) ** 2,
        "phiw_retr": np.unwrap(np.angle(result.spectrum)),
        "Et_retr": result.field,
        "trace_retr": result.trace,
        "omega": omega,
        "tau": result.delays,
        "wavelength": result.wavelength,
        "omega0_pulse": float(result.omega0),
        "error_final": float(result.error),
        "error": np.asarray(result.errors, dtype=float),
        "mu": np.atleast_1d(np.asarray(result.mu, dtype=float)),
        "interaction": result.interaction,
        "algorithm": result.algorithm,
        # fitted extras, for a full reload without re-running the solver
        "tau0": float(result.tau0),
    }
    if result.thickness is not None:
        d["thickness"] = float(result.thickness)
    if result.smear_scale is not None:
        d["smear_scale"] = float(result.smear_scale)
    if result.smear_scale_delta is not None:
        d["smear_scale_delta"] = float(result.smear_scale_delta)
    if processed is not None:
        d.update(
            t_oversampled=processed.t_over,
            It_retr_oversampled=processed.It_retr,
            t_TL_oversampled=processed.t_tl,
            It_TL_oversampled=processed.It_tl,
            phit_retr_oversampled=processed.phi_t,
            lam_retr=processed.wavelength,
            Ilam_retr=processed.Ilam,
            phiw_retr_proc=processed.phi_w,
            gdd_fs2=float(processed.gdd_fs2),
            tod_fs3=float(processed.tod_fs3),
            fod_fs4=float(processed.fod_fs4),
            fwhm_retr_fs=float(processed.fwhm_retr / 1e-15),
            fwhm_tl_fs=float(processed.fwhm_tl / 1e-15),
        )
        # absolute power (only when a measured pulse energy was supplied)
        if processed.peak_power is not None and processed.energy is not None:
            d["pulse_energy_J"] = float(processed.energy)
            d["peak_power_W"] = float(processed.peak_power)
            if processed.peak_power_tl is not None:
                d["peak_power_tl_W"] = float(processed.peak_power_tl)
        if processed.trace_meas is not None:
            d["trace_meas"] = processed.trace_meas
        # the independent measured spectrum (photon/λ domain, on lam_retr) — the
        # separately-measured fundamental, kept beside the retrieved Ilam_retr
        if processed.Iw_meas is not None:
            d["Ilam_meas"] = processed.Iw_meas
    # Known ground-truth pulse (synthetic / simulated workflows): intensities,
    # display phases, and the original complex spectrum when the source supplied
    # one. Keeping the latter makes the exact truth-seed diagnostic reproducible
    # from the saved result rather than only from the original scan.
    if truth is not None:
        d["t_truth"] = truth.t
        d["It_truth"] = truth.It
        d["fwhm_truth_fs"] = float(truth.fwhm / 1e-15)
        if truth.lam is not None and truth.Iw is not None:
            d["lam_truth"] = truth.lam
            d["Ilam_truth"] = truth.Iw
        if truth.phi_t is not None:
            d["phit_truth"] = truth.phi_t
        if truth.phi_w is not None:
            d["phiw_truth"] = truth.phi_w
        if truth.omega is not None and truth.Eomega is not None:
            d["omega_truth"] = truth.omega
            d["Ew_truth"] = truth.Eomega
    return d


def _write_flat(dest: h5py.Group, data: Mapping[str, Any]) -> None:
    """Write flat result entries into an open HDF5 file or group.

    Strings become attributes (HDF5 has no natural string dataset for short
    labels), scalars 0-d datasets and everything else an array dataset.
    """
    for key, value in data.items():
        if isinstance(value, str):
            dest.attrs[key] = value
        elif np.isscalar(value):
            dest[key] = value
        else:
            dest.create_dataset(key, data=np.asarray(value))


def save_result(
    result: RetrievalResult,
    path: str,
    *,
    processed: ProcessedResult | None = None,
    measured: np.ndarray | None = None,
    truth: TruthPulse | None = None,
    group: str | None = None,
    force: bool = False,
) -> str:
    """Write a retrieval result to an HDF5 file.

    Parameters
    ----------
    result : RetrievalResult
        The retrieval to save.
    path : str
        Output ``.h5`` path.
    processed : ProcessedResult, optional
        Pre-computed processing to include; if omitted and ``measured`` is
        given, it is computed.
    measured : numpy.ndarray, optional
        Measured trace, used to compute ``processed`` and stored.
    truth : TruthPulse, optional
        Known ground-truth pulse (synthetic / simulated workflows); its temporal
        intensity ``It_truth`` and, when present, spectral intensity, temporal /
        spectral phase, and the complex truth spectrum are stored.
    group : str, optional
        Write the (unchanged) flat schema into this **group** instead of the file
        root, appending to an existing file rather than truncating it. Use it to
        collect a series of retrievals — a parameter sweep, one group per run —
        in a single file; a slash-separated name (``"full/z00"``) nests. The
        default (``None``) writes at the root, one result per file.
    force : bool, optional
        Overwrite an existing file — or, with ``group``, an existing group of that
        name (other groups are left alone). Otherwise raise.

    Returns
    -------
    str
        The path written.

    Raises
    ------
    FileExistsError
        If the file (or, with ``group``, the group) already exists and ``force``
        is false.
    """
    if processed is None and measured is not None:
        processed = process_result(result, measured=measured)
    data = _result_dict(result, processed, truth)
    if group is None:
        if os.path.exists(path) and not force:
            raise FileExistsError(f"{path} exists; pass force=True to overwrite")
        with h5py.File(path, "w") as f:
            _write_flat(f, data)
        return path
    # Grouped: append to the file so sibling groups (and any root-level metadata
    # the caller wrote) survive, and only the named group is replaced.
    with h5py.File(path, "a" if os.path.exists(path) else "w") as f:
        if group in f:
            if not force:
                raise FileExistsError(
                    f"{path} already has a group {group!r}; pass force=True"
                )
            del f[group]
        _write_flat(f.create_group(group), data)
    return path


def _read_group(group: h5py.Group) -> dict[str, Any]:
    """Read an HDF5 group's attrs + datasets into a dict, recursing into subgroups."""
    out: dict[str, Any] = dict(group.attrs)
    for key in group:
        item = group[key]
        if isinstance(item, h5py.Group):
            out[str(key)] = _read_group(item)
        elif isinstance(item, h5py.Dataset):
            out[str(key)] = item[()]
    return out


def load_result(path: str) -> dict[str, Any]:
    """Read a saved result HDF5 file back into a (possibly nested) dict.

    Top-level attrs and datasets become keys; any subgroup (e.g. the ``uncertainty``
    group written by :func:`save_uncertainty`) becomes a nested dict.
    """
    with h5py.File(path, "r") as f:
        return _read_group(f)


def result_from_saved(
    saved: Mapping[str, Any], *, grid: Grid | None = None
) -> RetrievalResult:
    """Rebuild a :class:`~croak.result.RetrievalResult` from a saved-result dict.

    Reconstructs the retrieval (the retrieved spectrum, trace, error history and
    fitted extras) from a :func:`load_result` dict, so a previously saved
    retrieval can be reloaded **without re-running the solver**.

    Parameters
    ----------
    saved : mapping
        A :func:`load_result` dict (the flat top-level entries of ``result.h5``).
    grid : Grid, optional
        Grid to attach. Defaults to one matched to the saved ``omega`` axis
        (:meth:`croak.grid.Grid.from_omega`); pass the live preprocess grid when
        available so the rehydrated result shares it exactly.

    Returns
    -------
    RetrievalResult

    Raises
    ------
    ValueError
        If ``grid`` is supplied but its size does not match the saved spectrum.
    """
    omega = np.asarray(saved["omega"], dtype=float)
    g = Grid.from_omega(omega) if grid is None else grid
    spectrum = np.asarray(saved["Ew_retr"], dtype=complex)
    if g.n != spectrum.size:
        raise ValueError(
            f"grid size {g.n} does not match the saved spectrum ({spectrum.size})"
        )
    mu = np.atleast_1d(np.asarray(saved["mu"], dtype=float))
    errors = [
        float(e) for e in np.atleast_1d(np.asarray(saved.get("error", []), float))
    ]
    thickness = saved.get("thickness")
    smear_scale = saved.get("smear_scale")
    smear_scale_delta = saved.get("smear_scale_delta")
    return RetrievalResult(
        spectrum=spectrum,
        grid=g,
        delays=np.asarray(saved["tau"], dtype=float),
        interaction=str(saved["interaction"]),
        algorithm=str(saved["algorithm"]),
        error=float(saved["error_final"]),
        errors=errors,
        trace=(
            np.asarray(saved["trace_retr"], dtype=float)
            if saved.get("trace_retr") is not None
            else None
        ),
        mu=float(mu[0]) if mu.size == 1 else mu,
        omega0=float(saved["omega0_pulse"]),
        thickness=None if thickness is None else float(thickness),
        tau0=float(saved.get("tau0", 0.0)),
        smear_scale=None if smear_scale is None else float(smear_scale),
        smear_scale_delta=(
            None if smear_scale_delta is None else float(smear_scale_delta)
        ),
    )


def _uncertainty_group(grp: h5py.Group, u: UncertaintyResult) -> None:
    """Write one :class:`UncertaintyResult` into an open HDF5 group (fs / µm units)."""
    grp.attrs["method"] = u.method
    grp.attrs["interval_method"] = u.interval_method
    grp.attrs["summary"] = u.summary()
    if u.components:
        grp.attrs["components"] = ",".join(u.components)
    grp["point_estimate_fs"] = u.point_estimate / 1e-15
    grp["plus_minus_fs"] = u.plus_minus / 1e-15
    grp["interval68_fs"] = np.asarray(u.interval_68, dtype=float) / 1e-15
    grp["interval95_fs"] = np.asarray(u.interval_95, dtype=float) / 1e-15
    grp["n_converged"] = int(u.n_converged)
    grp["samples_fs"] = np.asarray(u.samples, dtype=float) / 1e-15
    if u.thicknesses is not None:
        grp["thicknesses_um"] = np.asarray(u.thicknesses, dtype=float) / 1e-6
    # Temporal-intensity confidence band (the 68 %/95 % shaded region of the plot),
    # stored as median + lower/upper envelopes rather than every replicate profile.
    if u.profiles is not None and u.t_profile is not None:
        prof = np.asarray(u.profiles, dtype=float)
        grp["t_profile_fs"] = np.asarray(u.t_profile, dtype=float) / 1e-15
        grp["band_median"] = np.median(prof, axis=0)
        lo68, hi68 = np.percentile(prof, [16.0, 84.0], axis=0)
        lo95, hi95 = np.percentile(prof, [2.5, 97.5], axis=0)
        grp["band_lo68"], grp["band_hi68"] = lo68, hi68
        grp["band_lo95"], grp["band_hi95"] = lo95, hi95


def save_uncertainty(
    results: Mapping[str, UncertaintyResult] | UncertaintyResult,
    path: str,
    *,
    force: bool = False,
) -> str:
    """Write FWHM-uncertainty estimates into an ``uncertainty`` group of an HDF5 file.

    Appends to an existing file (e.g. a ``result.h5`` from :func:`save_result`) when
    present, otherwise creates it. Each estimate becomes a subgroup
    ``uncertainty/<method>`` holding the point estimate, 68 %/95 % intervals and ``±``
    (femtoseconds), the raw FWHM samples, and — when the estimate carries them — the
    drawn thicknesses (µm) and the **68 %/95 % temporal-intensity confidence bands**
    (median and lower/upper envelopes) shown shaded in the plot.

    Parameters
    ----------
    results : mapping of str to UncertaintyResult, or UncertaintyResult
        The estimates to save, keyed by a label (e.g. ``"parametric"``,
        ``"thickness"``, ``"combined"``). A single result is wrapped under its method.
    path : str
        Output ``.h5`` path (appended to if it already exists).
    force : bool, optional
        Overwrite an existing ``uncertainty`` group (otherwise raise).

    Returns
    -------
    str
        The path written.

    Raises
    ------
    ValueError
        If ``results`` is empty.
    FileExistsError
        If an ``uncertainty`` group already exists and ``force`` is false.
    """
    if not isinstance(results, Mapping):
        results = {results.method: results}
    if not results:
        raise ValueError("no uncertainty results to save")
    mode = "a" if os.path.exists(path) else "w"
    with h5py.File(path, mode) as f:
        if "uncertainty" in f:
            if not force:
                raise FileExistsError(
                    f"{path} already has an 'uncertainty' group; pass force=True"
                )
            del f["uncertainty"]
        root = f.create_group("uncertainty")
        for label, u in results.items():
            _uncertainty_group(root.create_group(label), u)
    return path


# ---------------------------------------------------------------------------
# Session options (TOML)
# ---------------------------------------------------------------------------
#: Schema version of the options TOML; bump when the option layout changes.
#: 2 added the ``entry`` key and the ``[simulated]`` table (the simulated-scan
#: loader's options), so a simulated session records the trace it came from.
OPTIONS_VERSION = 2


def _croak_version() -> str:
    try:
        from importlib.metadata import version

        return version("croak")
    except Exception:  # pragma: no cover - metadata unavailable
        return "unknown"


def save_options(options: dict[str, Any], path: str) -> str:
    """Write GUI session options to a TOML file.

    A ``croak_version`` / ``options_version`` header is written first so older
    files can be reloaded (and migrated, if ever needed) gracefully. Nested
    dicts (one per stage) become TOML tables; numpy values are coerced.
    """
    # scalar header keys must precede the table sections in TOML
    data = {"croak_version": _croak_version(), "options_version": OPTIONS_VERSION}
    data.update(_tomlify(options))
    with open(path, "wb") as f:
        tomli_w.dump(data, f)
    return path


def load_options(path: str) -> dict[str, Any]:
    """Read GUI session options from a TOML file."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def _tomlify(value: Any) -> Any:
    """Coerce numpy types / arrays into TOML-serialisable Python values.

    TOML has no null, so ``None``-valued keys are **omitted** rather than
    written; reading them back falls through to the dataclass default, which is
    ``None`` for every optional option (e.g. an unset
    ``SimulatedLoadParams.z_thickness_um``), so the round-trip is exact.
    """
    if isinstance(value, dict):
        return {k: _tomlify(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [_tomlify(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_tomlify(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value
