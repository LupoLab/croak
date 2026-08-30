"""Loading experimental FROG/spectrum files and unit handling.

A thin, GUI-agnostic layer over HDF5 (``h5py``), NumPy ``.npz`` and delimited
text files. It enumerates the numeric datasets in a file, auto-matches the FROG
trace / wavelength / scan-axis arrays by name (mirroring
``ToolHelpers._match_dataset_index``), loads selected datasets, and converts the
loaded axes to SI units.

The actual file *choosing* is the GUI's job (``QFileDialog``); this module only
reads paths it is given, so the same code serves the library and the GUI.
"""

from __future__ import annotations

import contextlib
import os
import re
from dataclasses import dataclass

import h5py
import numpy as np
from numpy.typing import NDArray

from .constants import C

__all__ = [
    "DatasetInfo",
    "FileDatasets",
    "list_datasets",
    "load_dataset",
    "load_csv",
    "match_dataset",
    "guess_scanaxis_type",
    "guess_wavelength_unit",
    "guess_scanaxis_unit",
    "csv_wavelength_unit",
    "to_1d",
    "reduce_to_1d",
    "unit_to_si",
    "position_to_delay",
    "UNIT_FACTORS",
    "MaskWindowSpec",
    "SimulatedScan",
    "SIMULATED_WINDOW_KEYS",
    "SIMULATED_TRUTH_SOURCES",
    "read_simulated_scan",
    "read_simulated_z_positions",
    "read_simulated_window_keys",
    "read_simulated_mask_window",
    "window_def_prefix",
    "read_simulated_truth_keys",
    "read_simulated_wavelength_range",
]

# Name patterns used to auto-pick datasets (case-insensitive).
_IFROG_PATTERN = re.compile(r"spectr|trace|frog|image|intensity|signal", re.I)
_LAMBDA_PATTERN = re.compile(r"wavelength|wave|lambda|λ|wl|wvl|freq", re.I)
_SCAN_PATTERN = re.compile(r"pos|position|delay|z_|\bz\b|scan|stage|tau|τ|time", re.I)
_DELAY_PATTERN = re.compile(r"delay|tau|τ|time|dt", re.I)
_POSITION_PATTERN = re.compile(r"pos|position|z_|\bz\b|scan|stage", re.I)

PATTERNS = {
    "ifrog": _IFROG_PATTERN,
    "lambda": _LAMBDA_PATTERN,
    "scan": _SCAN_PATTERN,
}

#: Multiplicative factors converting a unit string to SI (metres or seconds).
UNIT_FACTORS: dict[str, float] = {
    "m": 1.0,
    "mm": 1e-3,
    "µm": 1e-6,
    "um": 1e-6,
    "nm": 1e-9,
    "s": 1.0,
    "ps": 1e-12,
    "fs": 1e-15,
}


@dataclass(frozen=True)
class DatasetInfo:
    """Metadata for one numeric dataset within a file."""

    name: str
    ndim: int
    shape: tuple[int, ...]

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        """Return ``"name shape"``."""
        return f"{self.name} {self.shape}"


@dataclass
class FileDatasets:
    """The numeric datasets discovered in a file, with lazy loading.

    Attributes
    ----------
    path : str
        File path.
    filetype : str
        ``"hdf5"`` or ``"npz"``.
    datasets : list of DatasetInfo
        All numeric datasets, in discovery order.
    """

    path: str
    filetype: str
    datasets: list[DatasetInfo]

    @property
    def names(self) -> list[str]:
        """Dataset names, in discovery order."""
        return [d.name for d in self.datasets]

    @property
    def matrix_names(self) -> list[str]:
        """Names of the 2-D datasets (candidate FROG traces)."""
        return [d.name for d in self.datasets if d.ndim == 2]

    @property
    def vector_names(self) -> list[str]:
        """Names of the 1-D datasets (candidate wavelength/scan axes)."""
        return [d.name for d in self.datasets if d.ndim == 1]

    @property
    def option_labels(self) -> list[str]:
        """``name  (shape)`` labels for every dataset (for spectrum pickers)."""
        return [f"{d.name}  {d.shape}" for d in self.datasets]

    def load(self, name: str) -> NDArray[np.float64]:
        """Load one dataset by name as a float array."""
        return load_dataset(self.path, name, self.filetype)

    def auto_index(self, kind: str) -> int | None:
        """Index of the best dataset for ``kind`` (ifrog/lambda/scan)."""
        return match_dataset(self.names, PATTERNS[kind])


# ---------------------------------------------------------------------------
# Enumeration & loading
# ---------------------------------------------------------------------------
def _walk_hdf5(group: h5py.Group, out: list[DatasetInfo], prefix: str = "") -> None:
    for name in group:
        obj = group[name]
        path = f"{prefix}/{name}" if prefix else str(name)
        if isinstance(obj, h5py.Dataset):
            # only numeric arrays, so callers can always load them as float
            if np.issubdtype(obj.dtype, np.number):
                out.append(DatasetInfo(path, obj.ndim, tuple(obj.shape)))
        elif isinstance(obj, h5py.Group):
            _walk_hdf5(obj, out, path)


def list_datasets(path: str) -> FileDatasets:
    """Enumerate the numeric datasets in an HDF5 or NPZ file.

    Parameters
    ----------
    path : str
        Path to a ``.h5``/``.hdf5`` or ``.npz`` file.

    Returns
    -------
    FileDatasets
    """
    ext = os.path.splitext(path)[1].lower()
    out: list[DatasetInfo] = []
    if ext in (".h5", ".hdf5"):
        with h5py.File(path, "r") as f:
            _walk_hdf5(f, out)
        return FileDatasets(path, "hdf5", out)
    if ext == ".npz":
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                arr = data[key]
                if np.issubdtype(arr.dtype, np.number):
                    out.append(DatasetInfo(key, arr.ndim, tuple(arr.shape)))
        return FileDatasets(path, "npz", out)
    raise ValueError(f"unsupported file type {ext!r} (expected .h5/.hdf5/.npz)")


def load_dataset(
    path: str, name: str, filetype: str | None = None
) -> NDArray[np.float64]:
    """Load a single dataset from an HDF5 or NPZ file as ``float64``.

    Parameters
    ----------
    path : str
        File path.
    name : str
        Dataset name/path within the file.
    filetype : str, optional
        ``"hdf5"`` or ``"npz"``; inferred from the extension if omitted.
    """
    if filetype is None:
        ext = os.path.splitext(path)[1].lower()
        filetype = "hdf5" if ext in (".h5", ".hdf5") else "npz"
    if filetype == "hdf5":
        with h5py.File(path, "r") as f:
            dset = f[name]
            if not isinstance(dset, h5py.Dataset):
                raise TypeError(f"{name!r} is not an HDF5 dataset")
            return np.asarray(dset[()], dtype=float)
    if filetype == "npz":
        with np.load(path, allow_pickle=False) as data:
            return np.asarray(data[name], dtype=float)
    raise ValueError(f"unsupported file type {filetype!r}")


def _detect_delimiter(path: str) -> str | None:
    """Return ``","`` for comma-separated files or ``None`` for whitespace."""
    with open(path) as fh:
        for line in fh:
            stripped = line.lstrip()
            if not stripped or stripped.startswith("#"):
                continue
            parts_c = line.split(",")
            if len(parts_c) >= 2 and all(_is_float(p) for p in parts_c):
                return ","
            parts_w = line.split()
            if len(parts_w) >= 2 and all(_is_float(p) for p in parts_w):
                return None
    return ","


def _is_float(s: str) -> bool:
    try:
        float(s.strip())
        return True
    except ValueError:
        return False


def load_csv(
    path: str, *, skip: int | None = None, delimiter: str | None = "auto"
) -> NDArray[np.float64]:
    """Load a delimited numeric text file (CSV or whitespace).

    Comment (``#``) and blank lines are skipped; the header is auto-detected
    when ``skip`` is omitted (first all-numeric row).

    Parameters
    ----------
    path : str
        File path.
    skip : int, optional
        Number of leading data rows to skip. Auto-detected if omitted.
    delimiter : str or None, optional
        ``","``, ``None`` (whitespace), or ``"auto"`` (default) to detect.

    Returns
    -------
    numpy.ndarray
        A 2-D ``(rows, columns)`` float array.
    """
    if delimiter == "auto":
        delimiter = _detect_delimiter(path)
    with open(path) as fh:
        lines = [ln for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    if skip is None:
        skip = 0
        for i, line in enumerate(lines):
            parts = line.split(delimiter) if delimiter else line.split()
            if len(parts) >= 2 and all(_is_float(p) for p in parts):
                skip = i
                break
            skip = i + 1
    data = np.genfromtxt(lines[skip:], delimiter=delimiter)
    return np.atleast_2d(data)


# ---------------------------------------------------------------------------
# Dataset matching & axis-type guessing
# ---------------------------------------------------------------------------
def match_dataset(names: list[str], pattern: re.Pattern) -> int | None:
    """Index of the shortest dataset name matching ``pattern`` (ties: first)."""
    best_idx: int | None = None
    best_len = np.inf
    for i, name in enumerate(names):
        if pattern.search(name) and len(name) < best_len:
            best_idx, best_len = i, len(name)
    return best_idx


def guess_scanaxis_type(name: str) -> str | None:
    """Guess whether a scan-axis name denotes a ``"delay"`` or ``"position"``."""
    if _DELAY_PATTERN.search(name):
        return "delay"
    if _POSITION_PATTERN.search(name):
        return "position"
    return None


# ---------------------------------------------------------------------------
# Unit guessing & conversion
# ---------------------------------------------------------------------------
def guess_wavelength_unit(values: NDArray[np.float64]) -> str:
    """Guess the unit (``"m"``/``"µm"``/``"nm"``) of a wavelength axis."""
    med = float(np.median(np.abs(values)))
    if med < 1e-3:
        return "m"
    if med < 50.0:
        return "µm"
    return "nm"


def guess_scanaxis_unit(values: NDArray[np.float64], scan_type: str) -> str:
    """Guess the unit of a scan axis given its type (``"delay"`` or ``"position"``)."""
    med = float(np.median(np.abs(values)))
    rng = float(np.max(np.abs(values)) - np.min(np.abs(values)))
    if scan_type == "position":
        if rng < 0.001:
            return "m"
        return "mm" if med < 100.0 else "µm"
    if med < 1e-6:
        return "s"
    return "ps" if med < 1.0 else "fs"


def unit_to_si(unit: str) -> float:
    """Multiplicative factor converting ``unit`` to SI (metres or seconds)."""
    return UNIT_FACTORS.get(unit, 1.0)


def position_to_delay(z: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Convert stage position (m) to double-pass delay (s), :math:`\tau = 2z/c`."""
    return 2.0 * np.asarray(z, dtype=float) / C


def csv_wavelength_unit(path: str) -> str:
    """Guess the wavelength unit of a 2-column CSV's first column (``nm`` on error)."""
    with contextlib.suppress(Exception):
        data = load_csv(path)
        if data.shape[1] >= 1 and data.shape[0] >= 2:
            return guess_wavelength_unit(data[:, 0])
    return "nm"


def to_1d(arr: NDArray, name: str = "") -> NDArray[np.float64]:
    """Flatten a near-1-D array (e.g. ``(N,1)``) to 1-D; error if genuinely 2-D.

    Used for wavelength axes loaded from HDF5/NPZ where the dataset may carry a
    singleton dimension.
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        return arr
    if np.count_nonzero(np.array(arr.shape) > 1) <= 1:
        return arr.reshape(-1)
    raise ValueError(
        f"dataset {name!r} has shape {arr.shape}; pick a 1-D wavelength axis"
    )


def reduce_to_1d(arr: NDArray, name: str, match_len: int) -> NDArray[np.float64]:
    """Reduce an intensity array to length ``match_len`` (averaging multi-shot axes).

    Mirrors ``Clean._meanspectrum``: a 2-D ``(N_lambda, N_shots)`` array is
    averaged over the non-matching axis. Square arrays default to averaging
    axis 1.
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        return arr
    if np.count_nonzero(np.array(arr.shape) > 1) <= 1:
        return arr.reshape(-1)
    if arr.ndim == 2:
        n0, n1 = arr.shape
        if n0 == match_len and n1 != match_len:
            return arr.mean(axis=1)
        if n1 == match_len and n0 != match_len:
            return arr.mean(axis=0)
        if n0 == match_len and n1 == match_len:
            return arr.mean(axis=1)
    raise ValueError(
        f"dataset {name!r} has shape {arr.shape}; cannot reduce to length {match_len}"
    )


# ---------------------------------------------------------------------------
# Numerically simulated FROG scans (Luna / ``scansave`` HDF5 layout)
# ---------------------------------------------------------------------------
#: HDF5 dataset names of the simulated-scan trace windows: the aperture-windowed
#: signal integrated over all k (``Iω_win``) or at the on-axis re-imaged pixel
#: (``Iω_win_reimaged``), and the full signal-beam collection without the aperture
#: crop (``Iω_full``, newer files only). The unicode ``ω`` is part of the on-disk
#: key string. ``read_simulated_scan`` reads any of these as the trace; not every
#: file stores every one (use :func:`read_simulated_window_keys` to list those
#: actually present).
SIMULATED_WINDOW_KEYS: tuple[str, ...] = ("Iω_win", "Iω_win_reimaged", "Iω_full")

#: Logical time-domain "truth" sources a simulated scan can provide for the
#: retrieval overlay: the post-mask gate **beamlet** (the beam that actually
#: gates, ``It_beamlet``/``Ito_beamlet``) or the pre-mask ideal **source** input
#: (``It``/``Ito``). Listed in preference order (``beamlet`` first), so a file
#: that stores the beamlet defaults to it. ``read_simulated_truth_keys`` lists the
#: subset a given file actually stores.
SIMULATED_TRUTH_SOURCES: tuple[str, ...] = ("beamlet", "source")


@dataclass(frozen=True)
class SimulatedScan:
    r"""A numerically simulated FROG scan read from a ``scansave`` HDF5 file.

    The arrays are returned in croak's conventions: the angular-frequency axis is
    **absolute** and sorted ascending, and the trace is oriented
    ``(Nomega, Ndelay)``. Spectral quantities (``trace``, ``Iomega``,
    ``Iomega_beamlet``) are **angular-frequency densities** :math:`|\tilde E|^2`,
    *not* the wavelength densities a spectrometer records — converting between the
    two is the caller's job (see
    :func:`croak.preprocess.omega_to_lambda_density`).

    Attributes
    ----------
    omega : numpy.ndarray
        Absolute angular-frequency axis (rad/s), ascending. Shape ``(Nomega,)``.
    omega0 : float
        Carrier angular frequency (rad/s).
    tau : numpy.ndarray
        Delay axis (s). Shape ``(Ndelay,)``.
    trace : numpy.ndarray
        Signal spectral intensity ``(Nomega, Ndelay)`` (ω-density), as simulated.
    Iomega : numpy.ndarray
        Reference (input) pulse spectrum ``(Nomega,)`` (ω-density).
    Iomega_beamlet : numpy.ndarray or None
        Vignetted gate-beam spectrum ``(Nomega,)`` for the mask correction, or
        ``None`` when the file does not provide it.
    t : numpy.ndarray
        Truth temporal axis (s); the oversampled ``To`` if present.
    It : numpy.ndarray
        Truth temporal intensity for the chosen ``truth_source`` (the post-mask
        ``It_beamlet`` by default, else the source ``It``); the oversampled
        ``Ito*`` if present.
    tau_fwhm : float
        Intensity FWHM of the input pulse (s).
    window_key : str
        The trace-window dataset that was read.
    z_index : int
        The resolved (non-negative) propagation-slice index that was loaded. The
        selected propagation distance is ``z_positions[z_index]`` when
        ``z_positions`` is available.
    z_positions : numpy.ndarray or None
        The saved propagation distances (m) of the slices stacked along the
        window's ``nz`` axis (the file's ``/grid/zsave``), strictly increasing
        from the entrance (``0``) to the substrate exit (``zmax``). ``None`` for
        the legacy single-thickness format that does not store ``zsave``.
    delay_convention : str
        The file's delay-axis convention: ``"gate"`` when the simulation stored
        the trace directly in the gate-delay (paper) frame (the file carries
        ``/grid/delay_convention``), else ``"legacy"`` — the probe-delayed
        frame of older files, whose delay axis must be negated on loading
        (``SimulatedLoadParams.reverse_trace``).
    Eomega_beamlet : numpy.ndarray or None
        The **complex** vignetted gate-beam spectrum (``Eω_beamlet_re/_im`` in
        newer files): the retrievable ground truth including its spectral
        phase, which carries any input chirp exactly (the mask is a real
        amplitude filter). Enables complex-field retrieval-error metrics and
        direct truth-GDD measurement, the temporal/spectral truth-phase
        overlays, and truth-seeded retrievals. ``None`` for files that store
        only intensities.
    Eomega : numpy.ndarray or None
        The complex pre-mask input (source) spectrum, same encoding; ``None``
        when absent.
    """

    omega: NDArray[np.float64]
    omega0: float
    tau: NDArray[np.float64]
    trace: NDArray[np.float64]
    Iomega: NDArray[np.float64]
    Iomega_beamlet: NDArray[np.float64] | None
    t: NDArray[np.float64]
    It: NDArray[np.float64]
    tau_fwhm: float
    window_key: str
    z_index: int
    z_positions: NDArray[np.float64] | None
    delay_convention: str = "legacy"
    Eomega_beamlet: NDArray[np.complex128] | None = None
    Eomega: NDArray[np.complex128] | None = None
    #: Mask geometry the scan was generated with (m), when the file records it.
    #: The smearing kernel is built from d/D = (spacing + D)/2D, and carrying
    #: that by hand between scripts is how a sweep ended up running a gap-500
    #: kernel against a gap-1000 trace — 25% too narrow, ~2% of retrieved
    #: duration, and invisible in the trace error. ``None`` for files written
    #: before the simulator recorded it.
    mask_diameter: float | None = None
    mask_spacing: float | None = None
    #: Focusing focal length (m), when recorded. The reduced smearing kernel
    #: never needs it (f cancels between spot size and tilt), but the explicit
    #: chromatic focal mixture (:mod:`croak.focal`) does.
    f_foc: float | None = None
    #: Interaction geometry the file was generated with (``"tg"`` / ``"sd"``),
    #: when recorded. It selects the smearing layout: a two-beam SD has no
    #: gate-shape channel at all, so the boxcars kernel is wrong for it.
    geometry: str | None = None
    #: The collection window the trace was recorded through, when the file stores
    #: it. Needed to model the finite collection aperture (:mod:`croak.collection`);
    #: ``None`` for files written before the simulator recorded the window.
    mask_window: MaskWindowSpec | None = None


@dataclass(frozen=True)
class MaskWindowSpec:
    """The collection window a simulated scan was recorded through.

    ``scansave`` files store the signal-extraction window as flattened
    ``/grid/window_def_*`` scalars rather than the (gigabyte-scale) window array, so
    the aperture can be rebuilt losslessly. This is the raw record; turning it into a
    quadrature is :func:`croak.collection.aperture_from_scan`.

    Attributes
    ----------
    window_type : str
        The simulator's window class, e.g. ``"PhysicalMaskWindow"``.
    hole_x, hole_y : float
        Hole centre in the mask plane (m).
    hole_diameter : float
        Hole diameter (m).
    z_mask : float
        Mask-plane distance from the focus (m).
    apod : str
        Apodisation form: ``"hard"``, ``"supergauss"`` or ``"tanh"``.
    apod_param : float or None
        Apodisation parameter, or ``None`` when the file recorded the simulator's
        ``"default"`` --- which depends on the transverse k-grid and so must be
        re-derived (:func:`croak.collection.resolve_tanh_width`) from ``delta_k``,
        ``omega`` and ``reference_wavelength``.
    delta_k : float
        Transverse k-grid spacing of the simulation (rad/m), from ``/grid/kx``.
    reference_wavelength : float
        The carrier wavelength (m) the simulator's defaults were evaluated at.
    omega_reference : float
        The simulation's own frequency grid point nearest ``2 pi c /
        reference_wavelength`` (rad/s) --- which is what the simulator evaluates its
        default apodisation width at, and on a coarse spectral grid is not the same
        number.
    """

    window_type: str
    hole_x: float
    hole_y: float
    hole_diameter: float
    z_mask: float
    apod: str
    apod_param: float | None
    delta_k: float
    reference_wavelength: float
    omega_reference: float


def _read_text(node: h5py.Group, name: str) -> str:
    """Read a 0-d string dataset, decoding the fixed-width bytes HDF5 stores."""
    raw = _require_dataset(node, name)[()]
    return raw.decode() if isinstance(raw, bytes) else str(raw)


def window_def_prefix(window: str | int | None) -> str:
    """Map a trace-window selector to its ``window_def`` key prefix.

    Multi-aperture scans store one record per collection hole:
    ``window_def_*`` for the first window (``Iω_win``/``Iω_win_reimaged``) and
    ``window_def_N_*`` for ``Iω_win_N`` (and its ``_reimaged`` partner).
    Accepts the dataset name (``"Iω_win_5"``, ``"Iω_win_5_reimaged"``), the
    bare index (``5``), or ``None``/``"Iω_win"`` for the first window.
    """
    if window is None:
        return "window_def_"
    if isinstance(window, int):
        return "window_def_" if window <= 1 else f"window_def_{window}_"
    name = window.removesuffix("_reimaged")
    m = re.fullmatch(r"Iω_win(?:_(\d+))?", name)
    if m is None:
        raise ValueError(
            f"cannot map trace window {window!r} to a window_def record; "
            "expected 'Iω_win', 'Iω_win_<n>' or their '_reimaged' partners"
        )
    return f"window_def_{m.group(1)}_" if m.group(1) else "window_def_"


def _mask_window_spec(
    g: h5py.Group, window: str | int | None = None
) -> MaskWindowSpec | None:
    """Build a :class:`MaskWindowSpec` from a ``/grid`` group, or ``None`` if absent.

    Every ingredient is required: a half-recorded window would rebuild to a
    confidently wrong aperture, and silence is worse than the ``None`` that says the
    file predates the record. ``window`` selects which collection hole of a
    multi-aperture scan (see :func:`window_def_prefix`).
    """
    p = window_def_prefix(window)
    needed = (
        f"{p}type",
        f"{p}holex",
        f"{p}holey",
        f"{p}holediam",
        f"{p}zmask",
        f"{p}apod",
        f"{p}apod_param",
        "kx",
        "ω",
        "referenceλ",
    )
    if any(name not in g for name in needed):
        return None
    kx = _read_array(g, "kx")
    if kx.size < 2:
        return None
    raw_param = _require_dataset(g, f"{p}apod_param")[()]
    param = raw_param.decode() if isinstance(raw_param, bytes) else raw_param
    # The simulator writes the string "default" when the parameter was resolved from
    # its own grid; anything else is the number it used.
    apod_param = None if isinstance(param, str) else float(np.asarray(param))
    # The simulator evaluates its defaults at the GRID POINT nearest the carrier, so
    # resolve that here rather than at 2 pi c / lambda0.
    grid_omega = _read_array(g, "ω")
    reference_wavelength = _read_scalar(g, "referenceλ")
    nearest = int(
        np.argmin(np.abs(grid_omega - 2.0 * np.pi * C / reference_wavelength))
    )
    return MaskWindowSpec(
        window_type=_read_text(g, f"{p}type"),
        hole_x=_read_scalar(g, f"{p}holex"),
        hole_y=_read_scalar(g, f"{p}holey"),
        hole_diameter=_read_scalar(g, f"{p}holediam"),
        z_mask=_read_scalar(g, f"{p}zmask"),
        apod=_read_text(g, f"{p}apod"),
        apod_param=apod_param,
        delta_k=float(kx[1] - kx[0]),
        reference_wavelength=reference_wavelength,
        omega_reference=float(grid_omega[nearest]),
    )


def read_simulated_mask_window(
    path: str, window: str | int | None = None
) -> MaskWindowSpec | None:
    """Read just the collection-window record of a simulated scan.

    A lightweight metadata reader, in the same spirit as
    :func:`read_simulated_z_positions`: it opens the ``scansave`` file and returns the
    ``/grid/window_def_*`` record without touching the (large) trace windows.

    Parameters
    ----------
    path : str
        Path to the ``.h5``/``.hdf5`` scan file.
    window : str or int, optional
        Which collection hole of a multi-aperture scan: the trace-window
        dataset name (``"Iω_win_5"``, ``_reimaged`` accepted) or the bare
        index; ``None`` is the first window (see :func:`window_def_prefix`).

    Returns
    -------
    MaskWindowSpec or None
        ``None`` when the file does not record a window (or records it only partly).

    Raises
    ------
    KeyError
        If the ``/grid`` group is missing (not a ``scansave`` file).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".h5", ".hdf5"):
        raise ValueError(f"simulated scans must be HDF5; got {ext!r}")
    with h5py.File(path, "r") as f:
        if "grid" not in f:
            raise KeyError(f"{path!r}: missing /grid group (not a scansave file?)")
        g = f["grid"]
        if not isinstance(g, h5py.Group):
            raise KeyError(f"{path!r}: /grid is not an HDF5 group")
        return _mask_window_spec(g, window)


def _require_dataset(node: h5py.Group, name: str) -> h5py.Dataset:
    """Return ``node[name]`` narrowed to an :class:`h5py.Dataset`."""
    obj = node[name]
    if not isinstance(obj, h5py.Dataset):
        raise TypeError(f"{name!r} is not an HDF5 dataset")
    return obj


def _read_array(node: h5py.Group, name: str) -> NDArray[np.float64]:
    """Read ``node[name]`` as a ``float64`` array."""
    return np.asarray(_require_dataset(node, name)[()], dtype=float)


def _read_scalar(node: h5py.Group, name: str) -> float:
    """Read a 0-d ``node[name]`` dataset as a Python float."""
    return float(np.asarray(_require_dataset(node, name)[()]))


def _truth_dataset_names(source: str) -> tuple[str, str]:
    """``(oversampled, native)`` intensity dataset names for a truth source.

    The post-mask gate ``"beamlet"`` is stored as ``Ito_beamlet``/``It_beamlet``;
    the pre-mask ``"source"`` input as ``Ito``/``It``. Both share the ``To``/``t``
    time axes.
    """
    suffix = "_beamlet" if source == "beamlet" else ""
    return f"Ito{suffix}", f"It{suffix}"


def _resolve_truth_keys(g: h5py.Group, truth_source: str) -> tuple[str, str]:
    """Resolve ``(time-axis, intensity)`` dataset names for the truth overlay.

    Prefers the requested ``truth_source`` (``"beamlet"`` = the post-mask gate
    beam, ``"source"`` = the ideal pre-mask input) and the oversampled time grid
    (``To``/``Ito*``) when present. Falls back to the other source when the
    requested one is absent — e.g. a Gaussian-beam run with no beamlet, or a
    legacy file storing only ``It`` — so the loader always returns *some* truth.
    """
    order = (
        ("beamlet", "source") if truth_source == "beamlet" else ("source", "beamlet")
    )
    for src in order:
        over, native = _truth_dataset_names(src)
        if "To" in g and over in g:
            return "To", over
        if native in g:
            return "t", native
    raise KeyError(
        "no time-domain truth dataset found (expected one of "
        "It/Ito/It_beamlet/Ito_beamlet in /grid)"
    )


def _orient_simulated_trace(
    window: NDArray[np.float64], n_omega: int, n_tau: int, z_index: int, key: str
) -> NDArray[np.float64]:
    """Select the propagation slice and orient a window to ``(Nomega, Ndelay)``.

    The ``scansave`` window is written by Julia (column-major) as
    ``(Nomega, nz, Ndelay)`` and so is read by h5py (row-major) reversed as
    ``(Ndelay, nz, Nomega)``; the propagation (``nz``) axis is the *middle* axis
    either way (a full axis reversal leaves the middle axis fixed). A 2-D window
    (no propagation axis) is accepted as-is. The final orientation is checked
    against the axis lengths and transposed if needed, so the result is always
    ``(Nomega, Ndelay)`` regardless of the stored order.
    """
    win = np.asarray(window, dtype=float)
    if win.ndim == 3:
        nz = win.shape[1]
        if not -nz <= z_index < nz:
            raise IndexError(f"z_index={z_index} out of range for {key!r} (nz={nz})")
        win = win[:, z_index, :]
    elif win.ndim != 2:
        raise ValueError(
            f"trace window {key!r} has shape {win.shape}; expected 2-D/3-D"
        )

    if win.shape == (n_omega, n_tau):
        return win
    if win.shape == (n_tau, n_omega):
        return win.T
    raise ValueError(
        f"trace window {key!r} oriented to {win.shape}, which matches neither "
        f"(Nomega, Ndelay)=({n_omega}, {n_tau}) nor its transpose"
    )


def _resolve_z_index(
    z_index: int,
    z_thickness: float | None,
    z_positions: NDArray[np.float64] | None,
    nz: int,
    path: str,
) -> int:
    """Resolve the propagation slice to load to a non-negative index.

    Precedence is ``z_thickness > z_index``:
    a non-``None`` ``z_thickness`` selects the slice whose saved position
    ``z_positions`` is nearest (``argmin|z_positions − z_thickness|``) and
    requires the multi-thickness ``zsave`` to be present; otherwise the explicit
    ``z_index`` is used. The result is normalised to ``[0, nz)`` so it can be
    stored on the returned :class:`SimulatedScan` and used to look up the chosen
    thickness as ``z_positions[z_index]``.
    """
    if z_thickness is not None:
        if z_positions is None:
            raise KeyError(
                f"{path!r}: z_thickness requested but the file has no /grid/zsave "
                "(legacy single-thickness scansave format)"
            )
        return int(np.argmin(np.abs(z_positions - z_thickness)))
    if not -nz <= z_index < nz:
        raise IndexError(f"z_index={z_index} out of range (nz={nz})")
    return z_index % nz


def read_simulated_scan(
    path: str,
    *,
    window_key: str = "Iω_win",
    z_index: int = -1,
    z_thickness: float | None = None,
    truth_source: str = "beamlet",
) -> SimulatedScan:
    """Read a numerically simulated FROG scan from a ``scansave`` HDF5 file.

    The file layout (a Luna ``scansave`` output) is a ``/grid`` group with the
    frequency/time axes and reference pulse, a ``/scanvariables/τ`` delay axis,
    and one or more top-level trace windows (``Iω_win``, ``Iω_win_reimaged``).
    This reader only *reads and orients*; the wavelength conversion, Jacobian,
    third-order and mask corrections live in
    :func:`croak.session.pipeline.assemble_simulated_load_data`.

    Parameters
    ----------
    path : str
        Path to the ``.h5``/``.hdf5`` scan file.
    window_key : str, optional
        Trace-window dataset to read: ``"Iω_win"`` (full-beam, default) or
        ``"Iω_win_reimaged"`` (on-axis re-imaged).
    z_index : int, optional
        Propagation-slice index along the window's ``nz`` axis. Default ``-1``
        (the last slice, i.e. the substrate exit), matching the Julia ``:end``.
        Ignored when ``z_thickness`` is given.
    z_thickness : float, optional
        Material thickness (m) to load from a multi-thickness scan. When given,
        the slice whose saved position ``/grid/zsave`` is nearest this value is
        selected (``argmin|zsave − z_thickness|``), taking precedence over
        ``z_index``. Requires the file to store ``/grid/zsave`` (the newer
        multi-thickness format); a value is invalid on the legacy format.
    truth_source : {"beamlet", "source"}, optional
        Which stored reference pulse to return as the time-domain truth (``t``,
        ``It``): the post-mask gate **beamlet** (``It_beamlet``/``Ito_beamlet``,
        default — the beam that actually gates) or the pre-mask ideal **source**
        input (``It``/``Ito``). The oversampled grid (``To``/``Ito*``) is used
        when present. Falls back to the other source when the requested one is
        absent (e.g. a Gaussian-beam run with no beamlet, or a legacy file with
        only ``It``). See :func:`read_simulated_truth_keys`.

    Returns
    -------
    SimulatedScan
        Axes and trace in croak conventions (absolute ω ascending,
        ``(Nomega, Ndelay)``), with the resolved ``z_index`` and (when present)
        the ``z_positions`` thickness axis.

    Raises
    ------
    KeyError
        If the ``/grid`` group, the ``/scanvariables/τ`` axis, or ``window_key``
        is missing (the error lists the available top-level datasets), or if
        ``z_thickness`` is requested but the file has no ``/grid/zsave``.
    IndexError
        If ``z_index`` is out of range for the window's ``nz`` axis.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".h5", ".hdf5"):
        raise ValueError(f"simulated scans must be HDF5; got {ext!r}")

    with h5py.File(path, "r") as f:
        if "grid" not in f:
            raise KeyError(f"{path!r}: missing /grid group (not a scansave file?)")
        g = f["grid"]
        if not isinstance(g, h5py.Group):
            raise KeyError(f"{path!r}: /grid is not an HDF5 group")
        omega = _read_array(g, "ω")
        omega0 = _read_scalar(g, "ω0")
        Iomega = _read_array(g, "Iω")
        beamlet = _read_array(g, "Iω_beamlet") if "Iω_beamlet" in g else None

        def _read_complex(prefix: str) -> NDArray[np.complex128] | None:
            """Assemble a complex spectrum stored as ``<prefix>_re``/``_im``."""
            if f"{prefix}_re" not in g or f"{prefix}_im" not in g:
                return None
            return _read_array(g, f"{prefix}_re") + 1j * _read_array(g, f"{prefix}_im")

        e_beamlet = _read_complex("Eω_beamlet")
        e_source = _read_complex("Eω")
        # Delay-convention marker: newer ModelPNPS files store the trace
        # directly in the gate-delay (paper) frame and say so; marker-less
        # files are the legacy probe-delayed frame (reversed on loading).
        if "delay_convention" in g:
            raw = _require_dataset(g, "delay_convention")[()]
            delay_convention = raw.decode() if isinstance(raw, bytes) else str(raw)
        else:
            delay_convention = "legacy"

        def _grid_scalar(name):
            """One stored scalar, or None when the file predates it."""
            if name not in g:
                return None
            try:
                return float(np.asarray(_require_dataset(g, name)).ravel()[0])
            except Exception:
                return None

        # Mask geometry, when recorded. Deliberately NOT reconstructed from the
        # signal-window fields: those pin it only for the boxcars layout, and
        # for self-diffraction the window sits at 1.5x the beam separation
        # rather than at d, so the same arithmetic would give a confidently
        # wrong answer on exactly the geometry that needs it most.
        mask_diameter = _grid_scalar("mask_diam")
        mask_spacing = _grid_scalar("mask_spacing")
        f_foc = _grid_scalar("f_foc")
        geometry = None
        if "geometry" in g:
            raw = _require_dataset(g, "geometry")[()]
            geometry = raw.decode() if isinstance(raw, bytes) else str(raw)
        mask_window = _mask_window_spec(g)
        # The multi-thickness format stacks the signal at several propagation
        # distances along the window's nz axis and records them in /grid/zsave
        # (strictly increasing, entrance 0 → exit zmax). The legacy single-
        # thickness format omits it; z_positions is then None.
        z_positions = _read_array(g, "zsave") if "zsave" in g else None
        # Time-domain truth overlay: the chosen source (post-mask beamlet by
        # default, else the pre-mask source), oversampled (To/Ito*) when present.
        t_key, it_key = _resolve_truth_keys(g, truth_source)
        t = _read_array(g, t_key)
        It = _read_array(g, it_key)
        tau_fwhm = _read_scalar(g, "τfwhm")

        sv = f["scanvariables"] if "scanvariables" in f else None
        if not isinstance(sv, h5py.Group) or "τ" not in sv:
            raise KeyError(f"{path!r}: missing /scanvariables/τ delay axis")
        tau = _read_array(sv, "τ")

        if window_key not in f:
            available = [
                k for k in f if k not in ("grid", "scanvariables", "scanorder")
            ]
            raise KeyError(
                f"{path!r}: window_key {window_key!r} not found; "
                f"available trace windows: {available}"
            )
        window = _read_array(f, window_key)
        # The propagation slices are the window's middle axis (a 2-D window has
        # none, i.e. a single implicit slice). Resolve which one to load.
        nz = window.shape[1] if window.ndim == 3 else 1
        z_index = _resolve_z_index(z_index, z_thickness, z_positions, nz, path)
        trace = _orient_simulated_trace(
            window, omega.size, tau.size, z_index, window_key
        )

    # Sort onto an ascending absolute-ω axis, keeping the trace rows and the
    # per-ω spectra aligned with their frequency bin.
    order = np.argsort(omega)
    omega = omega[order]
    trace = trace[order, :]
    Iomega = Iomega[order]
    if beamlet is not None:
        beamlet = beamlet[order]
    if e_beamlet is not None:
        e_beamlet = e_beamlet[order]
    if e_source is not None:
        e_source = e_source[order]

    return SimulatedScan(
        omega=omega,
        omega0=omega0,
        tau=tau,
        trace=trace,
        Iomega=Iomega,
        Iomega_beamlet=beamlet,
        t=t,
        It=It,
        tau_fwhm=tau_fwhm,
        window_key=window_key,
        z_index=z_index,
        z_positions=z_positions,
        delay_convention=delay_convention,
        Eomega_beamlet=e_beamlet,
        Eomega=e_source,
        mask_diameter=mask_diameter,
        mask_spacing=mask_spacing,
        f_foc=f_foc,
        geometry=geometry,
        mask_window=mask_window,
    )


def read_simulated_z_positions(path: str) -> NDArray[np.float64] | None:
    """Read just the propagation-thickness axis of a simulated scan.

    A lightweight metadata reader for populating a thickness selector without
    loading the (large) trace windows: it opens the ``scansave`` HDF5 file and
    returns only ``/grid/zsave``.

    Parameters
    ----------
    path : str
        Path to the ``.h5``/``.hdf5`` scan file.

    Returns
    -------
    numpy.ndarray or None
        The saved propagation distances (m), strictly increasing from the
        entrance (``0``) to the substrate exit (``zmax``), or ``None`` for the
        legacy single-thickness format that does not store ``zsave``.

    Raises
    ------
    KeyError
        If the ``/grid`` group is missing (not a ``scansave`` file).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".h5", ".hdf5"):
        raise ValueError(f"simulated scans must be HDF5; got {ext!r}")
    with h5py.File(path, "r") as f:
        if "grid" not in f:
            raise KeyError(f"{path!r}: missing /grid group (not a scansave file?)")
        g = f["grid"]
        if not isinstance(g, h5py.Group):
            raise KeyError(f"{path!r}: /grid is not an HDF5 group")
        return _read_array(g, "zsave") if "zsave" in g else None


def _window_key_order(name: str) -> tuple[int, int, int, str]:
    """Sort key placing numbered windows after the canonical trio, in order.

    Multi-aperture scans store one propagation reduced through several
    collection windows as ``Iω_win_2``, ``Iω_win_3``, … with matching
    ``_reimaged`` variants (and ``Iω_win_ωdep`` for the ω-dependent Planck
    pair). Numbered windows sort by index with each ``_reimaged`` directly
    after its base; anything else falls to the end alphabetically.
    """
    m = re.fullmatch(r"Iω_win_(\d+)(_reimaged)?", name)
    if m:
        return (1, int(m.group(1)), 1 if m.group(2) else 0, name)
    return (2, 0, 0, name)


def read_simulated_window_keys(path: str) -> list[str]:
    """List the trace windows actually stored in a simulated scan.

    A lightweight metadata reader for populating a trace-window selector. The
    canonical trio (:data:`SIMULATED_WINDOW_KEYS` — ``Iω_win``,
    ``Iω_win_reimaged``, ``Iω_full``) comes first in that order; any further
    trace datasets follow — in particular the numbered windows of a
    multi-aperture scan (``Iω_win_2`` … with their ``_reimaged`` partners),
    where one propagation was reduced through a whole series of collection
    holes. Every returned key is loadable via
    :func:`read_simulated_scan`'s ``window_key``.

    Parameters
    ----------
    path : str
        Path to the ``.h5``/``.hdf5`` scan file.

    Returns
    -------
    list of str
        The available trace-window dataset names (possibly empty).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".h5", ".hdf5"):
        raise ValueError(f"simulated scans must be HDF5; got {ext!r}")
    with h5py.File(path, "r") as f:
        canonical = [k for k in SIMULATED_WINDOW_KEYS if k in f]
        extra: list[str] = []
        for key in f.keys():
            name = str(key)
            if not name.startswith("Iω") or name in SIMULATED_WINDOW_KEYS:
                continue
            node = f[name]
            if isinstance(node, h5py.Dataset) and node.ndim >= 2:
                extra.append(name)
        extra.sort(key=_window_key_order)
    return canonical + extra


def read_simulated_truth_keys(path: str) -> tuple[str, ...]:
    """List the time-domain truth sources a simulated scan provides.

    A lightweight metadata reader for populating the truth selector (and choosing
    its default): it returns the subset of :data:`SIMULATED_TRUTH_SOURCES` the
    file stores, in preference order (``"beamlet"`` first). ``"beamlet"`` is
    listed when ``/grid/It_beamlet`` or ``/grid/Ito_beamlet`` is present;
    ``"source"`` when ``/grid/It`` or ``/grid/Ito`` is present.

    Parameters
    ----------
    path : str
        Path to the ``.h5``/``.hdf5`` scan file.

    Returns
    -------
    tuple of str
        The available truth sources (possibly empty for a malformed file).

    Raises
    ------
    KeyError
        If the ``/grid`` group is missing (not a ``scansave`` file).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".h5", ".hdf5"):
        raise ValueError(f"simulated scans must be HDF5; got {ext!r}")
    with h5py.File(path, "r") as f:
        if "grid" not in f:
            raise KeyError(f"{path!r}: missing /grid group (not a scansave file?)")
        g = f["grid"]
        if not isinstance(g, h5py.Group):
            raise KeyError(f"{path!r}: /grid is not an HDF5 group")
        present: list[str] = []
        for src in SIMULATED_TRUTH_SOURCES:
            over, native = _truth_dataset_names(src)
            if over in g or native in g:
                present.append(src)
    return tuple(present)


def read_simulated_wavelength_range(path: str) -> tuple[float, float]:
    """Return the full positive-frequency wavelength span of a simulated scan.

    A lightweight metadata reader (reads only ``/grid/ω``) for defaulting the
    load band to the file's full data range. The simulation grid is an FFT axis
    spanning positive and negative absolute frequencies; only the positive bins
    carry a wavelength, and ``λ = 2πc/ω`` maps the largest ω to the shortest λ
    and the smallest positive ω to the longest λ.

    Parameters
    ----------
    path : str
        Path to the ``.h5``/``.hdf5`` scan file.

    Returns
    -------
    tuple of float
        ``(lam_min_nm, lam_max_nm)`` — the shortest and longest wavelengths (nm)
        of the positive-frequency bins.

    Raises
    ------
    KeyError
        If the ``/grid`` group or its ``ω`` axis is missing.
    ValueError
        If the grid has no positive-frequency bins.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".h5", ".hdf5"):
        raise ValueError(f"simulated scans must be HDF5; got {ext!r}")
    with h5py.File(path, "r") as f:
        if "grid" not in f:
            raise KeyError(f"{path!r}: missing /grid group (not a scansave file?)")
        g = f["grid"]
        if not isinstance(g, h5py.Group):
            raise KeyError(f"{path!r}: /grid is not an HDF5 group")
        omega = _read_array(g, "ω")
    pos = omega[omega > 0.0]
    if pos.size == 0:
        raise ValueError(f"{path!r}: /grid/ω has no positive-frequency bins")
    lam_min = 2.0 * np.pi * C / float(pos.max()) / 1e-9
    lam_max = 2.0 * np.pi * C / float(pos.min()) / 1e-9
    return lam_min, lam_max
