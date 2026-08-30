r"""Chirped-mirror dispersion (reflectivity + group-delay data).

A port of the ``lookup_mirror`` / ``process_mirror_data`` routines of
`Luna.jl <https://github.com/LupoLab/Luna.jl>`_ (``PhysData.jl``), an
open-source (MIT) nonlinear pulse-propagation package, along with its bundled
mirror data. Each mirror is described by measured reflectivity ``R(λ)``
and group-delay dispersion ``GDD(λ)`` (or group delay ``GD(λ)``); the dispersion
is integrated to a spectral phase, its overall group delay removed, and a
Planck-taper window rolls it smoothly to zero outside the design band.

The public products are per-bounce spectral-phase functions
``phase(omega_abs) -> ndarray`` in **croak's** sign convention (``exp(+iφ)``,
matching :mod:`croak.dispersion`), so they drop straight into
:data:`croak.dispersion.MIRRORS` via :func:`croak.dispersion.register_mirror`.

Both the chirped compressor mirrors (:data:`BUILTIN_MIRRORS`) and the beam-path
coatings (:data:`BUILTIN_COATINGS`) return an amplitude (reflectivity) function
alongside the phase, so the measured loss per bounce *is* modelled: applying a
mirror attenuates by ``r(λ)`` and removing it divides by ``r(λ)`` (clamped). For
good chirped compressors ``R`` is ≳99 % in band, so this barely shifts the
peak-power auto-compressor; for the coatings (bare Si, MgF₂-coated Al, exported
by ``jaxmirror`` with per-polarisation reflectivity, phase and GDD) it matters,
as they are usually *removed* (back-propagated) to recover the pulse upstream.
See :func:`croak.dispersion.register_mirror`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import cumulative_trapezoid

from .maths import planck_taper, wlfreq

__all__ = [
    "BUILTIN_COATINGS",
    "BUILTIN_MIRRORS",
    "MirrorSpec",
    "load_builtin",
    "load_coating",
    "mirror_from_arrays",
    "process_mirror_data",
]

_DATA_DIR = Path(__file__).parent / "data" / "mirrors"

PhaseFn = Callable[[ArrayLike], NDArray[np.float64]]


@dataclass(frozen=True)
class MirrorSpec:
    """Description of a built-in mirror's data files and processing constants.

    Reflectivity (``%``) and dispersion come from one or two CSV/whitespace
    files; ``mode`` is ``"gdd"`` (fs², double-integrated) or ``"gd"`` (fs,
    single-integrated). ``taper_nm`` are the four Planck-taper corners
    ``(left0, left1, right1, right0)`` in nm; the inner pair is the valid range.
    """

    r_file: str  # reflectivity file ("" → lossless, r = 1)
    r_lam_col: int
    r_cols: tuple[int, ...]  # reflectivity columns (%) combined per `r_combine`
    r_combine: str  # "single" | "geom2"
    d_file: str  # dispersion file
    d_lam_col: int
    d_cols: tuple[int, ...]  # dispersion columns combined (averaged) per bounce
    mode: str  # "gdd" (fs²) | "gd" (fs)
    lam0_nm: float
    taper_nm: tuple[float, float, float, float]
    fitorder: int
    fit_window_fs: tuple[float, float] | None  # ωfs band used for the GD fit
    delim: str | None  # "," or None (whitespace)

    @property
    def valid_range_m(self) -> tuple[float, float]:
        """Valid wavelength range (m): the inner Planck-taper corners."""
        return self.taper_nm[1] * 1e-9, self.taper_nm[2] * 1e-9


#: Built-in mirrors ported from Luna.jl, keyed by name.
BUILTIN_MIRRORS: dict[str, MirrorSpec] = {
    # UltraFast Innovations / chirped-mirror pairs (GDD per reflection, fs²)
    "PC70": MirrorSpec(
        "PC70_R.csv",
        0,
        (1, 2),
        "geom2",
        "PC70_GDD.csv",
        0,
        (1, 2),
        "gdd",
        800.0,
        (400.0, 450.0, 1200.0, 1300.0),
        5,
        (2.0, 4.0),
        ",",
    ),
    "HD59": MirrorSpec(
        "",
        0,
        (),
        "single",
        "HD59_GDD.dat",
        0,
        (1,),
        "gdd",
        1030.0,
        (993.0, 1000.0, 1060.0, 1075.0),
        5,
        None,
        None,
    ),
    "PC147": MirrorSpec(
        "PC147.txt",
        0,
        (1,),
        "single",
        "PC147.txt",
        2,
        (3,),
        "gdd",
        1030.0,
        (640.0, 650.0, 1350.0, 1360.0),
        5,
        None,
        None,
    ),
    "PC1611": MirrorSpec(
        "PC1611.txt",
        0,
        (1,),
        "single",
        "PC1611.txt",
        2,
        (3,),
        "gdd",
        1030.0,
        (850.0, 855.0, 1195.0, 1200.0),
        5,
        None,
        None,
    ),
    "PC1821": MirrorSpec(
        "PC1821.txt",
        0,
        (1,),
        "single",
        "PC1821.txt",
        2,
        (3,),
        "gdd",
        1030.0,
        (800.0, 805.0, 1345.0, 1350.0),
        5,
        None,
        None,
    ),
    "HD120": MirrorSpec(
        "HD120.csv",
        0,
        (1,),
        "single",
        "HD120.csv",
        2,
        (3,),
        "gdd",
        1030.0,
        (880.0, 900.0, 1200.0, 1220.0),
        5,
        None,
        ",",
    ),
    # Thorlabs UMC ultrafast mirrors: group *delay* (fs), s-polarisation
    "ThorlabsUMC": MirrorSpec(
        "UCxx-15FS_R.csv",
        0,
        (2,),
        "single",
        "UCxx-15FS_GD.csv",
        0,
        (2,),
        "gd",
        800.0,
        (640.0, 650.0, 1050.0, 1100.0),
        3,
        (2.0, 4.0),
        ",",
    ),
}


#: Built-in beam-path coating mirrors (``jaxmirror`` exports), keyed by a
#: friendly ``"<coating> @ <angle>"`` name → CSV filename under ``data/mirrors/``.
#: These files share the column layout
#: ``wavelength_nm, R_s, R_p, phase_s_rad, phase_p_rad, GDD_s_fs2, GDD_p_fs2``
#: (phase convention sign ``+1``, matching croak). Loaded by :func:`load_coating`,
#: which builds the per-bounce phase from the GDD column (so the wrapped phase
#: columns and the s/p ambiguity at normal incidence never bite) and the
#: amplitude from the reflectivity column for the chosen polarisation.
BUILTIN_COATINGS: dict[str, str] = {
    "Si @ 0°": "Si_00deg.csv",
    "Si @ 45°": "Si_45deg.csv",
    "Si @ 74°": "Si_74deg.csv",
    "MgF2/Al @ 0°": "MgF2Al_00deg.csv",
    "MgF2/Al @ 45°": "MgF2Al_45deg.csv",
}

#: Floor on a mirror's amplitude reflectivity, so that *removing* a mirror
#: (dividing by ``r^N``) cannot hit a literal zero at an in-band reflectivity
#: dip. The net gain is bounded again by ``dispersion._MIRROR_AMPLITUDE_CAP``.
_AMPLITUDE_FLOOR = 1e-3

#: Column indices in a coating CSV for each polarisation: (reflectivity, GDD).
_COATING_COLS: dict[str, tuple[int, int]] = {"s": (1, 5), "p": (2, 6)}


def _amplitude_callable(
    lam_m: ArrayLike, r: ArrayLike
) -> Callable[[ArrayLike], NDArray[np.float64]]:
    r"""Build a per-bounce amplitude function ``r(omega_abs)`` for application.

    The returned callable interpolates the amplitude reflectivity ``r`` on
    wavelength, floored at :data:`_AMPLITUDE_FLOOR` (so removal cannot divide by
    zero) and returning ``1`` outside the data range (no effect where the mirror
    has no data — consistent with the phase being tapered to zero there).
    """
    lam_m = np.asarray(lam_m, dtype=float)
    r = np.clip(np.asarray(r, dtype=float), _AMPLITUDE_FLOOR, None)
    order = np.argsort(lam_m)
    lam_sorted, r_sorted = lam_m[order], r[order]

    def amplitude_fn(omega_abs: ArrayLike) -> NDArray[np.float64]:
        lam = wlfreq(np.asarray(omega_abs, dtype=float))
        return np.interp(lam, lam_sorted, r_sorted, left=1.0, right=1.0)

    return amplitude_fn


def _ones_amplitude(omega_abs: ArrayLike) -> NDArray[np.float64]:
    """Return unit amplitude (``r = 1``) — for mirrors that ship with no R data."""
    return np.ones_like(np.asarray(omega_abs, dtype=float))


def _read_columns(
    path: Path, cols: tuple[int, ...], delim: str | None
) -> NDArray[np.float64]:
    """Read selected numeric columns, skipping headers and ragged short rows.

    Robust to text header lines (e.g. quoted titles or ``"Simulated"``) and to
    rows with a varying number of trailing columns (as in ``PC147.txt``): a line
    is kept only if all requested columns parse as floats.
    """
    rows: list[list[float]] = []
    with open(path) as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = line.split(delim) if delim else line.split()
            try:
                rows.append([float(parts[c]) for c in cols])
            except ValueError, IndexError:
                continue  # header line or ragged short row
    if not rows:
        raise ValueError(f"no numeric data in {path}")
    return np.asarray(rows, dtype=float)


def _remove_group_delay(
    phi: NDArray[np.float64],
    omega_fs: NDArray[np.float64],
    omega_fs0: float,
    fitorder: int,
    fit_window: tuple[float, float] | None,
) -> NDArray[np.float64]:
    """Subtract the constant + linear (overall group-delay) part of ``phi``.

    Fits a polynomial of degree ``fitorder`` over ``fit_window`` (or all points),
    keeps only its order-0 and order-1 terms, and subtracts them everywhere. A
    constant phase and a linear phase (a pure group delay) are physically
    irrelevant to pulse shape — they only shift the carrier phase and the arrival
    time — so removing them leaves the GDD and higher orders the mirror actually
    imposes. Fitting to a higher ``fitorder`` than 1 and *then* discarding the
    high-order terms keeps the fit from being biased by curvature in the data.
    """
    x = omega_fs - omega_fs0
    mask = np.isfinite(phi)
    if fit_window is not None:
        lo, hi = fit_window
        mask = mask & (omega_fs > lo) & (omega_fs < hi)
    coeffs = np.polyfit(x[mask], phi[mask], fitorder)  # highest power first
    const, lin = coeffs[-1], coeffs[-2]
    return phi - (lin * x + const)


def process_mirror_data(
    lam_R: ArrayLike | None,
    R: ArrayLike | None,
    lam_disp: ArrayLike,
    disp: ArrayLike,
    *,
    mode: str,
    lam0_m: float,
    taper_m: tuple[float, float, float, float],
    fitorder: int = 5,
    fit_window_fs: tuple[float, float] | None = None,
) -> tuple[PhaseFn, Callable[[ArrayLike], NDArray[np.float64]]]:
    r"""Build per-bounce ``phase(omega_abs)`` and ``r(lambda_m)`` from mirror data.

    Parameters
    ----------
    lam_R, R : array_like or None
        Reflectivity wavelength (m) and *amplitude* reflectivity (0–1), or
        ``None`` for a lossless mirror (``r = 1``).
    lam_disp, disp : array_like
        Dispersion wavelength (m) and the dispersion value: GDD (s²) if
        ``mode == "gdd"`` or group delay (s) if ``mode == "gd"``.
    mode : {"gdd", "gd"}
        Whether ``disp`` is GDD (double-integrated to phase) or group delay
        (single-integrated).
    lam0_m : float
        Central wavelength used to reference the removed group delay.
    taper_m : tuple
        Planck-taper corners ``(left0, left1, right1, right0)`` in m.
    fitorder : int
        Degree of the group-delay-removal polynomial.
    fit_window_fs : tuple or None
        ``(lo, hi)`` band in ωfs over which to fit the group delay (edge kinks
        otherwise corrupt the fit). ``None`` fits all points.

    Returns
    -------
    (phase_fn, r_fn)
        ``phase_fn(omega_abs)`` is the per-bounce spectral phase in croak's
        ``exp(+iφ)`` convention; ``r_fn(lambda_m)`` is the amplitude reflectivity
        (metadata only — not applied by the dispersion engine).
    """
    lam_disp = np.asarray(lam_disp, dtype=float)
    disp = np.asarray(disp, dtype=float)
    omega = wlfreq(lam_disp)  # data order (descending ω for ascending λ)

    # Integrate dispersion → spectral phase by cumulative trapezoid, in data
    # order (GDD needs two integrations in ω, GD one).
    if mode == "gdd":
        phi = cumulative_trapezoid(
            cumulative_trapezoid(disp, omega, initial=0.0), omega, initial=0.0
        )
    elif mode == "gd":
        phi = cumulative_trapezoid(disp, omega, initial=0.0)
    else:
        raise ValueError(f"mode must be 'gdd' or 'gd', got {mode!r}")

    omega_fs = omega * 1e-15
    omega_fs0 = float(wlfreq(lam0_m)) * 1e-15
    phi = _remove_group_delay(phi, omega_fs, omega_fs0, fitorder, fit_window_fs)

    # interpolation needs ascending λ
    order = np.argsort(lam_disp)
    lam_sorted, phi_sorted = lam_disp[order], phi[order]

    def phase_fn(omega_abs: ArrayLike) -> NDArray[np.float64]:
        lam = wlfreq(np.asarray(omega_abs, dtype=float))
        phi_i = np.interp(lam, lam_sorted, phi_sorted, left=0.0, right=0.0)
        window = planck_taper(lam, *taper_m)
        # croak defines GDD = +d²φ/dω² and applies exp(+iφ), so integrating the
        # tabulated GDD gives the applied phase directly, with no sign flip. A
        # chirped mirror's anomalous (negative) GDD therefore carries the same
        # sign here as croak's own Taylor-series and material GDD. Codes using the
        # opposite field-time convention, exp(-iφ), need the other sign — worth
        # checking before comparing against tabulated data from elsewhere.
        return phi_i * window

    if lam_R is None or R is None:

        def r_fn(lam_m: ArrayLike) -> NDArray[np.float64]:
            return np.ones_like(np.asarray(lam_m, dtype=float))
    else:
        lam_R = np.asarray(lam_R, dtype=float)
        R = np.asarray(R, dtype=float)
        ro = np.argsort(lam_R)
        lam_R_s, R_s = lam_R[ro], R[ro]

        def r_fn(lam_m: ArrayLike) -> NDArray[np.float64]:
            return np.interp(
                np.asarray(lam_m, dtype=float), lam_R_s, R_s, left=0.0, right=0.0
            )

    return phase_fn, r_fn


def load_builtin(
    name: str,
) -> tuple[PhaseFn, Callable[[ArrayLike], NDArray[np.float64]], tuple[float, float]]:
    """Load a built-in mirror by name.

    Returns ``(phase_fn, amplitude_fn, valid_range_m)``. ``amplitude_fn`` is the
    per-bounce amplitude reflectivity ready for application (floored, and ``1``
    outside the design band where the phase is also tapered to zero); it is
    ``1`` everywhere for the lossless mirrors that ship without R data.
    """
    if name not in BUILTIN_MIRRORS:
        raise ValueError(
            f"unknown mirror {name!r}; available: {', '.join(BUILTIN_MIRRORS)}"
        )
    spec = BUILTIN_MIRRORS[name]

    # reflectivity → amplitude (per reflection)
    if spec.r_file:
        rmat = _read_columns(
            _DATA_DIR / spec.r_file, (spec.r_lam_col, *spec.r_cols), spec.delim
        )
        lam_R = rmat[:, 0] * 1e-9
        r_pct = rmat[:, 1:]
        if spec.r_combine == "geom2":
            # sqrt of the geometric mean of two angles' reflectivities
            r = (r_pct[:, 0] / 100 * r_pct[:, 1] / 100) ** 0.25
        else:
            r = np.sqrt(r_pct[:, 0] / 100)
    else:
        lam_R = r = None

    # dispersion → SI (GDD s² or GD s), averaged across the listed columns
    dmat = _read_columns(
        _DATA_DIR / spec.d_file, (spec.d_lam_col, *spec.d_cols), spec.delim
    )
    lam_disp = dmat[:, 0] * 1e-9
    raw = dmat[:, 1:].mean(axis=1)
    disp = raw * (1e-30 if spec.mode == "gdd" else 1e-15)

    phase_fn, _r_fn = process_mirror_data(
        lam_R,
        r,
        lam_disp,
        disp,
        mode=spec.mode,
        lam0_m=spec.lam0_nm * 1e-9,
        taper_m=(
            spec.taper_nm[0] * 1e-9,
            spec.taper_nm[1] * 1e-9,
            spec.taper_nm[2] * 1e-9,
            spec.taper_nm[3] * 1e-9,
        ),
        fitorder=spec.fitorder,
        fit_window_fs=spec.fit_window_fs,
    )
    # application-ready amplitude (the metadata r_fn above zeroes out-of-band,
    # which is unsafe to divide by; this one is floored and 1 outside the band)
    if lam_R is None or r is None:
        amplitude_fn = _ones_amplitude
    else:
        amplitude_fn = _amplitude_callable(lam_R, r)
    return phase_fn, amplitude_fn, spec.valid_range_m


def load_coating(
    name: str, pol: str = "s"
) -> tuple[PhaseFn, Callable[[ArrayLike], NDArray[np.float64]], tuple[float, float]]:
    r"""Load a built-in beam-path coating mirror for one polarisation.

    Unlike the chirped compressors of :func:`load_builtin`, these are simple
    optics whose dispersion is usually *removed*: the returned amplitude function
    therefore carries the measured reflectivity (applied — and divided out on
    removal — by :func:`croak.dispersion.apply_dispersion`).

    The per-bounce phase is built from the GDD column (double-integrated and
    group-delay-removed by :func:`mirror_from_arrays`), which is wrap-free and,
    at normal incidence where ``GDD_s == GDD_p``, sidesteps the ~π s/p ambiguity
    of the tabulated phase columns.

    Parameters
    ----------
    name : str
        A key of :data:`BUILTIN_COATINGS` (e.g. ``"MgF2/Al @ 45°"``).
    pol : {"s", "p"}
        Polarisation; selects the reflectivity and GDD columns. At 0° incidence
        the two are identical.

    Returns
    -------
    (phase_fn, amplitude_fn, valid_range_m)
        ``phase_fn(omega_abs)`` is the per-bounce spectral phase (croak's
        ``exp(+iφ)`` convention); ``amplitude_fn(omega_abs)`` is the amplitude
        reflectivity ``r = √R`` (1 outside the data band, floored in-band);
        ``valid_range_m`` is the usable wavelength range (m).

    Raises
    ------
    ValueError
        If ``name`` is not a built-in coating or ``pol`` is not ``"s"``/``"p"``.
    """
    if name not in BUILTIN_COATINGS:
        raise ValueError(
            f"unknown coating {name!r}; available: {', '.join(BUILTIN_COATINGS)}"
        )
    if pol not in _COATING_COLS:
        raise ValueError(f"pol must be 's' or 'p', got {pol!r}")
    r_col, d_col = _COATING_COLS[pol]
    mat = _read_columns(_DATA_DIR / BUILTIN_COATINGS[name], (0, r_col, d_col), ",")
    lam_m = mat[:, 0] * 1e-9
    refl = mat[:, 1]  # already a fraction (0–1) in these files
    gdd = mat[:, 2] * 1e-30  # fs² → s²

    phase_fn, valid_range_m = mirror_from_arrays(lam_m, gdd, mode="gdd")
    amplitude_fn = _amplitude_callable(lam_m, np.sqrt(np.clip(refl, 0.0, None)))
    return phase_fn, amplitude_fn, valid_range_m


def mirror_from_arrays(
    lam_m: ArrayLike,
    values: ArrayLike,
    *,
    mode: str,
    lam0_m: float | None = None,
    taper_m: tuple[float, float, float, float] | None = None,
    fitorder: int = 5,
) -> tuple[PhaseFn, tuple[float, float]]:
    """Build a per-bounce phase function from a custom mirror file's arrays.

    Parameters
    ----------
    lam_m : array_like
        Wavelength samples (m).
    values : array_like
        Either spectral phase (rad, ``mode="phase"``) or GDD (s²,
        ``mode="gdd"``).
    mode : {"phase", "gdd"}
        Interpretation of ``values``. ``"phase"`` is used as-is (after removing
        the overall group delay); ``"gdd"`` is double-integrated first.
    lam0_m : float, optional
        Reference wavelength for group-delay removal (defaults to the
        intensity-unweighted band centre).
    taper_m : tuple, optional
        Planck-taper corners (m). Defaults to a soft 2 %-margin taper just inside
        the data extents.
    fitorder : int
        Degree of the group-delay-removal polynomial.

    Returns
    -------
    (phase_fn, valid_range_m)
    """
    lam_m = np.asarray(lam_m, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(lam_m) & np.isfinite(values)
    lam_m, values = lam_m[finite], values[finite]
    if lam_m.size < 4:
        raise ValueError("need at least 4 valid (wavelength, value) samples")

    lo, hi = float(lam_m.min()), float(lam_m.max())
    if lam0_m is None:
        lam0_m = 0.5 * (lo + hi)
    if taper_m is None:
        margin = 0.02 * (hi - lo)
        taper_m = (lo, lo + margin, hi - margin, hi)

    if mode == "phase":
        # treat the phase as the result of integrating GDD: no integration,
        # just remove the overall group delay and taper.
        omega = wlfreq(lam_m)
        omega_fs = omega * 1e-15
        omega_fs0 = float(wlfreq(lam0_m)) * 1e-15
        phi = _remove_group_delay(values, omega_fs, omega_fs0, fitorder, None)
        order = np.argsort(lam_m)
        lam_sorted, phi_sorted = lam_m[order], phi[order]

        def phase_fn(omega_abs: ArrayLike) -> NDArray[np.float64]:
            lam = wlfreq(np.asarray(omega_abs, dtype=float))
            phi_i = np.interp(lam, lam_sorted, phi_sorted, left=0.0, right=0.0)
            return phi_i * planck_taper(lam, *taper_m)

        return phase_fn, (taper_m[1], taper_m[2])

    if mode == "gdd":
        # Distinct name from the ``def phase_fn`` in the "phase" branch above so the
        # two branches do not share an over-narrow inferred type.
        gdd_phase_fn, _ = process_mirror_data(
            None,
            None,
            lam_m,
            values,
            mode="gdd",
            lam0_m=lam0_m,
            taper_m=taper_m,
            fitorder=fitorder,
        )
        return gdd_phase_fn, (taper_m[1], taper_m[2])

    raise ValueError(f"mode must be 'phase' or 'gdd', got {mode!r}")
