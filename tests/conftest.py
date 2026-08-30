"""Shared fixtures for the phase-2 tests.

Provides helpers that synthesise *experimental-style* FROG data files (a
wavelength axis, a delay/position axis and a 2-D trace) from a known pulse, so
the loading/preprocessing/retrieval pipeline can be exercised end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np
import pytest

from croak.forward import maketrace
from croak.grid import Grid
from croak.maths import wlfreq
from croak.pulses import gaussian_pulse


@dataclass
class SyntheticExperiment:
    """A synthetic experimental FROG dataset and its ground truth."""

    lam_nm: np.ndarray  # wavelength axis (nm, ascending)
    delay_fs: np.ndarray  # delay axis (fs)
    trace: np.ndarray  # (Nlambda, Ndelay) intensity, max 1
    omega0: float  # pulse carrier (rad/s)
    grid: Grid  # grid the pulse was built on
    ew: np.ndarray  # ground-truth complex spectrum (centred)
    interaction: str
    lam_min: float  # suggested retrieval wavelength range (m)
    lam_max: float


def make_synthetic_experiment(
    *,
    n: int = 192,
    dt: float = 0.4e-15,
    fwhm: float = 6e-15,
    phases=(20e-30,),
    lambda0: float = 800e-9,
    ndelay: int = 80,
    delay_max: float = 60e-15,
    interaction: str = "pg",
) -> SyntheticExperiment:
    """Build a synthetic experimental FROG trace from a Gaussian (chirped) pulse.

    Uses a PG-type carrier mapping by default (``omega0_trace = omega0``), so the
    trace wavelengths sit around ``lambda0``.
    """
    g = Grid(n, dt=dt)
    omega0 = wlfreq(lambda0)
    ew = gaussian_pulse(g, fwhm, phases=phases)
    delays = np.linspace(-delay_max, delay_max, ndelay)
    trace = maketrace(g.omega, delays, ew, interaction)  # (Nomega, Ndelay)

    omega_abs = g.omega + omega0
    # Keep only the physically meaningful band: the full FFT grid maps its edges
    # (omega_abs near zero or negative) to nonsensical wavelengths.
    band = (omega_abs > 0.45 * omega0) & (omega_abs < 2.0 * omega0)
    lam = wlfreq(omega_abs[band])
    trace_band = trace[band, :]
    order = np.argsort(lam)
    lam_nm = lam[order] / 1e-9
    trace_exp = trace_band[order, :]

    # Suggested retrieval range: the wavelength marginal down to 1e-4 of its peak
    # (-40 dB), the level the docs recommend. A few-percent cut looks generous on a
    # linear scale but leaves regrid's edge taper chewing into live signal, which
    # both discards data and frees the retrieved field at the band edges.
    marg = trace_exp.sum(axis=1)
    sig = np.where(marg > 1e-4 * marg.max())[0]
    lam_min = lam_nm[sig[0]] * 1e-9
    lam_max = lam_nm[sig[-1]] * 1e-9

    return SyntheticExperiment(
        lam_nm=lam_nm,
        delay_fs=delays / 1e-15,
        trace=trace_exp,
        omega0=omega0,
        grid=g,
        ew=ew,
        interaction=interaction,
        lam_min=lam_min,
        lam_max=lam_max,
    )


def write_experiment_h5(path, exp: SyntheticExperiment) -> str:
    """Write a :class:`SyntheticExperiment` to an HDF5 file and return the path."""
    with h5py.File(path, "w") as f:
        f["wavelength"] = exp.lam_nm
        f["delay"] = exp.delay_fs
        f["trace"] = exp.trace
        # a decoy dataset to exercise shortest-name matching
        f["wavelength_std"] = np.zeros_like(exp.lam_nm)
    return str(path)


@pytest.fixture
def experiment() -> SyntheticExperiment:
    """A default synthetic experimental FROG dataset (PG, chirped Gaussian)."""
    return make_synthetic_experiment()


@pytest.fixture
def experiment_h5(tmp_path, experiment) -> str:
    """Path to a synthetic experimental ``.h5`` file (wavelength/delay/trace)."""
    return write_experiment_h5(tmp_path / "frog.h5", experiment)


# ---------------------------------------------------------------------------
# Numerically simulated FROG scans (Luna ``scansave`` HDF5 layout)
# ---------------------------------------------------------------------------
@dataclass
class SimulatedScanTruth:
    """A synthetic ``scansave``-style scan and its ground truth (ω-density)."""

    omega_abs: np.ndarray  # absolute angular frequency (rad/s, ascending)
    omega0: float  # carrier (rad/s)
    tau: np.ndarray  # delay axis (s, ascending)
    trace_omega: np.ndarray  # (Nomega, Ndelay) ω-density signal intensity
    Iomega: np.ndarray  # (Nomega,) reference spectrum (ω-density)
    Iomega_beamlet: np.ndarray  # (Nomega,) vignetted gate spectrum
    t: np.ndarray  # reference time axis (s)
    It: np.ndarray  # reference temporal intensity
    tau_fwhm: float  # input pulse FWHM (s)
    grid: Grid  # the internal grid
    ew: np.ndarray  # ground-truth complex spectrum (centred)
    interaction: str
    lam_min: float  # suggested retrieval wavelength range (m)
    lam_max: float


#: Group-delay dispersion (s²) of the default simulated fixture pulse. Named so
#: a test can assert the *signed* chirp the loader recovers rather than repeating
#: the literal: a missed conjugation flips this sign while leaving the curve
#: looking perfectly plausible.
SIMULATED_GDD = 15e-30


def make_simulated_scan(
    *,
    n: int = 256,
    dt: float = 0.7e-15,
    fwhm: float = 8e-15,
    phases=(SIMULATED_GDD,),
    lambda0: float = 350e-9,
    ndelay: int = 64,
    delay_max: float = 28e-15,
    interaction: str = "pg",
) -> SimulatedScanTruth:
    """Build a simulated TG-FROG-style scan from a known chirped Gaussian pulse.

    The trace is the raw angular-frequency density ``|psi(omega)|^2`` (what a
    numerical simulation stores), *without* any wavelength Jacobian — exactly the
    convention :func:`croak.io.read_simulated_scan` expects. The default sampling
    (``dt`` coarse enough to keep the whole grid at positive absolute frequency,
    fine enough to resolve the pulse) keeps the signal band well-sampled so the
    load → regrid → retrieve round-trip is faithful.
    """
    g = Grid(n, dt=dt)
    omega0 = wlfreq(lambda0)
    ew = gaussian_pulse(g, fwhm, phases=phases)
    delays = np.linspace(-delay_max, delay_max, ndelay)
    trace = maketrace(g.omega, delays, ew, interaction)  # (Nomega, Ndelay), ω-density
    omega_abs = g.omega + omega0
    Iomega = np.abs(ew) ** 2
    # a smooth chromatic vignette (off-centre Gaussian) for the mask correction
    vignette = np.exp(-0.5 * ((omega_abs - 1.05 * omega0) / (0.25 * omega0)) ** 2)
    beamlet = Iomega * np.clip(vignette, 1e-3, None)
    Et = g.ifft(ew)
    It = np.abs(Et) ** 2

    # Suggested band from the marginal at 1e-4 of its peak, as the docs advise: a
    # few-percent cut leaves regrid's edge taper cutting live signal.
    marg = trace.sum(axis=1)
    band = (omega_abs > 0) & (marg > 1e-4 * marg.max())
    lam_band = wlfreq(omega_abs[band])
    return SimulatedScanTruth(
        omega_abs=omega_abs,
        omega0=omega0,
        tau=delays,
        trace_omega=trace,
        Iomega=Iomega,
        Iomega_beamlet=beamlet,
        t=g.t,
        It=It,
        tau_fwhm=fwhm,
        grid=g,
        ew=ew,
        interaction=interaction,
        lam_min=float(lam_band.min()),
        lam_max=float(lam_band.max()),
    )


def write_simulated_h5(
    path,
    truth: SimulatedScanTruth,
    *,
    nz: int = 2,
    zsave: np.ndarray | None = None,
    delay_convention: str | None = None,
    store_complex: bool = False,
    geometry: tuple[float, float, str] | None = None,
    mask_window: dict[str, object] | None = None,
) -> str:
    """Write a :class:`SimulatedScanTruth` as a ``scansave`` HDF5 file.

    Axes and the trace window are stored in FFT/bin order (unshifted), as the
    real Luna output is, so :func:`croak.io.read_simulated_scan` must sort them
    back onto an ascending ω axis. The genuine trace sits in the **last** ``nz``
    propagation slice; the others carry a decoy so slice selection matters.

    Passing ``geometry`` as ``(mask_diam, mask_spacing, geometry_name)`` writes
    the instrument metadata newer simulator versions record; omitting it writes
    the older, geometry-free layout.

    Passing ``mask_window`` writes the flattened ``/grid/window_def_*`` collection
    record plus the ``kx`` and ``referenceλ`` needed to resolve a ``"default"``
    apodisation width. Keys are the simulator's own: ``type``, ``holex``, ``holey``,
    ``holediam``, ``zmask``, ``apod``, ``apod_param`` (a number, or the string
    ``"default"``), and ``delta_k`` / ``reference_wavelength`` for the grid record.

    Passing ``zsave`` (the propagation distances [m], strictly increasing with the
    exit at the end) writes the multi-thickness ``/grid/zsave`` axis and sets
    ``nz = len(zsave)``. Each non-exit slice then gets a *distinct spectral tilt*
    (not a global scale, which would cancel under the loader's normalisation), so
    selecting different thicknesses yields genuinely different traces.
    """
    n_tau = truth.tau.size
    n_omega = truth.omega_abs.size
    # FFT/bin order (the centred axes shifted back to DFT-bin layout)
    omega_store = np.fft.ifftshift(truth.omega_abs)
    trace_store = np.fft.ifftshift(truth.trace_omega, axes=0)  # (Nomega, Ntau)
    Iomega_store = np.fft.ifftshift(truth.Iomega)
    beamlet_store = np.fft.ifftshift(truth.Iomega_beamlet)

    if zsave is not None:
        zsave = np.asarray(zsave, dtype=float)
        nz = zsave.size

    # h5py reads what Julia wrote as (Nomega, nz, Ntau) reversed -> (Ntau, nz, Nomega)
    window = np.zeros((n_tau, nz, n_omega))
    for z in range(nz - 1):
        if zsave is None:
            window[:, z, :] = 0.5 * trace_store.T + 0.01  # decoy slices
        else:
            # a per-slice spectral tilt (=1 at the exit) so thicknesses differ in
            # shape and survive the loader's per-trace max normalisation
            tilt = np.linspace(1.0, 1.0 + 0.3 * (z + 1), n_omega)
            window[:, z, :] = trace_store.T * tilt
    window[:, nz - 1, :] = trace_store.T  # the genuine trace at the exit slice

    with h5py.File(path, "w") as f:
        gg = f.create_group("grid")
        gg["ω"] = omega_store
        gg["ω0"] = truth.omega0
        gg["Iω"] = Iomega_store
        gg["Iω_beamlet"] = beamlet_store
        gg["t"] = truth.t
        gg["It"] = truth.It
        gg["τfwhm"] = truth.tau_fwhm
        if store_complex:
            # ModelPNPS stores E(ω) in two conventions croak has to undo before
            # the field means what it looks like (see
            # croak.session.pipeline._simulated_truth_spectrum): the field is
            # centred in its own time window --- a (-1)**n alternation on the
            # ascending-ω axis --- and it uses the opposite Fourier sign to
            # croak's e^{+iωt}, so it is the conjugate. Write the fixture that
            # way round, so the loader's two corrections invert exactly back to
            # ``truth.ew``; storing croak's own convention here would make the
            # corrections corrupt the field and hide a real regression.
            alternate = (-1.0) ** np.arange(n_omega)
            e_src_ascending = np.conj(truth.ew) * alternate
            # complex beamlet: sqrt of the vignetted intensity carrying the same
            # (already conjugated and alternated) spectral phase
            e_beam_ascending = np.sqrt(truth.Iomega_beamlet) * np.exp(
                1j * np.angle(e_src_ascending)
            )
            e_beam = np.fft.ifftshift(e_beam_ascending)
            gg["Eω_beamlet_re"] = e_beam.real
            gg["Eω_beamlet_im"] = e_beam.imag
            e_src = np.fft.ifftshift(e_src_ascending)
            gg["Eω_re"] = e_src.real
            gg["Eω_im"] = e_src.imag
        if delay_convention is not None:
            gg["delay_convention"] = delay_convention
        if geometry is not None:
            gg["mask_diam"], gg["mask_spacing"], gg["geometry"] = geometry
        if mask_window is not None:
            for key in (
                "type",
                "holex",
                "holey",
                "holediam",
                "zmask",
                "apod",
                "apod_param",
            ):
                gg[f"window_def_{key}"] = mask_window[key]
            n_kx = 8
            gg["kx"] = np.arange(n_kx) * float(mask_window["delta_k"])
            gg["referenceλ"] = float(mask_window["reference_wavelength"])
        if zsave is not None:
            gg["zsave"] = zsave
        sv = f.create_group("scanvariables")
        sv["τ"] = truth.tau
        f["Iω_win"] = window
        f["Iω_win_reimaged"] = 0.7 * window  # a second window to choose from
        if zsave is not None:
            # the newer multi-thickness format also stores the full signal-beam
            # collection (no aperture crop), so Iω_win <= Iω_full
            f["Iω_full"] = window / 0.6
    return str(path)


@pytest.fixture
def simulated_truth() -> SimulatedScanTruth:
    """A default simulated TG-FROG scan (PG, chirped Gaussian)."""
    return make_simulated_scan()


@pytest.fixture
def simulated_scan_h5(tmp_path, simulated_truth) -> str:
    """Path to a synthetic ``scansave``-style ``.h5`` file (legacy, no ``zsave``)."""
    return write_simulated_h5(tmp_path / "scan.h5", simulated_truth)


#: Propagation distances [m] of the multi-thickness fixture (entrance → exit).
SIMULATED_ZSAVE = np.array([0.0, 5e-6, 10e-6, 20e-6, 40e-6])


@pytest.fixture
def simulated_scan_multi_h5(tmp_path, simulated_truth) -> str:
    """Path to a multi-thickness ``scansave`` file carrying ``/grid/zsave``.

    Five slices at :data:`SIMULATED_ZSAVE`; the 40 µm exit slice is the genuine
    trace and the thinner ones differ in spectral shape, so thickness selection
    is observable.
    """
    return write_simulated_h5(
        tmp_path / "scan_multi.h5", simulated_truth, zsave=SIMULATED_ZSAVE
    )
