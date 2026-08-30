"""High-level retrieval from cleaned :class:`~croak.preprocess.TraceData`.

One call that takes a cleaned trace all the way to a retrieved field: it builds
a physically sensible initial guess from the trace (and any independent
spectrum), selects the solver and pulse mode, runs the retrieval, and returns a
:class:`~croak.result.RetrievalResult` carrying the carrier frequency.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid

from .focal import FocalMixture
from .maths import gauss
from .preprocess import TraceData
from .processing import TruthPulse
from .result import RetrievalResult
from .retrieve import ALGORITHMS, algorithm_params
from .smearing import SmearingKernel

__all__ = ["retrieve_from_tracedata", "initial_guess"]

Complex = NDArray[np.complex128]


def _amplitude(td: TraceData) -> NDArray[np.float64]:
    """Spectral amplitude guess: from the independent spectrum or trace marginal."""
    if td.Iomega is not None:
        amp = np.sqrt(np.clip(td.Iomega, 0.0, None))
    else:
        marg = td.trace.sum(axis=1)
        amp = np.sqrt(np.clip(marg, 0.0, None))
    peak = amp.max()
    return amp / peak if peak > 0 else amp


def _centroid_phase(td: TraceData) -> NDArray[np.float64]:
    r"""Build the initial spectral phase from the per-frequency centroid delay.

    For each frequency the mean delay :math:`\langle\tau\rangle(\omega)` is the
    group delay; integrating it over :math:`\omega` gives a spectral phase that
    captures the trace's delay structure (the COPRA/LBFGS-friendly seed).
    """
    omega = td.omega
    tau = td.delays
    denom = np.trapezoid(td.trace, tau, axis=1)
    num = np.trapezoid(td.trace * tau[None, :], tau, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        group_delay = np.where(denom > 0, num / np.where(denom > 0, denom, 1.0), 0.0)
    return cumulative_trapezoid(group_delay, omega, initial=0.0)


def initial_guess(
    td: TraceData,
    *,
    mode: str = "auto",
    perfect_fwhm: float = 10e-15,
    truth: TruthPulse | None = None,
    rng: np.random.Generator | None = None,
) -> Complex:
    """Build an initial complex spectrum from a cleaned trace.

    Parameters
    ----------
    td : TraceData
        Cleaned trace.
    mode : {"auto", "tl", "perfect", "random", "truth"}
        The first four start from a spectral amplitude and differ in the phase, which
        is what selects the basin the optimizer falls into:

        ``"tl"``
            the measured amplitude with **flat** phase --- the transform limit
            of the spectrum actually measured. The natural experimental guess:
            it uses the real amplitude and assumes nothing about the duration.
        ``"perfect"``
            a synthetic transform-limited Gaussian of ``perfect_fwhm`` at the
            spectral peak. Note this *discards* the measured amplitude and
            encodes an assumed duration; ``"tl"`` is the better-justified
            version of the same flat-phase start and performs equivalently.
        ``"auto"``
            the measured amplitude with a centroid-delay phase, which already
            encodes the group-delay slope and so starts a *chirped* pulse close
            to its answer.
        ``"random"``
            the measured amplitude with a uniform random phase.
        ``"truth"``
            the known complex spectrum carried by ``truth``, aligned to the
            retrieval grid. ModelPNPS native-grid data are copied exactly;
            regridded data interpolate amplitude and unwrapped phase.

        The flat-phase starts (``"tl"``, ``"perfect"``) suit near-transform-
        limited pulses; a strongly chirped pulse started flat must build its
        quadratic phase from nothing and tends to retain temporal structure,
        where ``"auto"`` does not.
    perfect_fwhm : float, optional
        Temporal FWHM for the ``"perfect"`` seed (s).
    truth : TruthPulse, optional
        Known pulse carrying a complex spectrum; required for ``mode="truth"``.
    rng : numpy.random.Generator, optional
        Generator for ``"random"``.

    Returns
    -------
    numpy.ndarray
        Complex spectrum (centred order).
    """
    omega = td.omega
    if mode == "truth":
        if truth is None:
            raise ValueError("guess mode 'truth' requires a known TruthPulse")
        return truth.spectrum_on_grid(td.grid, td.omega0_pulse)
    if mode == "perfect":
        amp = _amplitude(td)
        omega_peak = omega[int(np.argmax(amp))]
        # time-bandwidth product 0.44 for a Gaussian: spectral FWHM in rad/s
        omega_fwhm = 2 * np.pi * 0.44 / perfect_fwhm
        Iw = gauss(omega, fwhm=omega_fwhm, x0=omega_peak)
        return np.sqrt(Iw).astype(complex)
    amp = _amplitude(td)
    if mode == "tl":
        return amp.astype(complex)
    if mode == "random":
        rng = np.random.default_rng() if rng is None else rng
        phase = 2 * np.pi * rng.random(omega.size)
    elif mode == "auto":
        phase = _centroid_phase(td)
    else:
        raise ValueError(f"unknown guess mode {mode!r}")
    return (amp * np.exp(1j * phase)).astype(complex)


def retrieve_from_tracedata(
    td: TraceData,
    *,
    algorithm: str = "warm-lbfgs",
    full: bool = True,
    R_omega: bool = False,
    guess="auto",
    perfect_fwhm: float = 10e-15,
    truth: TruthPulse | None = None,
    material: str | None = None,
    thickness: float = 0.0,
    npoints: int = 20,
    smearing: SmearingKernel | None = None,
    focal: FocalMixture | None = None,
    depth_weight=None,
    reg_amp: float = 0.0,
    reg_phase: float = 0.0,
    reg_spectrum: float = 0.0,
    reg_time: float = 0.0,
    time_window: tuple[float, float] | None = None,
    tau0: float = 0.0,
    smear_scale: float = 1.0,
    smear_scale_delta: float = 1.0,
    fit_thickness: bool = False,
    fit_tau0: bool = False,
    fit_smearing: bool = False,
    fit_smearing_split: bool = False,
    polish: bool = False,
    phase_basis: str = "pointwise",
    n_nodes: int = 20,
    strategy: str = "cma",
    popsize: int = 0,
    std_init: float = 0.5,
    seed: int | None = None,
    maxiters: int = 100,
    alpha: float = 0.25,
    stall_patience: int = 5,
    reltol: float = 1e-4,
    abstol: float = 1e-8,
    rng: np.random.Generator | None = None,
    callback=None,
    verbose: bool = False,
) -> RetrievalResult:
    """Retrieve a pulse from a cleaned :class:`~croak.preprocess.TraceData`.

    Parameters
    ----------
    td : TraceData
        Cleaned, regridded trace.
    algorithm : {"warm-lbfgs", "copra", "copra-jax", "lbfgs", "lbfgs-hand", \
"lbfgs-ad", "lbfgs-optx", "lm", "lm-optx", "cma-es"}
        Retrieval algorithm — any key in :data:`croak.retrieve.ALGORITHMS`.
        ``"warm-lbfgs"`` (the default) is ``"lbfgs-ad"`` warm-started from
        COPRA's local sweep;
        ``"copra"`` and ``"lbfgs"`` use hand-derived analytic gradients (pure
        NumPy); ``"copra-jax"`` and ``"lbfgs-hand"`` run the same hand-derived
        gradients jitted in JAX (no AD); ``"lbfgs-ad"`` and ``"lm"`` use JAX
        automatic differentiation. The ``"-optx"`` variants are JAX-native
        (Optimistix) twins of ``"lbfgs-ad"`` and ``"lm"`` that run the whole
        optimisation loop on-device, and ``"cma-es"`` is the derivative-free
        evolutionary strategy (evosax).
    full : bool, optional
        Retrieve amplitude **and** phase (``True``) or phase only (``False``,
        L-BFGS with the guess amplitude held fixed).
    R_omega : bool, optional
        Per-frequency adaptive scaling (supported by every algorithm).
    guess : str, array_like or Pulse, optional
        ``"auto"``/``"perfect"``/``"random"``/``"truth"`` (see
        :func:`initial_guess`) or an explicit spectrum/pulse passed straight
        through.
    perfect_fwhm : float, optional
        FWHM (s) for the ``"perfect"`` guess.
    truth : TruthPulse, optional
        Known complex pulse used when ``guess="truth"``.
    material, thickness, npoints
        Dispersive-slab parameters (dispersive propagation through a material).
    smearing : SmearingKernel or None, optional
        Geometric time-smearing kernel of a non-collinear BOXCARS geometry
        (``None`` = off). Only the autodiff solvers can model it; asking for it
        with any other algorithm raises rather than silently dropping it. See
        :mod:`croak.smearing` and :doc:`/howto/geometric_smearing`.
    focal : FocalMixture or None, optional
        Explicit chromatic focal-plane mixture, optionally carrying a
        collection aperture (``None`` = off). Replaces the reduced smearing
        kernel with the node-resolved mixture of :mod:`croak.focal`; like
        ``smearing`` it is a physical model term, so requesting it with a
        solver that cannot represent it raises rather than silently dropping
        it. See :doc:`/howto/collection_aperture`.
    reg_amp, reg_phase : float, optional
        L-BFGS/LM smoothness regularisation weights.
    reg_spectrum : float, optional
        Spectral-match regularisation weight (LBFGS/LBFGS-AD/LM, full mode):
        pulls the retrieved spectral amplitude towards the measured independent
        spectrum ``td.Iomega``. Ignored (with a warning) when ``reg_spectrum>0``
        but no independent spectrum is available.
    reg_time : float, optional
        Temporal (pedestal) regularisation weight (LBFGS-AD / cma-es): penalises
        the fraction of ``E(t)`` energy outside ``time_window``. A self-consistent
        alternative to the temporal post-filter — it keeps spectrum and field a
        Fourier pair, so introduces no spectral artefacts. Active in all modes.
    time_window : tuple of float, optional
        ``(lo, hi)`` window in seconds for ``reg_time``; defaults to the
        measurement delay range ``(td.delays.min(), td.delays.max())``.
    tau0, smear_scale : float, optional
        Starting centres for the two fitted extras that have no nominal argument
        of their own: the delay-zero offset (s, default ``0``) and the multiplier
        on the smearing-kernel widths (default ``1.0`` = the kernel as supplied).
        Pass the previous fit's values to **warm start** a re-run, as the GUI's
        "reuse previous result" does; the slab thickness is warm-started through
        ``thickness`` itself. Each is held at its centre when the matching
        ``fit_*`` flag is off.
    fit_thickness, fit_tau0 : bool, optional
        Fit the dispersive-slab thickness and/or a delay-zero offset as extra
        parameters alongside the pulse (``lbfgs-ad`` / ``lm`` only;
        ``fit_thickness`` needs a dispersive slab, PG/SD). See
        :doc:`/howto/fitting_thickness_tau0`.
    fit_smearing : bool, optional
        Fit the geometric-smearing strength (one dimensionless multiplier on the
        kernel widths) alongside the pulse. Requires ``smearing``. See
        :doc:`/howto/geometric_smearing`.
    fit_smearing_split : bool, optional
        Fit the gate-shape (``p``) and delay (``delta``) kernel widths as two
        independent multipliers instead of one joint one. Requires
        ``fit_smearing``; ``smear_scale_delta`` (default ``1.0``) is the delay
        channel's starting centre. See :doc:`/howto/geometric_smearing`.
    depth_weight : array_like or None, optional
        Complex generation-envelope weights, one per depth-quadrature node
        (``lbfgs-ad`` only): the transverse-geometry factor a 1D depth integral
        cannot see. See :func:`croak.forward_jax.make_param_trace_fn`.
    polish : bool, optional
        Two-phase polish for the extra parameters: retrieve the pulse with them
        fixed, then free them for a joint final phase.
    maxiters : int, optional
        Maximum iterations.
    reltol, abstol : float, optional
        Convergence tolerances forwarded to every solver (defaults ``1e-4`` /
        ``1e-8``). For the gradient/LM solvers these are the optimiser function
        tolerances; for COPRA they enable relative-plateau / absolute-target
        early stopping (off by default in the bare ``COPRA`` class).
    rng : numpy.random.Generator, optional
        Generator (random guess / COPRA permutation).
    callback : callable, optional
        ``callback(iteration, R, best_R, snapshot=...)`` progress hook; the
        optional ``snapshot`` is a zero-argument in-progress-result builder (or
        ``None``) for a live full-plot preview.
    verbose : bool, optional
        Print per-iteration progress.

    Returns
    -------
    RetrievalResult
    """
    if isinstance(guess, str):
        ew0 = initial_guess(
            td, mode=guess, perfect_fwhm=perfect_fwhm, truth=truth, rng=rng
        )
    else:
        ew0 = guess  # array or Pulse, passed through

    omega0 = td.omega0_pulse
    # Spectral-match target: the measured independent spectrum on the grid. Only
    # meaningful in full mode and when a spectrum was actually provided.
    spectrum_target = td.Iomega
    if reg_spectrum > 0 and (spectrum_target is None or not full):
        if spectrum_target is None:
            warnings.warn(
                "reg_spectrum > 0 but no independent spectrum is available "
                "(td.Iomega is None); spectral-match regularisation disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
        reg_spectrum = 0.0
    algo = algorithm.lower()
    if algo not in ALGORITHMS:
        raise ValueError(f"unknown algorithm {algorithm!r}")
    # One source of truth: build every candidate option, then forward only the
    # ones this solver's constructor accepts (e.g. COPRA has no reg_* / spectrum
    # target). Adding a new algorithm to ALGORITHMS makes it work here and in the
    # GUI with no per-solver branch to maintain.
    candidate = {
        "maxiters": maxiters,
        "reltol": reltol,
        "abstol": abstol,
        "material": material,
        "thickness": thickness,
        "npoints": npoints,
        "smearing": smearing,
        "focal": focal,
        "depth_weight": depth_weight,
        "omega0": omega0,
        "phase_only": not full,
        "phase_basis": phase_basis,
        "n_nodes": n_nodes,
        "R_omega": R_omega,
        "reg_amp": reg_amp,
        "reg_phase": reg_phase,
        "reg_spectrum": reg_spectrum,
        "spectrum_target": spectrum_target,
        "reg_time": reg_time,
        "time_window": time_window,
        "tau0": tau0,
        "smear_scale": smear_scale,
        "smear_scale_delta": smear_scale_delta,
        "fit_thickness": fit_thickness,
        "fit_tau0": fit_tau0,
        "fit_smearing": fit_smearing,
        "fit_smearing_split": fit_smearing_split,
        "polish": polish,
        # COPRA-only; filtered out for every other solver by `accepted` below.
        "alpha": alpha,
        "stall_patience": stall_patience,
        "strategy": strategy,
        "popsize": popsize,
        "std_init": std_init,
        "seed": seed,
        "verbose": verbose,
    }
    accepted = algorithm_params(algo)
    # Unsupported options are normally dropped silently, but a smearing kernel is a
    # physical model term: quietly ignoring it would return a confidently wrong pulse.
    # COPRA's magnitude-replacement projection and the hand-written adjoints cannot
    # represent the incoherent sum over quadrature nodes, so say so instead.
    if smearing is not None and "smearing" not in accepted:
        supported = sorted(n for n in ALGORITHMS if "smearing" in algorithm_params(n))
        raise ValueError(
            f"algorithm {algo!r} cannot model geometrical smearing; use one of "
            f"{', '.join(supported)}"
        )
    if focal is not None and "focal" not in accepted:
        supported = sorted(n for n in ALGORITHMS if "focal" in algorithm_params(n))
        raise ValueError(
            f"algorithm {algo!r} cannot model a chromatic focal mixture; use one "
            f"of {', '.join(supported)}"
        )
    if depth_weight is not None and "depth_weight" not in accepted:
        supported = sorted(
            n for n in ALGORITHMS if "depth_weight" in algorithm_params(n)
        )
        raise ValueError(
            f"algorithm {algo!r} cannot model a generation envelope; use one of "
            f"{', '.join(supported)}"
        )
    kwargs = {k: v for k, v in candidate.items() if k in accepted}
    solver = ALGORITHMS[algo](**kwargs)

    return solver.run(
        td.trace,
        td.omega,
        td.delays,
        td.interaction,
        guess=ew0,
        omega0=omega0,
        rng=rng,
        callback=callback,
    )
