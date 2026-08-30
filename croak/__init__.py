"""croak — complete ultrashort-pulse retrieval.

Reconstructs the full complex electric field of an ultrashort laser pulse —
spectral amplitude *and* phase — from a delay-scanned nonlinear-process spectrum.
The package ships the three FROG geometries (SHG, SD and PG/transient-grating)
in the thin-medium limit, and PG/SD additionally through a dispersive medium,
with the whole workflow around them: loading,
cleaning, retrieval, post-processing, uncertainty, plotting, saving and a GUI.

Ten solvers share one forward model. That model is differentiable end to end in
JAX, which is what makes geometric time smearing, fitted medium thickness and
delay-zero, the B-spline phase basis and Laplace covariance possible; for the core
model there is also a hand-derived Wirtinger adjoint, so :class:`COPRA` and
:class:`LBFGS` run in pure NumPy. The two are cross-checked against each other.

* :class:`COPRA` / :class:`COPRAJax` — Common Pulse Retrieval Algorithm,
* :class:`LBFGS` / :class:`LBFGSHand` — L-BFGS on the analytic adjoint,
* :class:`LBFGSAD` / :class:`OptxLBFGS` — L-BFGS on autodiff gradients,
* :class:`LM` / :class:`OptxLM` — Levenberg–Marquardt on a JAX Jacobian,
* :class:`CMAES` — derivative-free global search,
* :class:`WarmLBFGS` — the default: :class:`LBFGSAD` seeded from a COPRA
  local-projection sweep.

See :data:`croak.retrieve.ALGORITHMS` for the string keys :func:`retrieve` takes.

Quick start
-----------
>>> import numpy as np, croak
>>> g = croak.Grid(128, dt=0.3e-15)
>>> ew = croak.gaussian_pulse(g, 2.5e-15)
>>> delays = np.linspace(-15e-15, 15e-15, 87)
>>> trace = croak.maketrace(g.omega, delays, ew, "shg")
>>> res = croak.retrieve(trace, g.omega, delays, "shg", algorithm="warm-lbfgs")
>>> res.error < 1e-2
True
"""

from __future__ import annotations

# ``session`` is the headless replay subpackage; it builds on the stage modules
# below (preprocess, pipeline, processing, uncertainty, save) but imports them
# directly, so there is no import cycle even though it sorts before them here.
from . import (
    collection,
    constants,
    covariance,
    dispersion,
    focal,
    gases,
    io,
    marginal_checks,
    materials,
    maths,
    metrics,
    plotting,
    preprocess,
    processing,
    progress,
    save,
    session,
    smearing,
    uncertainty,
)
from .cmaes import CMAES
from .copra import COPRA
from .copra_jax import COPRAJax
from .covariance import (
    CovarianceResult,
    covariance_uncertainty,
    parameter_covariance,
)
from .dispersion import apply_dispersion
from .forward import ForwardModel, maketrace
from .forward_jax import make_signal_fn, make_trace_fn, maketrace_jax
from .grid import Grid, gridparams_lambda, gridparams_omega
from .interactions import PG, SD, SHG, get_interaction
from .lbfgs import LBFGS
from .lbfgs_ad import LBFGSAD
from .lbfgs_hand import LBFGSHand
from .lm import LM
from .marginal_checks import (
    MarginalPrediction,
    auto_third_order_exponent,
    predicted_centroid,
    predicted_marginal,
)
from .optimistix_lbfgs import OptxLBFGS
from .optimistix_lm import OptxLM
from .pipeline import initial_guess, retrieve_from_tracedata
from .plotting import (
    plot_frog_filter,
    plot_retrieval,
    plot_simulated_trace,
    plot_thickness_sensitivity,
    plot_uncertainty,
)
from .preprocess import (
    TraceData,
    arpls_baseline,
    defringe_carrier,
    load_and_clean,
)
from .processing import (
    ProcessedResult,
    TruthPulse,
    edge_energy_fraction,
    post_filter,
    process_result,
    resolve_time_direction,
)
from .pulses import (
    ArrayPulse,
    ComplexPulse,
    Pulse,
    gaussian_pulse,
    random_gaussian_spectrum,
    sech_pulse,
)
from .result import RetrievalResult
from .retrieve import ALGORITHMS, retrieve
from .save import save_options, save_result, save_uncertainty
from .session import SessionOptions, SessionResult, retarget, run_session
from .smearing import SmearingKernel, kernel_from_arms, square_boxcars_kernel
from .solver import Retriever
from .uncertainty import (
    CoverageResult,
    NoiseModel,
    UncertaintyResult,
    combine_uncertainties,
    coverage_calibration,
    estimate_fwhm_uncertainty,
    noise_from_background,
    noise_from_residual,
    parametric_bootstrap,
    resampling_bootstrap,
    thickness_bootstrap,
)
from .warm_lbfgs import WarmLBFGS

__all__ = [
    # submodules
    "constants",
    "maths",
    "materials",
    "gases",
    "metrics",
    "io",
    "preprocess",
    "processing",
    "marginal_checks",
    "progress",
    "dispersion",
    "plotting",
    "save",
    "uncertainty",
    "covariance",
    "session",
    "smearing",
    "focal",
    "collection",
    # headless sessions (replay / retarget)
    "SessionOptions",
    "SessionResult",
    "run_session",
    "retarget",
    # data loading & preprocessing
    "load_and_clean",
    "defringe_carrier",
    "arpls_baseline",
    "TraceData",
    # high-level pipeline
    "retrieve_from_tracedata",
    "initial_guess",
    # post-processing & dispersion
    "process_result",
    "ProcessedResult",
    "TruthPulse",
    "resolve_time_direction",
    "edge_energy_fraction",
    "post_filter",
    "apply_dispersion",
    # pre-retrieval marginal consistency checks
    "predicted_marginal",
    "predicted_centroid",
    "auto_third_order_exponent",
    "MarginalPrediction",
    # uncertainty estimation
    "estimate_fwhm_uncertainty",
    "parametric_bootstrap",
    "resampling_bootstrap",
    "thickness_bootstrap",
    "combine_uncertainties",
    "NoiseModel",
    "noise_from_residual",
    "noise_from_background",
    "UncertaintyResult",
    "coverage_calibration",
    "CoverageResult",
    # linearised covariance (analytic uncertainty)
    "covariance_uncertainty",
    "parameter_covariance",
    "CovarianceResult",
    # plotting & saving
    "plot_retrieval",
    "plot_frog_filter",
    "plot_simulated_trace",
    "plot_uncertainty",
    "plot_thickness_sensitivity",
    "save_result",
    "save_uncertainty",
    "save_options",
    # grid & transforms
    "Grid",
    "gridparams_omega",
    "gridparams_lambda",
    # interactions
    "SHG",
    "SD",
    "PG",
    "get_interaction",
    # forward model
    "ForwardModel",
    "maketrace",
    # geometric time smearing
    "SmearingKernel",
    "kernel_from_arms",
    "square_boxcars_kernel",
    # jax/autodiff forward model
    "make_signal_fn",
    "make_trace_fn",
    "maketrace_jax",
    # pulses
    "gaussian_pulse",
    "sech_pulse",
    "random_gaussian_spectrum",
    "Pulse",
    "ArrayPulse",
    "ComplexPulse",
    # retrieval — every solver in ``ALGORITHMS`` is reachable from the top level
    "COPRA",
    "COPRAJax",
    "LBFGS",
    "LBFGSHand",
    "LBFGSAD",
    "OptxLBFGS",
    "LM",
    "OptxLM",
    "CMAES",
    "WarmLBFGS",
    "Retriever",
    "retrieve",
    "ALGORITHMS",
    "RetrievalResult",
]
__version__ = "0.1.0"
