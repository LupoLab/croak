r"""Global retrieval by evolution strategy (``cma-es``), powered by evosax.

Every other retriever (except COPRA) is a *local* optimiser, so a poor initial
guess never converges. This retriever runs a **global** population search —
CMA-ES (default), Separable CMA-ES or Differential Evolution — over the same
differentiable parameterisation the gradient solvers use (full amplitude+phase,
phase-only, or the reduced cubic B-spline phase basis). Because the forward model
is JAX, the whole population is evaluated in a single jitted ``vmap`` device call,
so a generation costs roughly one batched forward pass.

The intended workflow is a global search here followed by a local refinement:
run ``cma-es`` (typically with the low-dimensional ``phase_basis="bspline"``),
then feed its result as the guess to ``lbfgs-ad`` (the GUI's "Reuse previous
result" initial guess). CMA-ES suits the ~20–30-parameter spline basis; Separable
CMA-ES / DE scale better if used on the full-dimensional parameterisation.

The same smoothness / spectral-match regularisation as the gradient solvers is
available (``reg_amp``/``reg_phase``/``reg_spectrum``): the penalties are added
to the FROG error to form the *total* the evolution strategy minimises, while
selection and reporting stay on the bare FROG error so results remain comparable
to the other retrievers. The penalties self-gate by mode/basis, so on the common
B-spline path only the phase penalty is active (amplitude/spectral are no-ops).

evosax is an optional dependency — ``pip install croak[evo]``.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from . import metrics_jax as mj
from ._jax_pulse import (
    ExtraParamSpec,
    augment,
    build_parameterisation,
    time_window_mask,
)
from .focal import FocalMixture
from .forward_jax import make_param_trace_fn
from .result import RetrievalResult
from .smearing import SmearingKernel
from .solver import (
    Retriever,
    assemble_result,
    spectral_target_amplitude,
    split_smear_value,
)

__all__ = ["CMAES", "run_cmaes"]

Complex = NDArray[np.complex128]

#: ``strategy`` name -> evosax class path. CMA/Sep-CMA are distribution-based
#: (seeded by a mean); DE is population-based (seeded by an initial population).
_STRATEGIES = {
    "cma": ("CMA_ES", "distribution"),
    "sep-cma": ("Sep_CMA_ES", "distribution"),
    "de": ("DifferentialEvolution", "population"),
}


def _import_strategy(strategy: str):
    """Return ``(evosax_class, family)`` for ``strategy`` (lazy evosax import)."""
    if strategy not in _STRATEGIES:
        raise ValueError(
            f"unknown strategy {strategy!r}; choose from {sorted(_STRATEGIES)}"
        )
    cls_name, family = _STRATEGIES[strategy]
    try:
        # Optional 'evo' extra, mirroring croak.refractive_db's handling of the
        # 'ridb' extra: the import is deliberately unresolvable in a base install.
        from evosax import (  # pyright: ignore[reportMissingImports]
            algorithms as evo_algos,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "the 'cma-es' retriever needs evosax — install with: pip install croak[evo]"
        ) from exc
    return getattr(evo_algos, cls_name), family


def _default_popsize(n_params: int) -> int:
    """evosax/CMA default population ``4 + floor(3 ln N)`` (min 4)."""
    return max(4, 4 + int(3 * np.log(max(n_params, 1))))


def run_cmaes(
    objective,
    u0: NDArray[np.float64],
    *,
    strategy: str = "cma",
    popsize: int = 0,
    std_init: float = 0.5,
    generations: int = 200,
    key,
    callback=None,
    make_snapshot: Callable[[NDArray[np.float64]], object] | None = None,
    err_log: list[float] | None = None,
    verbose: bool = False,
) -> tuple[NDArray[np.float64], list[float]]:
    """Minimise ``objective(u)`` with an evosax strategy; return ``(best_u, R_log)``.

    ``objective`` must be a **batched** map ``(P, N) -> ((P,), (P,))`` (i.e.
    already ``jit(vmap(...))``) returning, per population member, the
    ``(total, frog_error)`` pair: the *total* is the regularised objective the
    evolution strategy is driven by (``ask``/``tell``), while the bare
    *frog_error* is what is selected and reported on. The returned ``best_u`` is
    the member with the lowest **total** ever evaluated (consistent with the
    gradient solvers, which return the minimiser of the regularised objective),
    and ``R_log`` is that member's bare FROG error after each generation.

    ``callback`` is called as ``callback(generation, gen_R, best_R, snapshot=...)``
    where ``snapshot`` is a zero-argument builder of the current best
    :class:`~croak.result.RetrievalResult` (for the GUI live preview) when
    ``make_snapshot(best_u) -> RetrievalResult`` is supplied, else ``None``. Pass
    ``err_log`` to accumulate into an existing list a snapshot builder can read.
    """
    Strategy, family = _import_strategy(strategy)
    n = int(u0.size)
    P = popsize if popsize and popsize > 0 else _default_popsize(n)
    u0_j = jnp.asarray(u0, dtype=float)
    es = Strategy(population_size=P, solution=u0_j)
    params = es.default_params
    if hasattr(params, "std_init"):
        params = dataclasses.replace(params, std_init=float(std_init))

    key, key_init = jax.random.split(key)
    if family == "population":
        # Population-based (DE): seed an initial population around u0; the ES is
        # seeded with the regularised total (objective(...)[0]).
        pop0 = u0_j[None, :] + std_init * jax.random.normal(key_init, (P, n))
        state = es.init(key_init, pop0, objective(pop0)[0], params)
    else:
        state = es.init(key_init, u0_j, params)

    # Track the all-time best by the *total* (what the ES minimises), keeping its
    # paired bare FROG error — so selection follows the optimised objective while
    # reporting stays on the comparable bare error. Tracking the per-generation
    # argmin is equivalent to evosax's own ``state.best_solution`` (every member
    # is told), and pairs the ferr at no extra forward pass.
    best_total = np.inf
    best_u = np.asarray(u0, dtype=float)
    best_R = np.inf
    err_log = [] if err_log is None else err_log
    for gen in range(generations):
        key, key_ask, key_tell = jax.random.split(key, 3)
        population, state = es.ask(key_ask, state, params)
        total, ferr = objective(population)
        state, _ = es.tell(key_tell, population, total, state, params)
        j = int(jnp.argmin(total))
        gen_best_R = float(ferr[j])  # bare R of this generation's best-by-total
        if float(total[j]) < best_total:
            best_total = float(total[j])
            best_u = np.asarray(population[j], dtype=float)
            best_R = gen_best_R
        err_log.append(best_R)
        if verbose:
            print(f"  gen {gen + 1:4d}: best R = {best_R:.6e}")
        if callback is not None:
            # Snapshot the all-time best member for the live preview.
            snapshot = (
                functools.partial(make_snapshot, best_u)
                if make_snapshot is not None
                else None
            )
            callback(gen + 1, gen_best_R, best_R, snapshot=snapshot)

    return best_u, err_log


class CMAES(Retriever):
    r"""Global evolution-strategy retriever (``cma-es``); see the module docstring.

    Parameters
    ----------
    maxiters : int
        Number of generations.
    strategy : {"cma", "sep-cma", "de"}
        Evolution strategy (CMA-ES, Separable CMA-ES, Differential Evolution).
    popsize : int
        Population size (``0`` → evosax default ``4 + ⌊3 ln N⌋``).
    std_init : float
        Initial search std (CMA mean spread / DE population scatter).
    phase_only, phase_basis, n_nodes, R_omega
        Mode options shared with the gradient solvers (``phase_basis="bspline"``
        gives the reduced-dimensional phase search).
    material, thickness, npoints, omega0, quadrature
        Forward-model options shared with the gradient solvers.
    smearing : SmearingKernel or None, optional
        Geometric time-smearing kernel of a non-collinear BOXCARS geometry
        (default ``None`` = off). PG (TG) and SD only. See :mod:`croak.smearing`
        and :doc:`/howto/geometric_smearing`.
    reg_amp, reg_phase, reg_spectrum, spectrum_target
        Smoothness / spectral-match regularisation weights, identical to the
        gradient solvers (see :class:`croak.lbfgs_ad.LBFGSAD`). The penalties are
        added to the FROG error to form the *total* the search minimises, but the
        result is still selected and reported on the bare FROG error. The
        penalties self-gate by mode/basis, so amplitude/spectral terms are no-ops
        in phase-only or B-spline mode; ``reg_spectrum`` additionally needs an
        independent ``spectrum_target`` (full mode).
    reg_time, time_window
        Temporal (pedestal) penalty weight and ``(lo, hi)`` time window in
        **seconds**. Penalises the fraction of ``E(t) = ifft(spectrum)`` energy
        falling outside the window — a self-consistent, mode-agnostic alternative
        to the post-retrieval temporal filter (it keeps spectrum and field an
        exact Fourier pair, so it introduces no spectral artefacts). Active in
        all modes including B-spline. ``time_window=None`` defaults to the
        measurement delay range.
    tau0, smear_scale : float, optional
        Search centres for the delay-zero offset (s, default ``0``) and the
        smearing-width multiplier (default ``1.0`` = the kernel as supplied), e.g.
        the previous fit's values when warm-starting. Each is held at its centre
        when the matching ``fit_*`` flag is off.
    focal : FocalMixture or None, optional
        Chromatic focal mixture (:mod:`croak.focal`) in place of ``smearing``:
        the focal average is carried explicitly, each node applying its own
        wavelength-dependent amplitude filter, where the reduced kernel
        evaluates a single :math:`|A|^6` weight at the carrier. Mutually
        exclusive with ``smearing``, and incompatible with ``fit_smearing``
        (an explicit mixture has no kernel width to scale).

        Only worth its cost for a broadband pulse. The reduced kernel's widths
        scale as :math:`\sigma\propto\lambda`, so freezing them at the carrier
        is a -16%/+24% spread of width across a 1 fs deep-UV band and costs a
        mean 12% of retrieved duration there, against 3.5% at 2 fs. The mixture
        needs one field build per focal node rather than per Gauss-Hermite
        node, so it is one to two orders of magnitude slower.
    seed : int or None
        Optional explicit PRNG seed (otherwise derived from the framework rng).

    Notes
    -----
    The extra parameters are appended to the search vector as dimensionless,
    O(1)-scaled offsets (:func:`croak._jax_pulse.augment`) so the isotropic
    ``std_init`` spread treats them sensibly. The search is **unconstrained**, so
    — as for ``lm-optx`` — the thickness lower bound is not enforced; it relies on
    the prior as the search centre. There is no ``polish`` option (the global
    search frees the extras from the start); fit the thickness here only with a
    decent prior, or refine a pulse-only ``cma-es`` result with ``lbfgs-ad``
    (``polish=True``) as in the usual global-then-local workflow.
    """

    name = "cma-es"

    def __init__(
        self,
        *,
        maxiters: int = 200,
        strategy: str = "cma",
        popsize: int = 0,
        std_init: float = 0.5,
        material: str | None = None,
        thickness: float = 0.0,
        npoints: int = 20,
        omega0: float = 0.0,
        quadrature: str = "gausslegendre",
        smearing: SmearingKernel | None = None,
        focal: FocalMixture | None = None,
        phase_only: bool = False,
        phase_basis: str = "pointwise",
        n_nodes: int = 20,
        R_omega: bool = False,
        reg_amp: float = 0.0,
        reg_phase: float = 0.0,
        reg_spectrum: float = 0.0,
        spectrum_target=None,
        reg_time: float = 0.0,
        time_window: tuple[float, float] | None = None,
        tau0: float = 0.0,
        smear_scale: float = 1.0,
        smear_scale_delta: float = 1.0,
        fit_thickness: bool = False,
        fit_tau0: bool = False,
        fit_smearing: bool = False,
        fit_smearing_split: bool = False,
        seed: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(maxiters=maxiters, verbose=verbose)
        self.strategy = str(strategy)
        self.popsize = int(popsize)
        self.std_init = float(std_init)
        self.material = material
        self.thickness = float(thickness)
        self.npoints = int(npoints)
        self.omega0 = float(omega0)
        self.quadrature = quadrature
        self.smearing = smearing
        self.focal = focal
        self.phase_only = bool(phase_only)
        self.phase_basis = str(phase_basis)
        self.n_nodes = int(n_nodes)
        self.R_omega = bool(R_omega)
        self.reg_amp = float(reg_amp)
        self.reg_phase = float(reg_phase)
        self.reg_spectrum = float(reg_spectrum)
        self._spectral_target = spectral_target_amplitude(spectrum_target)
        self.reg_time = float(reg_time)
        self.time_window = time_window
        self.tau0 = float(tau0)
        self.smear_scale = float(smear_scale)
        self.smear_scale_delta = float(smear_scale_delta)
        self.fit_thickness = bool(fit_thickness)
        self.fit_tau0 = bool(fit_tau0)
        self.fit_smearing = bool(fit_smearing)
        self.fit_smearing_split = bool(fit_smearing_split)
        self.seed = seed

    def _solve(
        self, grid, t_meas, delays, interaction, ew0, weights, rng=None, callback=None
    ) -> RetrievalResult:
        # Forward model differentiable in the extra parameters (thickness/tau0);
        # with both flags off it reduces exactly to the standard trace map. (The
        # global search uses no gradients, but reuses the same shared map.)
        trace_param = make_param_trace_fn(
            grid.omega,
            delays,
            interaction,
            material=self.material,
            thickness=self.thickness,
            npoints=self.npoints,
            omega0=self.omega0,
            quadrature=self.quadrature,
            smearing=self.smearing,
            focal=self.focal,
            fit_thickness=self.fit_thickness,
            fit_tau0=self.fit_tau0,
            fit_smearing=self.fit_smearing,
        )
        # Temporal-penalty mask (None when disabled); window defaults to the
        # measurement delay range.
        time_mask = None
        if self.reg_time > 0:
            lo, hi = self.time_window or (float(delays.min()), float(delays.max()))
            time_mask = time_window_mask(grid.t, lo, hi, grid.dt)
        par = build_parameterisation(
            ew0,
            grid.omega,
            phase_only=self.phase_only,
            phase_basis=self.phase_basis,
            n_nodes=self.n_nodes,
            spectral_target=self._spectral_target,
            time_mask=time_mask,
        )
        spectrum_fn, u0_pulse, pen = par.spectrum_fn, par.u0, par.penalties

        # Extra searched scalars (slab thickness / delay-zero), appended to u as
        # dimensionless O(1)-scaled offsets so std_init treats them sensibly.
        scale_tau = (
            float(np.median(np.abs(np.diff(delays)))) if delays.size > 1 else 1.0
        )
        spec = ExtraParamSpec(
            fit_thickness=self.fit_thickness,
            thickness0=self.thickness,
            fit_tau0=self.fit_tau0,
            tau0_0=self.tau0,
            scale_tau=scale_tau,
            fit_smearing=self.fit_smearing,
            smear_scale0=self.smear_scale,
            fit_smearing_split=self.fit_smearing_split,
            smear_delta0=self.smear_scale_delta,
        )
        aug = augment(u0_pulse, spec)

        tm = jnp.asarray(t_meas)
        wj = jnp.asarray(weights)
        R_omega = self.R_omega
        reg_amp, reg_phase, reg_spectrum, reg_time = (
            self.reg_amp,
            self.reg_phase,
            self.reg_spectrum,
            self.reg_time,
        )

        # Unpack the (possibly augmented) candidate into pulse + extras. With no
        # extras fitted the search vector is the bare pulse and the extras stay
        # at their prior.
        def pulse_unpack(u):
            # Extras held at their centres, matching augment's unpack: a warm-started
            # or explicitly-set tau0/smear_scale still shifts the model here.
            smear0 = (
                np.array([self.smear_scale, self.smear_scale_delta])
                if self.fit_smearing_split
                else self.smear_scale
            )
            return u, float(self.thickness), self.tau0, smear0

        unpack = aug.unpack if spec.any else pulse_unpack
        u0_search = aug.u0 if spec.any else u0_pulse

        def objective_one(u):
            """Return ``(regularised total, bare FROG error)`` for one candidate."""
            u_pulse, thickness, tau0, smear = unpack(u)
            ferr = mj.frog_error(
                tm,
                trace_param(spectrum_fn(u_pulse), thickness, tau0, smear),
                wj,
                R_omega=R_omega,
            )
            total = ferr
            # Penalties self-gate by mode/basis (return 0 where N/A); they act on
            # the pulse block only, so the extra scalars are unpenalised.
            if reg_amp > 0:
                total = total + reg_amp * pen.amplitude(u_pulse)
            if reg_phase > 0:
                total = total + reg_phase * pen.phase(u_pulse)
            if reg_spectrum > 0:
                total = total + reg_spectrum * pen.spectral(u_pulse)
            if reg_time > 0:
                total = total + reg_time * pen.temporal(u_pulse)
            return total, ferr

        batched = jax.jit(jax.vmap(objective_one))

        err_log: list[float] = []

        def make_snapshot(u) -> RetrievalResult:
            """Assemble an in-progress result from a search vector (live preview).

            Unpacks the candidate, recomputes the spectrum and simulated trace and
            feeds them through the shared :func:`assemble_result`, so a live
            snapshot and the final result are built identically. Reused for the
            final return below (called on the all-time best member).
            """
            # `unpack` yields the physical extras for the forward model; coerce to
            # plain floats (they may be 0-d device scalars) so the result records
            # them as floats, matching the gradient/LM solvers.
            u_pulse, thickness, tau0, smear = unpack(u)
            thickness, tau0 = float(thickness), float(tau0)
            smear_p, smear_d = split_smear_value(smear)
            spectrum = np.asarray(spectrum_fn(jnp.asarray(u_pulse)), dtype=complex)
            t_sim = np.asarray(
                trace_param(jnp.asarray(spectrum), thickness, tau0, smear),
                dtype=float,
            )
            return assemble_result(
                spectrum=spectrum,
                t_sim=t_sim,
                grid=grid,
                t_meas=t_meas,
                delays=delays,
                weights=weights,
                interaction=interaction,
                algorithm=self.name,
                errors=err_log,
                R_omega=self.R_omega,
                thickness=thickness if self.fit_thickness else None,
                tau0=tau0 if self.fit_tau0 else 0.0,
                smear_scale=smear_p if self.fit_smearing else None,
                smear_scale_delta=smear_d if self.fit_smearing else None,
                omega0=self.omega0,
            )

        if self.seed is not None:
            key = jax.random.key(int(self.seed))
        else:
            rng = np.random.default_rng() if rng is None else rng
            key = jax.random.key(int(rng.integers(0, 2**31 - 1)))

        best_u, _ = run_cmaes(
            batched,
            u0_search,
            strategy=self.strategy,
            popsize=self.popsize,
            std_init=self.std_init,
            generations=self.maxiters,
            key=key,
            callback=callback,
            make_snapshot=make_snapshot,
            err_log=err_log,
            verbose=self.verbose,
        )

        # The final result is the all-time best member, assembled exactly as the
        # live snapshots (shared :func:`make_snapshot` / `assemble_result`).
        return make_snapshot(best_u)
