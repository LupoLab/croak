r"""JAX/autodiff twin of :mod:`croak.forward`.

This module re-implements the nonlinear-process forward model of
:class:`croak.forward.ForwardModel` in JAX, so that gradients and Jacobians of the
trace (and of any error metric built on it) can be obtained by **automatic
differentiation**. This is the general gradient route: it reaches model features
the hand-derived Wirtinger adjoints in :mod:`croak.forward` do not cover
(geometric smearing, fitted thickness/delay-zero/smearing width) and supplies the
Jacobians that Levenberg–Marquardt and :mod:`croak.covariance` need. It backs the
:class:`~croak.lbfgs_ad.LBFGSAD`, :class:`~croak.lm.LM`,
:class:`~croak.optimistix_lm.OptxLM` and :class:`~croak.cmaes.CMAES` retrievers,
and serves as an independent oracle in the tests that cross-check the analytic
adjoints.

The maths is identical to :meth:`croak.forward.ForwardModel.signal_single`; only
the array library changes. To reproduce the numpy model bit-for-bit the module

* enables 64-bit JAX (``jax_enable_x64``) at import — single precision would not
  match the ``~1e-10`` agreement the tests assert;
* works internally in **DFT-bin order** (``ifftshift`` on input, ``fftshift`` on
  output), using the unnormalised forward DFT for spectrum→time and the ``1/N``
  inverse DFT for time→spectrum, exactly as :func:`croak.forward._fft0` /
  ``_ifft0``;
* treats the slab propagation constant :math:`\beta(\omega)` as a **static
  constant**: it does not depend on the optimisation variables, so it is built
  once with the numpy :func:`croak.materials.beta` and captured (no need to port
  the Sellmeier formulae to JAX).

See :class:`croak.forward.ForwardModel` for the physical model and notation.
"""

from __future__ import annotations

from collections.abc import Callable

import jax

jax.config.update("jax_enable_x64", True)

# Everything below must be imported AFTER that call. Binding `jax.numpy` -- or any
# croak module that binds it -- while x64 is still off freezes float32 defaults for
# the process, and a retrieval then runs at single precision with no error message.
# Hence the E402 suppressions: the import order is the mechanism, not an oversight.
import jax.numpy as jnp  # noqa: E402 - must follow the x64 update above
import numpy as np  # noqa: E402 - must follow the x64 update above
from numpy.typing import ArrayLike, NDArray  # noqa: E402 - see the x64 note above

from . import materials  # noqa: E402 - binds jax.numpy; see the x64 note above
from .collection import (  # noqa: E402 - see the x64 note above
    CollectionAperture,
    transform_phases,
)
from .focal import FocalMixture  # noqa: E402 - see the x64 note above
from .forward import quadrature_nodes_weights  # noqa: E402 - see the x64 note above
from .interactions import get_interaction  # noqa: E402 - see the x64 note above
from .maths import wlfreq  # noqa: E402 - see the x64 note above
from .smearing import (  # noqa: E402 - see the x64 note above
    SmearingKernel,
    delay_frequency_grid,
)

__all__ = [
    "make_signal_fn",
    "make_trace_fn",
    "make_param_trace_fn",
    "make_forward_adjoint_fns",
    "maketrace_jax",
]

Complex = NDArray[np.complex128]

_C_LIGHT = 299_792_458.0

#: Nonlinear-interaction *signal* maps, in JAX. AD (in :class:`~croak.lbfgs_ad.LBFGSAD`
#: and :class:`~croak.lm.LM`) supplies the adjoint, but the hand-gradient retriever
#: :class:`~croak.lbfgs_hand.LBFGSHand` and a future jitted COPRA need the explicit
#: Wirtinger adjoints below. Keys match :data:`croak.interactions.INTERACTIONS`.
_JAX_INTERACTIONS: dict[str, Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = {
    "shg": lambda test, gate: test * gate,
    "sd": lambda test, gate: test**2 * jnp.conj(gate),
    "pg": lambda test, gate: test * jnp.abs(gate) ** 2,
}

#: JAX twin of :meth:`croak.interactions.Interaction.signal_adjoint`: map the output
#: cotangent ``s_bar`` to ``(test_bar, gate_bar)`` (Wirtinger). A direct port of
#: :mod:`croak.interactions` (``np.conj`` → ``jnp.conj`` etc.).
_JAX_INTERACTION_ADJOINTS: dict[
    str,
    Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]],
] = {
    "shg": lambda test, gate, s_bar: (jnp.conj(gate) * s_bar, jnp.conj(test) * s_bar),
    "sd": lambda test, gate, s_bar: (
        2.0 * jnp.conj(test) * gate * s_bar,
        test**2 * jnp.conj(s_bar),
    ),
    "pg": lambda test, gate, s_bar: (
        jnp.abs(gate) ** 2 * s_bar,
        2.0 * gate * jnp.real(jnp.conj(test) * s_bar),
    ),
}

#: JAX twin of :meth:`croak.interactions.Interaction.smeared_signal`: combine the
#: ``p``-independent replica with the pair at ``varying_base -/+ p/2``. SHG is absent —
#: geometric smearing is only defined for the three-arm PG/SD forms.
_JAX_SMEARED_INTERACTIONS: dict[
    str, Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
] = {
    "sd": lambda fixed, varying, swapped: varying * swapped * jnp.conj(fixed),
    "pg": lambda fixed, varying, swapped: fixed * varying * jnp.conj(swapped),
}


def _smear_scale_pair(smear_scale) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Split a smearing scale into its ``(p, delta)`` channel factors.

    A scalar scales both kernel channels together (the single ``fit_smearing``
    multiplier, equivalent to scaling the mask ratio ``d/D``); a length-2 vector
    ``(scale_p, scale_delta)`` scales the gate-shape and delay channels
    independently (``fit_smearing_split``). The branch is on the static array
    rank, so it is JIT- and autodiff-safe.
    """
    s = jnp.asarray(smear_scale)
    if s.ndim == 0:
        return s, s
    return s[0], s[1]


class _SignalConstants:
    """Static problem geometry shared by the JAX forward signal and its adjoint.

    Built once (with numpy) and promoted to JAX x64 arrays; captured in the
    forward / adjoint closures. Mirrors the attributes set in
    :meth:`croak.forward.ForwardModel.__init__`.
    """

    __slots__ = (
        "inter",
        "n",
        "omega_bin",
        "input_prop",
        "signal_prop",
        "weights",
        "signal_map",
        "adjoint_map",
    )

    def __init__(
        self, omega, interaction, *, material, thickness, npoints, omega0, quadrature
    ):
        inter = get_interaction(interaction)
        # Same physical guard as ForwardModel: a slab built at the fundamental
        # cannot consistently propagate an SHG signal that lives near 2*omega0.
        if material is not None and thickness and inter.omega0_scale != 1.0:
            raise ValueError(
                f"dispersive propagation is not supported for the "
                f"{inter.name.upper()} interaction (its signal is not at the "
                f"fundamental frequency); use a thin medium (material=None)"
            )
        omega = np.asarray(omega, dtype=float)
        omega_bin = np.fft.ifftshift(omega)
        beta = materials.beta(material, omega, omega0)
        beta_bin = np.fft.ifftshift(beta)
        nodes, weights = quadrature_nodes_weights(float(thickness), npoints, quadrature)
        # Propagation phase matrices (N, Q), DFT-bin order — identical to ForwardModel.
        input_prop = np.exp(1j * np.outer(beta_bin, nodes))
        signal_prop = np.exp(1j * np.outer(beta_bin, float(thickness) - nodes))

        self.inter = inter
        self.n = int(omega.size)
        self.omega_bin = jnp.asarray(omega_bin)
        self.input_prop = jnp.asarray(input_prop)
        self.signal_prop = jnp.asarray(signal_prop)
        self.weights = jnp.asarray(weights)
        self.signal_map = _JAX_INTERACTIONS[inter.name]
        self.adjoint_map = _JAX_INTERACTION_ADJOINTS[inter.name]


def _signal_with_tape(
    c: _SignalConstants, ew: jnp.ndarray, tau: jnp.ndarray | float
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Forward signal for one delay, returning ``(psi, test, gate)``.

    ``test``/``gate`` are the time-domain fields the adjoint needs (the JAX
    analogue of :meth:`ForwardModel.signal_single`'s ``record=True`` tape).
    """
    ew = jnp.asarray(ew)
    ew_bin = jnp.fft.ifftshift(ew)
    phase = jnp.exp(1j * c.omega_bin * tau)
    # (N, Q) propagated input spectra for the test (undelayed) and gate (delayed)
    e0 = ew_bin[:, None] * c.input_prop
    etau = (ew_bin * phase)[:, None] * c.input_prop
    test = jnp.fft.fft(e0, axis=0)
    gate = jnp.fft.fft(etau, axis=0)
    sig = c.signal_map(test, gate)
    psi_q = jnp.fft.ifft(sig, axis=0) * c.signal_prop
    psi_bin = jnp.sum(psi_q * c.weights, axis=1)
    return jnp.fft.fftshift(psi_bin), test, gate


def _adjoint(
    c: _SignalConstants,
    psi_bar: jnp.ndarray,
    test: jnp.ndarray,
    gate: jnp.ndarray,
    tau: jnp.ndarray | float,
) -> jnp.ndarray:
    """Adjoint of :func:`_signal_with_tape`: map ``psi_bar`` → ``ew_bar`` (centred).

    JAX port of :meth:`croak.forward.ForwardModel.adjoint_single`; reuses the
    ``test``/``gate`` tape returned by the matching forward pass.
    """
    psi_bar_bin = jnp.fft.ifftshift(psi_bar)
    # adjoint of psi_bin = sum_q w_q signal_prop ⊙ raw_ifft(sig)
    psi_q_bar = c.weights * jnp.conj(c.signal_prop) * psi_bar_bin[:, None]
    # adjoint of raw_ifft is (1/N) raw_fft
    sig_bar = jnp.fft.fft(psi_q_bar, axis=0) / c.n
    test_bar, gate_bar = c.adjoint_map(test, gate, sig_bar)
    # adjoint of raw_fft is N raw_ifft
    e0_bar = c.n * jnp.fft.ifft(test_bar, axis=0)
    etau_bar = c.n * jnp.fft.ifft(gate_bar, axis=0)
    phase = jnp.exp(1j * c.omega_bin * tau)
    ew_bar_bin = (
        jnp.conj(c.input_prop) * e0_bar
        + jnp.conj(phase[:, None] * c.input_prop) * etau_bar
    ).sum(axis=1)
    return jnp.fft.fftshift(ew_bar_bin)


def make_signal_fn(
    omega: ArrayLike,
    delays: ArrayLike,
    interaction: str,
    *,
    material: str | None = None,
    thickness: float = 0.0,
    npoints: int = 1,
    omega0: float = 0.0,
    quadrature: str = "gausslegendre",
) -> Callable[[jnp.ndarray, jnp.ndarray | float], jnp.ndarray]:
    r"""Build the single-delay signal map :math:`\tilde E \mapsto \psi(\omega)`.

    JAX twin of :meth:`croak.forward.ForwardModel.signal_single`. The returned
    function ``signal(ew, tau)`` takes a centred complex spectrum ``ew`` and a
    delay ``tau`` and returns the centred frequency-domain signal ``psi``. It is
    differentiable and ``vmap``/``jit``-friendly; the static problem geometry
    (grid, delays, interaction, dispersion) is captured in the closure.

    Parameters
    ----------
    omega : array_like
        Centred angular-frequency grid (rad/s), relative to ``omega0``.
    delays : array_like
        Delay axis (s); only its length/values inform the closure, ``tau`` is a
        runtime argument (kept for signature parity with the numpy model).
    interaction : str
        ``"shg"``, ``"sd"`` or ``"pg"``.
    material, thickness, npoints, omega0, quadrature
        Dispersive-slab parameters; see :class:`croak.forward.ForwardModel`.

    Returns
    -------
    callable
        ``signal(ew, tau) -> psi`` (centred order).
    """
    c = _SignalConstants(
        omega,
        interaction,
        material=material,
        thickness=thickness,
        npoints=npoints,
        omega0=omega0,
        quadrature=quadrature,
    )

    def signal(ew: jnp.ndarray, tau: jnp.ndarray | float) -> jnp.ndarray:
        psi, _test, _gate = _signal_with_tape(c, ew, tau)
        return psi

    return signal


def make_forward_adjoint_fns(
    omega: ArrayLike,
    delays: ArrayLike,
    interaction: str,
    *,
    material: str | None = None,
    thickness: float = 0.0,
    npoints: int = 1,
    omega0: float = 0.0,
    quadrature: str = "gausslegendre",
) -> tuple[
    Callable[
        [jnp.ndarray, jnp.ndarray | float], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ],
    Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray | float], jnp.ndarray],
]:
    r"""Build the matched JAX forward (with tape) and **hand-derived** adjoint.

    JAX twin of the forward/adjoint pair on :class:`croak.forward.ForwardModel`
    (:meth:`signal_single` + :meth:`adjoint_single`). Both share one set of static
    constants and are ``vmap``/``jit``-friendly, so a caller can backpropagate a
    trace cotangent **without automatic differentiation** — the basis of
    :class:`~croak.lbfgs_hand.LBFGSHand` and a future jitted COPRA.

    Returns
    -------
    signal_tape : callable
        ``signal_tape(ew, tau) -> (psi, test, gate)`` (centred ``psi``; ``test`` /
        ``gate`` are the time-domain fields the adjoint reuses).
    adjoint : callable
        ``adjoint(psi_bar, test, gate, tau) -> ew_bar`` (centred), the Wirtinger
        adjoint mapping :math:`\partial L/\partial\psi^*` to
        :math:`\partial L/\partial\tilde E^*`.
    """
    c = _SignalConstants(
        omega,
        interaction,
        material=material,
        thickness=thickness,
        npoints=npoints,
        omega0=omega0,
        quadrature=quadrature,
    )

    def signal_tape(
        ew: jnp.ndarray, tau: jnp.ndarray | float
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return _signal_with_tape(c, ew, tau)

    def adjoint(
        psi_bar: jnp.ndarray,
        test: jnp.ndarray,
        gate: jnp.ndarray,
        tau: jnp.ndarray | float,
    ) -> jnp.ndarray:
        return _adjoint(c, psi_bar, test, gate, tau)

    return signal_tape, adjoint


def make_trace_fn(
    omega: ArrayLike,
    delays: ArrayLike,
    interaction: str,
    *,
    material: str | None = None,
    thickness: float = 0.0,
    npoints: int = 1,
    omega0: float = 0.0,
    quadrature: str = "gausslegendre",
    smearing: SmearingKernel | None = None,
    focal: FocalMixture | None = None,
    normalize: bool = False,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    r"""Build the full-trace map :math:`\tilde E \mapsto T(\omega,\tau)`.

    JAX twin of :meth:`croak.forward.ForwardModel.trace`. The returned
    ``trace(ew)`` maps a centred spectrum to an ``(Nomega, Ndelay)`` intensity
    trace :math:`|\psi|^2`, vectorising the single-delay signal over ``delays``
    with :func:`jax.vmap`. It is differentiable and ``jit``-friendly.

    Parameters
    ----------
    smearing : SmearingKernel or None, optional
        Geometric time-smearing kernel (see :mod:`croak.smearing`). Because the delay
        offset acts across the whole delay axis, the smeared model has no single-delay
        form; this simply delegates to :func:`make_param_trace_fn` with the slab
        thickness and delay zero held fixed. The same applies to ``focal``.
    focal : FocalMixture or None, optional
        Chromatic focal mixture, optionally carrying a collection aperture. As for
        :func:`make_param_trace_fn`, to which this delegates when either it or
        ``smearing`` is set.
    normalize : bool, optional
        Scale the trace to unit maximum (default ``False``; retrieval works with
        the unnormalised trace and absorbs the scale through ``mu``).
    other parameters
        As :func:`make_signal_fn`.

    Returns
    -------
    callable
        ``trace(ew) -> T`` of shape ``(Nomega, Ndelay)``.
    """
    if smearing is not None or focal is not None:
        param_trace = make_param_trace_fn(
            omega,
            delays,
            interaction,
            material=material,
            thickness=thickness,
            npoints=npoints,
            omega0=omega0,
            quadrature=quadrature,
            smearing=smearing,
            focal=focal,
            normalize=normalize,
        )
        thickness_c = float(thickness)

        def smeared_trace(ew: jnp.ndarray) -> jnp.ndarray:
            return param_trace(ew, thickness_c, 0.0)

        return smeared_trace

    signal = make_signal_fn(
        omega,
        delays,
        interaction,
        material=material,
        thickness=thickness,
        npoints=npoints,
        omega0=omega0,
        quadrature=quadrature,
    )
    delays_j = jnp.asarray(np.asarray(delays, dtype=float))

    def trace(ew: jnp.ndarray) -> jnp.ndarray:
        # vmap over delays (tau varies, ew fixed) -> (Ndelay, Nomega), then transpose.
        psis = jax.vmap(lambda tau: signal(ew, tau))(delays_j)
        t = jnp.abs(psis.T) ** 2
        if normalize:
            t = t / jnp.max(t)
        return t

    return trace


def _depth_phase_rate(
    aperture: CollectionAperture,
    material: str,
    omega_abs: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Transverse dephasing rate :math:`g_j(\omega)` of the depth integral, in rad/m.

    Signal generated at depth :math:`z` and collected at aperture node :math:`j`
    accumulates, over the remaining slab :math:`L-z`, the transverse phase the
    on-axis propagator :math:`e^{i\beta(L-z)}` leaves out:

    .. math:: g_j(\omega)\,(L - z), \qquad
              g_j(\omega) = k_z(\omega, k_{\perp j}) - k_z(\omega, 0), \qquad
              k_z = \sqrt{(n\omega/c)^2 - k_\perp^2}.

    The full :math:`k_z` difference is used (no paraxial expansion — it costs
    nothing here) via the cancellation-safe rearrangement
    :math:`g = -k_\perp^2 / (k_{z0} + \sqrt{k_{z0}^2 - k_\perp^2})`: the naive
    subtraction loses ~4 digits at the reference geometry where
    :math:`k_\perp/k_{z0}\sim 10^{-2}`. Any :math:`k_\perp`-independent gauge on
    the on-axis propagation constant (the :math:`-\beta_1\omega` moving-frame term
    of :func:`croak.materials.beta`) cancels in the difference, so none is applied.

    The node wavevector is the **absolute** transverse wavevector: a chromatic
    hole's node at mask position :math:`(x_j, y_j)` selects
    :math:`k_{\perp j} = \omega\,\rho_j/(c\,z_{\text{mask}})` with
    :math:`\rho_j = \sqrt{x_j^2+y_j^2}` (positions are stored absolute, hole
    centre folded in); a fixed k window's nodes are wavevectors already. The
    carrier-removed frame is deliberately *not* used — the quadratic
    :math:`k_z` is not invariant under the carrier shift, and the difference (a
    pointing/delay term plus a node-common constant) is physical. The sign
    convention (``e^{+i beta z}`` propagator, so ``g < 0``) is pinned by the
    dense-FFT reference test in ``tests/test_collection.py``, not argued here.

    Parameters
    ----------
    aperture : CollectionAperture
        The collection aperture whose nodes define :math:`k_{\perp j}`.
    material : str
        Slab material name for :func:`croak.materials.refractive_index`.
    omega_abs : ndarray, shape (Nomega,)
        **Absolute** angular frequencies (rad/s), in the centred order the
        collection transform uses.

    Returns
    -------
    ndarray, shape (J, Nomega)
        :math:`g_j(\omega) \le 0` in rad/m. Bins where the material model is
        invalid (non-finite :math:`n`, non-positive frequency) or the node is
        evanescent (:math:`k_\perp \ge n\omega/c`) are zeroed — they carry no
        propagating signal, but a NaN would poison the coherent sums, the same
        policy as :func:`croak.materials.beta`.
    """
    rho = np.hypot(
        np.asarray(aperture.x, dtype=float), np.asarray(aperture.y, dtype=float)
    )
    omega_abs = np.asarray(omega_abs, dtype=float)
    if aperture.chromatic:
        # A mask position x selects k = (w/c) x / z_mask (see croak.collection).
        kperp = omega_abs[None, :] * rho[:, None] / (_C_LIGHT * aperture.z_mask)
    else:
        # A fixed k window's nodes are already transverse wavevectors (rad/m).
        kperp = np.repeat(rho[:, None], omega_abs.size, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        n = materials.refractive_index(material)(wlfreq(omega_abs))
        kz0 = n * omega_abs / _C_LIGHT
        kz_sq = kz0**2 - kperp**2
        g = -(kperp**2) / (kz0 + np.sqrt(np.maximum(kz_sq, 0.0)))
    g[~np.isfinite(g) | (kz_sq <= 0.0) | (omega_abs[None, :] <= 0.0)] = 0.0
    return np.ascontiguousarray(g)


def make_param_trace_fn(
    omega: ArrayLike,
    delays: ArrayLike,
    interaction: str,
    *,
    depth_weight: ArrayLike | None = None,
    depth_node_phase: ArrayLike | None = None,
    material: str | None = None,
    thickness: float = 0.0,
    npoints: int = 1,
    omega0: float = 0.0,
    quadrature: str = "gausslegendre",
    smearing: SmearingKernel | None = None,
    focal: FocalMixture | None = None,
    depth_transverse: bool = False,
    fit_thickness: bool = False,
    fit_tau0: bool = False,
    fit_smearing: bool = False,
    normalize: bool = False,
) -> Callable[..., jnp.ndarray]:
    r"""Build a trace map that also takes the slab thickness and a delay-zero shift.

    Like :func:`make_trace_fn`, but the returned ``trace(ew, thickness, tau0)``
    is differentiable in the **slab thickness** ``L`` and a **delay-zero offset**
    ``tau0`` as well as in the spectrum, so the AD retrievers
    (:class:`~croak.lbfgs_ad.LBFGSAD`, :class:`~croak.lm.LM`) and the covariance
    estimator can fit them alongside the pulse. See
    :func:`croak._jax_pulse.augment` for the parameter packing.

    The delay-zero offset enters as a shift of the gate's delay phase,
    :math:`e^{i\omega(\tau-\tau_0)}` — it slides the simulated trace along the
    delay axis (for SHG, its symmetry centre), correcting a mis-set time zero.

    When ``fit_thickness`` is set the depth quadrature uses **normalised**
    Gauss–Legendre nodes :math:`\xi\in[0,1]` and weights :math:`\hat w`
    (:math:`\sum\hat w=1`) so that, at call time, the depth nodes are
    :math:`z_q=L\,\xi_q`, the weights are :math:`L\,\hat w_q`, and the
    propagation phases :math:`e^{i\beta L\xi_q}`, :math:`e^{i\beta L(1-\xi_q)}`
    are differentiable in ``L``. When it is not set, the nodes/weights are baked
    once from :func:`quadrature_nodes_weights` (so the thin-medium single-node
    case ``material=None`` is preserved for a ``tau0``-only fit).

    Parameters
    ----------
    fit_thickness : bool, optional
        Make the trace differentiable in ``thickness``. Requires a dispersive
        slab (``material`` set and ``thickness > 0``); raises otherwise.
    fit_tau0 : bool, optional
        Make the trace differentiable in the delay-zero offset ``tau0``. Marker
        only — ``tau0`` is always a runtime argument; the flag documents intent
        and parallels ``fit_thickness``.
    smearing : SmearingKernel or None, optional
        Geometric time-smearing kernel (see :mod:`croak.smearing`). When set, the
        returned trace also accepts a fourth argument ``smear_scale`` (default ``1.0``)
        that multiplies the kernel widths, so the smearing strength — equivalently the
        mask ratio ``d/D`` — can be fitted. A scalar scales both channels together; a
        length-2 vector ``(scale_p, scale_delta)`` scales the gate-shape (``p``) and
        delay (``delta``) channels independently (see :func:`_smear_scale_pair`).
        ``None`` (default) leaves the model untouched.
    focal : FocalMixture or None, optional
        Chromatic focal mixture (:mod:`croak.focal`) in place of ``smearing``: the
        focal average is carried explicitly, each node applying its own
        :math:`A(r,\omega)` and its own definite ``(p, theta)`` rather than a
        Gaussian distribution over them. Mutually exclusive with ``smearing``.

        If the mixture carries a ``collection``
        (:class:`croak.collection.CollectionAperture`) the per-node fields are
        transformed to transverse wavevector and integrated over the aperture
        instead of summed incoherently — the difference between an instrument that
        collects the whole signal beam and one that collects it through a hole. PG
        (TG) only; see :mod:`croak.collection`.
    fit_smearing : bool, optional
        Make the trace differentiable in the smearing scale. Marker only, for the
        same reason as ``fit_tau0``; it does require ``smearing`` to be set.
    depth_weight : array_like or None, optional
        Complex generation-envelope weights, one per quadrature node, multiplying
        the depth-quadrature weights: models the transverse geometry a 1D depth
        integral cannot see (focal-spot evolution across the slab, beamlet
        walk-through, Gouy phase), which enters as a smooth per-depth factor
        :math:`g(z_q)` on the locally generated signal. The overall scale is
        irrelevant (absorbed by :math:`\mu`); only the shape matters. Requires a
        dispersive slab and length ``npoints``. With ``fit_thickness`` the
        envelope is pinned to the *normalized* nodes (it rescales with the fitted
        slab rather than staying fixed in absolute :math:`z`). ``None`` (default)
        is the uniform-generation model. Negligible for slabs much thinner than
        the beams' Rayleigh range.
    depth_transverse : bool, optional
        Model the transverse decoherence of the depth integral. croak's depth
        quadrature normally adds the signal generated at every depth with the
        *same* transverse phase — fully coherently in transverse wavevector.
        Real propagation dephases the angular spectrum: signal born at depth
        :math:`z_q` accumulates, over the remaining slab, the extra phase

        .. math:: \phi_{jq}(\omega) =
                  \bigl[k_z(\omega, k_{\perp j}) - k_z(\omega, 0)\bigr]\,
                  (L - z_q),

        applied per aperture node :math:`j` *inside* the collection transform,
        before the coherent depth sum (see :func:`_depth_phase_rate` for
        :math:`k_{\perp j}` and the exact :math:`k_z` used). Parameter-free and
        opt-in; the default ``False`` leaves the existing model bit-identical.
        Requires ``focal`` with a ``collection`` and a dispersive slab. For a
        full-collection variant use a collection window much wider than the
        signal footprint (a large chromatic hole, or a wide ``chromatic=False``
        k window) — there is deliberately no separate path. Costs one
        aperture-transform contraction per depth node instead of one total;
        with ``depth_weight`` and ``depth_node_phase`` it composes (both keep
        their existing meaning). See
        ``docs/howto/collection_aperture.md`` for when the missing physics
        matters (it grows with slab thickness).
    other parameters
        As :func:`make_trace_fn`.

    Returns
    -------
    callable
        ``trace(ew, thickness, tau0, smear_scale=1.0) -> T`` of shape
        ``(Nomega, Ndelay)``. ``thickness``/``tau0``/``smear_scale`` may be Python
        floats (held fixed) or JAX scalars (differentiated); ``smear_scale`` may
        also be a length-2 vector for split channel scaling.
    """
    inter = get_interaction(interaction)
    if smearing is not None and inter.name not in _JAX_SMEARED_INTERACTIONS:
        raise ValueError(
            f"geometrical smearing is not supported for the {inter.name.upper()} "
            f"interaction; only PG (TG) and SD have the required three-arm form"
        )
    # Same physical guard as the forward model: a slab built at the fundamental
    # cannot consistently propagate a signal that lives away from omega0 (SHG).
    if material is not None and thickness and inter.omega0_scale != 1.0:
        raise ValueError(
            f"dispersive propagation is not supported for the "
            f"{inter.name.upper()} interaction (its signal is not at the "
            f"fundamental frequency); use a thin medium (material=None)"
        )
    if fit_thickness and (material is None or thickness <= 0.0):
        raise ValueError(
            "fitting thickness requires a dispersive slab "
            "(material set and thickness > 0)"
        )
    if fit_smearing and smearing is None:
        if focal is not None:
            raise ValueError(
                "fit_smearing has no effect with `focal`: the explicit mixture "
                "carries a definite (p, theta) per node, so there is no kernel "
                "width to scale. Use `smearing` (the reduced kernel) if you "
                "want a fitted scale, or fit the mask geometry instead"
            )
        raise ValueError("fitting the smearing scale requires a smearing kernel")
    # EXPERIMENTAL, diagnostic only. A per-(focal node,
    # depth node) phase applied INSIDE the coherent depth sum, so the longitudinal
    # build-up is untouched while the transverse phase is allowed to vary with
    # depth. It exists to test whether the depth integral's transverse coherence
    # is what makes the collection model over-correct on thick slabs: croak adds
    # every depth with the SAME transverse phase, where 3-D diffraction does not.
    # Not part of the physical model; leave it None for any real retrieval.
    dnp = None
    if depth_node_phase is not None:
        if focal is None:
            raise ValueError("depth_node_phase applies only to the `focal` path")
        arr = np.asarray(depth_node_phase, dtype=complex)
        if arr.shape != (int(focal.nodes), npoints):
            raise ValueError(
                f"depth_node_phase must have shape (focal.nodes, npoints) = "
                f"({int(focal.nodes)}, {npoints}), got {arr.shape}"
            )
        dnp = jnp.asarray(arr)

    dw = None
    if depth_weight is not None:
        if material is None or thickness <= 0.0:
            raise ValueError(
                "a generation envelope (depth_weight) requires a dispersive slab "
                "(material set and thickness > 0)"
            )
        dw_np = np.asarray(depth_weight, dtype=complex)
        if dw_np.shape != (npoints,):
            raise ValueError(
                f"depth_weight must have one entry per quadrature node "
                f"(shape ({npoints},)), got {dw_np.shape}"
            )
        dw = jnp.asarray(dw_np)

    # Per-depth transverse dephasing (depth_transverse): validate up front and
    # build the static dephasing rate g_j(w) here, where the narrowing guards
    # keep `focal`, its collection and `material` provably non-None. The rate is
    # geometry only — the runtime thickness enters later, via depth_remaining.
    g_dt = None
    if depth_transverse:
        if focal is None or focal.collection is None:
            raise ValueError(
                "depth_transverse requires a `focal` mixture carrying a "
                "`collection` aperture: the per-depth transverse phase acts on "
                "the aperture nodes' wavevectors. For the full-collection "
                "variant use a collection window much wider than the signal "
                "footprint (a large chromatic hole, or a wide chromatic=False "
                "k window) instead of collection=None"
            )
        if material is None or thickness <= 0.0:
            raise ValueError(
                "depth_transverse requires a dispersive slab (material set and "
                "thickness > 0): the phase accumulates over the remaining "
                "slab (L - z_q)"
            )
        g_dt = _depth_phase_rate(
            focal.collection,
            material,
            np.asarray(omega, dtype=float) + float(omega0),
        )
    omega = np.asarray(omega, dtype=float)
    beta_bin_np = np.fft.ifftshift(materials.beta(material, omega, omega0))
    omega_bin = jnp.asarray(np.fft.ifftshift(omega))
    beta_bin = jnp.asarray(beta_bin_np)
    signal_map = _JAX_INTERACTIONS[inter.name]
    delays_j = jnp.asarray(np.asarray(delays, dtype=float))

    if fit_thickness:
        # Normalised quadrature on [0, 1]; thickness scales the nodes/weights at
        # call time, keeping the depth integral differentiable in L.
        nodes0, weights0 = quadrature_nodes_weights(1.0, npoints, quadrature)
        xi = jnp.asarray(nodes0)
        w_hat = jnp.asarray(weights0)

        def props(thk: jnp.ndarray | float):
            nodes = thk * xi
            input_prop = jnp.exp(1j * beta_bin[:, None] * nodes[None, :])
            signal_prop = jnp.exp(1j * beta_bin[:, None] * (thk - nodes)[None, :])
            weights = thk * w_hat
            return input_prop, signal_prop, weights if dw is None else weights * dw

        def depth_remaining(thk: jnp.ndarray | float) -> jnp.ndarray:
            """Remaining slab ``L - z_q`` per depth node, differentiable in ``L``."""
            return thk * (1.0 - xi)
    else:
        # Thickness held fixed: bake the propagation matrices once, exactly as the
        # standard forward model (handles the thin single-node case).
        nodes0, weights0 = quadrature_nodes_weights(
            float(thickness), npoints, quadrature
        )
        input_prop_c = jnp.asarray(np.exp(1j * np.outer(beta_bin_np, nodes0)))
        signal_prop_c = jnp.asarray(
            np.exp(1j * np.outer(beta_bin_np, float(thickness) - nodes0))
        )
        weights_c = jnp.asarray(weights0) if dw is None else jnp.asarray(weights0) * dw
        remaining_c = jnp.asarray(float(thickness) - np.asarray(nodes0))

        def props(thk: jnp.ndarray | float):
            return input_prop_c, signal_prop_c, weights_c

        def depth_remaining(thk: jnp.ndarray | float) -> jnp.ndarray:
            """Remaining slab ``L - z_q`` per depth node (thickness held fixed)."""
            return remaining_c

    def signal(ew, tau, thickness_v, tau0_v):
        ew_bin = jnp.fft.ifftshift(ew)
        phase = jnp.exp(1j * omega_bin * (tau - tau0_v))
        input_prop, signal_prop, weights = props(thickness_v)
        e0 = ew_bin[:, None] * input_prop
        etau = (ew_bin * phase)[:, None] * input_prop
        test = jnp.fft.fft(e0, axis=0)
        gate = jnp.fft.fft(etau, axis=0)
        sig = signal_map(test, gate)
        psi_q = jnp.fft.ifft(sig, axis=0) * signal_prop
        psi_bin = jnp.sum(psi_q * weights, axis=1)
        return jnp.fft.fftshift(psi_bin)

    # --- chromatic focal mixture --------------------------------------------------
    # The reduced kernel below keeps the PHASES of the focal average (p, theta) and
    # drops its AMPLITUDE structure, evaluating the |A|^6 weight at the carrier. Here
    # the sum over focal position stays explicit: each node applies its own chromatic
    # amplitude filter A(r, omega) to the field -- so the three-arm product supplies
    # A^3 and the trace |A|^6 with the frequency dependence intact -- and its own
    # DEFINITE (p, theta), theta entering as a shift of the effective delay because it
    # is a pure translation of the delay axis. Cost is one forward evaluation per node.
    # How those per-node fields are then combined is the mixture's `collection`: an
    # incoherent sum (full-beam collection, the default) or a transform to transverse
    # k followed by an aperture (see below and croak.collection).
    if focal is not None:
        if smearing is not None:
            raise ValueError(
                "pass either `smearing` (reduced kernel) or `focal` (explicit "
                "mixture), not both: the focal mixture already contains the "
                "geometric smearing the reduced kernel models"
            )
        focal_map = _JAX_SMEARED_INTERACTIONS[inter.name]
        omega_abs = np.asarray(omega, dtype=float) + float(omega0)
        filt = jnp.asarray(focal.spectral_filter(omega_abs))
        p_k = jnp.asarray(focal.p)
        theta_k = jnp.asarray(focal.theta)
        area_k = jnp.asarray(focal.area)

        # --- finite collection aperture --------------------------------------------
        # Summing |psi|^2 over focal position is exact ONLY under full-beam collection
        # (Parseval). Through an aperture the signal must be transformed to transverse
        # k BEFORE it is squared, which needs back the single phase the (p, theta)
        # reduction discards -- exactly exp(+i w p) once the signal's own phase-matched
        # carrier is removed. See croak.collection for the derivation.
        collect = focal.collection
        if collect is not None:
            if inter.name != "pg":
                raise ValueError(
                    f"a collection aperture is derived only for PG (TG), not "
                    f"{inter.name.upper()}: the phase the incoherent sum drops is "
                    f"+w*p for PG but -w*theta for SD, so SD needs its own derivation"
                )
            phases = transform_phases(
                focal, np.asarray(omega, dtype=float), float(omega0)
            )
            # M[j,k,w] = area_k exp(i (w Phi_jk - Theta_jk)) is a constant of the
            # model, so it is built once here rather than inside the traced function.
            # Memory is 16 J K Nomega bytes -- the aperture and mixture node counts
            # multiply, which is what makes both worth keeping modest.
            m_jkw = jnp.asarray(
                np.asarray(focal.area)[None, :, None]
                * np.exp(
                    1j
                    * (
                        omega_abs[None, None, :] * phases.chromatic_delay[:, :, None]
                        - phases.static_phase[:, :, None]
                    )
                )
            )
            weight_sq = jnp.asarray(phases.weight_sq)
            weight_amp = jnp.asarray(phases.weight_amp)

            # --- per-depth transverse dephasing (depth_transverse) --------------
            # The standard path sums the depth quadrature inside signal_focal and
            # transforms ONCE; here the depth sum moves INSIDE the transform, each
            # depth's contribution carrying exp(i g_j(w) (L - z_q)) at aperture
            # node j before the coherent sum over q. The mode reduction (|.|^2)
            # happens strictly AFTER that sum — that is the physics: the phase is
            # invisible to a single depth's modulus but decoheres the depth stack.
            # This branch leaves the flag-off path above byte-for-byte untouched
            # (the off = bit-identical contract in tests/test_depth_transverse.py).
            if depth_transverse:
                g_jw = jnp.asarray(g_dt)  # (J, Nomega), centred order

                def signal_focal_at(ew, tau, thickness_v, tau0_v, k, iq):
                    """One focal node's signal from depth node ``iq`` alone.

                    Identical to ``signal_focal`` below but for a single depth
                    node — no depth sum, quadrature weight folded in. Total FFT
                    work over the scan equals the batched (N, Q) FFTs of the
                    standard path.
                    """
                    # TODO(depth): walked per-arm filters A_j(|r_k - d_j(z_q)|, w)
                    # — the beamlet centroids walk ~ -z r_j/(f n_g) inside the
                    # slab (~0.7% weight modulation at 40 um). Amplitude-only and
                    # bounded small; deliberately not modelled (Feature B of the
                    # depth-decoherence brief).
                    ew_bin = jnp.fft.ifftshift(ew * filt[k])
                    input_prop, signal_prop, weights = props(thickness_v)
                    in_q = input_prop[:, iq]
                    fixed_shift, varying_base = inter.smeared_shifts(
                        tau - tau0_v - theta_k[k]
                    )
                    fixed = jnp.fft.fft(
                        ew_bin * jnp.exp(1j * omega_bin * fixed_shift) * in_q
                    )
                    lo = jnp.fft.fft(
                        ew_bin
                        * jnp.exp(1j * omega_bin * (varying_base - 0.5 * p_k[k]))
                        * in_q
                    )
                    hi = jnp.fft.fft(
                        ew_bin
                        * jnp.exp(1j * omega_bin * (varying_base + 0.5 * p_k[k]))
                        * in_q
                    )
                    sig = focal_map(fixed, lo, hi)
                    psi = jnp.fft.ifft(sig) * signal_prop[:, iq]
                    w_q = weights[iq] if dnp is None else weights[iq] * dnp[k, iq]
                    return jnp.fft.fftshift(psi * w_q)

                reimaged = collect.mode == "reimaged"

                def trace(ew, thickness_v, tau0_v, smear_scale=1.0):
                    rem = depth_remaining(thickness_v)  # (Q,)
                    # (Q, J, Nomega): unit-modulus, so it redistributes coherence
                    # rather than energy. Runtime because rem is differentiable
                    # in L under fit_thickness (constant-folded otherwise).
                    phase = jnp.exp(1j * g_jw[None, :, :] * rem[:, None, None])

                    def step(acc, iq):
                        # (Ndelay, K, Nomega) for this depth alone: contract k
                        # first, then phase-and-accumulate — the (D, K, Q, N)
                        # intermediate is never materialised (the memory trap).
                        psis = jax.vmap(
                            lambda tau: jax.vmap(
                                lambda k: signal_focal_at(
                                    ew, tau, thickness_v, tau0_v, k, iq
                                )
                            )(jnp.arange(p_k.size))
                        )(delays_j)
                        s_j = jnp.einsum("jkw,dkw->djw", m_jkw, psis)
                        return acc + phase[iq][None, :, :] * s_j, None

                    acc0 = jnp.zeros(
                        (delays_j.size, weight_sq.shape[0], omega_bin.size),
                        dtype=complex,
                    )
                    s, _ = jax.lax.scan(step, acc0, jnp.arange(npoints))
                    if reimaged:
                        t = jnp.abs(jnp.einsum("djw,jw->dw", s, weight_amp)).T ** 2
                    else:
                        t = jnp.einsum("djw,jw->dw", jnp.abs(s) ** 2, weight_sq).T
                    if normalize:
                        t = t / jnp.max(t)
                    return t

                return trace

        if collect is None:

            def reduce_nodes(psis: jnp.ndarray) -> jnp.ndarray:
                """Incoherent sum over focal position: the full-collection limit."""
                return jnp.tensordot(jnp.abs(psis) ** 2, area_k, axes=([1], [0])).T

        elif collect.mode == "reimaged":

            def reduce_nodes(psis: jnp.ndarray) -> jnp.ndarray:
                """Windowed field back at the focal origin: the coherent limit."""
                s = jnp.einsum("jkw,dkw->djw", m_jkw, psis)
                return jnp.abs(jnp.einsum("djw,jw->dw", s, weight_amp)).T ** 2

        else:

            def reduce_nodes(psis: jnp.ndarray) -> jnp.ndarray:
                """Intensity integrated over the aperture in transverse k."""
                s = jnp.einsum("jkw,dkw->djw", m_jkw, psis)
                return jnp.einsum("djw,jw->dw", jnp.abs(s) ** 2, weight_sq).T

        def signal_focal(ew, tau, thickness_v, tau0_v, k):
            # The node's chromatic filter multiplies the field ONCE; the three-replica
            # product then carries A^3, so |psi|^2 carries |A|^6 automatically.
            ew_k = ew * filt[k]
            ew_bin = jnp.fft.ifftshift(ew_k)
            input_prop, signal_prop, weights = props(thickness_v)
            # theta translates the delay axis for this node.
            fixed_shift, varying_base = inter.smeared_shifts(tau - tau0_v - theta_k[k])
            fixed = jnp.fft.fft(
                (ew_bin * jnp.exp(1j * omega_bin * fixed_shift))[:, None] * input_prop,
                axis=0,
            )
            lo = jnp.fft.fft(
                (ew_bin * jnp.exp(1j * omega_bin * (varying_base - 0.5 * p_k[k])))[
                    :, None
                ]
                * input_prop,
                axis=0,
            )
            hi = jnp.fft.fft(
                (ew_bin * jnp.exp(1j * omega_bin * (varying_base + 0.5 * p_k[k])))[
                    :, None
                ]
                * input_prop,
                axis=0,
            )
            sig = focal_map(fixed, lo, hi)
            psi_q = jnp.fft.ifft(sig, axis=0) * signal_prop
            w_q = weights if dnp is None else weights * dnp[k]
            return jnp.fft.fftshift(jnp.sum(psi_q * w_q, axis=1))

        def trace(ew, thickness_v, tau0_v, smear_scale=1.0):
            # (Ndelay, K, Nomega): delays vmapped as usual, focal nodes batched.
            psis = jax.vmap(
                lambda tau: jax.vmap(
                    lambda k: signal_focal(ew, tau, thickness_v, tau0_v, k)
                )(jnp.arange(p_k.size))
            )(delays_j)
            t = reduce_nodes(psis)
            if normalize:
                t = t / jnp.max(t)
            return t

        return trace

    if smearing is None:

        def trace(ew, thickness_v, tau0_v, smear_scale=1.0):
            psis = jax.vmap(lambda tau: signal(ew, tau, thickness_v, tau0_v))(delays_j)
            t = jnp.abs(psis.T) ** 2
            if normalize:
                t = t / jnp.max(t)
            return t

        return trace

    # --- geometric smearing -------------------------------------------------------
    # p (the shape parameter) needs an explicit quadrature; delta is a pure translation
    # of the delay axis and is applied exactly as a multiplication of the delay-axis
    # DFT. Both node positions and kernel widths are linear/quadratic in smear_scale,
    # so the whole construction stays differentiable in it. The two channels scale
    # independently: for a bivariate normal, scaling (sigma_p, sigma_delta) by
    # (s_p, s_d) at fixed rho maps the node positions to s_p*p_k and the conditional
    # delay moments to s_d*mu_k and s_d*sigma_cond — so a split scale is exact, not
    # an approximation.
    smeared_map = _JAX_SMEARED_INTERACTIONS[inter.name]
    p_nodes_np, p_weights_np = smearing.nodes_weights()
    mu_np, sigma_delta_cond = smearing.conditional_delay(p_nodes_np)
    p_nodes = jnp.asarray(p_nodes_np)
    p_weights = jnp.asarray(p_weights_np)
    mu_nodes = jnp.asarray(mu_np)
    omega_delay = jnp.asarray(delay_frequency_grid(delays, smearing.sigma_delta))

    def signal_smeared(ew, tau, thickness_v, tau0_v, scale_p):
        ew_bin = jnp.fft.ifftshift(ew)
        input_prop, signal_prop, weights = props(thickness_v)
        fixed_shift, varying_base = inter.smeared_shifts(tau - tau0_v)
        # (N, Q): the replica that does not move with p.
        fixed = jnp.fft.fft(
            (ew_bin * jnp.exp(1j * omega_bin * fixed_shift))[:, None] * input_prop,
            axis=0,
        )
        # (N, Q, K): the replicas at varying_base - p_k/2. Their p -> -p partners are
        # the same array reversed along K, because the quadrature nodes are exactly
        # antisymmetric — hence K + 1 rather than 2K + 1 field constructions.
        shifts = varying_base - 0.5 * scale_p * p_nodes
        varying_phase = jnp.exp(1j * omega_bin[:, None] * shifts[None, :])
        varying = jnp.fft.fft(
            ew_bin[:, None, None] * input_prop[:, :, None] * varying_phase[:, None, :],
            axis=0,
        )
        sig = smeared_map(fixed[:, :, None], varying, varying[:, :, ::-1])
        psi_q = jnp.fft.ifft(sig, axis=0) * signal_prop[:, :, None]
        psi_bin = jnp.sum(psi_q * weights[:, None], axis=1)
        return jnp.fft.fftshift(psi_bin, axes=0)

    def trace(ew, thickness_v, tau0_v, smear_scale=1.0):
        scale_p, scale_d = _smear_scale_pair(smear_scale)
        # (Ndelay, Nomega, K) — vmap over delays as in the unsmeared model, with the
        # quadrature nodes carried as a trailing batch axis.
        psis = jax.vmap(
            lambda tau: signal_smeared(ew, tau, thickness_v, tau0_v, scale_p)
        )(delays_j)
        per_node = jnp.abs(psis) ** 2
        # Characteristic function of the conditional delay offset N(mu_k, sigma^2),
        # shape (Ndelay, K); multiplying the delay-axis DFT by it performs the exact
        # convolution over delta. Only the delta channel scales here: the conditional
        # mean rho*(s_d*sigma_delta / s_p*sigma_p)*(s_p*p_k) = s_d*mu_k is
        # independent of the p scaling.
        omega_d = omega_delay[:, None]
        kernel = jnp.exp(
            -1j * omega_d * (scale_d * mu_nodes)[None, :]
            - 0.5 * (scale_d * sigma_delta_cond) ** 2 * omega_d**2
        )
        spec = jnp.fft.fft(per_node, axis=0)
        blurred = jnp.real(jnp.fft.ifft(spec * kernel[:, None, :], axis=0))
        t = jnp.tensordot(blurred, p_weights, axes=([2], [0])).T
        if normalize:
            t = t / jnp.max(t)
        return t

    return trace


def maketrace_jax(
    omega: ArrayLike,
    delays: ArrayLike,
    ew: Complex,
    interaction: str,
    *,
    material: str | None = None,
    thickness: float = 0.0,
    npoints: int = 1,
    omega0: float = 0.0,
    quadrature: str = "gausslegendre",
    smearing: SmearingKernel | None = None,
    normalize: bool = True,
) -> NDArray[np.float64]:
    """Generate a (normalised) trace with the JAX model, returned as numpy.

    Convenience wrapper around :func:`make_trace_fn` paralleling
    :func:`croak.forward.maketrace`, for tests and examples.
    """
    trace = make_trace_fn(
        omega,
        delays,
        interaction,
        material=material,
        thickness=thickness,
        npoints=npoints,
        omega0=omega0,
        quadrature=quadrature,
        smearing=smearing,
        normalize=normalize,
    )
    return np.asarray(trace(jnp.asarray(np.asarray(ew, dtype=complex))))
