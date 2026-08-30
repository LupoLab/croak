r"""The unified nonlinear-process forward model and its reverse-mode adjoint.

:class:`ForwardModel` is the single computational core shared by every retrieval
algorithm in croak. It maps a complex spectrum :math:`\tilde E(\omega)` to a
delay-scanned nonlinear-process spectrum (a "PNPS" trace — e.g. FROG, TDP),
optionally through a dispersive slab, and provides the exact reverse-mode
(Wirtinger) adjoint of that map.

**Thin medium** is simply the special case of the dispersive model with
:math:`\beta = 0`, zero thickness and a single quadrature node — so one code path
and one adjoint kernel serve COPRA and L-BFGS.

Dispersive model (depth integral over a slab of thickness ``L``), per delay
:math:`\tau`:

.. math::

    \psi(\omega) = \sum_q w_q\, e^{i\beta(L-z_q)} \odot
        \mathcal{F}^{-1}\!\big[\, \mathrm{sig}(
            \mathcal{F}[\tilde E e^{i\beta z_q}],\;
            \mathcal{F}[\tilde E e^{i\omega\tau} e^{i\beta z_q}]) \big]

with Gauss–Legendre nodes :math:`z_q` and weights :math:`w_q` on ``[0, L]``.
:math:`\mathcal{F}` here is the raw (unshifted) DFT; the model works internally in
DFT-bin order and exposes centred arrays at its boundary. The trace is
:math:`T(\omega,\tau)=|\psi(\omega,\tau)|^2`.
"""

from __future__ import annotations

import numpy as np
import scipy.fft
from numpy.polynomial.legendre import leggauss
from numpy.typing import ArrayLike, NDArray

from . import materials
from .interactions import Interaction, get_interaction
from .smearing import SmearingKernel, delay_frequency_grid

__all__ = ["ForwardModel", "quadrature_nodes_weights", "maketrace"]

Complex = NDArray[np.complex128]


def _fft0(x: Complex) -> Complex:
    """Unnormalised forward DFT along the frequency axis (axis 0)."""
    # np.asarray: scipy.fft's @_dispatchable stubs do not expose the ndarray return.
    return np.asarray(scipy.fft.fft(x, axis=0))


def _ifft0(x: Complex) -> Complex:
    """Inverse (1/N) DFT along the frequency axis (axis 0)."""
    return np.asarray(scipy.fft.ifft(x, axis=0))


def quadrature_nodes_weights(
    thickness: float, npoints: int, method: str = "gausslegendre"
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Nodes and weights for the depth integral over ``[0, thickness]``.

    Parameters
    ----------
    thickness : float
        Slab thickness (m). ``0`` returns the single thin-medium node.
    npoints : int
        Number of quadrature points.
    method : {"gausslegendre", "uniform"}
        ``"gausslegendre"`` (default) uses Gauss–Legendre quadrature mapped to
        ``[0, L]``; ``"uniform"`` uses equally spaced nodes with unit weights
        (a plain sum, useful for convergence checks).

    Returns
    -------
    tuple of numpy.ndarray
        ``(nodes, weights)``.
    """
    if thickness == 0.0:
        return np.array([0.0]), np.array([1.0])
    if method == "gausslegendre":
        x, w = leggauss(npoints)
        nodes = 0.5 * thickness * (x + 1.0)
        weights = 0.5 * thickness * w
        return nodes, weights
    if method == "uniform":
        return np.linspace(0.0, thickness, npoints), np.ones(npoints)
    raise ValueError(f"unknown quadrature method {method!r}")


class ForwardModel:
    """Nonlinear-process forward map and adjoint for fixed grid/delays/interaction.

    Parameters
    ----------
    omega : array_like
        Centred angular-frequency grid (rad/s), relative to ``omega0``.
    delays : array_like
        Delay axis (s).
    interaction : str or Interaction
        Nonlinear interaction (``"shg"``, ``"sd"``, ``"pg"``).
    material : str or None, optional
        Dispersive material for the slab (e.g. ``"SiO2"``); ``None`` (default)
        is a thin, dispersionless medium.
    thickness : float, optional
        Slab thickness (m). Default ``0``.
    npoints : int, optional
        Number of depth quadrature points (default ``1``).
    omega0 : float, optional
        Carrier angular frequency (rad/s), used to build ``beta``.
    quadrature : str, optional
        Quadrature method (see :func:`quadrature_nodes_weights`).
    smearing : SmearingKernel or None, optional
        Geometric time-smearing kernel of a non-collinear geometry (see
        :mod:`croak.smearing`). ``None`` (default) switches the correction off
        and leaves every code path exactly as it is without it.

    Notes
    -----
    All public array arguments/returns (spectra, cotangents) are in **centred**
    frequency order; the DFT-bin ordering is an internal detail.

    The hand-derived :meth:`adjoint_single` is **not** available with smearing; use the
    autodiff solvers (see :mod:`croak.forward_jax`) for that case.
    """

    def __init__(
        self,
        omega: ArrayLike,
        delays: ArrayLike,
        interaction: str | Interaction,
        *,
        material: str | None = None,
        thickness: float = 0.0,
        npoints: int = 1,
        omega0: float = 0.0,
        quadrature: str = "gausslegendre",
        smearing: SmearingKernel | None = None,
    ):
        """Build the forward model (see the class docstring for parameters)."""
        self.omega = np.asarray(omega, dtype=float)
        self.delays = np.asarray(delays, dtype=float)
        self.interaction: Interaction = get_interaction(interaction)
        self.n = self.omega.size
        self.m = self.delays.size
        self.material = material
        self.thickness = float(thickness)
        self.omega0 = float(omega0)

        # The slab propagation phase β(ω) is built at the *fundamental* carrier,
        # but the SHG signal lives near 2ω0 — propagating it with β(ω) is
        # physically inconsistent. Dispersive propagation is only meaningful for
        # interactions whose signal stays at the fundamental (PG, SD).
        if material is not None and thickness and self.interaction.omega0_scale != 1.0:
            raise ValueError(
                f"dispersive propagation is not supported for the "
                f"{self.interaction.name.upper()} interaction (its signal is not at "
                f"the fundamental frequency); use a thin medium (material=None)"
            )

        # Work internally in DFT-bin order.
        self.omega_bin = scipy.fft.ifftshift(self.omega)
        beta = materials.beta(material, self.omega, omega0)
        beta_bin = scipy.fft.ifftshift(beta)

        nodes, weights = quadrature_nodes_weights(self.thickness, npoints, quadrature)
        self.nodes = nodes
        self.weights = weights
        self.q = nodes.size

        # Propagation phase matrices, shape (N, Q), in DFT-bin order.
        self.input_prop = np.exp(1j * np.outer(beta_bin, nodes))
        self.signal_prop = np.exp(1j * np.outer(beta_bin, self.thickness - nodes))

        # Geometric smearing. The (p, delta) statistics are independent of depth — the
        # pulse-front tilts are set before the medium and the longitudinal walk-off
        # inside a thin slab is far below a femtosecond — so the kernel multiplies the
        # depth quadrature rather than coupling to it.
        self.smearing = smearing
        if smearing is not None:
            if self.interaction.name not in ("pg", "sd"):
                raise ValueError(
                    f"geometrical smearing is not supported for the "
                    f"{self.interaction.name.upper()} interaction; only PG (TG) and SD "
                    f"have the required three-arm form"
                )
            self.p_nodes, self.p_weights = smearing.nodes_weights()
            self.p_mu, self.p_sigma = smearing.conditional_delay(self.p_nodes)
            self.delay_omega = delay_frequency_grid(self.delays, smearing.sigma_delta)

        # Per-delay tape (forward time fields), reused between forward/adjoint.
        self._tape_test: Complex | None = None
        self._tape_gate: Complex | None = None
        self._tape_tau: float | None = None

    # -- forward -----------------------------------------------------------
    def signal_single(
        self, ew: Complex, tau: float, *, record: bool = False
    ) -> Complex:
        r"""Frequency-domain signal :math:`\psi(\omega)` for one delay (centred).

        Parameters
        ----------
        ew : numpy.ndarray
            Complex spectrum (centred order).
        tau : float
            Delay (s).
        record : bool, optional
            If ``True``, store the forward time-domain fields so a following
            :meth:`adjoint_single` call (same ``ew``, ``tau``) can reuse them.
        """
        ew_bin = scipy.fft.ifftshift(ew)
        phase = np.exp(1j * self.omega_bin * tau)
        # (N, Q) propagated input spectra for test (undelayed) and gate (delayed)
        e0 = ew_bin[:, None] * self.input_prop
        etau = (ew_bin * phase)[:, None] * self.input_prop
        test = _fft0(e0)
        gate = _fft0(etau)
        sig = self.interaction.signal(test, gate)
        psi_q = _ifft0(sig) * self.signal_prop
        psi_bin = (psi_q * self.weights).sum(axis=1)
        if record:
            self._tape_test = test
            self._tape_gate = gate
            self._tape_tau = float(tau)
        return scipy.fft.fftshift(psi_bin)

    def signal_smeared(self, ew: Complex, tau: float) -> Complex:
        r"""Per-node signals :math:`\psi_k(\omega)` for one delay, shape ``(N, K)``.

        One column per Gauss–Hermite node of the shape parameter ``p``; the delay
        offset ``delta`` is *not* applied here because it acts across the whole delay
        axis (see :meth:`trace`).

        The three replicas are built as exact spectral phases, so no interpolation and
        no delay-grid alignment is involved. Because the nodes are antisymmetric, the
        replica at ``varying_base + p_k/2`` is the one already computed for the mirrored
        node ``-p_k``: the ``[::-1]`` below is what keeps the cost at ``K + 1`` field
        constructions rather than ``2K + 1``.
        """
        if self.smearing is None:
            raise RuntimeError("signal_smeared requires a smearing kernel")
        ew_bin = scipy.fft.ifftshift(ew)
        fixed_shift, varying_base = self.interaction.smeared_shifts(tau)

        # (N, Q) replica that does not depend on p.
        fixed_phase = np.exp(1j * self.omega_bin * fixed_shift)
        fixed = _fft0((ew_bin * fixed_phase)[:, None] * self.input_prop)

        # (N, Q, K) replicas at varying_base - p_k/2. Frequency stays on axis 0 so the
        # raw-DFT helpers apply unchanged.
        shifts = varying_base - 0.5 * self.p_nodes
        varying_phase = np.exp(1j * np.outer(self.omega_bin, shifts))
        varying = _fft0(
            ew_bin[:, None, None] * self.input_prop[:, :, None] * varying_phase[:, None]
        )
        sig = self.interaction.smeared_signal(
            fixed[:, :, None], varying, varying[:, :, ::-1]
        )
        psi_q = _ifft0(sig) * self.signal_prop[:, :, None]
        psi_bin = (psi_q * self.weights[:, None]).sum(axis=1)
        return scipy.fft.fftshift(psi_bin, axes=0)

    def trace(self, ew: Complex, *, normalize: bool = True) -> NDArray[np.float64]:
        """Full FROG trace ``(Nomega, Ndelay)`` for spectrum ``ew`` (centred).

        Parameters
        ----------
        ew : numpy.ndarray
            Complex spectrum (centred order).
        normalize : bool, optional
            Scale the trace to unit maximum (default ``True``).
        """
        if self.smearing is not None:
            return self._trace_smeared(ew, normalize=normalize)
        out = np.empty((self.n, self.m), dtype=float)
        for j, tau in enumerate(self.delays):
            out[:, j] = np.abs(self.signal_single(ew, tau)) ** 2
        if normalize:
            out /= out.max()
        return out

    def _trace_smeared(
        self, ew: Complex, *, normalize: bool = True
    ) -> NDArray[np.float64]:
        r"""Trace with smearing: quadrature over ``p``, convolution over ``delta``.

        The measured trace is the incoherent average :math:`\sum_k w_k \int
        \rho(\delta\,|\,p_k)\,T_{p_k}(\omega,\tau-\delta)\,\mathrm d\delta`.
        The inner integral is a convolution along the delay axis, applied by multiplying
        the delay-axis DFT by the analytic Gaussian transform of the conditional
        density — exact, and independent of how many delay steps the kernel spans.
        """
        per_node = np.empty((self.n, self.m, self.p_nodes.size), dtype=float)
        for j, tau in enumerate(self.delays):
            per_node[:, j, :] = np.abs(self.signal_smeared(ew, tau)) ** 2
        # Characteristic function of N(mu_k, sigma^2), shape (M, K):
        #   exp(-i Omega mu - sigma^2 Omega^2 / 2)
        omega_delay = self.delay_omega[:, None]
        kernel = np.exp(
            -1j * omega_delay * self.p_mu[None, :]
            - 0.5 * self.p_sigma**2 * omega_delay**2
        )
        spec: Complex = np.asarray(scipy.fft.fft(per_node, axis=1))
        blurred: Complex = np.asarray(scipy.fft.ifft(spec * kernel[None], axis=1))
        out = blurred.real @ self.p_weights
        if normalize:
            out /= out.max()
        return out

    # -- adjoint -----------------------------------------------------------
    def adjoint_single(self, psi_bar: Complex, tau: float) -> Complex:
        r"""Adjoint of :meth:`signal_single`: map ``psi_bar`` → ``ew_bar`` (centred).

        Requires the forward tape for the same ``tau`` (call
        ``signal_single(ew, tau, record=True)`` first). ``psi_bar`` is the
        Wirtinger cotangent :math:`\partial L/\partial\psi^*`; the return is
        :math:`\partial L/\partial \tilde E^*` (centred order).
        """
        if self.smearing is not None:
            raise NotImplementedError(
                "the hand-derived adjoint does not support geometrical smearing; use "
                "an autodiff solver (lbfgs-ad, lm, lm-optx, lbfgs-optx, cma-es)"
            )
        # The tape fields (tau, test, gate) are written together by signal_single;
        # checking all three lets the type checker treat test/gate as non-None.
        if (
            self._tape_tau is None
            or self._tape_tau != float(tau)
            or self._tape_test is None
            or self._tape_gate is None
        ):
            raise RuntimeError(
                "adjoint_single requires a matching recorded forward pass"
            )
        test, gate = self._tape_test, self._tape_gate
        psi_bar_bin = scipy.fft.ifftshift(psi_bar)

        # adjoint of psi_bin = sum_q w_q signal_prop ⊙ raw_ifft(sig)
        psi_q_bar = self.weights * np.conj(self.signal_prop) * psi_bar_bin[:, None]
        # adjoint of raw_ifft is (1/N) raw_fft
        sig_bar = _fft0(psi_q_bar) / self.n
        test_bar, gate_bar = self.interaction.signal_adjoint(test, gate, sig_bar)
        # adjoint of raw_fft is N raw_ifft
        e0_bar = self.n * _ifft0(test_bar)
        etau_bar = self.n * _ifft0(gate_bar)
        phase = np.exp(1j * self.omega_bin * tau)
        ew_bar_bin = (
            np.conj(self.input_prop) * e0_bar
            + np.conj(phase[:, None] * self.input_prop) * etau_bar
        ).sum(axis=1)
        return scipy.fft.fftshift(ew_bar_bin)


def maketrace(
    omega: ArrayLike,
    delays: ArrayLike,
    ew: Complex,
    interaction: str | Interaction,
    *,
    material: str | None = None,
    thickness: float = 0.0,
    npoints: int = 1,
    omega0: float = 0.0,
    quadrature: str = "gausslegendre",
    smearing: SmearingKernel | None = None,
    normalize: bool = True,
) -> NDArray[np.float64]:
    """Generate a (normalised) FROG trace for spectrum ``ew``.

    Convenience wrapper around :class:`ForwardModel` for tests and examples, and the
    forward-only entry point for synthesising a geometrically smeared trace without
    running a retrieval. Returns an ``(Nomega, Ndelay)`` array. See
    :class:`ForwardModel` for the parameter meanings.
    """
    model = ForwardModel(
        omega,
        delays,
        interaction,
        material=material,
        thickness=thickness,
        npoints=npoints,
        omega0=omega0,
        quadrature=quadrature,
        smearing=smearing,
    )
    return model.trace(np.asarray(ew, dtype=complex), normalize=normalize)
