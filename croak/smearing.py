r"""Geometric (pulse-front-tilt) time smearing in a non-collinear BOXCARS geometry.

In a non-collinear geometry the input beams reach the interaction plane with tilted
pulse fronts, so the relative arrival time between arms varies linearly across the
transverse coordinate :math:`\mathbf r` of the focal spot. The detector integrates over
that coordinate, so the measured trace is an average over a distribution of *internal*
relative delays rather than a single one. This module turns a mask geometry into the
statistics of that distribution; :mod:`croak.forward` and :mod:`croak.forward_jax`
consume it.

The reduction
-------------
Write the third-order signal at focal-plane position :math:`\mathbf r` as the product of
three arm fields, one of them phase-conjugated. A beam apertured at mask position
:math:`\mathbf r_j` and focused by a lens of focal length :math:`f` crosses the focus at
angle :math:`\mathbf r_j/f`, so its pulse front makes it arrive at focal position
:math:`\mathbf r` at time :math:`t - \boldsymbol\alpha_j\cdot\mathbf r` with the
tilt coefficient

.. math:: \boldsymbol\alpha_j = -\mathbf r_j / (f c).

Because :math:`|\mathcal F\{\cdot\}|^2` is blind to a global time origin, and the origin
may be chosen per :math:`\mathbf r`, only **two** scalar combinations of the three
arrival times survive. In croak's own operator conventions they are

.. math::

    \mathrm{PG:}\quad S &= E(t)\,E(t-\tau+\delta+p/2)\,E^*(t-\tau+\delta-p/2) \\
    \mathrm{SD:}\quad S &= E(t+p/2)\,E(t-p/2)\,E^*(t-\tau+\delta)

with, for arms given in the interaction's role order
(:math:`\mathbf r_1,\mathbf r_2,\mathbf r_3`; see :func:`kernel_from_arms`),

.. math::

    \mathrm{PG:}\quad p &= (\alpha_3-\alpha_2)\cdot\mathbf r, &
    \delta &= \bigl(\alpha_1-\tfrac12(\alpha_2+\alpha_3)\bigr)\cdot\mathbf r \\
    \mathrm{SD:}\quad p &= (\alpha_2-\alpha_1)\cdot\mathbf r, &
    \delta &= \bigl(\tfrac12(\alpha_1+\alpha_2)-\alpha_3\bigr)\cdot\mathbf r.

Setting :math:`p=\delta=0` recovers ``test * |gate|**2`` and ``test**2 * conj(gate)``
exactly, so the correction vanishes identically when the kernel is switched off.

``p`` changes the *shape* of the signal and needs an explicit quadrature. ``delta`` is a
**pure translation of the delay axis** (:math:`\tau\to\tau-\delta`), so integrating over
it is a convolution along the measured delay axis, applied once to the finished trace.

The focal weight
----------------
The local signal amplitude is the product of the three focal-plane amplitudes, so the
weight with which position :math:`\mathbf r` contributes to the incoherent sum is
:math:`|A(\mathbf r)|^6`. For a uniformly illuminated circular hole of diameter ``D``,
:math:`A` is the Airy amplitude :math:`2J_1(u)/u` with
:math:`u=\pi D r/(\lambda f)`, and the rms of one Cartesian component of
:math:`\mathbf r` under that weight is

.. math:: \sigma_r = C\,\lambda f / D, \qquad C = 0.245673\ldots

(see :data:`AIRY6_RMS_COEFF`). Since :math:`\boldsymbol\alpha_j\propto 1/f`, the focal
length cancels: the kernel widths depend only on the mask ratio and the wavelength.

Since ``p`` and ``delta`` are both *linear* functionals of :math:`\mathbf r`, replacing
the focal weight by an isotropic Gaussian of the same second moment makes ``(p, delta)``
jointly Gaussian, fully described by :class:`SmearingKernel`. The leading trace
correction is quadratic in ``(p, delta)``, so only those second moments matter to first
order; the true :math:`|A|^6` weight has heavier tails, which
``tests/test_smearing.py`` quantifies.

Validity limits
---------------
* **Full-beam collection.** The incoherent sum over :math:`\mathbf r` follows from
  Parseval and holds only if the detector collects the whole signal beam. Under strong
  spatial filtering the average is partly *coherent* and the effective blur is narrower,
  so the widths computed here are an upper bound for a tightly apertured instrument.
  :mod:`croak.collection` lifts this: it transforms the focal field to transverse
  wavevector before squaring it, so a finite hole is modelled rather than bounded. The
  reduction to :math:`(p,\vartheta)` survives that intact --- the phase the incoherent
  sum discards is exactly :math:`\omega p` --- but the *kernel* in this module remains
  the full-collection form, deliberately: it is what all current results use.
* **Achromatic kernel.** :math:`\sigma\propto\lambda` because the focal spot scales
  with wavelength while the tilt does not; the kernel is evaluated at the carrier only.
  Across a 1 fs deep-UV pulse this is a ±15 % variation of width — a second-order
  refinement.
* **Uniform hole illumination**, beams collimated at the focusing lens, and a medium
  thin compared with the Rayleigh range (so the delay-vs-position map is linear).
* **No spatial chirp** in the input beam.

References
----------
Derivation, the arrangement-dependent widths and the discriminating experiments:
``docs/howto/geometric_smearing.md``. Note that the closed forms differ according
to which arm carries the scanned delay; :func:`square_boxcars_kernel` covers both
cases via its ``interaction`` argument.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from numpy.typing import ArrayLike, NDArray

from .interactions import Interaction, get_interaction

__all__ = [
    "collinear_sd_kernel",
    "AIRY6_RMS_COEFF",
    "deconvolve_delay",
    "SmearingKernel",
    "delay_frequency_grid",
    "kernel_from_arms",
    "square_boxcars_kernel",
    "square_boxcars_delay_width",
    "square_boxcars_spacing",
]

#: RMS of one Cartesian component of the focal position under the :math:`|A|^6` weight,
#: in units of :math:`\lambda f / D`. Obtained from the moments of the Airy amplitude
#: :math:`g(u)=(2J_1(u)/u)^6`::
#:
#:     <|u|^2> = int_0^inf u^3 g du / int_0^inf u g du = 1.19137
#:     sigma_u = sqrt(<|u|^2> / 2) = 0.771805        (one Cartesian component)
#:     C       = sigma_u / pi = 0.245673             (u = pi D r / (lambda f))
#:
#: Both integrals converge: :math:`g\sim u^{-9}` at large ``u``.
#: ``tests/test_smearing.py`` recomputes this from scipy's Bessel functions.
AIRY6_RMS_COEFF = 0.24567310004243084

# Speed of light in vacuum (m/s). Local copy so this module stays numpy-only.
_C_LIGHT = 299792458.0


@dataclass(frozen=True)
class SmearingKernel:
    r"""Bivariate-Gaussian geometric-smearing kernel in ``(p, delta)``.

    ``p`` splits the two unconjugated replicas (PG: the two gate replicas) and changes
    the *shape* of the nonlinear signal; ``delta`` offsets the conjugated replica and is
    a **pure translation of the delay axis**. Both are zero-mean.

    Parameters
    ----------
    sigma_p : float
        RMS of the shape parameter ``p`` (s). Must be ``>= 0``.
    sigma_delta : float
        RMS of the delay offset ``delta`` (s). Must be ``>= 0``.
    rho : float, optional
        Correlation coefficient between ``p`` and ``delta``, in ``(-1, 1)``. It is zero
        only for particular mask layouts (see :func:`square_boxcars_kernel`).
    npoints : int, optional
        Number of Gauss–Hermite nodes along ``p`` (default 5). The ``delta`` integral is
        exact and needs no nodes. Cost scales as ``(npoints + 1) / 2`` times the
        unsmeared model.

    Examples
    --------
    >>> k = SmearingKernel(sigma_p=0.32e-15, sigma_delta=0.36e-15, rho=0.447)
    >>> p, w = k.nodes_weights()
    >>> float(w.sum())
    1.0
    >>> bool(np.allclose(p, -p[::-1]))
    True
    """

    sigma_p: float
    sigma_delta: float
    rho: float = 0.0
    npoints: int = 5

    def __post_init__(self) -> None:
        """Validate the widths, the correlation and the node count."""
        if self.sigma_p < 0.0 or self.sigma_delta < 0.0:
            raise ValueError(
                f"smearing widths must be non-negative, got sigma_p={self.sigma_p!r}, "
                f"sigma_delta={self.sigma_delta!r}"
            )
        if not -1.0 < self.rho < 1.0:
            raise ValueError(f"rho must lie in (-1, 1), got {self.rho!r}")
        if self.npoints < 1:
            raise ValueError(f"npoints must be >= 1, got {self.npoints!r}")

    def nodes_weights(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""Gauss–Hermite nodes and normalised weights for the ``p`` integral.

        Uses the *probabilists'* Hermite rule, whose weight function is
        :math:`e^{-x^2/2}`, so the nodes are simply ``sigma_p * x``. The nodes are
        forced to be exactly antisymmetric (``p[k] == -p[-1-k]``) and the weights
        exactly symmetric, which the forward model relies on to reuse each shifted
        field twice.

        Returns
        -------
        tuple of numpy.ndarray
            ``(p, w)``, each of length :attr:`npoints`, with ``w.sum() == 1``.
        """
        if self.sigma_p == 0.0:
            # Degenerate kernel: p is deterministic, so one node carries all the weight.
            return np.zeros(1), np.ones(1)
        x, w = hermegauss(self.npoints)
        # hermegauss builds the nodes from an eigenvalue problem, so the exact ± pairing
        # can be lost in the last ulp. Symmetrising restores it, which makes the
        # field-reuse trick in the forward model exact rather than approximate.
        x = 0.5 * (x - x[::-1])
        w = 0.5 * (w + w[::-1])
        return self.sigma_p * x, w / math.sqrt(2.0 * math.pi)

    def conditional_delay(
        self, p: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], float]:
        r"""Mean and standard deviation of ``delta`` conditioned on ``p``.

        For a bivariate normal, ``delta | p`` is normal with mean
        :math:`\rho\,(\sigma_\delta/\sigma_p)\,p` and standard deviation
        :math:`\sigma_\delta\sqrt{1-\rho^2}`. Splitting the joint integral this way lets
        the ``delta`` direction stay an exact convolution even when the two are
        correlated.

        Parameters
        ----------
        p : numpy.ndarray
            Quadrature nodes along ``p`` (s).

        Returns
        -------
        tuple
            ``(mu, sigma)`` — the per-node conditional means (s, same shape as ``p``)
            and the single conditional standard deviation (s).
        """
        if self.sigma_p == 0.0:
            return np.zeros_like(p), self.sigma_delta
        mu = self.rho * (self.sigma_delta / self.sigma_p) * p
        sigma = self.sigma_delta * math.sqrt(1.0 - self.rho**2)
        return mu, sigma

    def scaled(self, factor: float) -> SmearingKernel:
        """Return a copy with both widths multiplied by ``factor``.

        Scaling the kernel is equivalent to scaling the mask ratio ``d/D``, which is the
        single physical lever on the effect. This is the parameterisation used when the
        smearing width is fitted.

        Parameters
        ----------
        factor : float
            Non-negative multiplier on ``sigma_p`` and ``sigma_delta``.
        """
        if factor < 0.0:
            raise ValueError(f"scale factor must be non-negative, got {factor!r}")
        return replace(
            self, sigma_p=self.sigma_p * factor, sigma_delta=self.sigma_delta * factor
        )


def delay_frequency_grid(delays: ArrayLike, sigma_delta: float) -> NDArray[np.float64]:
    r"""Angular-frequency grid conjugate to a **uniform** delay axis.

    The ``delta`` integral is a convolution along ``tau``, and it is applied by
    multiplying the delay-axis DFT of the trace by the analytic Gaussian transform
    :math:`e^{-i\Omega\mu-\sigma^2\Omega^2/2}`. Doing it in the transform domain rather
    than sampling the kernel on the delay grid makes the operation exact for any trace
    that is Nyquist-sampled in ``tau`` — which a measurable trace necessarily is — so
    there is no requirement that the kernel span several delay steps.

    The convolution is circular. That is harmless because the wrap-around weight is
    :math:`\exp[-(\text{delay span}/\sigma_\delta)^2/2]`; this function raises if the
    kernel is not comfortably narrower than the scan, which is the only regime where it
    would matter.

    Parameters
    ----------
    delays : array_like
        Delay axis (s). Must be uniformly spaced and have at least two points.
    sigma_delta : float
        RMS delay offset (s), used only for the sanity check above.

    Returns
    -------
    numpy.ndarray
        ``Omega`` (rad/s) in DFT-bin order, the same length as ``delays``.

    Raises
    ------
    ValueError
        If the delay axis is too short, is not uniformly spaced, or is not wide enough
        to contain the kernel.
    """
    tau = np.asarray(delays, dtype=float)
    if tau.size < 2:
        raise ValueError(
            "geometrical smearing needs a delay axis with at least two points"
        )
    steps = np.diff(tau)
    dtau = float(np.mean(steps))
    if not np.allclose(steps, dtau, rtol=1e-9, atol=0.0):
        raise ValueError(
            "geometrical smearing requires a uniformly spaced delay axis (the delay "
            "integral is applied as a convolution along tau); resample the trace first"
        )
    span = float(tau[-1] - tau[0])
    if sigma_delta > 0.0 and abs(span) < 12.0 * sigma_delta:
        raise ValueError(
            f"delay span {abs(span):.3e} s is too narrow for a smearing kernel of "
            f"sigma_delta = {sigma_delta:.3e} s; the delay-axis convolution needs a "
            f"span of at least 12 sigma"
        )
    return np.asarray(2.0 * np.pi * np.fft.fftfreq(tau.size, d=dtau), dtype=float)


def _pd_vectors(
    interaction: str | Interaction,
    r1: ArrayLike,
    r2: ArrayLike,
    r3: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Mask-plane coefficient vectors for ``p`` and ``delta`` (metres).

    The tilt coefficients are :math:`\alpha_j=-\mathbf r_j/(fc)`, so every combination
    below carries the same overall factor :math:`-1/(fc)`; dropping it changes neither
    the magnitudes (which are rescaled by ``sigma_r``) nor the correlation (which only
    sees the *relative* orientation).
    """
    inter = get_interaction(interaction)
    a = np.asarray(r1, dtype=float)
    b = np.asarray(r2, dtype=float)
    c = np.asarray(r3, dtype=float)
    for name, vec in (("r1", a), ("r2", b), ("r3", c)):
        if vec.shape != (2,):
            raise ValueError(f"{name} must be a 2-vector (x, y) in metres, got {vec!r}")
    if inter.name == "pg":
        # arms: (probe, gate unconjugated, gate conjugated)
        return c - b, a - 0.5 * (b + c)
    if inter.name == "sd":
        # arms: (unconjugated A, unconjugated B, conjugated)
        return b - a, 0.5 * (a + b) - c
    raise ValueError(
        f"geometrical smearing is not supported for the {inter.name.upper()} "
        f"interaction; only PG (TG) and SD have the required three-arm form"
    )


def collinear_sd_kernel(
    *,
    hole_diameter: float,
    hole_spacing: float,
    wavelength: float,
    npoints: int = 5,
) -> SmearingKernel:
    r"""Smearing kernel for a **two-beam** self-diffraction geometry.

    This is the SD an experiment actually builds: two collinear apertures, the
    probe entering the nonlinearity twice and the gate once with conjugation,
    signal at :math:`2k_E - k_G`. It is NOT what
    :func:`square_boxcars_kernel` with ``"sd"`` returns — that closed form
    assumes the two unconjugated arms come from *separate* holes, a three-hole
    layout, and gives a spurious ``sigma_p`` here.

    **The p channel vanishes identically.** ``p`` is proportional to
    :math:`\alpha_A - \alpha_B`, the difference of the two unconjugated arms'
    tilts, and in a two-beam SD those arms are the same beam. So

    .. math::

        \sigma_p = 0, \qquad
        \sigma_\vartheta = 0.49135\,(d/D)\,\lambda/c, \qquad \rho = 0,

    verified against :func:`kernel_from_arms` to six figures over
    ``d/D = 0.75``–``1.25``. Two consequences matter for retrieval. The
    smearing reduces to a pure one-dimensional convolution along the delay
    axis, so it is exactly deconvolvable — unlike the TG kernel, whose
    ``rho = 1/sqrt(5)`` makes it a correlated two-dimensional shear. And the p
    channel is the one that shifts the frequency axis in proportion to the gate
    chirp, so its absence removes the mechanism behind the chirped-arm
    difficulties of the TG geometry.

    The width coefficient is numerically the same 0.49135 that appears as
    ``sigma_p`` for PG, which is a coincidence of the geometry (both reduce to
    one arm-separation times the focal weight), not a shared mechanism.

    Parameters
    ----------
    hole_diameter : float
        Aperture diameter ``D`` (m).
    hole_spacing : float
        **Edge-to-edge** gap between the two holes (m); the beam offset from
        the axis is ``d = (spacing + D) / 2`` and their centre-to-centre
        separation is ``2d``.
    wavelength : float
        Carrier wavelength (m) — use the measured trace carrier.
    npoints : int, optional
        Gauss--Hermite nodes. Retained for interface symmetry; with
        ``sigma_p = 0`` the p quadrature is a single node regardless.

    Returns
    -------
    SmearingKernel
    """
    d = 0.5 * (hole_spacing + hole_diameter)
    return kernel_from_arms(
        (-d, 0.0),
        (-d, 0.0),
        (d, 0.0),
        interaction="sd",
        hole_diameter=hole_diameter,
        wavelength=wavelength,
        npoints=npoints,
    )


def kernel_from_arms(
    r1: ArrayLike,
    r2: ArrayLike,
    r3: ArrayLike,
    *,
    interaction: str | Interaction,
    hole_diameter: float,
    wavelength: float,
    npoints: int = 5,
) -> SmearingKernel:
    r"""Build a :class:`SmearingKernel` from three mask hole positions.

    Fully general: any three-arm layout, including unequal spacings. Positions are given
    in the **role order of the interaction**, which is what fixes the meaning of ``p``
    and ``delta``:

    * ``"pg"`` — ``(probe, gate unconjugated, gate conjugated)``. The probe is the arm
      that appears undelayed in ``E(t) |E(t - tau)|**2``.
    * ``"sd"`` — ``(unconjugated A, unconjugated B, conjugated)``. The conjugated arm is
      the delayed one in ``E(t)**2 conj(E(t - tau))``.

    Parameters
    ----------
    r1, r2, r3 : array_like
        Mask hole centres ``(x, y)`` in metres, in the role order above. Only
        differences matter, so the origin is arbitrary.
    interaction : str or Interaction
        ``"pg"`` (also used for TG) or ``"sd"``. SHG raises.
    hole_diameter : float
        Mask hole diameter (m); sets the focal spot size.
    wavelength : float
        Carrier wavelength (m).
    npoints : int, optional
        Gauss–Hermite nodes along ``p`` (default 5).

    Returns
    -------
    SmearingKernel

    Notes
    -----
    The focal length never appears: the pulse-front tilt scales as :math:`1/f` and the
    focal spot as :math:`f`, so the product is focal-length independent.
    """
    if hole_diameter <= 0.0:
        raise ValueError(f"hole_diameter must be positive, got {hole_diameter!r}")
    if wavelength <= 0.0:
        raise ValueError(f"wavelength must be positive, got {wavelength!r}")
    p_vec, d_vec = _pd_vectors(interaction, r1, r2, r3)
    # sigma_r = C lambda f / D and alpha ~ 1/(f c), so the product is
    #   sigma = C (lambda / c) |vector| / D
    # with the focal length cancelled out.
    scale = AIRY6_RMS_COEFF * wavelength / (_C_LIGHT * hole_diameter)
    norm_p = float(np.hypot(*p_vec))
    norm_d = float(np.hypot(*d_vec))
    rho = 0.0
    if norm_p > 0.0 and norm_d > 0.0:
        rho = float(np.dot(p_vec, d_vec) / (norm_p * norm_d))
        # Guard against |rho| == 1 from a degenerate (collinear) layout, which would
        # make the conditional delay distribution singular.
        rho = float(np.clip(rho, -1.0 + 1e-12, 1.0 - 1e-12))
    return SmearingKernel(
        sigma_p=scale * norm_p,
        sigma_delta=scale * norm_d,
        rho=rho,
        npoints=npoints,
    )


def _square_boxcars_arms(
    interaction: str | Interaction, d: float
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Arm positions for a folded square BOXCARS mask, in interaction role order.

    Holes sit at the four corners ``(±d, ±d)``. The signal emerges at
    ``r_s = r_a + r_b - r_conj``, which for a square forces the conjugated arm to be the
    corner *diagonally opposite* the signal hole. Taking the signal at ``(-d, -d)``
    puts the conjugated arm at ``(+d, +d)`` and the two remaining corners carry the
    unconjugated arms.

    The two supported layouts differ only in which arm the delay stage sits on:

    * ``"pg"`` — the delay is on an unconjugated arm (or, equivalently, on the gate
      pair). This is the arrangement of the ``ModelPNPS`` reference simulation.
    * ``"sd"`` — the delay is on the conjugated arm.
    """
    inter = get_interaction(interaction)
    conj = (d, d)
    if inter.name == "pg":
        # (probe, gate unconjugated, gate conjugated). Which of the two unconjugated
        # corners plays the probe is immaterial: the square's mirror symmetry gives the
        # same widths and the same correlation either way.
        return (-d, d), (d, -d), conj
    if inter.name == "sd":
        # (unconjugated A, unconjugated B, conjugated)
        return (d, -d), (-d, d), conj
    raise ValueError(
        f"geometrical smearing is not supported for the {inter.name.upper()} "
        f"interaction; only PG (TG) and SD have the required three-arm form"
    )


def square_boxcars_kernel(
    interaction: str | Interaction,
    *,
    hole_diameter: float,
    hole_spacing: float,
    wavelength: float,
    npoints: int = 5,
    reverse_delay: bool = False,
) -> SmearingKernel:
    r"""Kernel for the standard folded square BOXCARS mask.

    Four holes of diameter ``D`` at the corners ``(±d, ±d)``, with
    ``d = (hole_spacing + D) / 2``, uniformly illuminated. The closed forms are

    ==============  ===========================  ===========================  =========
    interaction     ``sigma_p``                  ``sigma_delta``              ``rho``
    ==============  ===========================  ===========================  =========
    ``"pg"``        ``0.49135 (d/D) lambda/c``   ``0.54933 (d/D) lambda/c``   ``1/√5``
    ``"sd"``        ``0.69487 (d/D) lambda/c``   ``0.34743 (d/D) lambda/c``   ``0``
    ==============  ===========================  ===========================  =========

    so the only physical lever is the mask ratio ``d/D`` (and the wavelength).

    Parameters
    ----------
    interaction : str or Interaction
        ``"pg"`` (also used for TG) or ``"sd"``; this also selects which arm carries the
        scanned delay — see :func:`kernel_from_arms`.
    hole_diameter : float
        Mask hole diameter ``D`` (m).
    hole_spacing : float
        **Edge-to-edge** gap between adjacent holes (m), so ``d = (spacing + D) / 2``.
    wavelength : float
        Carrier wavelength (m).
    npoints : int, optional
        Gauss–Hermite nodes along ``p`` (default 5).
    reverse_delay : bool, optional
        Negate ``rho`` (default ``False``). The sign of the ``p``–``delta`` correlation
        is tied to the orientation of the delay axis: a trace whose ``tau`` runs the
        other way relative to croak's operator needs the opposite sign. It affects only
        a small ``tau``-asymmetry of the smeared trace, not the bulk broadening.

    Returns
    -------
    SmearingKernel

    Examples
    --------
    An example DUV geometry (1 mm holes, 0.5 mm edge-to-edge, 260 nm):

    >>> k = square_boxcars_kernel("pg", hole_diameter=1e-3, hole_spacing=0.5e-3,
    ...                           wavelength=260e-9)
    >>> round(k.sigma_delta * 1e15, 3)
    0.357
    >>> round(k.sigma_p * 1e15, 3)
    0.32
    """
    if hole_spacing < 0.0:
        raise ValueError(f"hole_spacing must be non-negative, got {hole_spacing!r}")
    d = 0.5 * (hole_spacing + hole_diameter)
    r1, r2, r3 = _square_boxcars_arms(interaction, d)
    kernel = kernel_from_arms(
        r1,
        r2,
        r3,
        interaction=interaction,
        hole_diameter=hole_diameter,
        wavelength=wavelength,
        npoints=npoints,
    )
    if reverse_delay:
        kernel = replace(kernel, rho=-kernel.rho)
    return kernel


def deconvolve_delay(
    trace: ArrayLike,
    delays: ArrayLike,
    sigma: float,
    *,
    reg: float = 1e-3,
) -> NDArray[np.float64]:
    r"""Remove a Gaussian blur of width ``sigma`` from the delay axis of a trace.

    The :math:`\vartheta` channel of the smearing kernel enters the trace as a
    convolution along delay, and it factorises out of the :math:`p` mixture
    exactly. Writing the joint kernel as :math:`p` times :math:`\vartheta`
    conditioned on :math:`p` (:meth:`SmearingKernel.conditional_delay`), the
    conditional mean :math:`\mu(p)` is a per-node delay shift while the
    conditional width :math:`\sigma_c = \sigma_\vartheta\sqrt{1-\rho^2}` is
    the *same* for every node. It therefore commutes with the sum:

    .. math::
        \tilde T(\omega,\tau) = \mathcal N(0,\sigma_c) \ast
        \Big[\textstyle\sum_k w_k\, \tilde T_{p_k}(\omega, \tau - \mu_k)\Big].

    Deconvolving a measurement by :math:`\sigma_c` thus leaves exactly the
    :math:`p` mixture with its delay shifts --- no approximation. Note the width
    to remove is the **conditional** one, not the marginal
    :math:`\sigma_\vartheta`; using the marginal over-corrects by
    :math:`1/\sqrt{1-\rho^2}` (12\% for the square BOXCARS
    :math:`\rho = 1/\sqrt5`).

    This matters because it moves the delay-axis half of the smearing out of the
    forward model and into preprocessing, where any algorithm can use it ---
    including projection-based ones such as COPRA, whose local step is per-delay
    and cannot represent a convolution that couples delays.

    Deconvolution is ill-posed: dividing by a Gaussian amplifies high delay
    frequencies without bound, so a Wiener filter is used, trading exactness for
    noise control via ``reg``.

    Parameters
    ----------
    trace : array_like
        Trace ``(Nomega, Ndelay)``.
    delays : array_like
        Delay axis (s). Must be uniformly spaced.
    sigma : float
        Gaussian width to remove (s). Zero or negative returns the trace
        unchanged.
    reg : float, optional
        Wiener regularisation as a fraction of the filter's peak power. Larger
        leaves more blur but suppresses noise amplification; the default 1e-3
        recovers a noiseless blur to about a part in 1e3.

    Returns
    -------
    numpy.ndarray
        Deconvolved trace, same shape.
    """
    t = np.asarray(trace, dtype=float)
    d = np.asarray(delays, dtype=float)
    if t.ndim != 2:
        raise ValueError(f"trace must be 2-D (Nomega, Ndelay), got shape {t.shape}")
    if d.size != t.shape[1]:
        raise ValueError(
            f"delays has {d.size} points but the trace has {t.shape[1]} columns"
        )
    if sigma <= 0.0:
        return t.copy()
    steps = np.diff(d)
    if steps.size and not np.allclose(steps, steps[0], rtol=1e-6, atol=0.0):
        raise ValueError(
            "deconvolve_delay needs a uniformly spaced delay axis; resample first"
        )
    dt = float(steps[0]) if steps.size else 1.0
    n = t.shape[1]
    omega = 2.0 * np.pi * np.fft.rfftfreq(n, d=abs(dt))
    # Transfer function of the blur, and its Wiener inverse.
    h = np.exp(-0.5 * (sigma * omega) ** 2)
    inv = h / (h * h + reg)
    return np.fft.irfft(np.fft.rfft(t, axis=1) * inv[None, :], n=n, axis=1)


def square_boxcars_delay_width(
    interaction: str | Interaction,
    *,
    hole_diameter: float,
    hole_spacing: float,
    wavelength: float,
) -> float:
    """Delay-offset width ``sigma_delta`` (s) of a square BOXCARS mask.

    Thin wrapper over :func:`square_boxcars_kernel` for callers (the GUI) that need the
    geometry → width map on its own. See that function for the parameters.
    """
    return square_boxcars_kernel(
        interaction,
        hole_diameter=hole_diameter,
        hole_spacing=hole_spacing,
        wavelength=wavelength,
    ).sigma_delta


def square_boxcars_spacing(
    interaction: str | Interaction,
    *,
    hole_diameter: float,
    sigma_delta: float,
    wavelength: float,
) -> float:
    """Invert :func:`square_boxcars_delay_width` for the edge-to-edge hole spacing.

    ``sigma_delta`` is proportional to ``d = (spacing + D) / 2``, so a single reference
    evaluation fixes the slope and no constants are duplicated. The result may be
    negative when the requested width is below the geometric floor reached with the
    holes touching (``spacing = 0``); callers should clamp.

    Parameters
    ----------
    interaction : str or Interaction
        ``"pg"`` or ``"sd"``.
    hole_diameter : float
        Mask hole diameter ``D`` (m).
    sigma_delta : float
        Target delay-offset width (s).
    wavelength : float
        Carrier wavelength (m).

    Returns
    -------
    float
        Edge-to-edge hole spacing (m).
    """
    # Reference point: spacing = D gives d = D.
    reference = square_boxcars_delay_width(
        interaction,
        hole_diameter=hole_diameter,
        hole_spacing=hole_diameter,
        wavelength=wavelength,
    )
    d = hole_diameter * sigma_delta / reference
    return 2.0 * d - hole_diameter
