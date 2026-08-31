r"""Chromatic focal mixture: the smearing kernel without its achromatic reduction.

:mod:`croak.smearing` reduces the incoherent sum over focal position
:math:`\mathbf r` to two arrival-time parameters :math:`(p,\vartheta)` carried by a
bivariate Gaussian, with the focal weight :math:`|A(\mathbf r)|^6` evaluated at the
carrier wavelength alone. That reduction keeps the *phases* the focal average imposes
and discards its *amplitude* structure, because the mask maps to transverse wavevector
as :math:`x = k_\perp z_{\text{mask}} c/\omega`: the focal amplitude
:math:`A(r,\omega)` therefore depends on the product :math:`r\omega`, and the local
field at radius :math:`r` is the retrieved field passed through a
frequency-dependent filter. Different focal positions see *different spectra*, hence
different local pulses, and the measured trace is a generation-weighted mixture of them.

For a transform-limited pulse that costs a few percent of duration. For a chirped one
it is much larger, because reshaping the spectrum changes how far the local pulse
stretches: full 3D simulations of an apertured instrument under-report a
:math:`+2\,\mathrm{fs}^2` pulse by 23%, against 8% for an otherwise identical
instrument whose beams carry no chromatic transverse structure.

This module keeps the mixture explicit. The focal plane is sampled by a quadrature
and each node carries

* the chromatic amplitude filter :math:`A(r,\omega)`, applied to the field so that
  the three-arm product supplies :math:`A^3` and the trace :math:`|A|^6`
  automatically --- the weight of the reduced model, but now inside the spectrum
  rather than outside it; and
* the *definite* arrival-time offsets :math:`(p,\vartheta)` of that node, in place
  of the reduced model's distribution over them.

The reduced kernel is recovered when :math:`A` is held at the carrier: the second
moments of :math:`(p,\vartheta)` under the :math:`|A|^6` weight are exactly the
closed-form :math:`\sigma_p`, :math:`\sigma_\vartheta`, :math:`\rho` of
:mod:`croak.smearing`, which :func:`reduced_moments` checks.

Cost is one forward evaluation per node instead of one per Gauss--Hermite
:math:`p`-node, so a few tens of times the reduced model. That is the price of not
approximating the mixture.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import j0, j1

from . import materials
from .collection import CollectionAperture
from .maths import wlfreq

__all__ = [
    "EvolveSpec",
    "FocalMixture",
    "airy_amplitude",
    "evolved_arm_filters",
    "evolved_profile_table",
    "focal_mixture",
    "mixture_from_nodes",
    "reduced_moments",
]

_C_LIGHT = 299_792_458.0

#: Hole-centre offsets in units of the spacing ``d``, in the arm order the PG/TG
#: kernel expects: arm 1 is the test (probe) beam and arms 2, 3 are the gate pair
#: that ``p`` splits. The layout matches the simulated instrument -- test at
#: ``(-1, +1)``, gates at ``(+1, -1)`` and ``(+1, +1)``, signal at ``(-1, -1)`` --
#: so the conjugated gate sits DIAGONALLY OPPOSITE the signal, which is the
#: arrangement phase matching selects.
#:
#: This ordering is load-bearing and easy to get wrong: permuting the arms turns the
#: PG constants into the SD ones (sigma_p 0.695, sigma_theta 0.347, rho 0) or flips
#: the sign of rho. :func:`reduced_moments` exists to catch exactly that.
_ARMS_PG = ((-1.0, +1.0), (+1.0, -1.0), (+1.0, +1.0))


def airy_amplitude(u: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Airy amplitude :math:`2J_1(u)/u`, with the removable singularity handled.

    Parameters
    ----------
    u : ndarray
        Reduced focal coordinate :math:`u = \pi D r/(\lambda f) = D r \omega/(2 f c)`.

    Returns
    -------
    ndarray
        Amplitude, normalised to 1 at ``u = 0``.

    Warnings
    --------
    The normalisation is unit **peak**, not unit **power**. Because the focal
    spot shrinks as :math:`1/\omega`, the power integral carries a spurious
    frequency dependence,

    .. math:: \int |A|^2 \, \mathrm{d}^2 r \;\propto\; \omega^{-2},

    whereas an unclipped aperture physically passes the same power at every
    frequency. Any quantity formed by integrating over the radial coordinate
    therefore inherits a factor of :math:`\omega^{-2}` that is a convention of
    this function, not physics. Weighting a "beamlet truth" by
    :math:`\sum_k a_k |A_k(\omega)|^2`, for instance, imposes an order-of-
    magnitude chromatic transmission across a few-femtosecond DUV band and
    shortens the apparent pulse; the correct truth for a mixture built from
    this amplitude is the *unfiltered* source spectrum.

    This does **not** affect the mixture physics. At fixed :math:`\omega` the
    relative weighting between radii---the chromatic focal structure the
    mixture exists to represent---is exactly :math:`|A(r,\omega)|^2` and is
    correct. The :math:`\omega^{-2}` is a *global* spectral factor on the
    mixture output, which per-frequency response factors (``R_omega``) absorb
    in retrieval. It matters only when a mixture trace or spectrum is compared
    to an external reference in absolute terms.

    Note also what the mixture consequently does *not* contain: an unclipped
    hole passes all its power, so this amplitude carries chromatic focal
    *structure* but no chromatic *vignetting*. An instrument whose mask clips a
    guided mode has both.
    """
    u = np.asarray(u, dtype=float)
    out = np.ones_like(u)
    nz = u != 0.0
    out[nz] = 2.0 * j1(u[nz]) / u[nz]
    return out


@dataclass(frozen=True)
class EvolveSpec:
    """Slab parameters for the depth-resolved mixture (fc-z).

    The evolved per-arm filters need the depth-quadrature nodes, which are set
    by the slab; storing the parameters here (rather than a prebuilt table)
    lets the table be built at model-build time on the model's own frequency
    grid, and lets :func:`croak.forward_jax.make_param_trace_fn` verify that
    the mixture and the trace map agree about the slab — a silent mismatch
    between the filter depths and the propagation depths would be a wrong
    model with no error message.

    Attributes
    ----------
    material : str
        Slab material name (must match the trace map's ``material``).
    thickness : float
        Slab thickness in metres (must match the trace map's ``thickness``).
    npoints : int
        Depth-quadrature node count (must match the trace map's ``npoints``).
    quadrature : str
        Depth-quadrature rule (must match the trace map's ``quadrature``).
    """

    material: str
    thickness: float
    npoints: int
    quadrature: str = "gausslegendre"


@dataclass(frozen=True)
class FocalMixture:
    """Quadrature over the focal plane, carrying the chromatic amplitude filter.

    Attributes
    ----------
    r : ndarray, shape (K,)
        Radial coordinate of each node (m).
    phi : ndarray, shape (K,)
        Azimuth of each node (rad).
    area : ndarray, shape (K,)
        Quadrature weight including the ``r dr dphi`` Jacobian (m^2). The
        :math:`|A|^6` weight is NOT folded in here: it arrives through
        :meth:`spectral_filter`, which is the whole point of the model.
    p : ndarray, shape (K,)
        Gate-splitting arrival offset at each node (s).
    theta : ndarray, shape (K,)
        Delay-axis arrival offset at each node (s).
    hole_diameter : float
        Aperture diameter (m).
    f_foc : float
        Focal length (m).
    arms : ndarray, shape (3, 2), or None
        Mask hole centres (m) in the interaction's role order, as passed to
        :func:`croak.smearing.kernel_from_arms`. The incoherent sum never needs them
        -- only the differences that make ``p`` and ``theta`` -- but a finite
        collection aperture does, because the absolute tilts fix where the signal goes
        (see :mod:`croak.collection`). ``None`` for a mixture built without them.
    collection : CollectionAperture or None
        Finite collection aperture. ``None`` (default) is the full-beam incoherent sum
        the model has always done; anything else transforms the focal field to k,
        windows it and integrates there.
    evolve : EvolveSpec or None
        Depth-resolved mixture (fc-z). ``None`` (default) is the entrance-face
        model: one filter per node, shared by all three arms and all depth
        nodes. An :class:`EvolveSpec` makes each arm at each depth node carry
        its own linearly evolved complex profile — see
        :func:`evolved_arm_filters`. The ``(p, theta)`` offsets stay
        depth-independent either way (exact: transverse wavevector is
        conserved).
    """

    r: NDArray[np.float64]
    phi: NDArray[np.float64]
    area: NDArray[np.float64]
    p: NDArray[np.float64]
    theta: NDArray[np.float64]
    hole_diameter: float
    f_foc: float
    profile: str = "airy"
    waist: float = 0.0
    arms: NDArray[np.float64] | None = None
    collection: CollectionAperture | None = None
    evolve: EvolveSpec | None = None

    @property
    def nodes(self) -> int:
        """Number of quadrature nodes."""
        return int(self.r.size)

    @property
    def signal_position(self) -> NDArray[np.float64]:
        r"""Mask-plane position of the phase-matched signal, :math:`r_1+r_2-r_3` (m).

        Phase matching sends the third-order signal out along
        :math:`\mathbf k_1+\mathbf k_2-\mathbf k_3`, which maps back to this mask
        position. For the square BOXCARS layout it is the fourth corner -- the one the
        collection hole sits on.

        Raises
        ------
        ValueError
            If the mixture was built without ``arms``.
        """
        if self.arms is None:
            raise ValueError(
                "this mixture has no arm positions, so the signal direction is unknown"
            )
        return self.arms[0] + self.arms[1] - self.arms[2]

    def spectral_filter(self, omega: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Amplitude filter :math:`A(r_k, \omega)` for every node.

        Parameters
        ----------
        omega : ndarray, shape (N,)
            ABSOLUTE angular frequency (rad/s), not the centred grid: the chromatic
            scaling is in :math:`r\omega`, so an offset grid would mis-scale it.

        Returns
        -------
        ndarray, shape (K, N)
            One filter per node. Apply to the field once; the three-arm product then
            supplies :math:`A^3` and the trace :math:`|A|^6`.
        """
        omega = np.asarray(omega, dtype=float)
        if self.profile == "gaussian":
            # A Gaussian beam's focal profile is ACHROMATIC: its waist is set by
            # the beam, not by an aperture that k-space scales with frequency.
            # This is the whole point of the Gaussian control -- no r-structure.
            return np.repeat(
                np.exp(-(self.r[:, None] ** 2) / self.waist**2), omega.size, axis=1
            )
        u = (
            self.hole_diameter
            * self.r[:, None]
            * omega[None, :]
            / (2.0 * self.f_foc * _C_LIGHT)
        )
        return airy_amplitude(u)

    def reduced_weight(self, wavelength: float) -> NDArray[np.float64]:
        r""":math:`|A|^6` weight at one wavelength, times the area element.

        This is what the reduced kernel of :mod:`croak.smearing` integrates against;
        it exists here so the two models can be compared on identical quadratures.
        """
        omega = 2.0 * np.pi * _C_LIGHT / float(wavelength)
        a = self.spectral_filter(np.array([omega]))[:, 0]
        return self.area * a**6


def focal_mixture(
    *,
    hole_diameter: float,
    hole_spacing: float,
    f_foc: float,
    wavelength: float,
    n_radial: int = 24,
    n_azimuth: int = 16,
    r_max_units: float = 6.0,
    profile: str = "airy",
    waist: float | None = None,
    collection: CollectionAperture | None = None,
    evolve_profiles: bool = False,
    material: str | None = None,
    thickness: float = 0.0,
    npoints: int = 0,
    quadrature: str = "gausslegendre",
) -> FocalMixture:
    r"""Build a focal-plane quadrature for the square BOXCARS PG/TG geometry.

    Parameters
    ----------
    hole_diameter, hole_spacing : float
        Aperture diameter and edge-to-edge gap (m). The hole-centre offset is
        ``d = (hole_spacing + hole_diameter)/2``, matching :mod:`croak.smearing`.
    f_foc : float
        Focal length (m).
    wavelength : float
        Carrier wavelength (m), used ONLY to set the radial extent of the
        quadrature in physical units. The model itself is chromatic.
    n_radial, n_azimuth : int, optional
        Quadrature nodes. The azimuthal integrand is smooth and periodic, so uniform
        sampling converges spectrally and 16 nodes is generous; the radial integrand
        carries the Airy oscillations and wants more.
    r_max_units : float, optional
        Radial cut in units of :math:`\lambda f/D`. :math:`|A|^6` falls as
        :math:`u^{-9}`, so 6 captures the weight to ~1e-4. Note that a *coherent*
        collection integrand is not :math:`|A|^6`, so a mixture carrying a
        ``collection`` may want more radial nodes -- see :mod:`croak.collection`.
    collection : CollectionAperture or None, optional
        Finite collection aperture (:mod:`croak.collection`). ``None`` (default) keeps
        the full-beam incoherent sum.
    evolve_profiles : bool, optional
        Depth-resolved mixture (fc-z): each arm at each depth node carries its
        own linearly evolved complex profile, evaluated at the arm's walked
        radius, instead of the single shared entrance-face filter. Requires
        the slab parameters below, which must MATCH the ones later passed to
        :func:`croak.forward_jax.make_param_trace_fn` (verified there). See
        :func:`evolved_arm_filters` for the physics and conventions.
    material : str or None, optional
        Slab material for ``evolve_profiles`` (in-glass evolution and walk).
    thickness : float, optional
        Slab thickness in metres for ``evolve_profiles``.
    npoints : int, optional
        Depth-quadrature node count for ``evolve_profiles``.
    quadrature : str, optional
        Depth-quadrature rule for ``evolve_profiles``.

    Returns
    -------
    FocalMixture
    """
    if n_radial < 2 or n_azimuth < 2:
        raise ValueError("n_radial and n_azimuth must both be >= 2")
    evolve = None
    if evolve_profiles:
        if material is None or thickness <= 0.0 or npoints < 1:
            raise ValueError(
                "evolve_profiles needs the slab: material set, thickness > 0 "
                "and npoints >= 1 (they must match the trace map's arguments)"
            )
        evolve = EvolveSpec(
            material=material,
            thickness=float(thickness),
            npoints=int(npoints),
            quadrature=quadrature,
        )
    if profile == "gaussian":
        if waist is None:
            raise ValueError("the gaussian profile needs `waist` (the 1/e^2 radius w0)")
        scale = float(waist)
    elif profile == "airy":
        scale = wavelength * f_foc / hole_diameter
    else:
        raise ValueError(f"profile must be 'airy' or 'gaussian', got {profile!r}")
    r_max = r_max_units * scale

    # Gauss-Legendre in r (the Airy oscillations live here), uniform in phi.
    x, w = np.polynomial.legendre.leggauss(int(n_radial))
    r = 0.5 * r_max * (x + 1.0)
    wr = 0.5 * r_max * w
    phi = 2.0 * np.pi * np.arange(int(n_azimuth)) / float(n_azimuth)
    wphi = 2.0 * np.pi / float(n_azimuth)

    rr2, pp2 = np.meshgrid(r, phi, indexing="ij")
    area = np.ascontiguousarray((wr[:, None] * rr2 * wphi).ravel(), dtype=np.float64)
    rr = np.ascontiguousarray(rr2.ravel(), dtype=np.float64)
    pp = np.ascontiguousarray(pp2.ravel(), dtype=np.float64)

    # Holes at the four corners (+/-d, +/-d) of the folded square mask.
    d = 0.5 * (hole_spacing + hole_diameter)
    arms = np.array(_ARMS_PG, dtype=float) * d
    p, theta = _arm_offsets(arms, f_foc, rr * np.cos(pp), rr * np.sin(pp))
    return FocalMixture(
        r=rr,
        phi=pp,
        area=area,
        p=p,
        theta=theta,
        hole_diameter=float(hole_diameter),
        f_foc=float(f_foc),
        profile=profile,
        waist=float(waist) if waist is not None else 0.0,
        arms=arms,
        collection=collection,
        evolve=evolve,
    )


def evolved_profile_table(
    mixture: FocalMixture,
    omega: NDArray[np.float64],
    omega0: float,
    z_nodes: NDArray[np.float64],
    n_kperp: int = 256,
) -> NDArray[np.complex128]:
    r"""Per-arm, per-depth complex focal profiles at explicit depths (fc-z).

    Between generation events the beamlet evolution is linear, so the profile
    of arm :math:`j` at depth :math:`z` is the entrance profile propagated in
    transverse wavevector and evaluated at the arm's walked radius:

    .. math::

        A_j(r_k, \omega; z) \;=\; N(\omega) \int_0^{k_{\max}(\omega)}
            S(k_\perp, \omega)\, J_0\!\bigl(k_\perp \rho_{jk}(z)\bigr)\,
            e^{\,i\,[k_z(\omega,k_\perp) - k_z(\omega,0)]\, z}\,
            k_\perp\, \mathrm{d}k_\perp ,
        \qquad \rho_{jk}(z) = \lvert \mathbf r_k - \bm\delta_j(z)\rvert ,

    with the in-glass walk :math:`\bm\delta_j(z) = -z\,\mathbf r_j/(f\,n_0)`
    (Snell: transverse wavevector conserved, so the ray angle divides by the
    index; the sign follows the ``+r_j/(f c)`` tilt convention of
    :func:`_arm_offsets`) and
    :math:`k_z = \sqrt{(n_0\,\omega/c)^2 - k_\perp^2}` in the glass — the
    full square root, computed cancellation-safely as
    :math:`k_z - k_{z0} = -k_\perp^2/(k_{z0} + \sqrt{k_{z0}^2 - k_\perp^2})`.
    Only the TRANSVERSE part :math:`k_z - k_z(\omega,0)` appears: the on-axis
    dispersion stays in the trace map's ``input_prop``, so there is no double
    counting and :math:`z \to 0` reduces exactly to the entrance filter.

    The entrance k-space :math:`S` is the aperture's: a unit disc of radius
    :math:`k_R(\omega) = (\omega/c)\,(D/2)/f` for ``profile="airy"`` (whose
    closed form at :math:`z=0` is :func:`airy_amplitude`), or the Gaussian
    :math:`e^{-k_\perp^2 w_0^2/4}` for ``profile="gaussian"`` (closed form
    :math:`e^{-\rho^2/w_0^2}`). :math:`N` normalises to unit peak at
    :math:`(\rho, z) = (0, 0)`, preserving :func:`airy_amplitude`'s unit-peak
    convention and its documented :math:`\omega^{-2}` caveat.

    Numerically the table is built in the DIFFERENCE form

    .. math:: A_j = A_0(\rho_{jk}(z), \omega) \;+\;
              N \int S J_0 k_\perp \bigl(e^{i[\cdot]z} - 1\bigr)\,
              \mathrm{d}k_\perp ,

    with :math:`A_0` the closed-form entrance profile: the quadrature error
    of the Gauss–Legendre :math:`k_\perp` integral cancels between the two
    terms as :math:`z \to 0`, so a zero-phase/zero-walk table is BIT-identical
    to the entrance filter — the reduction the tests assert.

    Two v1 approximations, both documented deliberately: the glass index is
    evaluated at the carrier (:math:`n_0 = n(\omega_0)`) in both the walk and
    :math:`k_z` (the chromatic correction is second order in the small
    quantities), and bins with :math:`\omega_{\rm abs} \le 0` or a non-finite
    index keep the unevolved entrance profile (they carry no signal; a NaN
    would poison the sums — the :func:`croak.materials.beta` policy).

    Parameters
    ----------
    mixture : FocalMixture
        Must carry ``arms`` (the walk needs the hole centres).
    omega : ndarray, shape (N,)
        CENTRED angular-frequency grid (rad/s), as the forward model uses.
    omega0 : float
        Carrier angular frequency (rad/s).
    z_nodes : ndarray, shape (Q,)
        Depths (m) at which to evaluate the evolved profiles.
    n_kperp : int, optional
        Gauss–Legendre nodes of the transverse-wavevector integral. The
        evolved profiles oscillate more than the entrance Airy; 256 holds the
        propagator-truth test at ~1e-6 of peak over 40 µm.

    Returns
    -------
    ndarray, shape (3, K, Q, N), complex
        One profile per (arm, node, depth) on the centred frequency axis, in
        the PG arm order of ``_ARMS_PG`` (probe, gate, conjugated gate).
    """
    if mixture.arms is None:
        raise ValueError(
            "evolved profiles need the mask hole positions: build the mixture "
            "with croak.focal.focal_mixture (which records them)"
        )
    if mixture.evolve is None:
        raise ValueError(
            "this mixture has no EvolveSpec: build it with evolve_profiles=True"
        )
    omega = np.asarray(omega, dtype=float)
    omega_abs = omega + float(omega0)
    z_nodes = np.asarray(z_nodes, dtype=float)
    lam0 = wlfreq(np.array([float(omega0)]))[0]
    n0 = float(materials.refractive_index(mixture.evolve.material)(lam0))

    # Walked radii rho[j, k, q]: node positions minus the in-glass ray walk.
    rx = mixture.r * np.cos(mixture.phi)
    ry = mixture.r * np.sin(mixture.phi)
    walk = -np.asarray(mixture.arms, dtype=float) / (mixture.f_foc * n0)  # (3,2)/m
    rho = np.hypot(
        rx[None, :, None] - z_nodes[None, None, :] * walk[:, 0, None, None],
        ry[None, :, None] - z_nodes[None, None, :] * walk[:, 1, None, None],
    )  # (3, K, Q)
    # At exactly z = 0 the walk vanishes and rho IS the node radius; writing it
    # so removes the last-ulp noise of hypot(r cos, r sin) and makes the
    # zero-evolution table bit-identical to the entrance filter (the reduction
    # contract in tests/test_fcz.py).
    rho[:, :, z_nodes == 0.0] = mixture.r[None, :, None]
    kk, qq, nn = mixture.nodes, z_nodes.size, omega.size
    flat = rho.reshape(3 * kk * qq)

    # Closed-form entrance profile at the walked radii (the A0 of the
    # difference form; exactly spectral_filter's formulas).
    if mixture.profile == "gaussian":
        a0 = np.repeat(np.exp(-(flat[:, None] ** 2) / mixture.waist**2), nn, axis=1)
        k_edge = np.full(nn, 8.0 / mixture.waist)  # e^{-16}: support captured
    else:
        u = (
            mixture.hole_diameter
            * flat[:, None]
            * omega_abs[None, :]
            / (2.0 * mixture.f_foc * _C_LIGHT)
        )
        a0 = airy_amplitude(u)
        k_edge = omega_abs * (0.5 * mixture.hole_diameter) / (_C_LIGHT * mixture.f_foc)

    # Gauss-Legendre in k_perp on [0, k_edge(omega)], one rule shared by all
    # frequencies through the substitution k = k_edge * x.
    x, w = np.polynomial.legendre.leggauss(int(n_kperp))
    x = 0.5 * (x + 1.0)
    w = 0.5 * w

    table = np.empty((3 * kk * qq, nn), dtype=complex)
    kz0_all = n0 * omega_abs / _C_LIGHT
    good = (omega_abs > 0.0) & np.isfinite(kz0_all) & (k_edge > 0.0)
    table[:, ~good] = a0[:, ~good]  # unphysical bins: entrance profile, unevolved
    for i in np.nonzero(good)[0]:
        k = k_edge[i] * x  # (nk,)
        wk = (k_edge[i] ** 2) * w * x  # k dk Jacobian
        if mixture.profile == "gaussian":
            s = np.exp(-(k**2) * mixture.waist**2 / 4.0)
        else:
            s = 1.0
        wks = wk * s
        norm = 1.0 / np.sum(wks)
        kz0 = kz0_all[i]
        # cancellation-safe kz - kz0; clamp the (never physically reached)
        # evanescent corner so the sqrt stays real
        dkz = -(k**2) / (kz0 + np.sqrt(np.maximum(kz0**2 - k**2, 0.0)))
        bess = j0(np.outer(flat, k))  # (3KQ, nk)
        ring = np.exp(1j * np.outer(z_nodes, dkz)) - 1.0  # (Q, nk)
        corr = (bess * wks[None, :]).reshape(3, kk, qq, -1) * ring[None, None]
        table[:, i] = a0[:, i] + norm * corr.sum(axis=3).reshape(3 * kk * qq)
    return np.ascontiguousarray(table.reshape(3, kk, qq, nn))


def evolved_arm_filters(
    mixture: FocalMixture,
    omega: NDArray[np.float64],
    omega0: float,
) -> NDArray[np.complex128]:
    """Spec-driven fc-z filter table on the mixture's own depth quadrature.

    Thin wrapper over :func:`evolved_profile_table` using the depth nodes of
    the mixture's :class:`EvolveSpec` — the same
    :func:`croak.forward.quadrature_nodes_weights` nodes the trace map
    propagates on, which is what makes the per-depth filters line up with
    ``input_prop`` column for column.
    """
    if mixture.evolve is None:
        raise ValueError(
            "this mixture has no EvolveSpec: build it with evolve_profiles=True"
        )
    from .forward import quadrature_nodes_weights

    nodes, _ = quadrature_nodes_weights(
        mixture.evolve.thickness, mixture.evolve.npoints, mixture.evolve.quadrature
    )
    return evolved_profile_table(mixture, omega, omega0, np.asarray(nodes))


def _arm_offsets(
    arms: NDArray[np.float64],
    f_foc: float,
    rx: NDArray[np.float64],
    ry: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the ``(p, theta)`` arrival offsets of PG arms at focal positions.

    Arrival-time offsets: ``alpha_j = x_j / (f c)`` with ``x_j`` the hole centre, and
    the PG combinations are ``p = (alpha_3 - alpha_2).r`` and ``theta = (alpha_1 -
    (alpha_2 + alpha_3)/2).r``, as in :mod:`croak.smearing`. Shared so that a polar and
    a Cartesian node set cannot disagree about the sign of either.

    Sign convention: this is the OPPOSITE overall sign to
    :mod:`croak.smearing`'s ``alpha_j = -x_j / (f c)`` (the physically correct
    one: a beamlet from mask position ``x_j`` crosses the focus travelling
    along ``-x_j/f``, so its pulse front arrives at ``r`` earlier by
    ``(x_j . r)/(f c)``). Negating every tilt is a point inversion of the
    focal plane — it flips ``p_k`` and ``theta_k`` jointly — which is
    unobservable everywhere croak uses them: the reduced kernel keeps its
    widths and correlation (second moments are even), and the mixture's
    incoherent sums are unchanged because the quadrature node set is
    inversion-symmetric and the ``A(r, omega)`` filters and areas are even in
    ``r``. For the *collection* path the inversion maps the model onto one
    with the aperture reflected through the phase-matched signal direction —
    still unobservable for a circularly symmetric hole centred on that
    direction (the physical case; ``aperture_offset()`` measures the
    centring), but NOT for an off-centre aperture. Two rules follow: the sign
    must stay consistent between this module and :mod:`croak.collection`
    (``transform_phases`` assumes one convention linking ``p_k`` to the
    aperture ramp — ``tests/test_collection.py`` asserts the symmetry), and a
    deliberately off-centre collection model would promote this convention
    from bookkeeping to physics.
    """
    alpha = arms / (f_foc * _C_LIGHT)
    dot = lambda a: a[0] * rx + a[1] * ry  # noqa: E731 - one dot product, used twice
    p = np.ascontiguousarray(dot(alpha[2] - alpha[1]), dtype=np.float64)
    theta = np.ascontiguousarray(
        dot(alpha[0] - 0.5 * (alpha[1] + alpha[2])), dtype=np.float64
    )
    return p, theta


def mixture_from_nodes(
    *,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    area: NDArray[np.float64],
    arms: NDArray[np.float64],
    hole_diameter: float,
    f_foc: float,
    profile: str = "airy",
    waist: float | None = None,
    collection: CollectionAperture | None = None,
) -> FocalMixture:
    r"""Build a mixture on **arbitrary** focal-plane nodes from three arm positions.

    :func:`focal_mixture` is this with a polar Gauss--Legendre quadrature and the square
    BOXCARS layout filled in. Splitting them out keeps the ``(p, \vartheta)`` algebra in
    one place: a Cartesian node set (which the collection tests need, to compare against
    a plain 2-D FFT and to check Parseval exactly) then cannot drift from it.

    Parameters
    ----------
    x, y : ndarray, shape (K,)
        Focal-plane node positions (m).
    area : ndarray, shape (K,)
        Quadrature weights including any Jacobian (m^2).
    arms : ndarray, shape (3, 2)
        Mask hole centres (m) in the PG role order ``(probe, gate unconjugated, gate
        conjugated)``. Only differences enter ``(p, theta)``, but the absolute
        positions fix the signal direction and so the aperture centring.
    hole_diameter, f_foc : float
        Mask hole diameter and focal length (m).
    profile : {"airy", "gaussian"}, optional
        Focal amplitude model; see :meth:`FocalMixture.spectral_filter`.
    waist : float or None, optional
        1/e^2 radius (m); required for ``profile="gaussian"``.
    collection : CollectionAperture or None, optional
        Finite collection aperture (:mod:`croak.collection`).

    Returns
    -------
    FocalMixture
    """
    if profile not in ("airy", "gaussian"):
        raise ValueError(f"profile must be 'airy' or 'gaussian', got {profile!r}")
    if profile == "gaussian" and waist is None:
        raise ValueError("the gaussian profile needs `waist` (the 1/e^2 radius w0)")
    arms = np.asarray(arms, dtype=float)
    if arms.shape != (3, 2):
        raise ValueError(f"arms must have shape (3, 2), got {arms.shape}")
    rx = np.asarray(x, dtype=float)
    ry = np.asarray(y, dtype=float)
    p, theta = _arm_offsets(arms, f_foc, rx, ry)
    return FocalMixture(
        r=np.ascontiguousarray(np.hypot(rx, ry), dtype=np.float64),
        phi=np.ascontiguousarray(np.arctan2(ry, rx), dtype=np.float64),
        area=np.ascontiguousarray(area, dtype=np.float64),
        p=p,
        theta=theta,
        hole_diameter=float(hole_diameter),
        f_foc=float(f_foc),
        profile=profile,
        waist=float(waist) if waist is not None else 0.0,
        arms=arms,
        collection=collection,
    )


def reduced_moments(mix: FocalMixture, wavelength: float) -> dict[str, float]:
    r"""Second moments of :math:`(p,\vartheta)` under the achromatic weight.

    Self-test of the geometry and quadrature: these must reproduce the closed forms
    of :mod:`croak.smearing` --- :math:`\sigma_p = 0.49135\,(d/D)\lambda/c`,
    :math:`\sigma_\vartheta = 0.54933\,(d/D)\lambda/c`, :math:`\rho = 1/\sqrt5`
    --- to the quadrature's accuracy. Agreement means the mixture is sampling the
    same focal average the reduced kernel models, and differs from it only by
    keeping the frequency dependence.

    Returns
    -------
    dict
        ``sigma_p``, ``sigma_theta`` (s) and ``rho``.
    """
    w = mix.reduced_weight(wavelength)
    w = w / w.sum()
    sp = float(np.sqrt(np.sum(w * mix.p**2)))
    st = float(np.sqrt(np.sum(w * mix.theta**2)))
    rho = float(np.sum(w * mix.p * mix.theta) / (sp * st)) if sp * st > 0 else 0.0
    return {"sigma_p": sp, "sigma_theta": st, "rho": rho}
