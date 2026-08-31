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
from scipy.special import j1

from .collection import CollectionAperture

__all__ = [
    "FocalMixture",
    "airy_amplitude",
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

    Returns
    -------
    FocalMixture
    """
    if n_radial < 2 or n_azimuth < 2:
        raise ValueError("n_radial and n_azimuth must both be >= 2")
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
    )


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
