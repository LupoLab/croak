r"""Finite collection aperture: the focal mixture without full-beam collection.

:mod:`croak.focal` sums **incoherently** over focal-plane nodes. By Parseval that is
exact only if the detector collects the *entire* signal beam: the measured spectral
intensity is

.. math:: T(\omega,\tau) = \frac{1}{(2\pi)^2}\int_{\text{aperture}}
          \bigl|\tilde S(\mathbf k,\omega,\tau)\bigr|^2\,\mathrm d^2k ,

with :math:`\tilde S` the *transverse* Fourier transform of the focal-plane signal
field, and only an unbounded aperture turns that into
:math:`\int|S|^2\,\mathrm d^2r`. A finite
aperture must transform **before** squaring. This module supplies the aperture; the
transform itself lives in :func:`croak.forward_jax.make_param_trace_fn`.

The phase the incoherent sum throws away
----------------------------------------
Write the arrival-time offset of arm :math:`j` at focal position :math:`\mathbf r` as
:math:`\delta t_j=\boldsymbol\alpha_j\cdot\mathbf r` with
:math:`\boldsymbol\alpha_j=\mathbf r_j/(fc)`, as
:func:`croak.focal.focal_mixture` already does (the opposite overall sign to
:mod:`croak.smearing` — a pure point inversion of the focal plane, unobservable
for the centred circular aperture modelled here; see
:func:`croak.focal._arm_offsets` for the full statement). The exact PG signal there is

.. math:: S(t,\mathbf r) = E(t-\delta t_1)\,E(t-\tau-\delta t_2)\,
          E^*(t-\tau-\delta t_3)\,A^3(r,\omega),

while croak's per-node build is
:math:`E(t)E(t-\tau+\vartheta+p/2)E^*(t-\tau+\vartheta-p/2)A^3` with croak's own
:math:`\vartheta=\delta t_1-\tfrac12(\delta t_2+\delta t_3)` and
:math:`p=\delta t_3-\delta t_2`. The two are **identical up to a rigid time shift**,
:math:`S(t,\mathbf r)=S_{\text{croak}}(t-\delta t_1,\mathbf r)`, so in croak's
convention (a delay :math:`s` is a spectral factor :math:`e^{+i\omega s}`)

.. math:: \Psi(\omega,\mathbf r) = \Psi_{\text{croak}}(\omega,\mathbf r)\,
          e^{+i\omega\,\boldsymbol\alpha_1\cdot\mathbf r}.

The :math:`(p,\vartheta)` reduction therefore loses **nothing**: the incoherent
assumption enters only in discarding that one phase. Removing the signal's own
phase-matched carrier
:math:`\boldsymbol\alpha_s=\boldsymbol\alpha_1+\boldsymbol\alpha_2-\boldsymbol\alpha_3`
leaves :math:`\boldsymbol\alpha_1-\boldsymbol\alpha_s=\boldsymbol\alpha_3-
\boldsymbol\alpha_2`, which is precisely the ``p`` vector, so

.. math:: \tilde S(\boldsymbol\kappa,\omega) = \sum_k a_k\,
          \Psi_{\text{croak}}(\omega,\mathbf r_k)\,e^{+i\omega p_k}\,
          e^{-i\boldsymbol\kappa\cdot\mathbf r_k},
          \qquad \boldsymbol\kappa = \mathbf k - \omega_{\text{abs}}
          \boldsymbol\alpha_s .

**The discarded phase is exactly :math:`\omega` times one of the two reduced smearing
parameters** --- ``p`` for PG; by the same algebra :math:`-\vartheta` for SD, which
is why only PG is implemented here.

.. warning::

   That :math:`\omega` is the **centred** (envelope) frequency, while the aperture's
   own position :math:`\boldsymbol\kappa` scales with the **absolute** one. croak's
   forward model applies every time shift as :math:`e^{+i\omega s}` on a centred
   spectrum, so each replica drops a constant :math:`e^{+i\omega_0 s}`; the three
   together drop :math:`e^{-i\omega_0 p}`, which :math:`|\Psi|^2` never sees but a
   coherent sum does. Written against the absolute frequency the transform phase is
   :math:`\omega_{\text{abs}}(p_k - \mathbf u_j\cdot\mathbf r_k) - \omega_0 p_k`,
   and that trailing term is the whole difference between a model that reproduces a
   dense 2-D FFT and one that is wrong by four orders of magnitude.

Where the aperture sits
-----------------------
A hole at mask-plane position :math:`\mathbf x` a distance :math:`z_{\text{mask}}`
from the focus selects transverse wavevectors about
:math:`\mathbf k=(\omega/c)\mathbf x/z_{\text{mask}}`, so in the carrier-removed
frame the aperture is centred at :math:`(\omega/c)(\mathbf x_{\text{hole}}-\mathbf r_s)/
z_{\text{mask}}` with half-width :math:`(\omega/c)(D_{\text{hole}}/2)/z_{\text{mask}}`:
both scale with :math:`\omega`, i.e. **the aperture is chromatic**. For a square BOXCARS
mask collected at the signal corner the centre is identically zero, which
``tests/test_collection.py`` asserts rather than assumes.

Two collection models are supported, and both are functionals of the same
:math:`\tilde S`:

``"integrated"``
    :math:`\sum_j \mathrm dk_j\,W_j^2\,|\tilde S_j|^2/(2\pi)^2` --- a spectrometer
    fed by everything the hole passes (the reference simulator's ``Iω_win``).
``"reimaged"``
    :math:`\bigl|\sum_j \mathrm dk_j\,W_j\,\tilde S_j/(2\pi)^2\bigr|^2` --- the on-axis
    re-imaged pixel, i.e. the windowed field back at the focal origin, and the fully
    coherent limit (``Iω_win_reimaged``).

Validity
--------
The collection model is exact given croak's focal-plane field, but that field comes from
a **one-dimensional** depth quadrature: signal generated at every depth is added with
the same transverse phase, i.e. fully coherently in k. A real slab decorrelates depth
from depth (diffraction, Gouy phase, walk-through), so this model is *too* coherent, and
increasingly so the thicker the slab. Measured at the known truth against the reference
3-D traces, it improves the residual by up to 24-30 % at 2-9.5 um and over-corrects
beyond about 20 um. Prefer it for thin slabs; see
``docs/howto/collection_aperture.md`` for the depth table. The leading missing term —
the per-depth transverse phase ``[k_z(w, k_perp) - k_z(w, 0)](L - z)`` — has an
optional parameter-free model: ``depth_transverse=True`` on
:func:`croak.forward_jax.make_param_trace_fn` moves the depth sum inside the aperture
transform with that phase applied per depth node (same how-to for usage and caveats).

Regime
------
For the reference instrument at 260 nm the aperture passes a k half-width of
6.04e4 rad/m against a signal footprint of 3.63e5 rad/m --- six times narrower, about
3 % of the solid angle. The corresponding real-space coherence length (~63 µm)
*exceeds* the region that actually emits (~15--30 µm, where :math:`|p|,|\vartheta|`
stay inside the pulse duration), so this collection is much closer to the coherent
limit than to the incoherent one, and a coherent sum **sharpens** where an incoherent
one blurs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a focal <-> collection cycle
    from .focal import FocalMixture

__all__ = [
    "APODISATIONS",
    "COLLECTION_MODES",
    "CollectionAperture",
    "DEFAULT_N_AZIMUTH",
    "DEFAULT_N_RADIAL",
    "DEFAULT_PAD",
    "TransformPhases",
    "aperture_from_scan",
    "mask_hole_aperture",
    "mask_transmission",
    "resolve_tanh_width",
    "transform_phases",
]

_C_LIGHT = 299_792_458.0

#: Mask apodisations, ported from ``ModelPNPS.makemask`` so a simulated window can be
#: reproduced exactly rather than approximated by a top hat. The reference instrument's
#: ``tanh`` edge is 96.9 um on a 500 um hole -- 19 % of the diameter -- so the
#: difference is not a detail.
APODISATIONS = ("hard", "supergauss", "tanh")

#: Which functional of the transformed field the detector measures.
COLLECTION_MODES = ("integrated", "reimaged")

#: Default aperture quadrature: nodes per radial panel, azimuthal nodes, and the outer
#: radial cut in hole radii. Chosen by the convergence measurement in
#: ``tests/test_collection.py`` -- on the reference geometry these reproduce a
#: 6x denser quadrature to better than 1e-5 of peak in both collection modes. Named
#: constants rather than repeated literals so :func:`mask_hole_aperture` and
#: :func:`aperture_from_scan` cannot drift apart.
DEFAULT_N_RADIAL = 6
DEFAULT_N_AZIMUTH = 12
DEFAULT_PAD = 2.5


def mask_transmission(
    radius: NDArray[np.float64],
    *,
    hole_diameter: float,
    apod: str = "tanh",
    apod_param: float | None = None,
) -> NDArray[np.float64]:
    r"""Amplitude transmission of an apodised mask hole at a mask-plane radius.

    A direct port of ``ModelPNPS.makemask``; the three forms are

    .. math::

        \text{hard} &: \; \mathbb 1[r \le D/2] \\
        \text{supergauss} &: \; \exp\bigl[-(2r/D)^n\bigr] \\
        \text{tanh} &: \; \tfrac12\bigl[1-\tanh\bigl((r-D/2)/\Delta\bigr)\bigr]

    Parameters
    ----------
    radius : ndarray
        Distance from the hole centre in the mask plane (m).
    hole_diameter : float
        Hole diameter :math:`D` (m).
    apod : {"hard", "supergauss", "tanh"}, optional
        Apodisation form.
    apod_param : float or None, optional
        The super-Gaussian exponent :math:`n` (default 16) or the ``tanh``
        smoothing width :math:`\Delta` in **mask-plane metres**. Required for
        ``"tanh"``: its default in the simulator depends on the simulation's own
        k-grid, so it must be resolved with :func:`resolve_tanh_width` rather than
        guessed here.

    Returns
    -------
    ndarray
        Amplitude (not intensity) transmission, same shape as ``radius``.

    Raises
    ------
    ValueError
        If ``apod`` is unknown, ``hole_diameter`` is not positive, or a ``tanh``
        window is requested without a width.
    """
    if hole_diameter <= 0.0:
        raise ValueError(f"hole_diameter must be positive, got {hole_diameter!r}")
    if apod not in APODISATIONS:
        raise ValueError(f"apod must be one of {APODISATIONS}, got {apod!r}")
    r = np.asarray(radius, dtype=float)
    if apod == "hard":
        return (r <= 0.5 * hole_diameter).astype(float)
    if apod == "supergauss":
        n = 16.0 if apod_param is None else float(apod_param)
        return np.exp(-((2.0 * r / hole_diameter) ** n))
    if apod_param is None:
        raise ValueError(
            "the 'tanh' apodisation needs `apod_param` (the smoothing width in "
            "mask-plane metres); use resolve_tanh_width() to reproduce the "
            "simulator's own default, which depends on its transverse k-grid"
        )
    return 0.5 * (1.0 - np.tanh((r - 0.5 * hole_diameter) / float(apod_param)))


def resolve_tanh_width(
    *, delta_k: float, z_mask: float, omega_reference: float
) -> float:
    r"""Reproduce ``ModelPNPS``' default ``tanh`` smoothing width.

    The simulator sets :math:`\Delta = 3\,\Delta x_{\text{mask}}` with
    :math:`\Delta x_{\text{mask}} = \Delta k\, z_{\text{mask}} c/\omega_0`. Files
    record the resolved window as the string ``"default"``, so it has to be re-derived
    from the stored k-grid.

    Parameters
    ----------
    delta_k : float
        Transverse k-grid spacing of the simulation (rad/m).
    z_mask : float
        Mask-plane distance from the focus (m).
    omega_reference : float
        The frequency the simulator evaluated the default at (rad/s): its own grid
        point **nearest** :math:`2\pi c/\lambda_0`, not that value itself. On a
        coarse spectral grid the two differ, which is why
        :class:`croak.io.MaskWindowSpec` resolves it at read time.

    Returns
    -------
    float
        Smoothing width in mask-plane metres.
    """
    if delta_k <= 0.0 or z_mask <= 0.0 or omega_reference <= 0.0:
        raise ValueError(
            "delta_k, z_mask and omega_reference must all be positive, got "
            f"{delta_k!r}, {z_mask!r}, {omega_reference!r}"
        )
    return 3.0 * float(delta_k) * float(z_mask) * _C_LIGHT / float(omega_reference)


@dataclass(frozen=True)
class CollectionAperture:
    """Quadrature over the collecting aperture, in the mask (far-field) plane.

    Attributes
    ----------
    x, y : ndarray, shape (J,)
        Node positions. Mask-plane metres when ``chromatic`` (a physical hole, whose
        k-space footprint scales with frequency), else **absolute** transverse
        wavevectors in rad/m (a frequency-independent k window).
    weight : ndarray, shape (J,)
        Quadrature weight including the ``r dr dphi`` Jacobian: m^2 when ``chromatic``,
        else rad^2/m^2. The transmission is NOT folded in -- ``"integrated"`` needs its
        square and ``"reimaged"`` its first power, so it is kept separate.
    transmission : ndarray, shape (J,)
        **Amplitude** transmission :math:`W` at each node (see
        :func:`mask_transmission`).
    z_mask : float
        Distance from the focus to the mask plane (m). Unused when not ``chromatic``.
    chromatic : bool
        Whether the aperture is a physical hole (``True``, the usual case) or a fixed
        window in k. The latter exists because that is the frame a signal-quadrant or
        Planck k-window lives in, and it is what makes the exact Parseval test possible.
    mode : {"integrated", "reimaged"}
        Which functional of the transformed field the detector measures; see the module
        docstring.
    """

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    weight: NDArray[np.float64]
    transmission: NDArray[np.float64]
    z_mask: float
    chromatic: bool = True
    mode: str = "integrated"

    def __post_init__(self) -> None:
        """Validate shapes and enumerations at construction, not at trace time."""
        shapes = {np.shape(a) for a in (self.x, self.y, self.weight, self.transmission)}
        if len(shapes) != 1 or len(next(iter(shapes))) != 1:
            raise ValueError(
                "x, y, weight and transmission must all be 1-D of the same length, got "
                f"{np.shape(self.x)}, {np.shape(self.y)}, {np.shape(self.weight)}, "
                f"{np.shape(self.transmission)}"
            )
        if self.mode not in COLLECTION_MODES:
            raise ValueError(
                f"mode must be one of {COLLECTION_MODES}, got {self.mode!r}"
            )
        if self.chromatic and self.z_mask <= 0.0:
            raise ValueError(
                f"a chromatic aperture needs a positive z_mask, got {self.z_mask!r}"
            )

    @property
    def nodes(self) -> int:
        """Number of quadrature nodes."""
        return int(np.size(self.x))


def _polar_panel(
    r_inner: float, r_outer: float, n_radial: int, n_azimuth: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Gauss-Legendre in radius over an annulus, uniform in azimuth.

    Returns ``(x, y, area)`` flattened over the (radius, azimuth) product. Uniform
    azimuthal sampling of a smooth periodic integrand converges spectrally, so the
    radial direction is where the nodes are worth spending.
    """
    x, w = np.polynomial.legendre.leggauss(int(n_radial))
    half = 0.5 * (r_outer - r_inner)
    r = half * (x + 1.0) + r_inner
    wr = half * w
    phi = 2.0 * np.pi * np.arange(int(n_azimuth)) / float(n_azimuth)
    wphi = 2.0 * np.pi / float(n_azimuth)
    rr, pp = np.meshgrid(r, phi, indexing="ij")
    area = (wr[:, None] * rr * wphi).ravel()
    return rr.ravel() * np.cos(pp.ravel()), rr.ravel() * np.sin(pp.ravel()), area


def mask_hole_aperture(
    *,
    hole_x: float,
    hole_y: float,
    hole_diameter: float,
    z_mask: float,
    apod: str = "tanh",
    apod_param: float | None = None,
    n_radial: int = DEFAULT_N_RADIAL,
    n_azimuth: int = DEFAULT_N_AZIMUTH,
    pad: float = DEFAULT_PAD,
    mode: str = "integrated",
) -> CollectionAperture:
    r"""Build the quadrature for a single apodised collection hole.

    The radial quadrature is **split at the hole edge** into ``[0, D/2]`` and
    ``[D/2, pad\,D/2]``. The integrand is smooth on each panel but has a knee at the
    edge, so two panels of ``n_radial`` nodes resolve an apodised rim that a single
    panel of ``2 n_radial`` would smear.

    Parameters
    ----------
    hole_x, hole_y : float
        Hole centre in the mask plane (m).
    hole_diameter : float
        Hole diameter (m).
    z_mask : float
        Distance from the focus to the mask plane (m).
    apod : {"hard", "supergauss", "tanh"}, optional
        Apodisation form; see :func:`mask_transmission`.
    apod_param : float or None, optional
        Apodisation parameter; see :func:`mask_transmission` and
        :func:`resolve_tanh_width`.
    n_radial, n_azimuth : int, optional
        Quadrature nodes **per radial panel** and in azimuth, so the total is
        ``2 n_radial n_azimuth`` (``n_radial n_azimuth`` for a hard edge, which has no
        outer panel to integrate).
    pad : float, optional
        Outer radial cut in units of the hole radius. A soft edge leaks well past the
        nominal rim -- the reference ``tanh`` window still passes 0.6 % of amplitude at
        ``r = D`` -- so the default carries the quadrature to 2.5 times the radius.
        The ``"reimaged"`` mode sums the amplitude rather than its square and so feels
        that tail ten times more strongly: measured on the 1 fs geometry, dropping to
        ``pad = 2`` costs it 4e-5 of peak against 2e-7 for ``"integrated"``. Ignored
        (clamped to 1) for a hard edge, which transmits nothing outside the rim.
    mode : {"integrated", "reimaged"}, optional
        Collection model; see the module docstring.

    Returns
    -------
    CollectionAperture
    """
    if n_radial < 2 or n_azimuth < 2:
        raise ValueError("n_radial and n_azimuth must both be >= 2")
    if pad < 1.0:
        raise ValueError(f"pad must be >= 1 (the hole itself), got {pad!r}")
    radius = 0.5 * hole_diameter
    # A hard edge transmits nothing outside the rim, so the outer panel would be an
    # exactly-zero contribution bought at full cost.
    panels = (
        [(0.0, radius)] if apod == "hard" else [(0.0, radius), (radius, pad * radius)]
    )
    xs, ys, ws = [], [], []
    for r_inner, r_outer in panels:
        px, py, pw = _polar_panel(r_inner, r_outer, n_radial, n_azimuth)
        xs.append(px)
        ys.append(py)
        ws.append(pw)
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    weight = np.concatenate(ws)
    transmission = mask_transmission(
        np.hypot(x, y), hole_diameter=hole_diameter, apod=apod, apod_param=apod_param
    )
    return CollectionAperture(
        x=np.ascontiguousarray(x + float(hole_x)),
        y=np.ascontiguousarray(y + float(hole_y)),
        weight=np.ascontiguousarray(weight),
        transmission=np.ascontiguousarray(transmission),
        z_mask=float(z_mask),
        chromatic=True,
        mode=mode,
    )


def aperture_from_scan(
    path: str,
    *,
    mode: str = "integrated",
    window: str | int | None = None,
    n_radial: int = DEFAULT_N_RADIAL,
    n_azimuth: int = DEFAULT_N_AZIMUTH,
    pad: float = DEFAULT_PAD,
) -> CollectionAperture:
    """Build the aperture the reference simulation actually collected through.

    Reads the flattened ``window_def_*`` scalars a ``scansave`` file records (see
    :func:`croak.io.read_simulated_mask_window`) and rebuilds the same hole, resolving
    a ``"default"`` ``tanh`` width from the file's own transverse k-grid.

    Parameters
    ----------
    path : str
        Path to the simulated scan file.
    mode : {"integrated", "reimaged"}, optional
        Which collection model to build; the two correspond to the file's ``Iω_win``
        and ``Iω_win_reimaged`` datasets.
    window : str or int, optional
        Which collection hole of a multi-aperture scan (one propagation reduced
        through several holes, ``Iω_win_2`` …): the trace-window dataset name
        (``_reimaged`` accepted) or the bare index. ``None`` is the first
        window — the pre-existing behaviour.
    n_radial, n_azimuth, pad : optional
        Quadrature controls; see :func:`mask_hole_aperture`.

    Returns
    -------
    CollectionAperture

    Raises
    ------
    KeyError
        If the file records no mask window.
    """
    from .io import read_simulated_mask_window

    spec = read_simulated_mask_window(path, window)
    if spec is None:
        raise KeyError(
            f"{path!r}: no /grid/window_def_* scalars for window {window!r}, so "
            f"the collection aperture cannot be rebuilt from the file"
        )
    apod_param = spec.apod_param
    if apod_param is None and spec.apod == "tanh":
        apod_param = resolve_tanh_width(
            delta_k=spec.delta_k,
            z_mask=spec.z_mask,
            omega_reference=spec.omega_reference,
        )
    return mask_hole_aperture(
        hole_x=spec.hole_x,
        hole_y=spec.hole_y,
        hole_diameter=spec.hole_diameter,
        z_mask=spec.z_mask,
        apod=spec.apod,
        apod_param=apod_param,
        n_radial=n_radial,
        n_azimuth=n_azimuth,
        pad=pad,
        mode=mode,
    )


@dataclass(frozen=True)
class TransformPhases:
    r"""Precomputed geometry of the focal-plane-to-aperture transform.

    The transform matrix is

    .. math:: M_{jk}(\omega) = a_k\,
              \exp\bigl[i(\omega\,\Phi_{jk} - \Theta_{jk})\bigr],

    with :math:`a_k` the mixture's area weights, so that
    :math:`\tilde S_j = \sum_k M_{jk}\Psi_k` is the transform of the carrier-removed
    field at aperture node :math:`j`. Splitting the phase into a part linear in
    :math:`\omega` and a constant part is what lets a *chromatic* hole and a *fixed* k
    window share one implementation.

    Attributes
    ----------
    chromatic_delay : ndarray, shape (J, K)
        :math:`\Phi_{jk}` in seconds: the node's own arrival offset minus the
        aperture node's phase ramp, both of which scale with frequency.
    static_phase : ndarray, shape (J, K)
        :math:`\Theta_{jk}` in radians; identically zero for a chromatic hole.
    weight_sq : ndarray, shape (J, Nomega)
        :math:`\mathrm dk_j W_j^2/(2\pi)^2` --- the ``"integrated"`` reduction weight.
    weight_amp : ndarray, shape (J, Nomega)
        :math:`\mathrm dk_j W_j/(2\pi)^2` --- the ``"reimaged"`` reduction weight.
    """

    chromatic_delay: NDArray[np.float64]
    static_phase: NDArray[np.float64]
    weight_sq: NDArray[np.float64]
    weight_amp: NDArray[np.float64]


def _mixture_geometry(
    mixture: FocalMixture,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(rx, ry, signal_position)``, or raise if the mixture has no arms."""
    if mixture.arms is None:
        raise ValueError(
            "the collection model needs the mask hole positions: build the mixture "
            "with croak.focal.focal_mixture (which records them) rather than "
            "constructing FocalMixture without `arms`"
        )
    rx = mixture.r * np.cos(mixture.phi)
    ry = mixture.r * np.sin(mixture.phi)
    return rx, ry, mixture.signal_position


def aperture_offset(mixture: FocalMixture) -> NDArray[np.float64]:
    r"""Mask-plane offset of the aperture from the phase-matched signal direction.

    The transmission-weighted centroid of the aperture nodes minus
    :math:`\mathbf r_s=\mathbf r_1+\mathbf r_2-\mathbf r_3`, in metres. Divided by
    :math:`z_{\text{mask}}` and multiplied by :math:`\omega/c` it is where the aperture
    sits in the carrier-removed k frame, so a value of zero means the instrument
    collects exactly on the phase-matched direction.

    Returns
    -------
    ndarray, shape (2,)
        ``(dx, dy)`` in metres.

    Raises
    ------
    ValueError
        If the mixture carries no collection aperture, or the aperture is not
        chromatic (a fixed k window has no mask-plane position).
    """
    aperture = mixture.collection
    if aperture is None:
        raise ValueError("the mixture carries no collection aperture")
    if not aperture.chromatic:
        raise ValueError(
            "a non-chromatic aperture is specified directly in k, so it has no "
            "mask-plane offset; compare its nodes with (omega/c) r_s / f instead"
        )
    w = aperture.weight * aperture.transmission**2
    total = float(np.sum(w))
    if total <= 0.0:
        raise ValueError("the aperture transmits nothing: its weights sum to zero")
    centroid = np.array([np.sum(w * aperture.x), np.sum(w * aperture.y)]) / total
    return centroid - _mixture_geometry(mixture)[2]


def transform_phases(
    mixture: FocalMixture, omega: NDArray[np.float64], omega0: float
) -> TransformPhases:
    r"""Build the transform geometry for a mixture and its collection aperture.

    Parameters
    ----------
    mixture : FocalMixture
        The focal quadrature. Must carry ``arms`` and a ``collection``.
    omega : ndarray, shape (Nomega,)
        **Centred** angular-frequency grid (rad/s), as the forward model uses.
    omega0 : float
        Carrier angular frequency (rad/s). Both are needed and they are not
        interchangeable: the aperture's k footprint scales with the absolute
        frequency, while the phase croak's envelope bookkeeping dropped scales with
        the centred one. See the warning in the module docstring.

    Returns
    -------
    TransformPhases

    Notes
    -----
    Memory is ``16 J K Nomega`` bytes for the complex matrix the caller builds from
    this, so the aperture and mixture node counts multiply.
    """
    aperture = mixture.collection
    if aperture is None:
        raise ValueError("the mixture carries no collection aperture")
    rx, ry, r_s = _mixture_geometry(mixture)
    omega_abs = np.asarray(omega, dtype=float) + float(omega0)
    p = np.asarray(mixture.p, dtype=float)
    # exp(+i w_centred p) = exp(+i w_abs p) exp(-i w0 p): the trailing constant is the
    # carrier phase croak's envelope shifts leave out, invisible to |Psi|^2 and not to
    # a coherent sum.
    carrier_phase = float(omega0) * p
    if aperture.chromatic:
        # A hole at mask position x selects k = (w/c) x / z_mask, so its phase ramp
        # across the focal plane is w (x - r_s).r / (c z_mask) once the signal's own
        # carrier at r_s is removed -- linear in w, hence a pure delay.
        ux = (aperture.x - r_s[0]) / (_C_LIGHT * aperture.z_mask)
        uy = (aperture.y - r_s[1]) / (_C_LIGHT * aperture.z_mask)
        chromatic_delay = p[None, :] - (
            ux[:, None] * rx[None, :] + uy[:, None] * ry[None, :]
        )
        static_phase = np.repeat(carrier_phase[None, :], aperture.nodes, axis=0)
        # d^2k = (w / (c z_mask))^2 d^2x
        dk = (
            aperture.weight[:, None]
            * (omega_abs[None, :] / (_C_LIGHT * aperture.z_mask)) ** 2
        )
    else:
        # A fixed k window lives in the ABSOLUTE frame, so the carrier is not removed:
        # the node phase is the probe arm's own arrival offset,
        # alpha_1.r = p + alpha_s.r.
        alpha_s = r_s / (mixture.f_foc * _C_LIGHT)
        chromatic_delay = np.repeat(
            (p + alpha_s[0] * rx + alpha_s[1] * ry)[None, :], aperture.nodes, axis=0
        )
        static_phase = (
            aperture.x[:, None] * rx[None, :]
            + aperture.y[:, None] * ry[None, :]
            + carrier_phase[None, :]
        )
        dk = np.repeat(aperture.weight[:, None], omega_abs.size, axis=1)
    # croak's inverse transform carries the 1/(2 pi)^d, so the collected intensity is
    # int |S~|^2 d^2k / (2 pi)^2 -- which is what makes an unbounded aperture reproduce
    # the incoherent sum EXACTLY rather than up to a factor (tests/test_collection.py).
    norm = 1.0 / (2.0 * np.pi) ** 2
    return TransformPhases(
        chromatic_delay=np.ascontiguousarray(chromatic_delay),
        static_phase=np.ascontiguousarray(static_phase),
        weight_sq=np.ascontiguousarray(dk * (aperture.transmission**2)[:, None] * norm),
        weight_amp=np.ascontiguousarray(dk * aperture.transmission[:, None] * norm),
    )
