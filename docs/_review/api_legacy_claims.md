# Earlier API claims retained for review

These extracts retain earlier benchmark and implementation-history prose. They
are not current validation claims. See the scientific audit for corrections.

## croak.lm

Levenberg–Marquardt retrieval via :func:`scipy.optimize.least_squares` (``lm``).

A least-squares retriever that drives the flattened FROG residual
:math:`(T_\mathrm{meas}-\mu T_\mathrm{sim})\,w/\sqrt{\mathrm{denom}}` to zero
(its Euclidean norm is exactly the FROG error :math:`R`). By default it uses
SciPy's trust-region reflective solver (``method="trf"``, of the
Levenberg–Marquardt family) with the **Jacobian supplied by JAX**
(:func:`jax.jacfwd` — forward mode, since there are far fewer parameters than
residuals); MINPACK's classic ``method="lm"`` is honoured with a numeric
Jacobian instead (see *Choice of* ``method`` below).

It shares the JAX forward model (:mod:`croak.forward_jax`), the residual builder
(:mod:`croak.metrics_jax`) and the pulse parameterisation (:mod:`croak._jax_pulse`)
with :class:`~croak.lbfgs_ad.LBFGSAD`, and supports the same forward-model
options: thin or dispersive propagation, phase-only retrieval, ``R_omega``
adaptive scaling, and amplitude / phase smoothness regularisation (added as extra
residual rows so that ``||rows||^2 == reg * penalty``).

Choice of ``method``
--------------------
The default is SciPy's trust-region least-squares ``method="trf"`` (the
Levenberg–Marquardt family), driven by the **analytic JAX Jacobian**. We do
*not* default to MINPACK's classic ``method="lm"`` because its Fortran driver
calls the user Jacobian back from compiled code, and evaluating the JAX/XLA
Jacobian program from inside that callback segfaults the interpreter (a hard
XLA-reentrancy crash; the JAX *residual* is fine, only the Jacobian program
crashes). ``"trf"`` is pure-Python SciPy, accepts the analytic Jacobian, and for
these unconstrained problems is numerically equivalent. If ``method="lm"`` is
requested explicitly it is honoured **with a numeric (finite-difference)
Jacobian** of the JAX residual, so the process cannot crash.

Because the method minimises the *sum of squares*, the regularisation weight
scales the penalty relative to :math:`R^2` (whereas :class:`croak.lbfgs.LBFGS`
adds it to :math:`R`); the penalty *functions* themselves are identical.

## croak.cmaes

Global retrieval by evolution strategy (``cma-es``), powered by evosax.

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

## croak.warm_lbfgs

L-BFGS-AD seeded from COPRA's local projection sweep.

COPRA has two stages that behave very differently. Its *local* sweep visits
delays in random order and replaces each one's signal modulus with the measured
value, which injects far more measured information per iteration than a
gradient step; its *global* stage is a gradient descent, and on the cases
measured here engaging it makes the result worse. Running the local sweep alone
therefore produces a good, cheap estimate — but one that cannot represent
geometrical smearing, because a smeared trace is an incoherent mixture with no
single signal whose modulus to replace.

This solver uses that estimate as the starting point for the full extended
model under L-BFGS. The forward model is unchanged; only the initial guess is.

What it buys, measured across the Gaussian-beam control and the apertured mask
at transform limit and +-2 fs^2 (and the mask +-0.625 pair).

Accuracy, all eight cases at the 9.5 um experimental substrate: mean complex
error 0.333 -> 0.208 (38% lower), median 0.408 -> 0.150 (63% lower), mean
``|dFWHM|`` 17.7% -> 8.5% (52% lower), worst ``|dFWHM|`` 69.3% -> 20.8%. The complex
error improved by more than 20% in four of the eight cases and degraded by more
than 20% in none; the duration improved in seven of eight. The pattern is that
it does little where the answer was already good and a great deal where it was
not: it rescues bad basins rather than refining good solutions, raising the
floor without lowering the ceiling.

Across depth, over a 1-40 um ladder, the complex-error ranges improve on most
arms -- clearly on both chirped Gaussian arms and on mask -2 fs^2 (0.33-0.55 ->
0.15-0.46), with the mask +2 arm unchanged. The retrieved-duration ranges over
depth are mixed and slightly wider on the mask arms; that alone is not the whole
picture and should not be read as one.

Variance over starting points is the largest effect. The spread in retrieved
duration collapses on every chirped arm -- five-fold on Gaussian +2 and mask -2,
twelve-fold on Gaussian -2 (+-36.9 -> +-3.1 points) and mask +2 (+-6.0 -> +-0.5)
-- while the transform-limited arms, which were never start-sensitive, are
unchanged. A retrieval whose answer moves by +-37% with the draw of the initial
guess is not a measurement whatever its mean error, and on real data one never
learns which draw one got.

One regression is known: the Gaussian transform-limited arm at depths away from
9.5 um, where the warm complex error reaches 0.63 against a cold maximum of
0.18. That is the case where the cold start was already reliable.

## croak.smearing

Geometric (pulse-front-tilt) time smearing in a non-collinear BOXCARS geometry.

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

## croak.focal

Chromatic focal mixture: the smearing kernel without its achromatic reduction.

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

## croak.collection

Finite collection aperture: the focal mixture without full-beam collection.

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

## Earlier session parameter comments

```python
class RetrieveParams:
    """Stage 3: the solver, initial guess, dispersive slab and regularisation.

    Covers the algorithm and its tolerances, the guess mode, optional dispersive
    propagation through a slab (the forward model, distinct from the stage-4
    dispersion *compensation*), the regularisation weights, the phase
    parameterisation, the CMA-ES global-search knobs and the post-filter toggles.
    """

    #: "warm-lbfgs" is the extended model seeded from COPRA's local projection
    #: sweep: same forward model as "lbfgs-ad", better starting point. It is the
    #: default because it is close to neutral where the landscape is benign and
    #: substantially better where it is not, so it is the safer choice for a
    #: user who does not want to reason about initial guesses.
    solver: str = "warm-lbfgs"
    #: 300 iterations and a 1e-5 relative tolerance: the settings every result
    #: in the smearing/solver study was measured with. 100/1e-4 stopped some
    #: dispersive retrievals short of convergence.
    maxiters: int = 300
    reltol: float = 1e-5
    abstol: float = 1e-8
    full: bool = True
    R_omega: bool = True
    random_init: bool = False
    perfect_init: bool = False
    #: Seed from the known complex field carried by a synthetic or ModelPNPS
    #: load. The selected truth must contain complex spectral datasets; intensity-
    #: only legacy files cannot provide this initial condition.
    truth_init: bool = False
    #: Start from the measured spectrum with flat phase: its transform limit.
    #: Preferred over ``perfect_init`` for near-transform-limited pulses --- the
    #: two perform equivalently, but this one uses the amplitude that was
    #: actually measured and assumes no duration. Ignored if ``random_init`` or
    #: ``perfect_init`` is set.
    tl_init: bool = False
    reuse_result: bool = False  # seed from the previous retrieval's spectrum
    # Live full-plot preview during retrieval (GUI only): redraw the visible plot page
    # ~1/s as the solver converges, rather than only the convergence curve. When
    # off, the cheap convergence curve is shown. Ignored by the headless pipeline.
    live_preview: bool = True
    perfect_init_fs: float = 10.0
    dispersive: bool = False  # dispersive propagation through a slab
    material: str = "SiO2-Franta"
    thickness_um: float = 10.0
    # Default node count kept consistent with the tied defaults below:
    # thickness_um / delta_z_um = 10 µm / 1 µm = 10.
    npoints: int = 10
    # Nominal depth step (µm) the GUI uses to pick ``npoints`` for the slab
    # integral: ``npoints = round(thickness_um / delta_z_um)``. The two are tied
    # so editing the thickness grows ``npoints`` to hold the step fixed; editing
    # ``npoints`` back-computes the step. Only ``npoints`` reaches the forward
    # model (:class:`croak.forward.ForwardModel`) — ``delta_z_um`` is the GUI knob.
    delta_z_um: float = 1.0
    # Geometric time smearing of a non-collinear BOXCARS geometry (PG/SD only,
    # autodiff solvers only). The mask hole diameter and edge-to-edge spacing are
    # the physical inputs; the resulting delay-offset width is tied to them via
    # croak.smearing.square_boxcars_delay_width at the measured carrier, so editing
    # either geometry field updates the width and editing the width back-computes
    # the spacing — the same coupling as thickness / Delta z / npoints above. Only
    # the derived kernel reaches the forward model.
    smearing: bool = False
    smear_hole_diameter_mm: float = 1.0
    smear_hole_spacing_mm: float = 0.5
    smear_sigma_delay_fs: float = 0.357
    #: COPRA step-size heuristic (Geib's alpha). Measured to be inert: a factor
    #: of five in it moves the complex error by a few percent, non-monotonically.
    copra_alpha: float = 0.25
    #: Iterations without improvement before COPRA's local projection sweep
    #: hands over to its global descent. Not a smooth knob but a switch between
    #: the two stages, and with a smeared model the stages differ in whether
    #: they can represent the physics at all: the local projection cannot (a
    #: mixture has no single signal to project onto) while the global descent
    #: can. A very large value never leaves the local sweep.
    copra_stall_patience: int = 5
    smear_nodes: int = 5
    #: Multiplier on the geometric kernel's widths, applied as a FIXED value
    #: (unlike ``fit_smearing``, which optimises it). Exists because the mask
    #: geometry can only express d/D = (gap + D)/2D >= 0.5, so a kernel
    #: narrower than that — which Gaussian-beam data wants, see below — is not
    #: reachable by adjusting the gap at all.
    #:
    #: 1.0 is correct for the apertured instrument: scanning the width against
    #: the 3D simulation at the known truth puts the optimum at 0.85-1.00 at
    #: both 9.5 and 20 um and at both d/D = 1.0 and 1.5. Gaussian-beam traces
    #: want about 0.5, on top of the 0.748 focal-weight rescale their effective
    #: gap already carries.
    smear_scale: float = 1.0
    #: Aperture layout the kernel is built from. ``"boxcars"`` is the four-hole
    #: BOXCARS mask used for TG/PG (and for a three-hole SD). ``"sd2"`` is a
    #: genuine TWO-beam self-diffraction: two collinear apertures, the probe
    #: entering twice and the gate once with conjugation.
    #:
    #: The distinction is not cosmetic. In a two-beam SD the two unconjugated
    #: arms are the SAME beam, so ``p``—which is proportional to the difference
    #: of their tilts—is identically zero and the smearing collapses to a pure
    #: delay-axis convolution. Building such a measurement's kernel from the
    #: boxcars closed form instead attaches a spurious ``sigma_p`` of order
    #: 0.4 fs that the experiment does not have.
    smear_layout: str = "boxcars"
    #: Chromatic focal mixture (``croak.focal``) instead of the reduced
    #: smearing kernel — the full model. Each focal-plane node
    #: carries the chromatic Airy filter ``A(r, omega)`` and its own definite
    #: ``(p, theta)``, so the r-dependent (bluer-on-axis) local spectra that
    #: the reduced kernel averages away are modelled explicitly. Mutually
    #: exclusive with ``smearing`` (the mixture REPLACES the kernel; asking
    #: for both raises). Autodiff solvers only. Roughly 40x the reduced
    #: kernel's cost per iteration at the default 24x16 node grid.
    focal: bool = False
    focal_n_radial: int = 24
    #: 16, not 8: the coherent (apertured) sum needs the finer azimuthal
    #: quadrature — 8 costs 4e-5 of trace peak.
    focal_n_azimuth: int = 16
    focal_rmax_units: float = 4.0
    #: Focusing focal length (mm). The reduced kernel never needs it (f
    #: cancels between spot size and tilt); the explicit mixture does. For
    #: simulated scans it is stored in the file (``f_foc``, files written
    #: after 2026-08-26) and adopted with the rest of the geometry.
    focal_f_mm: float = 100.0
    #: Collection model riding on the mixture: ``"off"`` sums incoherently
    #: over the focal plane (exact only for full-beam collection), ``"file"``
    #: rebuilds the collection aperture from a simulated scan's own
    #: ``window_def_*`` record, ``"manual"`` builds a centred hole at the
    #: phase-matched corner from ``collection_diam_mm``. A tightly apertured
    #: instrument sits far from the incoherent limit and the mixture
    #: over-blurs without this.
    collection: str = "off"
    collection_mode: str = "integrated"
    collection_diam_mm: float = 0.5
    #: Spectral-frame exponent ``p``: the measured spectrum used for the
    #: initial guess and the ``reg_spectrum`` target is reweighted by
    #: ``(omega/omega0)^(2p)`` in intensity before retrieval. The focal
    #: mixture's ``ew`` is the ON-AXIS focal field, while a measured/beamlet
    #: spectrum is spatially integrated — bluer on axis by omega^2 for an
    #: aperture-diffraction beamlet, reduced towards 0 as the collection
    #: opens (measured: ~1.0 at gap 0.5 mm, ~0.5 at 1.0 mm).
    #: Leave 0 for the reduced kernel, whose implicit frame is the
    #: mixture-averaged one. GDD/span are insensitive to it; durations move
    #: by a few %.
    spectrum_frame_p: float = 0.0
    # extra fitted parameters (lbfgs-ad / lm / lm-optx / cma-es): slab thickness
    # (dispersive PG/SD only), a delay-zero offset and the smearing strength;
    # two-phase polish is a local-solver option (cma-es frees the extras from the
    # start).
    # Generation envelope g(z) on the depth quadrature (lbfgs-ad only): the
    # transverse-geometry factor of the crossing focused beams — focal droop
    # over the Rayleigh range, beamlet walk-through over the overlap length,
    # and a net Gouy phase. Negligible for slabs much thinner than the
    # (in-medium) Rayleigh range; see docs/explanation/forward_model.md.
    # Lengths in um from the slab entrance; in-medium values.
    envelope: bool = False
    envelope_zr_um: float = 0.0
    envelope_lw_um: float = 0.0
    envelope_zf_um: float = 0.0
    envelope_gouy: float = 1.0
    fit_thickness: bool = False
    fit_tau0: bool = False
    fit_smearing: bool = False
    # fit the gate-shape (p) and delay (delta) kernel widths as two independent
    # multipliers instead of the single joint one — the diagnostic for which
    # smearing channel deviates from the geometric prediction; needs fit_smearing
    fit_smearing_split: bool = False
    polish_two_phase: bool = False
    #: Where the model's delay zero sits (AD solvers): ``"coincidence"`` (gate–probe
    #: coincidence, the forward model's native convention) or ``"marginal_peak"``
    #: (every model trace re-centred on its own delay-marginal peak, the convention
    #: the preprocessing gives the data — no free parameter). The latter is the
    #: single-cycle protocol: a coherently collected trace peaks tens of as from
    #: coincidence, growing with the chirp, and a fixed origin costs ~10 % of
    #: duration while a free ``fit_tau0`` wanders on chirped pulses (FROG N102).
    delay_origin: str = "coincidence"
    #: ``|τ0| ≤ tau0_bound_fs`` (fs) for a fitted delay offset; 0 = unbounded.
    tau0_bound_fs: float = 0.0
    #: Amplitude-smoothness weight (2nd difference of ``|E(w)|``, max-normalised).
    #: Used *together with* ``reg_spectrum``: 0.03 alongside a spectrum weight of
    #: 0.01 beat every single-penalty setting tested, on every metric at once and
    #: across eight arms at two depths -- 12% better in complex error, 2.4 points
    #: better in retrieved duration, and 2.5x smoother in the worst case, than
    #: ``reg_spectrum`` = 0.1 alone. Adding it never hurt in any configuration
    #: measured, noiseless or noisy. Same objective-scale caveat as
    #: ``reg_spectrum`` below.
    reg_amp: float = 0.03
    reg_phase: float = 0.0
    #: Spectrum-divergence weight. 1e-2 is the measured optimum for the
    #: gradient solvers: scanning it over three decades on a dispersive
    #: TG-FROG trace moves the trace error by only 14% but the pulse error by
    #: 44%, with a clear interior optimum here. The penalty does not remove
    #: misfit so much as move it out of the spectral amplitude, where it is
    #: constrained, and into the phase, where it is not -- so both a weight
    #: that is too small (amplitude error dominates) and one too large (phase
    #: error dominates) are worse. Do NOT tune this by watching the trace
    #: error: it barely responds.
    #:
    #: NOTE this value is for the L-BFGS/gradient family, which minimises the
    #: scalar R + lambda P. The LM solvers add the penalty as residual rows
    #: (R^2 + lambda' P), so the same number is far stronger there; see
    #: :data:`REG_DEFAULTS` for the per-family values and the lambda' = 2 R
    #: lambda mapping between them. croak does not translate automatically, so
    #: reset both weights when changing `solver` to "lm"/"lm-optx" (the GUI
    #: does this for you).
    reg_spectrum: float = 1e-2
    # temporal (pedestal) penalty (LBFGS-AD / cma-es) + its time window (fs)
    reg_time: float = 0.0
    reg_time_lo_fs: float = -500.0
    reg_time_hi_fs: float = 500.0
    # phase parameterisation (AD solvers + cma-es): pointwise or reduced B-spline
    phase_basis: str = "pointwise"
    n_nodes: int = 20
    # global search (cma-es)
    cma_strategy: str = "cma"
    cma_popsize: int = 0  # 0 -> evosax default
    cma_std_init: float = 0.5
    # post-retrieval cleanup (toggled on the result pane, no re-run needed)
    post_filter_lam: bool = False
    post_filter_tau: bool = False



```

## Former regularisation-default benchmark comment

```python
#: Default regularisation weights ``(reg_spectrum, reg_amp)`` per solver family.
#:
#: The two families do not minimise the same objective, so the same number does
#: not mean the same thing to them. The gradient solvers minimise
#: :math:`R + \lambda P`; the Levenberg-Marquardt solvers append the penalty as
#: residual rows, hence minimise :math:`R^2 + \lambda' P`. Matching the
#: stationary conditions pairs them as :math:`\lambda' = 2 R \lambda`, i.e. at
#: the :math:`R \approx 2\times10^{-4}` of a good dispersive TG-FROG retrieval
#: the LM weight is some three orders of magnitude smaller.
#:
#: That is not a cosmetic difference. On the 2 fs, 9.5 um dispersive trace
#: (2026-08-25), sweeping the LM weight over four decades:
#:
#: =====================  ========  ========  =========  ======
#: LM (spectrum, amp)     R (%)     eps       dFWHM (%)  rough.
#: =====================  ========  ========  =========  ======
#: 1e-2, 3e-2 (as LBFGS)  0.0237    0.305     +1.11      1.00
#: 1e-5, 3e-5 (mapped)    0.0193    0.112     -0.11      1.01
#: 0, 0                   0.0174    0.466     +3.12      2.57
#: =====================  ========  ========  =========  ======
#:
#: Both failure modes are visible: the gradient weights over-regularise LM (a
#: worse pulse *and* a worse trace error), and no penalty at all overfits (the
#: best trace error of any arm, the worst pulse -- 2.6x the truth's spectral
#: roughness and +3.1% on duration). The measured optimum, 1e-5, sits within
#: 30% of the 2 R lambda prediction, which is the evidence that the mapping is
#: the right rule rather than a coincidence of one trace.
#:
#: So the LM entry scales with the trace error you actually reach: multiply it
#: by :math:`R/2\times10^{-4}` on a noisier measurement.
```
