"""The :class:`RetrievalResult` returned by every retrieval algorithm."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .grid import Grid

__all__ = ["RetrievalResult"]


@dataclass
class RetrievalResult:
    r"""Outcome of a pulse retrieval, uniform across all algorithms.

    Attributes
    ----------
    spectrum : numpy.ndarray
        Retrieved complex spectrum :math:`\tilde E(\omega)` (centred order).
    grid : Grid
        Time/frequency grid the retrieval ran on.
    delays : numpy.ndarray
        Delay axis (s).
    interaction : str
        Interaction name (``"shg"``/``"sd"``/``"pg"``).
    algorithm : str
        Algorithm name — one of the keys of
        :data:`croak.retrieve.ALGORITHMS` (``"copra"``, ``"copra-jax"``,
        ``"lbfgs"``, ``"lbfgs-hand"``, ``"lbfgs-ad"``, ``"lbfgs-optx"``,
        ``"lm"``, ``"lm-optx"``, ``"cma-es"``).
    error : float
        Final FROG error ``R``.
    errors : list of float
        FROG error at each iteration.
    trace : numpy.ndarray
        Retrieved (simulated) trace, ``(Nomega, Ndelay)``, optimally scaled.
    mu : float or numpy.ndarray
        Final intensity scale factor (scalar, or per-frequency in ``Rω`` mode).
    thickness : float or None
        Fitted dispersive-slab thickness (m) when the retriever fitted it as an
        extra parameter (``lbfgs-ad``/``lm`` with ``fit_thickness``); ``None``
        when thickness was held fixed.
    tau0 : float
        Fitted delay-zero offset (s) when the retriever fitted it
        (``fit_tau0``); ``0.0`` otherwise. Sign convention: the true delay-zero
        of the measured trace lies at ``+tau0``, i.e. the simulated gate uses the
        delay phase ``exp(iω(τ − τ0))``.
    smear_scale : float or None
        Fitted multiplier on the geometric-smearing kernel widths when the
        retriever fitted it (``fit_smearing``); ``None`` when the kernel was held
        fixed or absent. ``1.0`` means the supplied kernel was exactly right.
        For a split fit (``fit_smearing_split``) this is the gate-shape
        (``p``) channel multiplier.
    smear_scale_delta : float or None
        Fitted multiplier on the delay (``delta``) kernel width when the two
        smearing channels were fitted separately (``fit_smearing_split``);
        ``None`` for a joint fit (where ``smear_scale`` covers both channels)
        or when smearing was not fitted.
    stage_boundaries : list of int
        Indices into ``errors`` where one stage of a multi-stage solver handed
        over to the next, so a convergence plot can mark the join. Empty for the
        single-stage solvers. A spliced curve needs this: ``warm-lbfgs`` counts
        one COPRA *iteration* (a full local sweep over every delay) before the
        handover and one L-BFGS *function evaluation* after it, and those are
        not the same unit of work — an unmarked kink there reads as convergence
        behaviour rather than as a change of algorithm.
    """

    spectrum: NDArray[np.complex128]
    grid: Grid
    delays: NDArray[np.float64]
    interaction: str
    algorithm: str
    error: float
    errors: list[float] = field(default_factory=list)
    trace: NDArray[np.float64] | None = None
    mu: float | NDArray[np.float64] = 1.0
    omega0: float = 0.0
    thickness: float | None = None
    tau0: float = 0.0
    smear_scale: float | None = None
    smear_scale_delta: float | None = None
    stage_boundaries: list[int] = field(default_factory=list)

    # -- derived quantities -------------------------------------------------
    @property
    def omega(self) -> NDArray[np.float64]:
        """Angular-frequency axis (rad/s, centred)."""
        return self.grid.omega

    @property
    def wavelength(self) -> NDArray[np.float64]:
        r"""Absolute wavelength axis (m): :math:`2\pi c/(\omega+\omega_0)`."""
        from .maths import wlfreq

        return wlfreq(self.grid.omega + self.omega0)

    @property
    def t(self) -> NDArray[np.float64]:
        """Time axis (s)."""
        return self.grid.t

    @property
    def field(self) -> NDArray[np.complex128]:
        """Retrieved temporal field :math:`E(t)`."""
        return self.grid.ifft(self.spectrum)

    @property
    def intensity_omega(self) -> NDArray[np.float64]:
        r"""Spectral intensity :math:`|\tilde E(\omega)|^2`."""
        return np.abs(self.spectrum) ** 2

    @property
    def phase_omega(self) -> NDArray[np.float64]:
        r"""Spectral phase :math:`\arg \tilde E(\omega)`, unwrapped."""
        return np.unwrap(np.angle(self.spectrum))

    @property
    def intensity_t(self) -> NDArray[np.float64]:
        """Temporal intensity :math:`|E(t)|^2`."""
        return np.abs(self.field) ** 2

    @property
    def phase_t(self) -> NDArray[np.float64]:
        r"""Temporal phase :math:`\arg E(t)`, unwrapped."""
        return np.unwrap(np.angle(self.field))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        """Return a concise summary repr (algorithm, interaction, error)."""
        return (
            f"RetrievalResult(algorithm={self.algorithm!r}, "
            f"interaction={self.interaction!r}, error={self.error:.3e}, "
            f"iterations={len(self.errors)})"
        )
