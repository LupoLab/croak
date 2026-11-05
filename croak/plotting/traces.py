"""FROG-trace images: colormaps, signed images, residual, marginals and composites.

Custom colormaps (white at the low end; a red band for negatives), the
sign-splitting :func:`signed_pcolormesh`, the residual and marginal panels, and
the trace composites :func:`plot_frog_filter` (with its artist-reusing twin
:class:`FrogFilterView`) and :func:`plot_simulated_trace`.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

from ..processing import marginals
from .artists import MarginalArtists
from .figure import _prepare_fig
from .units import _TWOPI_PHZ

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Colormaps
# ---------------------------------------------------------------------------
def cmap_white(base: str = "viridis", *, n: int = 8, white: bool = True, N: int = 512):
    """Build a base colormap with (optionally) a white point at the low end."""
    import matplotlib as mpl

    cm = mpl.colormaps[base]
    vals = np.linspace(0, 1, n)
    cols = cm(vals)
    if white:
        cols[0] = [1, 1, 1, 1]
    xi = np.linspace(0, 1, N)
    out = np.empty((N, 4))
    for k in range(4):
        out[:, k] = np.interp(xi, vals, cols[:, k])
    return ListedColormap(out)


def cmap_negwhite(
    base: str = "viridis",
    *,
    n: int = 8,
    white: bool = True,
    negfrac: float = 0.0,
    N: int = 512,
):
    """Like :func:`cmap_white` but with a red gradient for negative values."""
    if negfrac <= 0:
        return cmap_white(base, n=n, white=white, N=N)
    n_pos = max(round(N * (1 - negfrac)), 2)
    n_neg = N - n_pos
    pos = cmap_white(base, n=n, white=white, N=n_pos)(np.linspace(0, 1, n_pos))
    neg = np.empty((n_neg, 4))
    for i in range(n_neg):
        t = i / max(n_neg - 1, 1)
        neg[i] = [1.0, t, t, 1.0]
    return ListedColormap(np.vstack([neg, pos]))


def _white_red():
    """White→red colormap used to render negative trace values by magnitude."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("white_red", ["white", "red"])


def _signed_magnitude(
    C, *, db: bool, tracedb: float, vmax: float | None
) -> tuple[NDArray[np.float64], NDArray[np.bool_], float, float]:
    """Split ``C`` into magnitude, negative mask and the shared ``(vmin, vmax)``.

    One code path for drawing and for updating in place.
    """
    C = np.asarray(C, dtype=float)
    neg = C < 0
    if db:
        mag = np.clip(10 * np.log10(np.maximum(np.abs(C), 1e-30)), -tracedb, 0.0)
        return mag, neg, -tracedb, 0.0
    mag = np.abs(C)
    return mag, neg, 0.0, float(np.nanmax(mag)) if vmax is None else vmax


def signed_pcolormesh(ax, x, y, C, *, db, tracedb=30.0, cmap_pos, vmax=None):
    """Pcolormesh that splits an image's sign onto two colormaps.

    Positive samples are drawn with ``cmap_pos`` (e.g. viridis); negative
    samples are drawn by *magnitude* with a white→red map. Both maps share the
    **same** scale (``[0, vmax]`` linear, ``[-tracedb, 0]`` dB), so a negative is
    only as saturated as a positive of equal magnitude — small negatives stay
    faint instead of dominating — and zero is white in both (pair ``cmap_pos``
    with :func:`cmap_white`). ``C`` follows the usual
    :external+matplotlib:meth:`~matplotlib.axes.Axes.pcolormesh` convention —
    shape ``(len(y), len(x))`` (a FROG ``(Nλ, Nτ)`` array already has this shape
    for x = delay and y = wavelength).

    Returns ``(mappable_pos, mappable_neg)`` (the second is ``None`` when there
    are no negative samples).
    """
    mag, neg, vmin, vhi = _signed_magnitude(C, db=db, tracedb=tracedb, vmax=vmax)
    mpos = ax.pcolormesh(
        x,
        y,
        np.ma.masked_where(neg, mag),
        vmin=vmin,
        vmax=vhi,
        cmap=cmap_pos,
        rasterized=True,
        shading="auto",
    )
    mneg = None
    if neg.any():
        mneg = ax.pcolormesh(
            x,
            y,
            np.ma.masked_where(~neg, mag),
            vmin=vmin,
            vmax=vhi,
            cmap=_white_red(),
            rasterized=True,
            shading="auto",
        )
    return mpos, mneg


def plot_residual(ax, omega, delays, resid, omega0, *, halfwidth=None, flim=None):
    """Plot the measured−retrieved residual with a symmetric blue-white-red map.

    ``flim`` optionally bounds the frequency axis, as ``(f_min, f_max)`` in Hz.
    """
    mr = float(np.max(np.abs(resid))) or 1.0
    freq = (np.asarray(omega) + omega0) / _TWOPI_PHZ
    m = ax.pcolormesh(
        np.asarray(delays) / 1e-15,
        freq,
        resid,
        vmin=-mr,
        vmax=mr,
        cmap=cmap_white("bwr", n=32, white=False),
        rasterized=True,
        shading="auto",
    )
    ax.set_xlabel("Delay (fs)")
    ax.set_ylabel("Frequency (PHz)")
    ax.set_title("Residuals")
    if halfwidth is not None:
        ax.set_xlim(-halfwidth / 1e-15, halfwidth / 1e-15)
    if flim is not None:
        ax.set_ylim(flim[0] / 1e15, flim[1] / 1e15)
    return m


def plot_marginal(
    ax, x, retrieved, measured=None, *, xlabel="", scale=1.0, halfwidth=None
) -> MarginalArtists:
    """Plot a (retrieved vs measured) trace marginal."""
    (line_retr,) = ax.plot(np.asarray(x) / scale, retrieved, c="C0", label="R")
    line_meas = None
    if measured is not None:
        (line_meas,) = ax.plot(np.asarray(x) / scale, measured, c="C1", label="M")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Marginal (a.u.)")
    ax.set_ylim(0, 1.05)
    if halfwidth is not None:
        ax.set_xlim(-halfwidth / scale, halfwidth / scale)
    legend = ax.legend(
        loc="upper left", handlelength=1.0, frameon=False, fontsize="small"
    )
    return MarginalArtists(line_retr=line_retr, legend=legend, line_meas=line_meas)


def filter_view_vmax(td) -> float:
    """Return the shared positive colour scale for the six-panel filter view.

    Take the largest **positive** sample across the measured, filtered and
    regridded traces, restricting the two wavelength-domain arrays to the
    measurement window ``td.lamm_lims`` when set.

    Both restrictions matter on real data. An absolute-calibration curve
    amplifies the background-subtracted noise far outside the signal band, so
    scaling by ``max|·|`` lets a large *negative* excursion set the *positive*
    scale: on one of our measured deep-UV traces, whose signal peaks at ``+1``
    but dips to ``-5.7`` at 1097 nm, every linear panel rendered its signal at
    18 % of the colour range.

    The three panels share this one number so the row's single colourbar means
    something; each array is peak-normalised upstream, so they are comparable.
    If no positive sample remains, fall back to the largest absolute value,
    then 1.0, so the returned scale is always positive.
    """

    def _positive_max(arr, lam=None) -> float:
        a = np.asarray(arr, dtype=float)
        if lam is not None and td.lamm_lims is not None:
            lo, hi = td.lamm_lims
            band = (np.asarray(lam, dtype=float) >= lo) & (
                np.asarray(lam, dtype=float) <= hi
            )
            if band.any():
                a = a[band]
        hi = float(np.nanmax(a)) if a.size else 0.0
        return hi if hi > 0.0 else 0.0

    vmax = max(
        _positive_max(td.Ifrog_meas, td.lam_frog),
        _positive_max(td.Ifrog_filt, td.lam_frog),
        _positive_max(td.trace),
    )
    if vmax > 0.0:
        return vmax
    # Degenerate (all-negative or all-zero) trace: keep a usable, positive scale.
    fallback = max(
        float(np.nanmax(np.abs(td.Ifrog_meas))),
        float(np.nanmax(np.abs(td.Ifrog_filt))),
        float(np.nanmax(np.abs(td.trace))),
    )
    return fallback if fallback > 0.0 else 1.0


def plot_frog_filter(
    td,
    *,
    tracedb: float = 30.0,
    cmap: str = "viridis",
    figsize=(9.45, 4.72),
    fig: Figure | None = None,
) -> Figure:
    """6-panel measured/filtered (wavelength-domain) + regridded (frequency-domain).

    Positive values use the viridis (white-low) map; negative values — kept
    through filtering rather than clamped — are drawn by magnitude on a
    white→red map so they are visible and not mistaken for signal. Each row
    (the dB row and the linear row) carries one positive + one negative
    colorbar on the right-hand side.
    """
    fig = _prepare_fig(fig, figsize)
    axd = fig.subplot_mosaic("abe\ncdf")
    cm = cmap_white(cmap)
    # The linear row shares a single scale so its one colorbar is meaningful.
    lin_vmax = filter_view_vmax(td)

    def _panel(ax, x, y, C, *, db, title, xlabel):
        mpos, mneg = signed_pcolormesh(
            ax,
            x,
            y,
            C,
            db=db,
            tracedb=tracedb,
            cmap_pos=cm,
            vmax=None if db else lin_vmax,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Delay (fs)")
        ax.set_title(title)
        return mpos, mneg

    def _row_cbars(axes, mpos, negs):
        """One positive + (if any) one negative colorbar at the row's right."""
        fig.colorbar(mpos, ax=axes, shrink=0.5)
        mneg = next((m for m in negs if m is not None), None)
        if mneg is not None:
            fig.colorbar(mneg, ax=axes, shrink=0.5, label="neg |·|")

    lam_nm = td.lam_frog / 1e-9
    freq = (td.omega + td.omega0_trace) / _TWOPI_PHZ
    WL = "Wavelength (nm)"
    HZ = "Frequency (PHz)"

    # top row: dB (log) — measured, filtered, regridded
    pa, na = _panel(
        axd["a"],
        lam_nm,
        td.tau_meas / 1e-15,
        td.Ifrog_meas.T,
        db=True,
        title="Measured (dB)",
        xlabel=WL,
    )
    _pb, nb = _panel(
        axd["b"],
        lam_nm,
        td.tau_filt / 1e-15,
        td.Ifrog_filt.T,
        db=True,
        title="Filtered (dB)",
        xlabel=WL,
    )
    _pe, ne = _panel(
        axd["e"],
        freq,
        td.delays / 1e-15,
        td.trace.T,
        db=True,
        title="Regridded (dB)",
        xlabel=HZ,
    )
    _row_cbars([axd["a"], axd["b"], axd["e"]], pa, (nb, ne, na))

    # bottom row: linear — measured, filtered, regridded
    pc, nc = _panel(
        axd["c"],
        lam_nm,
        td.tau_meas / 1e-15,
        td.Ifrog_meas.T,
        db=False,
        title="Measured (lin)",
        xlabel=WL,
    )
    _pd, nd = _panel(
        axd["d"],
        lam_nm,
        td.tau_filt / 1e-15,
        td.Ifrog_filt.T,
        db=False,
        title="Filtered (lin)",
        xlabel=WL,
    )
    _pf, nf = _panel(
        axd["f"],
        freq,
        td.delays / 1e-15,
        td.trace.T,
        db=False,
        title="Regridded (lin)",
        xlabel=HZ,
    )
    _row_cbars([axd["c"], axd["d"], axd["f"]], pc, (nd, nf, nc))
    return fig


def _decimate_for_display(C, x, y, *, cap=1000):
    """Stride a ``(ny, nx)`` image and its axes down to at most ``cap`` per side.

    The filter figure panels are only a few hundred screen pixels wide, so
    rasterising a multi-thousand-sample trace is wasted work. Plain strided
    subsampling is enough for a diagnostic preview (no anti-alias needed — the
    physics-correct arrays are kept in ``td``; this only thins what is drawn).
    """
    ny, nx = C.shape
    sy = max(ny // cap, 1)
    sx = max(nx // cap, 1)
    if sy == 1 and sx == 1:
        return C, x, y
    return C[::sy, ::sx], x[::sx], y[::sy]


def _resample_uniform_x(x, C):
    """Resample ``C`` (``(ny, nx)``) along axis 1 onto a uniform ``x`` grid.

    :external+matplotlib:class:`~matplotlib.image.AxesImage` (``imshow``) assumes evenly
    spaced pixels and is positioned by a single linear ``extent``. A *wavelength* axis
    derived from a uniform **frequency** grid (the simulated-trace case) is strongly
    non-uniform, so a feature would otherwise be painted at the wrong wavelength (e.g. a
    260 nm signal drawn near 480 nm over a 150–800 nm span). Linearly interpolating each
    delay row onto an evenly spaced ``x`` restores the correct positions; an
    already-uniform axis (the frequency panels, or a uniform-λ spectrometer trace) is
    returned unchanged. ``x`` must be ascending.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return x, C
    d = np.diff(x)
    if np.allclose(d, d[0], rtol=1e-3, atol=0.0):
        return x, C  # already uniform — no interpolation needed
    xu = np.linspace(x[0], x[-1], x.size)
    Cu = np.empty_like(C)
    for i in range(C.shape[0]):
        Cu[i] = np.interp(xu, x, C[i])
    return xu, Cu


class FrogFilterView:
    """Maintain a preprocessing figure for repeated GUI updates.

    The artist-reusing twin of :func:`plot_frog_filter`, which clears and
    rebuilds the whole figure (6 panels × up to 2 ``pcolormesh`` + 4
    colourbars) on every call: ~150–400 ms, the dominant cost of the GUI
    preprocessing redraw. Call :meth:`attach` once to create the panels, images
    and colourbars, then :meth:`update` to push new data via
    ``set_data``/``set_extent``/``set_clim`` onto the persistent images, which
    benchmarks ~4× faster. Positive and negative samples use the same
    signed-magnitude convention as :func:`signed_pcolormesh`. Non-uniform
    wavelength arrays are interpolated onto a uniform display grid; the
    underlying :class:`croak.preprocess.TraceData` arrays are unchanged.
    """

    _PANELS = (
        # key, title, xlabel-kind ("wl"/"hz"), db
        ("a", "Measured (dB)", "wl", True),
        ("b", "Filtered (dB)", "wl", True),
        ("e", "Regridded (dB)", "hz", True),
        ("c", "Measured (lin)", "wl", False),
        ("d", "Filtered (lin)", "wl", False),
        ("f", "Regridded (lin)", "hz", False),
    )

    def __init__(
        self, *, tracedb: float = 30.0, cmap: str = "viridis", cap: int = 1000
    ):
        self.tracedb = tracedb
        self.cap = cap
        self._cmap_pos = cmap_white(cmap)
        self._cmap_pos.set_bad(alpha=0.0)  # masked (negative) pixels stay clear
        self._cmap_neg = _white_red()
        self._cmap_neg.set_bad(alpha=0.0)
        self.fig = None
        self.axd = None
        self.images: dict[str, tuple] = {}

    @property
    def attached(self) -> bool:
        """Whether a figure is currently attached to this view."""
        return self.fig is not None

    def attach(self, fig: Figure) -> None:
        """Build the mosaic, images and colourbars once into ``fig``."""
        self.fig = fig
        fig.clear()
        with contextlib.suppress(Exception):
            fig.set_layout_engine("constrained")
        self.axd = fig.subplot_mosaic("abe\ncdf")
        empty = np.zeros((1, 1))
        for key, title, kind, _db in self._PANELS:
            ax = self.axd[key]
            pos = ax.imshow(
                empty,
                origin="lower",
                aspect="auto",
                cmap=self._cmap_pos,
                interpolation="nearest",
            )
            neg = ax.imshow(
                empty,
                origin="lower",
                aspect="auto",
                cmap=self._cmap_neg,
                interpolation="nearest",
            )
            self.images[key] = (pos, neg)
            ax.set_title(title)
            ax.set_xlabel("Wavelength (nm)" if kind == "wl" else "Frequency (PHz)")
            ax.set_ylabel("Delay (fs)")
        # One positive + one negative colorbar per row (dB row a/b/e, lin row c/d/f).
        self._cb_top = fig.colorbar(
            self.images["a"][0], ax=[self.axd[k] for k in "abe"], shrink=0.5
        )
        self._cb_top_neg = fig.colorbar(
            self.images["a"][1],
            ax=[self.axd[k] for k in "abe"],
            shrink=0.5,
            label="neg |·|",
        )
        self._cb_bot = fig.colorbar(
            self.images["c"][0], ax=[self.axd[k] for k in "cdf"], shrink=0.5
        )
        self._cb_bot_neg = fig.colorbar(
            self.images["c"][1],
            ax=[self.axd[k] for k in "cdf"],
            shrink=0.5,
            label="neg |·|",
        )

    def _set_panel(self, key, x, y, C, *, db, vmax=None):
        if self.axd is None:
            raise RuntimeError("live plot is not attached to a figure")
        pos_im, neg_im = self.images[key]
        C = np.asarray(C, dtype=float)
        C, x, y = _decimate_for_display(C, np.asarray(x), np.asarray(y), cap=self.cap)
        # imshow paints evenly spaced pixels; a wavelength axis built from a
        # uniform frequency grid (simulated traces) is non-uniform, so resample
        # onto an even axis to place features at the right wavelength.
        x, C = _resample_uniform_x(x, C)
        neg = C < 0
        if db:
            mag = np.clip(
                10 * np.log10(np.maximum(np.abs(C), 1e-30)), -self.tracedb, 0.0
            )
            vmin, vhi = -self.tracedb, 0.0
        else:
            mag = np.abs(C)
            vmin = 0.0
            vhi = float(np.nanmax(mag)) if vmax is None else vmax
        extent = (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))
        for im, masked in (
            (pos_im, np.ma.masked_where(neg, mag)),
            (neg_im, np.ma.masked_where(~neg, mag)),
        ):
            im.set_data(masked)
            im.set_clim(vmin, vhi)
            im.set_extent(extent)
        ax = self.axd[key]
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    def update(self, td) -> None:
        """Push a new :class:`~croak.preprocess.TraceData` onto the existing artists."""
        if not self.attached:
            raise RuntimeError("attach(fig) before update(td)")
        lam_nm = td.lam_frog / 1e-9
        freq = (td.omega + td.omega0_trace) / _TWOPI_PHZ
        lin_vmax = filter_view_vmax(td)
        self._set_panel("a", lam_nm, td.tau_meas / 1e-15, td.Ifrog_meas.T, db=True)
        self._set_panel("b", lam_nm, td.tau_filt / 1e-15, td.Ifrog_filt.T, db=True)
        self._set_panel("e", freq, td.delays / 1e-15, td.trace.T, db=True)
        self._set_panel(
            "c", lam_nm, td.tau_meas / 1e-15, td.Ifrog_meas.T, db=False, vmax=lin_vmax
        )
        self._set_panel(
            "d", lam_nm, td.tau_filt / 1e-15, td.Ifrog_filt.T, db=False, vmax=lin_vmax
        )
        self._set_panel(
            "f", freq, td.delays / 1e-15, td.trace.T, db=False, vmax=lin_vmax
        )
        # The linear row's scale changes with the data; refresh its colorbars.
        self._cb_bot.update_normal(self.images["c"][0])
        self._cb_bot_neg.update_normal(self.images["c"][1])


def plot_simulated_trace(
    omega,
    delays,
    trace,
    omega0_trace,
    *,
    tracedb: float = 40.0,
    cmap: str = "viridis",
    figsize=(7.87, 5.51),
    fig: Figure | None = None,
) -> Figure:
    """4-panel view of a simulated trace: dB, linear, and the two marginals."""
    trace = np.asarray(trace, dtype=float)
    fig = _prepare_fig(fig, figsize)
    axd = fig.subplot_mosaic("ab\ncd")
    cm = cmap_white(cmap)
    freq = (omega + omega0_trace) / _TWOPI_PHZ

    def _panel(ax, *, db, title):
        mpos, mneg = signed_pcolormesh(
            ax, delays / 1e-15, freq, trace, db=db, tracedb=tracedb, cmap_pos=cm
        )
        ax.set_xlabel("Delay (fs)")
        ax.set_ylabel("Frequency (PHz)")
        ax.set_title(title)
        return mpos, mneg

    ma, na = _panel(axd["a"], db=True, title="FROG (dB)")
    mb, nb = _panel(axd["b"], db=False, title="FROG (lin)")
    fig.colorbar(ma, ax=[axd["a"]], shrink=0.6)
    if na is not None:
        fig.colorbar(na, ax=[axd["a"]], shrink=0.6, label="neg |·|")
    fig.colorbar(mb, ax=[axd["b"]], shrink=0.6)
    if nb is not None:
        fig.colorbar(nb, ax=[axd["b"]], shrink=0.6, label="neg |·|")
    wmarg, tmarg = marginals(trace)
    axd["c"].plot(delays / 1e-15, tmarg, c="C0")
    axd["c"].set_xlabel("Delay (fs)")
    axd["c"].set_ylabel("Marginal")
    axd["c"].set_ylim(0, 1.05)
    axd["d"].plot((omega + omega0_trace) / _TWOPI_PHZ, wmarg, c="C0")
    axd["d"].set_xlabel("Frequency (PHz)")
    axd["d"].set_ylabel("Marginal")
    axd["d"].set_ylim(0, 1.05)
    return fig
