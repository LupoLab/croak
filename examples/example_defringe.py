"""De-fringe a DUV TG-FROG trace (Takeda FTSI).

Builds a synthetic deep-ultraviolet transient-grating FROG trace that carries
interferometric fringes on top of the signal (see :mod:`synthetic_duv_trace`),
then removes them with :func:`croak.preprocess.defringe_carrier` — a smooth
per-wavelength low-pass that keeps the slow magnitude band below the optical
carrier ``f_c(λ) = c/λ`` and discards the ``±f_c`` sidebands. Run::

    uv run python examples/example_defringe.py

Because the trace is synthetic, the fringe-free magnitude ``a = |E_s|² + |E_b|²``
is known exactly, so the example reports the *actual* recovery error rather than
only showing that something happened. Writes ``defringe_check.png``; read it as:

* **residual** (measured − de-fringed) should be the diagonal fringe pattern
  *only* — any slow envelope leaking in means ``CARRIER_FRACTION`` is too low;
* **row cuts** — the thick de-fringed curve should ride the centre of the
  measured oscillation and sit on the dashed truth;
* **delay / wavelength marginals** should overlay (signal preserved, only the
  zero-mean fringe removed).

Note the de-fringed trace still contains the broad leakage background |E_b|²;
removing that is a separate step, shown in ``example_arpls_baseline.py``.
"""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from synthetic_duv_trace import make_fringed_trace

from croak.preprocess import defringe_carrier

# De-fringe cutoff as a fraction of the carrier f_c = c/λ (in (0, 1)). It must
# sit above the trace's own slow delay-bandwidth and below the fringe carrier;
# anywhere in ~0.3–0.6 works here, and the recovery error is insensitive to it.
CARRIER_FRACTION = 0.5


def main() -> None:
    """Build the fringed trace, de-fringe it and write the comparison figure."""
    ft = make_fringed_trace()

    # The DUV carrier (≈0.83 fs period at 250 nm) needs δτ < λ/(2c) to avoid
    # aliasing. synthetic_duv_trace.DELAY_STEP satisfies this with margin, so the
    # default on_alias="raise" is the right setting: on real data a raise here
    # means the scan was stepped too coarsely to separate the fringes at all.
    defr = defringe_carrier(
        ft.measured, ft.lam, ft.tau, carrier_fraction=CARRIER_FRACTION
    )

    peak = ft.magnitude.max()
    print(f"trace: {ft.lam.size} wavelengths × {ft.tau.size} delays")
    print(
        f"max error vs true magnitude: {np.abs(defr - ft.magnitude).max() / peak:.2%}"
    )
    print(f"RMS error vs true magnitude: {np.std(defr - ft.magnitude) / peak:.2%}")

    _plot(ft, defr)


def _plot(ft, defr) -> None:
    """Six-panel measured-vs-defringed diagnostic; writes ``defringe_check.png``."""
    tau_fs = ft.tau / 1e-15
    lam_nm = ft.lam / 1e-9
    trace = ft.measured

    # Zoom the 2-D maps to the signal-bearing wavelength band (with a margin), so
    # the compact signal and the diagonal fringes are not a thin strip.
    marg_lam = trace.sum(axis=1)
    sig = np.where(marg_lam > 0.05 * marg_lam.max())[0]
    lo, hi = lam_nm[sig[0]], lam_nm[sig[-1]]
    pad = 0.25 * (hi - lo)
    band = (lam_nm >= lo - pad) & (lam_nm <= hi + pad)
    vmax = float(trace[band].max())

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))

    ax[0, 0].pcolormesh(tau_fs, lam_nm, trace, shading="auto", vmin=0, vmax=vmax)
    ax[0, 0].set_title("measured (fringed)")
    ax[0, 0].set_ylabel("λ (nm)")
    ax[0, 1].pcolormesh(tau_fs, lam_nm, defr, shading="auto", vmin=0, vmax=vmax)
    ax[0, 1].set_title("de-fringed a(λ,τ)")
    rvmax = float(np.abs((trace - defr)[band]).max())
    ax[0, 2].pcolormesh(
        tau_fs,
        lam_nm,
        trace - defr,
        shading="auto",
        cmap="RdBu_r",
        vmin=-rvmax,
        vmax=rvmax,
    )
    ax[0, 2].set_title("residual = measured − de-fringed (the fringes)")
    for a in ax[0]:
        a.set_ylim(lo - pad, hi + pad)
        a.set_xlabel("delay (fs)")

    # Row cuts at the three brightest signal wavelengths, against the known truth.
    rows = np.argsort(trace.sum(axis=1))[-3:]
    for n, i in enumerate(rows):
        (line,) = ax[1, 0].plot(tau_fs, trace[i], lw=0.7, alpha=0.4)
        ax[1, 0].plot(
            tau_fs, defr[i], lw=1.6, color=line.get_color(), label=f"{lam_nm[i]:.0f} nm"
        )
        # One legend entry for the truth, not one per row — they are all black.
        ax[1, 0].plot(
            tau_fs,
            ft.magnitude[i],
            "--",
            lw=1.0,
            color="k",
            label="true" if n == 0 else None,
        )
    ax[1, 0].set_title("row cuts: measured (thin) · de-fringed (thick) · true (dashed)")
    ax[1, 0].set_xlabel("delay (fs)")
    ax[1, 0].legend(fontsize=7)

    ax[1, 1].plot(tau_fs, trace.sum(axis=0), label="measured")
    ax[1, 1].plot(tau_fs, defr.sum(axis=0), label="de-fringed")
    ax[1, 1].plot(tau_fs, ft.magnitude.sum(axis=0), "k--", lw=1.0, label="true")
    ax[1, 1].set_title("delay marginal (signal preserved?)")
    ax[1, 1].set_xlabel("delay (fs)")
    ax[1, 1].legend()

    ax[1, 2].plot(lam_nm, trace.sum(axis=1), label="measured")
    ax[1, 2].plot(lam_nm, defr.sum(axis=1), label="de-fringed")
    ax[1, 2].plot(lam_nm, ft.magnitude.sum(axis=1), "k--", lw=1.0, label="true")
    ax[1, 2].set_title("wavelength marginal")
    ax[1, 2].set_xlabel("λ (nm)")
    ax[1, 2].legend()

    fig.tight_layout()
    fig.savefig("defringe_check.png", dpi=120)
    print("wrote defringe_check.png")


if __name__ == "__main__":
    main()
