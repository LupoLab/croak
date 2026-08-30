r"""Reduce a ModelPNPS ``scansave`` scan file to a small, shippable dataset.

The full output of a 3D instrument simulation stores several collection
windows, every saved propagation slice, and the transverse grids — hundreds of
megabytes. The retrieval loader (:func:`croak.io.read_simulated_scan`) needs
none of the transverse data and usually only one window and a few thicknesses.
This script copies just that subset into a new HDF5 file:

* one trace window, stored under the default name ``Iω_win`` together with its
  ``window_def_*`` record (a multi-window scan's ``Iω_win_5`` can be selected
  and is renamed on the way);
* the propagation slices nearest the requested thicknesses, with ``grid/zsave``
  subset to match;
* the whole ``/grid`` group except the large transverse arrays
  (``r``, ``xywin``, ``x``, ``y``, ``sidx``);
* ``/scanvariables`` and ``/scanorder`` unchanged.

Optionally the mask geometry (``mask_diam``, ``mask_spacing``, ``f_foc``,
``geometry``) can be written into files that predate the simulator recording
it, so the reduced file is self-describing for the smearing kernel.

Example (the datasets shipped in ``examples/data/`` were made this way)::

    python tools/reduce_scansave.py scan_collected.h5 examples/data/out.h5 \\
        --window "Iω_win" --z-um 4 9.5 40 \\
        --mask-geometry 1e-3 1e-3 0.1 tg
"""

import argparse

import h5py
import numpy as np

# Transverse-grid datasets the retrieval never reads; dropping them is most of
# the size reduction. ``kx``/``ky`` stay: the collection-window record needs the
# k-grid spacing, and they are a few kilobytes.
DROP = {"r", "xywin", "x", "y", "sidx"}


def window_def_prefix(window_key: str) -> str:
    """Return the ``window_def`` prefix matching a trace-window dataset name."""
    base = window_key.removesuffix("_reimaged")
    if base == "Iω_win":
        return "window_def"
    return "window_def_" + base.removeprefix("Iω_win_")


def main() -> None:
    """Run the reduction from the command line."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("infile")
    ap.add_argument("outfile")
    ap.add_argument("--window", default="Iω_win", help="trace window to keep")
    ap.add_argument(
        "--z-um",
        type=float,
        nargs="+",
        default=None,
        help="thicknesses (µm) to keep; nearest saved slices are selected",
    )
    ap.add_argument(
        "--mask-geometry",
        nargs=4,
        metavar=("DIAM", "SPACING", "F_FOC", "GEOMETRY"),
        default=None,
        help="write mask_diam/mask_spacing/f_foc (m) and geometry if absent",
    )
    args = ap.parse_args()

    with h5py.File(args.infile, "r") as fin, h5py.File(args.outfile, "w") as fout:
        gin = fin["grid"]
        if not isinstance(gin, h5py.Group):
            raise SystemExit("not a scansave file: /grid is not a group")

        # Resolve the propagation slices to keep.
        zsave = np.asarray(gin["zsave"]) if "zsave" in gin else None
        if args.z_um is not None:
            if zsave is None:
                raise SystemExit("--z-um given but the file has no /grid/zsave")
            idx = sorted({int(np.argmin(np.abs(zsave - z * 1e-6))) for z in args.z_um})
        else:
            idx = None

        # The chosen window, subset along the middle (nz) axis and stored under
        # the default name so the loader finds it without arguments.
        win = np.asarray(fin[args.window])
        if idx is not None and win.ndim == 3:
            win = win[:, idx, :]
        fout.create_dataset("Iω_win", data=win, compression="gzip", shuffle=True)

        # /grid, minus the transverse arrays and the other windows' records.
        keep_prefix = window_def_prefix(args.window)
        gout = fout.create_group("grid")
        for k in gin:
            if k in DROP:
                continue
            if k.startswith("window_def"):
                continue  # rewritten below under the unprefixed name
            if k == "zsave" and idx is not None:
                gout.create_dataset("zsave", data=zsave[idx])
                continue
            gin.copy(k, gout)
        for k in gin:
            if k.startswith(keep_prefix + "_"):
                suffix = k.removeprefix(keep_prefix + "_")
                gin.copy(k, gout, name="window_def_" + suffix)

        if args.mask_geometry is not None:
            diam, spacing, f_foc, geometry = args.mask_geometry
            for name, value in (
                ("mask_diam", float(diam)),
                ("mask_spacing", float(spacing)),
                ("f_foc", float(f_foc)),
            ):
                if name not in gout:
                    gout.create_dataset(name, data=value)
            if "geometry" not in gout:
                gout.create_dataset("geometry", data=np.bytes_(geometry))

        for k in ("scanvariables", "scanorder"):
            if k in fin:
                fin.copy(k, fout)

    print(f"wrote {args.outfile}")


if __name__ == "__main__":
    main()
