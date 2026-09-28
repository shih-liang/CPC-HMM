"""Convert supplied per-hemisphere eigenmodes and cortical masks to analysis arrays.

Inputs are geometry-derived eigenmodes, not CPCA estimates. Their mesh, boundary
conditions and eigenvalue ordering must come from the generating protocol.
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def array(path):
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix == ".gii":
        parts = nib.load(path).darrays
        return np.column_stack([a.data for a in parts]).squeeze()
    return np.loadtxt(path)


def prepare(args):
    prepared = []
    for hemi in ("L", "R"):
        mask = array(args.mask_pattern.format(hemi=hemi)).ravel()
        modes = array(args.modes_pattern.format(hemi=hemi))
        values = array(args.values_pattern.format(hemi=hemi)).ravel()
        if mask.shape != (2562,) or not np.isin(mask, (0, 1)).all():
            raise ValueError("Supply a binary 2562-vertex fsaverage4 cortical mask")
        indices = np.flatnonzero(mask)
        if modes.ndim != 2 or modes.shape[0] not in (2562, len(indices)):
            raise ValueError("Eigenmode rows must follow full fsaverage4 or masked vertex order")
        if modes.shape[1] < 200 or len(values) < 200:
            raise ValueError("At least 200 ordered geometric modes and eigenvalues are required")
        if not np.isfinite(modes).all() or not np.isfinite(values).all() or (np.diff(values) < 0).any():
            raise ValueError("Require finite modes and ascending eigenvalues")
        if modes.shape[0] == 2562:
            modes = modes[indices]
        prepared.append((hemi, indices, modes[:, :200], values[:200]))
    if args.output.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    args.output.mkdir(parents=True)
    for hemi, indices, modes, values in prepared:
        np.save(args.output / f"{hemi}.fs4_idx.npy", indices)
        np.savetxt(args.output / f"eigenmodes_fs4_{hemi}.txt", modes)
        np.savetxt(args.output / f"eigenvalues_fs4_{hemi}.txt", values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes-pattern", required=True, help="fsaverage4 eigenmodes; {hemi}=L/R")
    parser.add_argument("--values-pattern", required=True)
    parser.add_argument("--mask-pattern", required=True)
    parser.add_argument("--output", required=True, type=Path)
    prepare(parser.parse_args())
