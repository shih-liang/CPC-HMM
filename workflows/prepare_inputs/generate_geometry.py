"""Generate fsaverage4 geometric eigenmodes from white/pial surfaces and functional masks."""

import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from lapy import Solver, TriaMesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


NUM_MODES = 200

HEMISPHERES = {
    "L": "left",
    "R": "right",
}


def load_surface(path):
    surface = nib.load(path)
    vertices = surface.get_arrays_from_intent("NIFTI_INTENT_POINTSET")[0].data
    faces = surface.get_arrays_from_intent("NIFTI_INTENT_TRIANGLE")[0].data
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def load_medial_wall_mask(mask_dir, hemi, expected_vertices):
    masks = [np.load(mask_dir / f"REST1_{run}.{hemi}.mask.npy", allow_pickle=False)
             for run in ("LR", "RL")]
    for mask in masks:
        if mask.shape != (expected_vertices,) or not np.isin(mask, (0, 1)).all():
            raise ValueError(f"{hemi}: expected a binary {expected_vertices}-vertex mask")
    if not np.array_equal(*masks):
        raise ValueError(f"{hemi}: LR and RL functional masks do not match")
    return masks[0].astype(bool)


def load_midthickness_and_faces(hemi, surface_name, surface_dir, templates):
    white_path = os.path.join(surface_dir, f"white_{surface_name}.gii.gz")
    pial_path = os.path.join(surface_dir, f"pial_{surface_name}.gii.gz")
    sphere_path = os.path.join(surface_dir, f"sphere_{surface_name}.gii.gz")
    target_sphere_path = os.path.join(
        templates,
        f"fsaverage4_std_sphere.{hemi}.3k_fsavg_{hemi}.surf.gii",
    )

    white_vertices, white_faces = load_surface(white_path)
    pial_vertices, pial_faces = load_surface(pial_path)
    _, sphere_faces = load_surface(sphere_path)
    _, target_faces = load_surface(target_sphere_path)

    if white_vertices.shape != pial_vertices.shape:
        raise ValueError(
            f"{hemi}: white/pial vertex shapes differ: "
            f"{white_vertices.shape} != {pial_vertices.shape}"
        )
    if white_vertices.shape != (2562, 3) or pial_faces.shape != (5120, 3):
        raise ValueError(
            f"{hemi}: expected fsaverage4 geometry (2562 vertices, 5120 faces), "
            f"got {white_vertices.shape}, {pial_faces.shape}"
        )
    for name, faces in (
        ("white", white_faces),
        ("sphere", sphere_faces),
        ("HCP target sphere", target_faces),
    ):
        if not np.array_equal(pial_faces, faces):
            raise ValueError(f"{hemi}: pial and {name} topology/order do not match")

    vertices = (white_vertices + pial_vertices) / 2.0
    if not np.isfinite(vertices).all():
        raise ValueError(f"{hemi}: midthickness contains non-finite coordinates")
    return vertices, pial_faces


def mask_surface(vertices, faces, cortex_mask, hemi):
    if cortex_mask.shape != (vertices.shape[0],):
        raise ValueError(
            f"{hemi}: mask shape {cortex_mask.shape} does not match {vertices.shape[0]} vertices"
        )

    keep_idx = np.flatnonzero(cortex_mask)
    old_to_new = np.full(vertices.shape[0], -1, dtype=np.int64)
    old_to_new[keep_idx] = np.arange(keep_idx.size)
    kept_faces = faces[np.all(cortex_mask[faces], axis=1)]
    remapped_faces = old_to_new[kept_faces]
    masked_vertices = vertices[keep_idx]

    edges = np.vstack(
        (
            remapped_faces[:, (0, 1)],
            remapped_faces[:, (1, 2)],
            remapped_faces[:, (2, 0)],
        )
    )
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    cols = np.concatenate((edges[:, 1], edges[:, 0]))
    adjacency = coo_matrix(
        (np.ones(rows.size, dtype=np.uint8), (rows, cols)),
        shape=(keep_idx.size, keep_idx.size),
    ).tocsr()
    isolated = np.count_nonzero(np.diff(adjacency.indptr) == 0)
    components, _ = connected_components(adjacency, directed=False)
    if isolated or components != 1:
        raise ValueError(
            f"{hemi}: invalid masked mesh: components={components}, isolated={isolated}"
        )

    return TriaMesh(masked_vertices, remapped_faces), keep_idx


def calculate_eigenmodes(tria, hemi):
    evals, emodes = Solver(tria).eigs(k=NUM_MODES)
    evals = np.asarray(evals, dtype=np.float64)
    emodes = np.asarray(emodes, dtype=np.float64)

    if evals.shape != (NUM_MODES,) or emodes.shape != (tria.v.shape[0], NUM_MODES):
        raise ValueError(
            f"{hemi}: unexpected eigensystem shapes: {evals.shape}, {emodes.shape}"
        )
    if not np.isfinite(evals).all() or not np.isfinite(emodes).all():
        raise ValueError(f"{hemi}: eigensystem contains non-finite values")
    if np.any(np.diff(evals) < -1e-10) or evals[0] < -1e-8:
        raise ValueError(f"{hemi}: eigenvalues are not non-negative and ordered")
    if abs(evals[0]) > 1e-7:
        raise ValueError(f"{hemi}: first eigenvalue is not near zero: {evals[0]:.6e}")
    return evals, emodes


def output_paths(hemi, output):
    return {
        "evals": os.path.join(output, f"eigenvalues_fs4_{hemi}.txt"),
        "emodes": os.path.join(output, f"eigenmodes_fs4_{hemi}.txt"),
        "mask": os.path.join(output, f"{hemi}.fs4_mask.txt"),
        "idx": os.path.join(output, f"{hemi}.fs4_idx.npy"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface-dir", type=Path, required=True,
                        help="Nilearn fsaverage4 directory containing white/pial/sphere GIFTIs")
    parser.add_argument("--templates", type=Path, required=True,
                        help="HCP resampling templates used in cortical preprocessing")
    parser.add_argument("--mask-dir", type=Path, required=True,
                        help="geometry_masks directory written by prepare_hcp.py")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    prepared = []
    for hemi, name in HEMISPHERES.items():
        vertices, faces = load_midthickness_and_faces(hemi, name, args.surface_dir, args.templates)
        mask = load_medial_wall_mask(args.mask_dir, hemi, len(vertices))
        tria, indices = mask_surface(vertices, faces, mask, hemi)
        values, modes = calculate_eigenmodes(tria, hemi)
        prepared.append((hemi, mask, indices, values, modes))
    args.output.mkdir(parents=True)
    for hemi, mask, indices, values, modes in prepared:
        paths = output_paths(hemi, args.output)
        np.savetxt(paths["evals"], values)
        np.savetxt(paths["emodes"], modes)
        np.savetxt(paths["mask"], mask.astype(np.uint8), fmt="%d")
        np.save(paths["idx"], indices)
        print(f"{hemi}: saved {modes.shape[0]} vertices × {modes.shape[1]} modes")


if __name__ == "__main__":
    main()
