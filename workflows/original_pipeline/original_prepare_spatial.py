"""Extract cortical surfaces and ICA maps into figure-source arrays.

Requires original spatial inputs and writes derived files; executes on import.
"""

import os  # Public release: configurable data roots.
from pathlib import Path
import numpy as np
import nibabel as nib
import json

P = Path(
    os.path.join(
        os.environ.get("HCP_DERIVATIVES", "/configure/HCP_DERIVATIVES"),
        "wave_rsn_revision_20260906",
    )
)
out = {}
E = Path(
    os.path.join(
        os.environ.get("HCP_DERIVATIVES", "/configure/HCP_DERIVATIVES"),
        "wmy/geometry/Eigenmodes_fs4",
    )
)
for hemi, h in [("lh", "L"), ("rh", "R")]:
    for surface in ["inflated", "pial", "sphere"]:
        v, f = nib.freesurfer.read_geometry(
            f"/opt/freesurfer/subjects/fsaverage4/surf/{hemi}.{surface}"
        )
        out[f"{h}_{surface}_vertices"] = v
        out[f"{h}_faces"] = f
    v = out[f"{h}_pial_vertices"]
    f = out[f"{h}_faces"]
    area = np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1) / 2
    va = np.bincount(f.ravel(), weights=np.repeat(area / 3, 3), minlength=len(v))
    out[f"{h}_vertex_area"] = va
    out[f"{h}_indices"] = np.load(E / f"{h}.fs4_idx.npy")
    out[f"{h}_eigenmodes"] = np.loadtxt(E / f"eigenmodes_fs4_{h}.txt")
    out[f"{h}_eigenvalues"] = np.loadtxt(E / f"eigenvalues_fs4_{h}.txt")
np.savez(P / "data/surfaces_and_eigenmodes.npz", **out)
f = Path(os.environ.get("HCP_ICA_SPATIAL_FILE", "/configure/HCP_ICA_SPATIAL_FILE"))
img = nib.load(f)
maps = np.asarray(img.dataobj)
print("ICA", maps.shape, flush=True)
ax = img.header.get_axis(1)
records = []
summ = []
for name, sl, bm in ax.iter_structures():
    power = np.square(maps[:, sl]).sum(1)
    records.append({"structure": name, "grayordinates": int(maps[:, sl].shape[1])})
    summ.append(power)
np.savez(
    P / "data/ica_structure_energy.npz",
    energy=np.array(summ).T,
    structures=np.array([r["structure"] for r in records]),
)
(P / "results/ica_structure_metadata.json").write_text(
    json.dumps(
        {
            "source": str(f),
            "structures": records,
            "caution": "Squared group ICA map values per grayordinate; no surface area or voxel volume harmonization. Not a measure of physiological contribution.",
        },
        indent=2,
    )
)
print("COMPLETE", flush=True)
