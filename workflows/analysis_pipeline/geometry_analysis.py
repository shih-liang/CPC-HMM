"""Project complex spatial patterns onto nested geometric subspaces.

Reads archived basis/geometry inputs and writes energy summaries; executes on import.
"""

import os  # Public release: configurable data roots.
from pathlib import Path
import numpy as np
import pandas as pd
import json

P = Path(
    os.path.join(
        os.environ.get("HCP_DERIVATIVES", "/configure/HCP_DERIVATIVES"),
        "wave_rsn_revision_20260906",
    )
)
for directory in (P / "data", P / "results"):
    directory.mkdir(parents=True, exist_ok=True)
b = np.load(P / "data/training_bases.npz")
s = np.load(P / "data/surfaces_and_eigenmodes.npz")
c = b["complex_vectors"].conj().astype(complex)
evals = b["complex_eigenvalues"]
weights = np.concatenate([s[h + "_vertex_area"][s[h + "_indices"]] for h in ["L", "R"]])
weights /= weights.mean()
rows = []
allincr = {}
allrecon = {}
gram = {}
for measure, w in [("vertex", np.ones(len(c))), ("area", weights)]:
    den = (abs(c) ** 2 * w[:, None]).sum(0)
    incr = np.zeros((200, 50))
    recons = {m: np.zeros_like(c) for m in [10, 50, 200]}
    offset = 0
    for h in ["L", "R"]:
        E = s[h + "_eigenmodes"]
        n = len(E)
        root = np.sqrt(w[offset : offset + n])
        q, _ = np.linalg.qr(root[:, None] * E, mode="reduced")
        z = root[:, None] * c[offset : offset + n]
        coef = q.T @ z
        incr += abs(coef) ** 2
        G = E.T @ (w[offset : offset + n, None] * E)
        norm = G / np.sqrt(np.diag(G)[:, None] * np.diag(G)[None, :])
        gram[measure + "_" + h] = {
            "max_normalized_offdiagonal": float(abs(norm - np.eye(200)).max())
        }
        for m in recons:
            recons[m][offset : offset + n] = (q[:, :m] @ coef[:m]) / root[:, None]
        offset += n
    incr /= den[None, :]
    cum = incr.cumsum(0)
    allincr[measure] = incr
    for j in range(50):
        for m in range(1, 201):
            rows.append(
                {
                    "metric": measure,
                    "complex_component": j + 1,
                    "modes_per_hemisphere": m,
                    "captured_spatial_energy": cum[m - 1, j],
                    "incremental_ordered_projection_energy": incr[m - 1, j],
                }
            )
    for m, r in recons.items():
        allrecon[measure + str(m)] = r
        dphi = np.angle(c * np.conj(r))
        ww = w[:, None] * abs(c) * abs(r)
        phase = (ww * np.cos(dphi)).sum(0) / np.maximum(ww.sum(0), 1e-12)
        for j in range(50):
            rows.append(
                {
                    "metric": measure + "_phase_cosine",
                    "complex_component": j + 1,
                    "modes_per_hemisphere": m,
                    "captured_spatial_energy": phase[j],
                    "incremental_ordered_projection_energy": np.nan,
                }
            )
pd.DataFrame(rows).to_csv(P / "results/geometric_reconstruction.csv", index=False)
np.savez(P / "data/geometric_spatial_reconstructions.npz", original=c, **allrecon)
np.savez(P / "results/geometry_incremental_energy.npz", **allincr)
travel = []
for j in range(50):
    sv = np.linalg.svd(np.c_[c[:, j].real, c[:, j].imag], compute_uv=False)
    sa = np.linalg.svd(
        np.sqrt(weights[:, None]) * np.c_[c[:, j].real, c[:, j].imag], compute_uv=False
    )
    travel.append(
        {
            "complex_component": j + 1,
            "analytic_variance_fraction": float(evals[j] / evals.sum()),
            "quadrature_axis_ratio": float(sv[1] / sv[0]),
            "area_weighted_quadrature_axis_ratio": float(sa[1] / sa[0]),
        }
    )
pd.DataFrame(travel).to_csv(P / "results/cpca_component_properties.csv", index=False)
(P / "results/geometry_metadata.json").write_text(
    json.dumps(
        {
            "basis_gram": gram,
            "reconstruction": "Nested QR projections into the first M supplied eigenmodes separately per hemisphere. Fractions are incremental projection energy, not squared coefficients in a presumed Euclidean-orthogonal basis. Area sensitivity uses lumped fsaverage4 pial vertex areas, not the unavailable original FEM mass matrix.",
            "phase": "Amplitude-product-weighted cosine of original minus reconstructed complex spatial phase; global CPCA phase is fixed consistently.",
            "quadrature": "Smaller/larger singular value of spatial [real, imaginary] matrix. Zero indicates a standing pattern, one balanced quadrature. Nonzero ratio alone does not prove spatial propagation or neural causality.",
            "spatial_convention": "Rows of the spatial reconstruction are conjugated covariance eigenvectors, since temporal scores are Z times U and reconstruction is scores times U conjugate transpose.",
        },
        indent=2,
    )
)
print("COMPLETE", flush=True)

# Compact group table consumed by the current geometric-basis figure.
energy = allincr["vertex"][:, :30].cumsum(axis=0)
pd.DataFrame(
    [
        dict(modes_per_hemisphere=m, unweighted_mean_captured_energy=float(energy[m - 1].mean()))
        for m in [10, 50, 200]
    ]
).to_csv(P / "results/Figure2_geometry_summary.csv", index=False)
