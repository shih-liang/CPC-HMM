"""CPC compression of observed activity in a fixed cortical geometry subspace.

Extract only group covariance sufficient statistics, or recompute the result
locally from those statistics. No participant time series or basis refitting.
"""

from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def extract(source, out):
    paths = {
        name: source / "data" / filename
        for name, filename in {
            "covariance": "extended_training_complex_covariance.npy",
            "basis": "extended_training_bases.npz",
            "geometry": "surfaces_and_eigenmodes.npz",
        }.items()
    }
    inputs = {k: {"path": str(p), "sha256": sha(p)} for k, p in paths.items()}
    C = np.load(paths["covariance"]).astype(np.complex128)
    b, s = np.load(paths["basis"]), np.load(paths["geometry"])
    U = b["complex_vectors"].astype(np.complex128)
    lam = b["complex_eigenvalues"].astype(float)
    assert C.shape == (4801, 4801) and U.shape == (4801, 200)
    indices = np.r_[s["L_indices"], len(s["L_pial_vertices"]) + s["R_indices"]]
    assert np.array_equal(indices, b["vertex_indices"])
    assert np.array_equal(b["training_scans"], np.arange(1003) * 4)
    Q = np.zeros((4801, 400))
    offset, qr_errors = 0, {}
    for j, hemi in enumerate(["L", "R"]):
        E = s[hemi + "_eigenmodes"].astype(float)
        q, _ = np.linalg.qr(E, mode="reduced")
        Q[offset : offset + len(E), j * 200 : (j + 1) * 200] = q
        qr_errors[hemi] = float(np.max(abs(q.T @ q - np.eye(200))))
        offset += len(E)
    assert offset == 4801
    hermitian = float(np.linalg.norm(C - C.conj().T) / np.linalg.norm(C))
    orthogonal = float(np.max(abs(U.conj().T @ U - np.eye(200))))
    CU, CQ = C @ U, C @ Q
    eig_residual = float(np.linalg.norm(CU - U * lam[:200]) / np.linalg.norm(U * lam[:200]))
    total = float(np.trace(C).real)
    trace_error = float(abs(lam.sum() - total) / total)
    assert hermitian < 1e-6 and orthogonal < 1e-4
    assert eig_residual < 1e-4 and trace_error < 1e-5
    assert max(qr_errors.values()) < 1e-10
    np.savez_compressed(
        out / "geometry_activity_sufficient_stats.npz",
        geometry_covariance=Q.T @ CQ,
        cpc_geometry_overlap=U.conj().T @ Q,
        cpc_geometry_cross_covariance=U.conj().T @ CQ,
        cpc_covariance=U.conj().T @ CU,
        eigenvalues=lam,
        total_cortical_variance=total,
    )
    for key, path in paths.items():
        assert sha(path) == inputs[key]["sha256"], f"Changed source: {path}"
    report = dict(
        status="PASS",
        sources=inputs,
        source_hashes_unchanged=True,
        scope="REST1_LR training covariance; 1003 participants; 1200 frames per run",
        metric="Equal vertex weights; separate hemispheric QR; 200 modes per hemisphere",
        covariance_hermitian_relative_error=hermitian,
        cpc_orthogonality_max_error=orthogonal,
        qr_orthogonality_max_error=qr_errors,
        covariance_eigenvector_relative_residual=eig_residual,
        trace_vs_eigenvalue_sum_relative_error=trace_error,
        script_sha256=sha(Path(__file__)),
    )
    (out / "geometry_activity_extraction.json").write_text(json.dumps(report, indent=2) + "\n")


def analyze(stats, out):
    s = np.load(stats)
    A = s["cpc_geometry_overlap"]
    D, H = s["cpc_geometry_cross_covariance"], s["cpc_covariance"]
    lam, total = s["eigenvalues"], float(s["total_cortical_variance"])
    geo = float(np.trace(s["geometry_covariance"]).real)
    overlap_energy = np.sum(abs(A) ** 2, axis=1)
    retained = np.cumsum(lam[: len(A)] * overlap_energy) / geo
    cortical = np.cumsum(lam) / total
    geometry_increments = s["geometry_covariance"].diagonal().real.reshape(2, 200).sum(axis=0)
    geometry_cortical = geometry_increments.cumsum() / total
    # Exact quadratic residual identity; this does not assume eigenvectors are exact.
    cross = np.cumsum(np.sum(A.conj() * D, axis=1).real)
    terms = H * (A @ A.conj().T).T
    predicted = terms.cumsum(axis=0).cumsum(axis=1).diagonal().real
    residual = geo - 2 * cross + predicted
    exact = 1 - residual / geo
    error = float(abs(exact - retained).max())
    assert error < 1e-4, error
    assert geo > 0 and geo <= total * (1 + 1e-5)
    assert np.all((retained >= -1e-5) & (retained <= 1 + 1e-5))
    rows = [
        dict(
            CPCs=k + 1,
            geometry_activity_variance_fraction=float(retained[k]),
            cortical_activity_variance_fraction=float(cortical[k]),
            exact_residual_variance_fraction=float(exact[k]),
            incremental_geometry_activity_fraction=float(lam[k] * overlap_energy[k] / geo),
            geometric_subspace_overlap=float(overlap_energy[k]),
        )
        for k in range(len(A))
    ]
    with (out / "geometry_activity_retention.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    geometry_rows = [
        dict(
            modes_per_hemisphere=m + 1,
            total_coefficients=2 * (m + 1),
            cortical_activity_variance_fraction=float(geometry_cortical[m]),
        )
        for m in range(200)
    ]
    with (out / "geometry_cortical_variance.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=geometry_rows[0].keys())
        writer.writeheader()
        writer.writerows(geometry_rows)
    with (out / "cpca_cortical_variance.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["CPCs", "cortical_activity_variance_fraction"])
        writer.writeheader()
        writer.writerows(
            dict(CPCs=k + 1, cortical_activity_variance_fraction=float(cortical[k]))
            for k in range(400)
        )
    matched = np.flatnonzero(geometry_cortical >= cortical[29])
    summary = dict(
        status="PASS",
        scope="Training covariance, REST1_LR",
        total_geometric_dimensions=400,
        modes_per_hemisphere=200,
        cortical_vertices=4801,
        geometry_subspace_fraction_of_total_cortical_variance=geo / total,
        exact_vs_eigenvalue_formula_max_absolute_error=error,
        selected_K=[rows[k - 1] for k in [3, 10, 30, 50, 100, 200]],
        geometry_selected_M=[geometry_rows[m - 1] for m in [10, 50, 100, 200]],
        CPCA30_vs_geometry200=dict(
            cpc_cortical_variance_fraction=float(cortical[29]),
            geometry_cortical_variance_fraction=float(geometry_cortical[-1]),
            ratio=float(cortical[29] / geometry_cortical[-1]),
            difference_percentage_points=float(100 * (cortical[29] - geometry_cortical[-1])),
            geometry_modes_per_hemisphere_to_match_CPC30=int(matched[0] + 1)
            if len(matched)
            else None,
            CPCs_to_match_geometry200=int(np.flatnonzero(cortical >= geometry_cortical[-1])[0] + 1),
        ),
        first_K_at_80_90_95_percent={
            str(t): int(np.flatnonzero(retained >= t)[0] + 1) if np.any(retained >= t) else None
            for t in [0.8, 0.9, 0.95]
        },
        mean_orthonormal_geometry_basis_overlap_with_CPCA30=float(overlap_energy[:30].sum() / 400),
        stats_sha256=sha(stats),
        script_sha256=sha(Path(__file__)),
    )
    (out / "geometry_activity_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", type=Path)
    group.add_argument("--stats", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.source:
        extract(args.source, args.output)
    analyze(args.stats or args.output / "geometry_activity_sufficient_stats.npz", args.output)
