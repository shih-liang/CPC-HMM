"""Generate the state-map panels of Supplementary Figure 3.

Uses the state-map calculations from the original correspondence analysis;
no transition-control inputs are needed for these panels.
"""

import argparse
import csv
from pathlib import Path
import numpy as np


def mean_valid(a, axis=0):
    a = np.asarray(a)
    valid = np.isfinite(a)
    n = valid.sum(axis)
    return np.divide(
        np.where(valid, a, 0).sum(axis),
        n,
        out=np.full(np.shape(n), np.nan, dtype=np.result_type(a.dtype, float)),
        where=n > 0,
    )


def rows_mean(x, label, n=144):
    counts = np.bincount(label, minlength=n)
    sums = np.zeros((n, x.shape[1]), dtype=np.result_type(x.dtype, float))
    np.add.at(sums, label, x)
    return np.divide(
        sums, counts[:, None], out=np.full_like(sums, np.nan), where=counts[:, None] > 0
    ), counts


def real_coordinates(c):
    """Concatenate real then imaginary coefficients along the last axis."""
    return np.concatenate([c.real, c.imag], axis=-1)


def metrics(observed, coordinates, basis):
    """Compute spatial correlation and zero-baseline reconstruction R-squared.

    observed is (..., vertices), coordinates is (..., features), and basis
    is (features, vertices). Reconstructed maps equal coordinates @ basis.
    Correlation centers maps across vertices; R-squared uses uncentered
    observed energy as its denominator and can be negative.
    """
    b = basis
    bc = b - b.mean(1, keepdims=True)
    g = b @ b.T
    gc = bc @ bc.T
    observed = np.asarray(observed, float)
    mean = observed.mean(-1)
    norm = (observed**2).sum(-1)
    centered_norm = norm - observed.shape[-1] * mean**2
    cross = observed @ b.T
    cross_centered = observed @ bc.T
    rec_norm = np.einsum("...i,ij,...j->...", coordinates, g, coordinates)
    rec_centered_norm = np.einsum("...i,ij,...j->...", coordinates, gc, coordinates)
    dot = np.einsum("...i,...i->...", coordinates, cross)
    dot_centered = np.einsum("...i,...i->...", coordinates, cross_centered)
    corr = np.divide(
        dot_centered,
        np.sqrt(np.maximum(centered_norm * rec_centered_norm, 0)),
        out=np.full(rec_norm.shape, np.nan),
        where=(centered_norm * rec_centered_norm) > 1e-12,
    )
    r2 = np.divide(
        2 * dot - rec_norm, norm, out=np.full(rec_norm.shape, np.nan), where=norm > 1e-12
    )
    return corr, r2


def main(scores_path, posterior_path, basis_path, bold_path, out):
    out.mkdir(parents=True, exist_ok=True)
    if (out / "state_associated_cortical_maps.npz").exists():
        raise FileExistsError("Use a new output directory")
    scores = np.load(scores_path, mmap_mode="r")
    alpha = np.load(posterior_path, mmap_mode="r")
    raw = np.load(bold_path, mmap_mode="r")
    with np.load(basis_path) as a:
        u = a["complex_vectors"][:, :30]
        idx = a["vertex_indices"]
    assert scores.shape == (4012, 1200, 200)
    assert alpha.shape == (4012, 1200, 12) and raw.shape == (4814400, 5124)
    b = np.concatenate([u.real.T, u.imag.T], axis=0).astype(float)
    # Confirm the fixed basis and coefficients use the same phase convention.
    from scipy.signal import hilbert

    x = np.asarray(raw[2400:3600, idx], dtype=float)
    x -= x.mean(0)
    target = np.asarray(scores[2, :, :30])
    assert np.linalg.norm(hilbert(x, axis=0) @ u - target) / np.linalg.norm(target) < 1e-4
    total = np.zeros((2, 12, len(idx)))
    count = np.zeros(12, int)
    rows = []
    participant_metrics = []
    for subject in range(1003):
        maps, values = [], []
        for acquisition in [2, 3]:
            scan = subject * 4 + acquisition
            x = np.asarray(raw[scan * 1200 : (scan + 1) * 1200, idx], dtype=float)
            x -= x.mean(0)
            labels = alpha[scan, 100:1100].argmax(1)
            observed, n = rows_mean(x[100:1100], labels, 12)
            coordinates, _ = rows_mean(
                real_coordinates(scores[scan, 100:1100, :30]).astype(float), labels, 12
            )
            reconstructed = coordinates @ b
            corr, r2 = metrics(observed, coordinates, b)
            maps.append(np.stack([observed, reconstructed]))
            values.append(np.stack([corr, r2], axis=-1))
            for k in range(12):
                rows.append(
                    dict(
                        scan=scan,
                        subject=subject,
                        state=k + 1,
                        frames=int(n[k]),
                        spatial_r=corr[k],
                        reconstruction_r2=r2[k],
                    )
                )
        average = mean_valid(maps, 0)
        valid = np.isfinite(average[0, :, 0])
        total[:, valid] += average[:, valid]
        count += valid
        participant_metrics.append(mean_valid(values, 0))
    maps = np.divide(
        total, count[None, :, None], out=np.full_like(total, np.nan), where=count[None, :, None] > 0
    )
    np.savez_compressed(
        out / "state_associated_cortical_maps.npz",
        state_observed=maps[0],
        state_wave=maps[1],
        state_subjects=count,
    )
    values = np.asarray(participant_metrics)
    summaries = [
        dict(state=k + 1, metric=name, mean=mean_valid(values[:, k, j], 0))
        for k in range(12)
        for j, name in enumerate(["spatial_r", "reconstruction_r2"])
    ]
    for name, data in [
        ("state_map_by_run.csv", rows),
        ("archived_state_map_summary.csv", summaries),
    ]:
        with (out / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ["scores", "posterior", "basis", "bold", "out"]:
        parser.add_argument("--" + key, required=True, type=Path)
    args = parser.parse_args()
    main(args.scores, args.posterior, args.basis, args.bold, args.out)
