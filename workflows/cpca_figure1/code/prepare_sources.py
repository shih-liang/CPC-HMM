"""Reuse published-package statistics; participant bootstrap for all 30 CPCs."""

from pathlib import Path
import argparse
import csv
import json
import hashlib
import numpy as np


def main(source, root):
    target = root / "source_data"
    target.mkdir(parents=True, exist_ok=True)
    (root / "provenance").mkdir(exist_ok=True)
    names = [
        "cpca_test_variance_summary.csv",
        "original_cpca_component_properties.csv",
        "cpca_acquisition_mode_overlaps.csv",
    ]
    for name in names:
        (target / name).write_bytes((source / "source_data" / name).read_bytes())
    e = np.load(source / "figure_data/cpca_basis_REST1_LR.npz")["eigenvalues"].astype(float)
    (target / "training_variance.json").write_text(
        json.dumps(
            dict(
                cumulative=(e.cumsum() / e.sum())[:50].tolist(),
                individual=(e / e.sum())[:50].tolist(),
            ),
            indent=2,
        )
    )
    values = np.full((1003, 2, 30), np.nan)
    with (source / "source_data/cpca_test_score_reproducibility.csv").open() as f:
        for r in csv.DictReader(f):
            k = int(r["component"]) - 1
            if k < 30:
                values[int(r["participant_index"]), int(r["scan_index"]) % 4 - 2, k] = float(
                    r["phase_aligned_score_correlation"]
                )
    assert np.isfinite(values).all()
    v = values.mean(1)
    rng = np.random.default_rng(20260917)
    bs = np.array([v[rng.integers(0, len(v), len(v))].mean(0) for _ in range(2000)])
    with (target / "coordinate_agreement_summary.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(["component", "mean", "ci_low", "ci_high", "n_participants"])
        w.writerows(
            zip(
                range(1, 31),
                v.mean(0),
                np.quantile(bs, 0.025, axis=0),
                np.quantile(bs, 0.975, axis=0),
                [1003] * 30,
            )
        )
    names.append("cpca_test_score_reproducibility.csv")
    (root / "provenance/local_sources.json").write_text(
        json.dumps(
            [
                dict(
                    path=str(source / "source_data" / n),
                    sha256=hashlib.sha256((source / "source_data" / n).read_bytes()).hexdigest(),
                )
                for n in names
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    main(a.source, a.root)
