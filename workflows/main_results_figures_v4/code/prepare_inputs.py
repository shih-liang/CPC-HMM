"""Place generated Figure 2 summaries and its fixed example beside the renderer."""

import argparse
import csv
import shutil
from pathlib import Path
import numpy as np


def main(analysis, revision, derivatives, output):
    output.mkdir(parents=True, exist_ok=True)
    names = [
        "decoding_summary.json",
        "state_profiles.npz",
        "fixed_example_agreement.csv",
        "cpc_count_accuracy.csv",
        "confusion.csv",
        "aggregation_complete.json",
        "comparison_complete.json",
    ]
    sources = [(analysis / "results" / n, output / n) for n in names]
    sources += [
        (revision / "results" / n, output / n)
        for n in ["cpca_acquisition_subspaces.csv", "hmm_reliability_pairs.csv"]
    ]
    for source, target in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
    alpha = np.load(derivatives / "data/hmm_alpha_robust_seed2.npy", mmap_mode="r")
    pred = np.load(
        derivatives
        / "data/lower_components_agreement_20260909/complex_30_agreement_accuracy_test_predictions.npy",
        mmap_mode="r",
    )
    with (output / "Figure3_example_posteriors.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["frame"]
            + [f"HMM_state{k}" for k in range(1, 13)]
            + [f"CPCA30_state{k}" for k in range(1, 13)]
        )
        writer.writerows(np.c_[np.arange(300, 500), alpha[2, 300:500], pred[0, 200:400]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["analysis", "revision", "derivatives", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    a = parser.parse_args()
    main(a.analysis, a.revision, a.derivatives, a.output)
