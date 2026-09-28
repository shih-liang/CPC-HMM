"""REST2 per-CPC variance with participant bootstrap, using the frozen basis."""

from pathlib import Path
import argparse
import csv
import hashlib
import json
import time
import numpy as np
from scipy.signal import hilbert
from make_figure import BASE, BASIS, SCORES, RAW, fingerprint, read_csv

RUN_VARIANCE = BASE / "revision_20260910/results/cpca_test_variance_participants.csv"


def main(root):
    started = time.time()
    sources = [BASIS, SCORES, RAW, RUN_VARIANCE]
    before = [fingerprint(p) for p in sources]
    reference = {}
    for row in read_csv(RUN_VARIANCE):
        scan = int(row["scan_index"])
        rank = int(row["rank"])
        assert int(row["participant_index"]) == scan // 4
        assert (scan, rank) not in reference
        reference[scan, rank] = float(row["analytic_variance_fraction"])
    scans = np.sort(np.r_[np.arange(1003) * 4 + 2, np.arange(1003) * 4 + 3])
    assert len(reference) == len(scans) * 4
    scores = np.load(SCORES, mmap_mode="r")
    assert scores.shape == (4012, 1200, 200)
    fractions = np.empty((len(scans), 30), dtype=float)
    cumulative_error = {k: 0.0 for k in [3, 10, 30]}
    # The archived rank-50 fraction supplies the original full-field denominator:
    # E_k / E_total = (E_k / E_1:50) * archived_fraction_50.
    for j, scan in enumerate(scans):
        c = np.asarray(scores[scan, 100:1100, :50], dtype=np.complex128)
        energy = np.sum(c.real * c.real + c.imag * c.imag, axis=0)
        values = energy / energy.sum() * reference[int(scan), 50]
        fractions[j] = values[:30]
        for k in cumulative_error:
            error = abs(values[:k].sum() - reference[int(scan), k])
            cumulative_error[k] = max(cumulative_error[k], float(error))
        if j % 400 == 0:
            print("Variance contributions", j, "/", len(scans), flush=True)
    assert max(cumulative_error.values()) < 5e-5, cumulative_error
    assert np.isfinite(fractions).all() and (fractions > 0).all()
    assert np.all(scans.reshape(1003, 2) // 4 == np.arange(1003)[:, None])
    participants = fractions.reshape(1003, 2, 30).mean(axis=1)
    original_total = (
        np.array([reference[int(scan), 30] for scan in scans]).reshape(1003, 2).mean(axis=1)
    )
    # Reproduce the original rank-30 bootstrap draws: ranks 3 and 10 consumed
    # the first two groups of 2,000 draws in cpca_reliability.py.
    rng = np.random.default_rng(20260910)
    for _ in range(4000):
        rng.choice(1003, 1003, replace=True)
    boot = np.empty((2000, 30))
    original_boot = np.empty(2000)
    for b in range(2000):
        ids = rng.choice(1003, 1003, replace=True)
        boot[b] = participants[ids].mean(axis=0)
        original_boot[b] = original_total[ids].mean()
    old = next(
        row
        for row in read_csv(root / "source_data/cpca_test_variance_summary.csv")
        if int(row["rank"]) == 30
    )
    original_summary = np.r_[original_total.mean(), np.quantile(original_boot, [0.025, 0.975])]
    expected = np.array([float(old[key]) for key in ["mean", "ci_low", "ci_high"]])
    assert np.allclose(original_summary, expected, rtol=0, atol=1e-12)
    new_total = np.r_[participants.sum(1).mean(), np.quantile(boot.sum(1), [0.025, 0.975])]
    assert np.allclose(new_total, expected, rtol=0, atol=5e-6)
    basis = np.load(BASIS)
    u = basis["complex_vectors"][:, :50].astype(np.complex128)
    raw = np.load(RAW, mmap_mode="r")
    idx = basis["vertex_indices"]
    projection_errors = {}
    for scan in [2, 2006, 4011]:
        x = np.asarray(raw[scan * 1200 : (scan + 1) * 1200][:, idx], dtype=float)
        x -= x.mean(axis=0)
        replay = hilbert(x, axis=0) @ u
        saved = np.asarray(scores[scan, :, :50], dtype=np.complex128)
        error = float(np.linalg.norm(replay - saved) / np.linalg.norm(saved))
        assert error < 5e-5, (scan, error)
        projection_errors[scan] = error
    mean = participants.mean(0)
    low, high = np.quantile(boot, [0.025, 0.975], axis=0)
    dest = root / "source_data/cpca_component_variance_rest2.csv"
    with dest.open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["component", "mean", "ci_low", "ci_high", "n_participants", "n_runs"])
        writer.writerows(zip(range(1, 31), mean, low, high, [1003] * 30, [2006] * 30))
    sd = participants.std(axis=0, ddof=1)
    q25, median, q75 = np.quantile(participants, [0.25, 0.5, 0.75], axis=0)
    distribution = root / "source_data/cpca_component_variance_distribution.csv"
    with distribution.open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["component", "mean", "sd", "q25", "median", "q75", "n_participants"])
        writer.writerows(zip(range(1, 31), mean, sd, q25, median, q75, [1003] * 30))
    cumulative = []
    for rank in [3, 10, 30, 50]:
        values = (
            np.array([reference[int(scan), rank] for scan in scans]).reshape(1003, 2).mean(axis=1)
        )
        ref = next(
            row
            for row in read_csv(root / "source_data/cpca_test_variance_summary.csv")
            if int(row["rank"]) == rank
        )
        assert np.isclose(values.mean(), float(ref["mean"]), rtol=0, atol=1e-12)
        cumulative.append([rank, values.mean(), values.std(ddof=1), 1003, 2006])
        if rank == 30:
            assert np.isclose(
                values.std(ddof=1), participants.sum(axis=1).std(ddof=1), rtol=0, atol=5e-6
            )
    cumulative_file = root / "source_data/cpca_cumulative_variance_distribution.csv"
    with cumulative_file.open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "mean", "sd", "n_participants", "n_runs"])
        writer.writerows(cumulative)
    analysis = root / "analysis"
    analysis.mkdir(exist_ok=True)
    np.save(analysis / "component_variance_participants.npy", participants)
    after = [fingerprint(p) for p in sources]
    assert before == after
    report = dict(
        status="PASS",
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        estimator="Mean of participant-average REST2 LR/RL per-CPC analytic energy fractions",
        basis="Fixed REST1_LR basis; no refitting",
        core_frames=[100, 1099],
        n_participants=1003,
        n_runs=2006,
        bootstrap=dict(
            unit="participant",
            resamples=2000,
            seed=20260910,
            skipped_prior_draws=4000,
            percentiles=[2.5, 97.5],
            runs_kept_together=True,
        ),
        denominator="Archived rank-50 fraction times each component share of first-50 score energy",
        maximum_cumulative_fraction_error=cumulative_error,
        raw_projection_relative_errors=projection_errors,
        original_rank30_summary_reproduced=original_summary.tolist(),
        sum_of_component_means_and_total_CI=new_total.tolist(),
        source_preserved=True,
        input_fingerprints=before,
        run_variance_sha256=hashlib.sha256(RUN_VARIANCE.read_bytes()).hexdigest(),
        source_summary_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
        distribution_summary_sha256=hashlib.sha256(distribution.read_bytes()).hexdigest(),
        cumulative_distribution_sha256=hashlib.sha256(cumulative_file.read_bytes()).hexdigest(),
        sd_definition="Sample standard deviation across 1003 participant means; ddof=1",
        seconds=time.time() - started,
    )
    (root / "provenance/component_variance_validation.json").write_text(
        json.dumps(report, indent=2)
    )
    print(
        json.dumps(
            dict(
                status="PASS",
                seconds=report["seconds"],
                maximum_cumulative_fraction_error=cumulative_error,
                total_mean_and_ci=new_total.tolist(),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
