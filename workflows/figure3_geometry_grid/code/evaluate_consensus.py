"""Score fixed geometric decoders against the existing six-HMM common labels."""

import os  # Public release: configurable data roots.
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np

SOURCE = Path(
    os.path.join(
        os.environ.get("HCP_DERIVATIVES", "/configure/HCP_DERIVATIVES"),
        "wave_rsn_revision_20260906",
    )
)
UPDATED = SOURCE / "revision_20260910"
DECODERS = Path(
    os.path.join(
        os.environ.get("HCP_DERIVATIVES", "/configure/HCP_DERIVATIVES"),
        "wave_geometry_decoder_20260918_01/results",
    )
)
ADDITIONAL = Path(
    os.path.join(
        os.environ.get("HCP_DERIVATIVES", "/configure/HCP_DERIVATIVES"),
        "wave_geometry_decoder_grid_20260918_01/results",
    )
)
TAGS = [
    "robust_seed2",
    "robust",
    "consensus_seed3",
    "consensus_seed4",
    "consensus_seed5",
    "consensus_seed6",
]
CONDITIONS = {"geo15": 15, "geo50": 50, "geo100": 100, "geo150": 150, "geo200": 200}
N, K = 1003, 12
TEST = np.sort(np.r_[np.arange(N) * 4 + 2, np.arange(N) * 4 + 3])


def fingerprint(path):
    size = path.stat().st_size
    h = hashlib.sha256()
    with path.open("rb") as f:
        offsets = (
            [0] if size <= 32 * 2**20 else [0, size // 4, size // 2, 3 * size // 4, size - 2**20]
        )
        for offset in offsets:
            f.seek(offset)
            h.update(f.read() if len(offsets) == 1 else f.read(2**20))
    return dict(
        path=str(path),
        bytes=size,
        mtime_ns=path.stat().st_mtime_ns,
        sha256=h.hexdigest(),
        method="full" if len(offsets) == 1 else "five 1-MiB samples",
    )


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(out):
    export = out / "export"
    export.mkdir(parents=True)
    private = out / "participant_metrics"
    private.mkdir()
    posterior_paths = [
        (SOURCE if tag in ["robust_seed2", "consensus_seed5"] else UPDATED)
        / f"data/hmm_alpha_{tag}.npy"
        for tag in TAGS
    ]
    folders = {
        name: (ADDITIONAL if name in ["geo100", "geo150"] else DECODERS) / name
        for name in CONDITIONS
    }
    prediction_paths = {name: folder / "test_predictions.npy" for name, folder in folders.items()}
    for folder in folders.values():
        assert json.loads((folder / "complete.json").read_text())["status"] == "PASS"
    for name in ["geo100", "geo150"]:
        assert (
            json.loads((folders[name] / "fitting_validation.json").read_text())["status"] == "PASS"
        )
    prediction_paths["archived_cpc30"] = (
        SOURCE
        / "data/lower_components_agreement_20260909/complex_30_agreement_accuracy_test_predictions.npy"
    )
    mapping_path = UPDATED / "results/updated_hmm_mappings.json"
    mask_path = UPDATED / "data/updated_consensus_masks.npz"
    metric_paths = [folder / "scan_metrics.csv" for folder in folders.values()]
    paths = (
        posterior_paths
        + list(prediction_paths.values())
        + metric_paths
        + [mapping_path, mask_path]
    )
    before = [fingerprint(p) for p in paths]
    mappings = json.loads(mapping_path.read_text())
    with np.load(mask_path) as saved:
        assert np.array_equal(saved["test_scan_indices"], TEST)
        archived_mask = saved["fits6"]
    common = np.ones((len(TEST), 1000), dtype=bool)
    truth = None
    for tag, path in zip(TAGS, posterior_paths):
        a = np.load(path, mmap_mode="r")
        assert a.shape == (4012, 1200, K)
        perm = np.array(mappings[tag])
        assert np.array_equal(np.sort(perm), np.arange(K))
        p = np.asarray(a[TEST, 100:1100])[..., perm]
        assert np.isfinite(p).all() and np.max(abs(p.sum(2) - 1)) < 2e-5
        labels = p.argmax(2)
        if truth is None:
            truth = labels
        common &= labels == truth
        del a, p, labels
    assert np.array_equal(common, archived_mask)
    assert common.shape == (2006, 1000)
    counts = common.sum(1)
    participant_counts = counts.reshape(N, 2).sum(1)
    rng = np.random.default_rng(20260918)
    boot = rng.integers(0, N, size=(2000, N))
    rows = []
    state_rows = []
    all_frame_errors = {}
    probability_errors = {}
    benchmark = None
    for name, path in prediction_paths.items():
        p = np.load(path, mmap_mode="r")
        assert p.shape == (2006, 1000, K) and np.isfinite(p).all()
        probability_errors[name] = float(abs(p.sum(2) - 1).max())
        assert probability_errors[name] < 1e-5
        guess = p.argmax(2)
        correct = guess == truth
        hits = (correct & common).sum(1)
        run_accuracy = np.divide(hits, counts, out=np.full(len(counts), np.nan), where=counts > 0)
        # A run with no unanimous frames has no accuracy estimate.
        values = np.nanmean(run_accuracy.reshape(N, 2), axis=1)
        participant_hits = hits.reshape(N, 2).sum(1)
        mean_ci = np.quantile(np.nanmean(values[boot], axis=1), [0.025, 0.975])
        pooled_ci = np.quantile(
            participant_hits[boot].sum(1) / participant_counts[boot].sum(1), [0.025, 0.975]
        )
        cf = np.zeros((K, K), dtype=np.int64)
        np.add.at(cf, (truth[common], guess[common]), 1)
        assert cf.sum() == common.sum()
        row = dict(
            condition=name,
            participants=int(np.isfinite(values).sum()),
            runs=len(TEST),
            frames=int(common.sum()),
            coverage=float(common.mean()),
            accuracy_mean=float(np.nanmean(values)),
            accuracy_sd=float(np.nanstd(values, ddof=1)),
            accuracy_ci_low=float(mean_ci[0]),
            accuracy_ci_high=float(mean_ci[1]),
            pooled_accuracy=float(hits.sum() / counts.sum()),
            pooled_ci_low=float(pooled_ci[0]),
            pooled_ci_high=float(pooled_ci[1]),
            balanced_accuracy=float(np.nanmean(np.divide(
                np.diag(cf), cf.sum(1), out=np.full(K, np.nan), where=cf.sum(1) > 0
            ))),
            present_states=int((cf.sum(1) > 0).sum()),
        )
        write_csv(
            private / (name + ".csv"),
            [
                dict(
                    participant_index=i,
                    frames=int(participant_counts[i]),
                    correct=int(participant_hits[i]),
                    mean_run_accuracy=float(values[i]),
                )
                for i in range(N)
            ],
        )
        if name == "archived_cpc30":
            benchmark = row
        else:
            row.update(
                representation="geometry",
                complex_coordinates=2 * CONDITIONS[name],
                modes_per_hemisphere=CONDITIONS[name],
            )
            with (folders[name] / "scan_metrics.csv").open() as f:
                metrics = list(csv.DictReader(f))
            assert np.array_equal([int(r["scan"]) for r in metrics], TEST)
            all_frame_accuracy = sum(int(r["correct"]) for r in metrics) / sum(
                int(r["frames"]) for r in metrics
            )
            all_frame_errors[name] = abs(float(correct.mean()) - all_frame_accuracy)
            assert all_frame_errors[name] < 1e-12
            rows.append(row)
            for k in range(K):
                support = int(cf[k].sum())
                full = int((truth == k).sum())
                state_rows.append(
                    dict(
                        condition=name,
                        state=k + 1,
                        selected_frames=support,
                        all_frames=full,
                        retention=support / full if full else float("nan"),
                        recall=float(cf[k, k] / support) if support else float("nan"),
                    )
                )
        print(json.dumps(row), flush=True)
        del p, guess, correct, cf
    assert before == [fingerprint(p) for p in paths]
    write_csv(export / "direct_decoding_summary.csv", rows)
    write_csv(export / "consensus_state_metrics.csv", state_rows)
    write_csv(export / "archived_CPC30_consensus_verification.csv", [benchmark])
    histories = []
    selections = []
    fitting_checks = []
    for name, folder in folders.items():
        with (folder / "validation_history.csv").open() as f:
            histories.extend(csv.DictReader(f))
        selections.append(json.loads((folder / "complete.json").read_text())["selection"])
        if name in ["geo100", "geo150"]:
            fitting_checks.append(json.loads((folder / "fitting_validation.json").read_text()))
    write_csv(export / "grid_validation_histories.csv", histories)
    (export / "additional_fitting_validation.json").write_text(
        json.dumps(fitting_checks, indent=2) + "\n"
    )
    validation = dict(
        status="PASS",
        sources=before,
        source_fingerprints_unchanged=True,
        script_fingerprint=fingerprint(Path(__file__)),
        fit_tags=TAGS,
        training_derived_mappings={t: mappings[t] for t in TAGS},
        definition="All six aligned posterior argmax labels identical at the evaluated frame",
        archived_mask_exact_match=True,
        scans_exact_match=True,
        frames=int(common.sum()),
        total_evaluation_frames=truth.size,
        coverage=float(common.mean()),
        participants=N,
        runs=len(TEST),
        minimum_selected_frames_per_run=int(counts.min()),
        probability_sum_max_errors=probability_errors,
        all_frame_replay_max_errors=all_frame_errors,
        selected_checkpoints=selections,
        CPC30_pooled_accuracy=benchmark["pooled_accuracy"],
        displayed_conditions=list(CONDITIONS),
        displayed_estimator="Participant mean of available run-specific consensus accuracies; unsupported runs omitted",
        plot_error_bars="Between-participant SD",
        bootstrap="2000 participant resamples; both runs together; seed 20260918",
        models_refitted=False,
        newly_fitted_conditions=["geo100", "geo150"],
        reused_conditions=["geo15", "geo50", "geo200"],
        consensus_used_for_training=False,
        family_dependence_controlled=False,
    )
    (export / "consensus_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "PASS",
                "selected_frames": int(common.sum()),
                "coverage": float(common.mean()),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
