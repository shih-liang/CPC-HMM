"""Rescore saved CPC ranks and aggregate direct HMM transition controls."""

from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from aggregate import P, PAIRS, bootstrap, fingerprint, mean_valid, ratio, write_csv

RANKS = [3, 5, 10, 20, 30, 40, 50]


def transition_counts(y):
    return np.bincount(12 * y[:-1] + y[1:], minlength=144).reshape(12, 12)


def normalize(a):
    d = a.sum(-1, keepdims=True)
    return np.divide(a, d, out=np.full(a.shape, np.nan, dtype=float), where=d > 0)


def correlation(a, b):
    a = a - a.mean(-1, keepdims=True)
    b = b - b.mean(-1, keepdims=True)
    return np.sum(a * b, axis=-1) / np.sqrt(np.sum(a * a, axis=-1) * np.sum(b * b, axis=-1))


def self_test():
    c = transition_counts(np.array([0, 0, 1, 0, 1]))
    assert c.sum() == 4 and c[0, 0] == 1 and c[0, 1] == 2 and c[1, 0] == 1
    assert np.allclose(normalize(c)[0, :2], [1 / 3, 2 / 3])
    assert np.isclose(correlation(np.array([0, 1, 2]), np.array([2, 1, 0])), -1)
    print("ADDITION SELF_TEST PASS", flush=True)


def main(out):
    start = time.time()
    out.mkdir(parents=True, exist_ok=True)
    r = out / "results"
    r.mkdir(exist_ok=True)
    private = out / "run_data"
    private.mkdir(exist_ok=True)
    if (r / "comparison_complete.json").exists():
        raise RuntimeError("Completed analysis exists; reuse it.")
    pred_dir = P / "data/lower_components_agreement_20260909"
    paths = [
        P / "data/hmm_alpha_robust_seed2.npy",
        P / "revision_20260910/data/updated_consensus_masks.npz",
    ]
    paths += [pred_dir / f"complex_{k}_agreement_accuracy_test_predictions.npy" for k in RANKS]
    before = [fingerprint(p) for p in paths]
    alpha = np.load(paths[0], mmap_mode="r")
    maskfile = np.load(paths[1])
    mask = maskfile["fits6"]
    scans = np.sort(np.r_[np.arange(1003) * 4 + 2, np.arange(1003) * 4 + 3])
    assert np.array_equal(scans, maskfile["test_scan_indices"]) and mask.sum() == 582874
    y = np.asarray(alpha[scans, 100:1100]).argmax(2)
    assert y.shape == (2006, 1000)
    rng = np.random.default_rng(20260919)
    w = np.stack(
        [np.bincount(rng.integers(1003, size=1003), minlength=1003) for _ in range(2000)]
    ).astype(float)
    rank_rows = []
    guesses = {}
    private_rank = {}
    for k, path in zip(RANKS, paths[2:]):
        pred = np.load(path, mmap_mode="r")
        assert pred.shape == (2006, 1000, 12)
        assert np.isfinite(pred).all() and np.max(abs(pred.sum(2) - 1)) < 1e-5
        guess = pred.argmax(2)
        if k == 30:
            guesses[k] = guess
        correct = guess == y
        for label, keep in [("six_fit", mask), ("all", np.ones_like(mask))]:
            n = keep.sum(1)
            hit = (correct & keep).sum(1)
            pn = n.reshape(1003, 2).sum(1)
            ph = hit.reshape(1003, 2).sum(1)
            draws = (w @ ph) / (w @ pn)
            part = mean_valid(ratio(hit, n).reshape(1003, 2), 1)
            rank_rows.append(
                dict(
                    components=k,
                    subset=label,
                    frames=int(n.sum()),
                    correct=int(hit.sum()),
                    accuracy=float(hit.sum() / n.sum()),
                    low=float(np.quantile(draws, 0.025)),
                    high=float(np.quantile(draws, 0.975)),
                    participant_mean=float(part.mean()),
                    participant_sd=float(part.std(ddof=1)),
                    selector="agreement_accuracy",
                    participants=1003,
                )
            )
            private_rank[f"{k}_{label}_hits"] = hit
            private_rank[f"{k}_{label}_frames"] = n
        print("RANK", k, "accuracy", rank_rows[-2]["accuracy"], flush=True)
    thirty = next(
        row for row in rank_rows if row["components"] == 30 and row["subset"] == "six_fit"
    )
    assert abs(thirty["accuracy"] - 0.7789093354652978) < 1e-12
    write_csv(r / "cpc_count_accuracy.csv", rank_rows)
    np.savez_compressed(private / "rank_counts.npz", **private_rank)

    pred = np.load(pred_dir / "complex_30_agreement_accuracy_test_predictions.npy", mmap_mode="r")
    counts = np.zeros((2, 2006, 12, 12), np.int64)
    windows = np.full((2, 2006, 6, 11, 2), np.nan)
    event_counts = np.zeros((2006, 6), np.int64)
    lags = np.arange(-5, 6)
    for pos, scan in enumerate(scans):
        truth = y[pos]
        guess = guesses[30][pos]
        counts[0, pos] = transition_counts(truth)
        counts[1, pos] = transition_counts(guess)
        a = np.asarray(alpha[scan, 100:1100], float)
        p = np.asarray(pred[pos], float)
        arrivals = np.flatnonzero(truth[1:] != truth[:-1]) + 1
        arrivals = arrivals[(arrivals >= 5) & (arrivals < 995)]
        for q, (i, j, k) in enumerate(PAIRS):
            t = arrivals[(truth[arrivals - 1] == i - 1) & (truth[arrivals] == j - 1)]
            event_counts[pos, q] = len(t)
            if len(t):
                ix = t[:, None] + lags[None, :]
                windows[0, pos, q] = a[ix][:, :, [i - 1, j - 1]].mean(0)
                windows[1, pos, q] = p[ix][:, :, [i - 1, j - 1]].mean(0)
    total = counts.sum(1)
    matrix = normalize(total)
    np.savez_compressed(r / "hmm_cpc_transition_matrices.npz", count=total, probability=matrix)
    assert counts.sum((2, 3)).min() == 999 and counts.sum((2, 3)).max() == 999
    switches = total.sum((1, 2)) - np.trace(total, axis1=1, axis2=2)
    assert switches.tolist() == [225160, 284055]
    matrix_rows = []
    for method in range(2):
        for i in range(12):
            for j in range(12):
                matrix_rows.append(
                    dict(
                        method=["HMM", "CPC30"][method],
                        source=i + 1,
                        target=j + 1,
                        count=int(total[method, i, j]),
                        probability=float(matrix[method, i, j]),
                        source_frames=int(total[method, i].sum()),
                    )
                )
    write_csv(r / "hmm_cpc_transition_matrices.csv", matrix_rows)
    off = ~np.eye(12, dtype=bool)
    point = float(correlation(matrix[0, off], matrix[1, off]))
    participants = counts.reshape(2, 1003, 2, 12, 12).sum(2)
    draws = np.stack(
        [normalize((w @ participants[k].reshape(1003, 144)).reshape(-1, 12, 12)) for k in range(2)]
    )
    corr = correlation(draws[0][:, off], draws[1][:, off])
    switch_rows = []
    for k in range(2):
        by_part = (participants[k].sum((1, 2)) - np.trace(participants[k], axis1=1, axis2=2)) / 1998
        mean, bs, n = bootstrap(by_part, w)
        switch_rows.append(
            dict(
                method=["HMM", "CPC30"][k],
                switches=int(switches[k]),
                mean_switch_probability=float(mean),
                low=float(np.quantile(bs, 0.025)),
                high=float(np.quantile(bs, 0.975)),
            )
        )
    summary = dict(
        offdiagonal_correlation=point,
        low=float(np.quantile(corr, 0.025)),
        high=float(np.quantile(corr, 0.975)),
        switch_rates=switch_rows,
        pairs=132,
        normalization="pooled empirical counts, each source row including self-transitions",
        adjacent_pairs_per_method=int(total[0].sum()),
    )
    (r / "transition_comparison_summary.json").write_text(json.dumps(summary, indent=2))
    posterior_rows = []
    for method, name in enumerate(["HMM", "CPC30"]):
        a = mean_valid(windows[method].reshape(1003, 2, 6, 11, 2), 1)
        mean, bs, n = bootstrap(a, w)
        lo, hi = np.nanquantile(bs, [0.025, 0.975], axis=0)
        for q, (i, j, k) in enumerate(PAIRS):
            for h, lag in enumerate(lags):
                for role in range(2):
                    posterior_rows.append(
                        dict(
                            method=name,
                            source=i,
                            target=j,
                            lag_TR=int(lag),
                            time_seconds=float(lag * 0.72),
                            role=["source", "target"][role],
                            mean=float(mean[q, h, role]),
                            low=float(lo[q, h, role]),
                            high=float(hi[q, h, role]),
                            participants=int(n[q, h, role]),
                            events=int(event_counts[:, q].sum()),
                        )
                    )
    write_csv(r / "transition_triggered_posteriors.csv", posterior_rows)
    np.savez_compressed(
        private / "transition_by_run.npz",
        count=counts,
        posterior_windows=windows,
        event_counts=event_counts,
    )
    assert before == [fingerprint(p) for p in paths]
    (r / "comparison_complete.json").write_text(
        json.dumps(
            dict(
                status="PASS",
                source_fingerprints=before,
                sources_unchanged=True,
                models_refitted=False,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                helper_sha256=hashlib.sha256(
                    Path(__file__).with_name("aggregate.py").read_bytes()
                ).hexdigest(),
                seed=20260919,
                bootstrap_draws=2000,
                seconds=time.time() - start,
                rank_counts=RANKS,
                mask_frames=int(mask.sum()),
                HMM_events=int(switches[0]),
                CPC30_events=int(switches[1]),
            ),
            indent=2,
        )
    )
    print("COMPARISON COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test()
    if not args.self_test:
        if args.output is None:
            parser.error("--output is required")
        main(args.output)
