"""Reuse frozen CPC30/HMM outputs, covering every directed transition."""

import os  # Public release: configurable data roots.
from pathlib import Path
import argparse
import csv
import hashlib
import json
import time
import numpy as np

P = Path(
    os.path.join(
        os.environ.get("HCP_DERIVATIVES", "/configure/HCP_DERIVATIVES"),
        "wave_rsn_revision_20260906",
    )
)
CORE = slice(100, 1100)
LAGS = np.arange(-5, 6)


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def ratio(a, b):
    shape = np.broadcast_shapes(np.shape(a), np.shape(b))
    return np.divide(a, b, out=np.full(shape, np.nan), where=np.asarray(b) > 0)


def mean_valid(a, axis=0):
    ok = np.isfinite(a)
    return ratio(np.where(ok, a, 0).sum(axis), ok.sum(axis))


def grouped(values, keys):
    """Average events by source*12 + target key; unobserved pairs remain NaN."""
    sums = np.zeros((144,) + values.shape[1:], dtype=np.result_type(values.dtype, float))
    np.add.at(sums, keys, values)
    counts = np.bincount(keys, minlength=144)
    return ratio(sums, counts.reshape((144,) + (1,) * (values.ndim - 1)))


def phase_prob(y, bins, exclude=None):
    """Return P(next state | source state, source-frame phase), shape (144, 24, 30).

    Denominators include both staying and switching frames. Input is one run;
    exclude indexes adjacent-frame pairs to omit, not individual phase bins.
    """
    use = np.ones(len(y) - 1, bool)
    if exclude is not None:
        use[exclude] = False
    # k, source, phase-bin, target; bincount includes all stay/switch outcomes.
    keys = (
        ((np.arange(30)[:, None] * 12 + y[:-1][None, :]) * 24 + bins[:-1].T) * 12 + y[1:][None, :]
    )[:, use]
    joint = np.bincount(keys.ravel(), minlength=30 * 12 * 24 * 12).reshape(30, 12, 24, 12)
    probs = ratio(joint, joint.sum(3, keepdims=True)).transpose(1, 3, 2, 0).reshape(144, 24, 30)
    return probs


def fingerprint(path):
    s = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(65536))
        f.seek(max(0, s.st_size - 65536))
        h.update(f.read(65536))
    return dict(path=str(path), size=s.st_size, mtime_ns=s.st_mtime_ns, edge_sha256=h.hexdigest())


def self_test():
    y = np.array([0, 0, 1, 0, 1])
    b = np.tile(np.array([0, 1, 1, 0, 0])[:, None], (1, 30))
    p = phase_prob(y, b)
    assert np.allclose(p[1, :2, 0], [0.5, 1])
    assert np.isnan(p[1, 2:, 0]).all()
    assert np.allclose(phase_prob(y, b, exclude=3)[1, :2, 0], [0, 1])
    a = grouped(np.array([[1.0, 2.0], [3.0, 4.0], [6.0, 8.0]]), np.array([1, 1, 2]))
    assert np.allclose(a[1], [2, 3]) and np.isnan(a[0]).all()
    print("SELF_TEST PASS", flush=True)


def main(out):
    start = time.time()
    out.mkdir(parents=True, exist_ok=True)
    result = out / "results"
    result.mkdir(exist_ok=True)
    if (result / "complete.json").exists():
        raise RuntimeError("Completed output exists; reuse it.")
    files = [P / "data/cpca_scores_200.npy", P / "data/hmm_alpha_robust_seed2.npy"]
    before = [fingerprint(p) for p in files]
    scores, alpha = [np.load(p, mmap_mode="r") for p in files]
    assert scores.shape == (4012, 1200, 200) and alpha.shape == (4012, 1200, 12)
    train = np.arange(1003) * 4
    test = np.sort(np.r_[train + 2, train + 3])
    # The previous component-selection rule, now for all 132 directions.
    sumsq = np.zeros(30)
    n_rms = 0
    effect = np.zeros((144, 30), complex)
    train_counts = np.zeros(144, np.int64)
    for scan in train:
        c = np.asarray(scores[scan, :, :30], complex)
        y = np.asarray(alpha[scan]).argmax(1)
        rms_sample = c[100:1100:4]
        sumsq += (abs(rms_sample) ** 2).sum(0)
        n_rms += len(rms_sample)
        t = np.flatnonzero(y[1:] != y[:-1]) + 1
        t = t[(t >= 105) & (t < 1095)]
        key = y[t - 1] * 12 + y[t]
        np.add.at(effect, key, c[t + 2] - c[t - 2])
        train_counts += np.bincount(key, minlength=144)
    standardized = effect / np.maximum(train_counts[:, None], 1) / np.sqrt(sumsq / n_rms)
    chosen = abs(standardized).argmax(1)
    fallback = train_counts == 0
    chosen[fallback] = 0
    frozen = [(1, 9, 1), (7, 1, 1), (11, 5, 2), (4, 11, 4), (5, 6, 3), (9, 7, 1)]
    assert [int(chosen[(i - 1) * 12 + j - 1]) + 1 for i, j, k in frozen] == [
        k for i, j, k in frozen
    ]
    print("SELECTION COMPLETE", round(time.time() - start), flush=True)
    accumulator = {}
    total = np.zeros((12, 12), np.int64)
    window_counts = np.zeros(144, np.int64)

    def update(name, a):
        ok = np.isfinite(a)
        x = np.where(ok, a, 0)
        if name not in accumulator:
            accumulator[name] = [np.zeros_like(x), np.zeros_like(x), np.zeros(a.shape, np.int64)]
        v = accumulator[name]
        v[0] += x
        v[1] += x * x
        v[2] += ok

    rng = np.random.default_rng(20260918)
    shifts = np.stack([rng.choice(np.arange(100, 901), 32, replace=False) for _ in test])
    by_run = []
    for pos, scan in enumerate(test):
        c = np.asarray(scores[scan, :, :30], complex)
        a = np.asarray(alpha[scan], float)
        y = a.argmax(1)
        yc = y[CORE]
        count = np.bincount(yc[:-1] * 12 + yc[1:], minlength=144).reshape(12, 12)
        total += count
        t = np.flatnonzero(y[1:] != y[:-1]) + 1
        t = t[(t >= 105) & (t < 1095)]
        key = y[t - 1] * 12 + y[t]
        window_counts += np.bincount(key, minlength=144)
        windows = c[t[:, None] + LAGS]
        amp = abs(windows) / abs(c[CORE]).mean(0)
        inc = np.angle(windows[:, 1:] * windows[:, :-1].conj())
        ph = np.concatenate([np.zeros_like(inc[:, :1]), np.cumsum(inc, axis=1)], axis=1)
        ph -= ph[:, 4:5]
        posterior = grouped(a[t[:, None] + LAGS], key)
        ids = np.arange(144)
        post = np.stack([posterior[ids, :, ids // 12], posterior[ids, :, ids % 12]], axis=-1)
        bins = np.floor(((np.angle(c[CORE]) + np.pi) % (2 * np.pi)) * 24 / (2 * np.pi)).astype(int)
        observed = phase_prob(yc, bins)
        nullsum = np.zeros_like(observed)
        nulln = np.zeros_like(observed, dtype=np.int32)
        for shift in shifts[pos]:
            v = phase_prob(yc, np.roll(bins, int(shift), axis=0), exclude=int(shift) - 1)
            ok = np.isfinite(v)
            nullsum += np.where(ok, v, 0)
            nulln += ok
        null = ratio(nullsum, nulln)
        common = np.isfinite(observed) & np.isfinite(null)
        run = dict(
            amplitude=grouped(amp, key),
            phase=grouped(ph, key),
            posterior=post,
            probability=np.where(common, observed, np.nan),
            shifted=np.where(common, null, np.nan),
            baseline=ratio(count, count.sum(1, keepdims=True)).reshape(144),
        )
        by_run.append(run)
        if len(by_run) == 2:
            for name in run:
                update(name, mean_valid(np.stack([r[name] for r in by_run]), axis=0))
            by_run = []
        if pos % 100 == 0:
            print("RUN", pos + 1, len(test), "seconds", round(time.time() - start), flush=True)
    assert not by_run and total.sum() == 2006 * 999
    assert int(total.sum() - np.trace(total)) == 225160
    fields = {}
    for name, (s, ss, n) in accumulator.items():
        avg = ratio(s, n)
        variance = ratio(ss - ratio(s * s, n), n - 1)
        fields[name + "_mean"] = avg
        fields[name + "_sd"] = np.sqrt(np.maximum(variance, 0))
        fields[name + "_participants"] = n
    probability = ratio(total, total.sum(1, keepdims=True))
    np.savez_compressed(
        result / "all_transition_profiles.npz",
        **fields,
        chosen=chosen,
        selection_fallback=fallback,
        training_counts=train_counts,
        window_counts=window_counts,
        transition_counts=total,
        transition_probability=probability,
        lags_TR=LAGS,
        phase_degrees=np.linspace(-172.5, 172.5, 24),
    )
    rows = []
    for i in range(12):
        for j in range(12):
            if i == j:
                continue
            key = 12 * i + j
            rows.append(
                dict(
                    source=i + 1,
                    target=j + 1,
                    component=int(chosen[key]) + 1,
                    training_events=int(train_counts[key]),
                    display_fallback=bool(fallback[key]),
                    test_events=int(total[i, j]),
                    complete_window_events=int(window_counts[key]),
                    transition_probability=float(probability[i, j]),
                    profile_participants=int(fields["amplitude_participants"][key, 0, chosen[key]]),
                )
            )
    write_csv(result / "transition_coverage.csv", rows)
    assert before == [fingerprint(p) for p in files]
    (result / "complete.json").write_text(
        json.dumps(
            dict(
                status="PASS",
                models_refitted=False,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                source_fingerprints=before,
                source_preserved=True,
                participants=1003,
                evaluation_runs=2006,
                nonself_types=132,
                observed_types=int(
                    sum(total[i, j] > 0 for i in range(12) for j in range(12) if i != j)
                ),
                complete_window_events=int(window_counts.sum()),
                all_transitions=int(total.sum() - np.trace(total)),
                errors="sample SD across participant means; two runs averaged within participant",
                numpy=np.__version__,
                seconds=time.time() - start,
                display_selection="training-only; CPC1 fallback if no training events",
                shift_seed=20260918,
                shift_count=32,
            ),
            indent=2,
        )
    )
    print("COMPLETE", round(time.time() - start), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    self_test()
    if not args.self_test:
        if args.output is None:
            p.error("--output required")
        main(args.output)
