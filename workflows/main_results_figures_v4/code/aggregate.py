"""Aggregate existing HCP outputs for the two main results figures; no fitting."""

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
PAIRS = [(1, 9, 1), (7, 1, 1), (11, 5, 2), (4, 11, 4), (5, 6, 3), (9, 7, 1)]
SEED = 20260918


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def mean_valid(a, axis=0):
    ok = np.isfinite(a)
    n = ok.sum(axis)
    return np.divide(
        np.where(ok, a, 0).sum(axis),
        n,
        out=np.full(n.shape, np.nan, dtype=np.result_type(a.dtype, float)),
        where=n > 0,
    )


def bootstrap(a, w):
    a = np.asarray(a)
    shape = a.shape[1:]
    flat = a.reshape(len(a), -1)
    ok = np.isfinite(flat)
    den = w @ ok.astype(float)
    draws = np.divide(
        w @ np.where(ok, flat, 0),
        den,
        out=np.full(den.shape, np.nan, dtype=np.result_type(a.dtype, float)),
        where=den > 0,
    )
    return mean_valid(a), draws.reshape((len(w),) + shape), ok.sum(0).reshape(shape)


def fingerprint(p):
    st = p.stat()
    h = hashlib.sha256()
    with p.open("rb") as f:
        h.update(f.read(65536))
        f.seek(max(0, st.st_size - 65536))
        h.update(f.read(65536))
    return dict(
        path=str(p), size=st.st_size, mtime_ns=st.st_mtime_ns, first_last_64KiB_sha256=h.hexdigest()
    )


def hazard_counts(labels, bins, source, target, keep=None):
    eligible = labels[:-1] == source
    if keep is not None:
        eligible &= keep
    den = np.bincount(bins[:-1][eligible], minlength=24)
    num = np.bincount(bins[:-1][eligible & (labels[1:] == target)], minlength=24)
    assert np.all(num <= den)
    return num, den


def ratio(num, den):
    return np.divide(num, den, out=np.full(np.shape(den), np.nan), where=den > 0)


def self_test():
    y = np.array([0, 0, 1, 0, 1])
    b = np.array([0, 1, 1, 0, 0])
    n, d = hazard_counts(y, b, 0, 1)
    assert n.tolist()[:2] == [1, 1] and d.tolist()[:2] == [2, 1]
    assert np.allclose(ratio(n, d)[:2], [0.5, 1.0]) and np.isnan(ratio(n, d)[2:]).all()
    n, d = hazard_counts(y, b, 0, 1, np.array([True, True, True, False]))
    assert n.tolist()[:2] == [0, 1] and d.tolist()[:2] == [1, 1]
    assert np.isclose(np.angle(mean_valid(np.exp(1j * np.deg2rad([179, -179])))), np.pi)
    print("SELF_TEST PASS", flush=True)


def main(out):
    started = time.time()
    out.mkdir(parents=True, exist_ok=True)
    result = out / "results"
    private = out / "run_data"
    result.mkdir(exist_ok=True)
    private.mkdir(exist_ok=True)
    complete = result / "aggregation_complete.json"
    if complete.exists():
        raise RuntimeError("Completed output exists; reuse it or select a new output directory.")
    files = [
        P / "data/cpca_scores_200.npy",
        P / "data/hmm_alpha_robust_seed2.npy",
        P
        / "data/lower_components_agreement_20260909/complex_30_agreement_accuracy_test_predictions.npy",
        P / "revision_20260910/data/updated_consensus_masks.npz",
    ]
    before = [fingerprint(p) for p in files]
    scores, alpha, pred = [np.load(p, mmap_mode="r") for p in files[:3]]
    masks = np.load(files[3])
    scans = np.sort(np.r_[np.arange(1003) * 4 + 2, np.arange(1003) * 4 + 3])
    mask = masks["fits6"]
    assert np.array_equal(scans, masks["test_scan_indices"]) and mask.shape == (2006, 1000)
    assert (
        scores.shape == (4012, 1200, 200)
        and alpha.shape == (4012, 1200, 12)
        and pred.shape == (2006, 1000, 12)
    )
    rng = np.random.default_rng(SEED)
    shifts = np.stack([rng.choice(np.arange(100, 901), 32, replace=False) for _ in scans])
    np.save(private / "circular_shifts.npy", shifts)
    amp = np.full((2, 2006, 12, 30), np.nan)
    phase = np.full(amp.shape, np.nan, dtype=complex)
    state_count = np.zeros((2, 2006, 12), np.int64)
    confusion = np.zeros((2006, 12, 12), np.int64)
    hazard_num = np.zeros((2006, 6, 24), np.int64)
    hazard_den = np.zeros_like(hazard_num)
    null = np.full((2006, 6, 24), np.nan)
    baseline = np.full((2006, 6), np.nan)
    zero_amp = 0
    all_hits = 0
    all_events = 0
    for pos, scan in enumerate(scans):
        c = np.asarray(scores[scan, 100:1100, :30], dtype=np.complex128)
        a = np.asarray(alpha[scan, 100:1100])
        y = a.argmax(1)
        guess = pred[pos].argmax(1)
        assert np.isfinite(c).all() and np.isfinite(a).all()
        amplitude = abs(c)
        zero_amp += int((amplitude == 0).sum())
        normalized = amplitude / np.maximum(amplitude.mean(0), 1e-12)
        unit = c / np.maximum(amplitude, 1e-12)
        keep = mask[pos]
        confusion[pos] = np.bincount((12 * y + guess)[keep], minlength=144).reshape(12, 12)
        all_hits += int((guess == y).sum())
        all_events += int((y[1:] != y[:-1]).sum())
        for kind, valid in enumerate([keep, np.ones(1000, bool)]):
            for s in range(12):
                use = valid & (y == s)
                state_count[kind, pos, s] = use.sum()
                if use.any():
                    amp[kind, pos, s] = normalized[use].mean(0)
                    phase[kind, pos, s] = unit[use].mean(0)
        bins = np.floor(((np.angle(c) + np.pi) % (2 * np.pi)) * (24 / (2 * np.pi))).astype(int)
        for k, (i, j, component) in enumerate(PAIRS):
            n, d = hazard_counts(y, bins[:, component - 1], i - 1, j - 1)
            hazard_num[pos, k] = n
            hazard_den[pos, k] = d
            baseline[pos, k] = ratio(n.sum(), d.sum())
        shift_values = np.full((32, 6, 24), np.nan)
        for h, shift in enumerate(shifts[pos]):
            shifted = np.roll(bins, int(shift), axis=0)
            keep_step = np.arange(999) != shift - 1
            for k, (i, j, component) in enumerate(PAIRS):
                n, d = hazard_counts(y, shifted[:, component - 1], i - 1, j - 1, keep_step)
                shift_values[h, k] = ratio(n, d)
        null[pos] = mean_valid(shift_values)
        if pos == 0:
            write_csv(
                result / "fixed_example_agreement.csv",
                [dict(frame=t, unanimous=int(mask[0, t - 100])) for t in range(300, 500)],
            )
        if pos % 200 == 0:
            print("RUNS", pos + 1, len(scans), "seconds", round(time.time() - started), flush=True)
    cm = confusion.sum(0)
    frames = int(cm.sum())
    correct = int(np.trace(cm))
    accuracy = correct / frames if frames else float("nan")
    assert np.array_equal(cm.sum(1), state_count[0].sum(0))
    np.savez_compressed(
        private / "by_run.npz",
        scan=scans,
        amplitude=amp,
        phase=phase,
        state_count=state_count,
        confusion=confusion,
        hazard_numerator=hazard_num,
        hazard_denominator=hazard_den,
        null_probability=null,
        baseline=baseline,
    )
    rng = np.random.default_rng(SEED + 1)
    w = np.stack(
        [np.bincount(rng.integers(1003, size=1003), minlength=1003) for _ in range(2000)]
    ).astype(float)
    hits = confusion.diagonal(axis1=1, axis2=2).sum(1).reshape(1003, 2).sum(1)
    counts = confusion.sum((1, 2)).reshape(1003, 2).sum(1)
    draws = (w @ hits) / (w @ counts)
    stat = dict(
        accuracy=accuracy,
        ci_low=float(np.quantile(draws, 0.025)),
        ci_high=float(np.quantile(draws, 0.975)),
        frames=frames,
        total_frames=2006000,
        coverage=frames / 2006000,
        all_frame_accuracy=all_hits / 2006000,
        balanced_accuracy=float(np.nanmean(ratio(cm.diagonal(), cm.sum(1)))),
        participants=1003,
        states=12,
    )
    (result / "decoding_summary.json").write_text(json.dumps(stat, indent=2))
    write_csv(
        result / "confusion.csv",
        [
            dict(
                state=i + 1,
                predicted_state=j + 1,
                count=int(cm[i, j]),
                row_fraction=cm[i, j] / cm[i].sum(),
                support=int(cm[i].sum()),
            )
            for i in range(12)
            for j in range(12)
        ],
    )
    rows = []
    profile_arrays = {}
    for kind, name in enumerate(["six_fit", "all_reference"]):
        ap = mean_valid(amp[kind].reshape(1003, 2, 12, 30), 1)
        zp = mean_valid(phase[kind].reshape(1003, 2, 12, 30), 1)
        am, ad, ns = bootstrap(ap, w)
        zm, zd, _ = bootstrap(zp, w)
        alo, ahi = np.quantile(ad, [0.025, 0.975], axis=0)
        angle = np.angle(zm)
        plen = abs(zm)
        phase_offsets = np.angle(zd * np.exp(-1j * angle))
        plo, phi = np.quantile(phase_offsets, [0.025, 0.975], axis=0)
        rlo, rhi = np.quantile(abs(zd), [0.025, 0.975], axis=0)
        profile_arrays[name + "_amplitude"] = am
        profile_arrays[name + "_phase"] = zm
        for s in range(12):
            for k in range(30):
                rows.append(
                    dict(
                        subset=name,
                        state=s + 1,
                        component=k + 1,
                        amplitude_ratio=am[s, k],
                        amplitude_sd=np.nanstd(ap[:, s, k], ddof=1),
                        amplitude_low=alo[s, k],
                        amplitude_high=ahi[s, k],
                        phase_degrees=np.degrees(angle[s, k]),
                        resultant_length=plen[s, k],
                        phase_offset_low_degrees=np.degrees(plo[s, k]),
                        phase_offset_high_degrees=np.degrees(phi[s, k]),
                        resultant_low=rlo[s, k],
                        resultant_high=rhi[s, k],
                        participants=int(ns[s, k]),
                        frames=int(state_count[kind, :, s].sum()),
                    )
                )
    write_csv(result / "state_profiles.csv", rows)
    np.savez_compressed(result / "state_profiles.npz", **profile_arrays)
    observed = ratio(hazard_num, hazard_den)
    common = np.isfinite(observed) & np.isfinite(null)
    observed = np.where(common, observed, np.nan)
    null = np.where(common, null, np.nan)
    arrays = {}
    rows = []
    summary = []
    for name, values in [
        ("observed", observed),
        ("shifted", null),
        ("difference", observed - null),
    ]:
        participant = mean_valid(values.reshape(1003, 2, 6, 24), 1)
        mean, draws, ns = bootstrap(participant, w)
        lo, hi = np.nanquantile(draws, [0.025, 0.975], axis=0)
        arrays[name + "_mean"] = mean
        arrays[name + "_low"] = lo
        arrays[name + "_high"] = hi
        arrays[name + "_subjects"] = ns
        for q, (i, j, k) in enumerate(PAIRS):
            for b in range(24):
                rows.append(
                    dict(
                        series=name,
                        source=i,
                        target=j,
                        component=k,
                        phase_degrees=-172.5 + 15 * b,
                        mean=mean[q, b],
                        low=lo[q, b],
                        high=hi[q, b],
                        participants=int(ns[q, b]),
                        observed_source_frames=int(hazard_den[:, q, b].sum()),
                        observed_transitions=int(hazard_num[:, q, b].sum()),
                    )
                )
    bm, bd, bns = bootstrap(mean_valid(baseline.reshape(1003, 2, 6), 1), w)
    arrays["baseline_mean"] = bm
    arrays["baseline_low"] = np.nanquantile(bd, 0.025, axis=0)
    arrays["baseline_high"] = np.nanquantile(bd, 0.975, axis=0)
    for q, (i, j, k) in enumerate(PAIRS):
        v = arrays["observed_mean"][q]
        peak = int(np.nanargmax(v)) if np.isfinite(v).any() else 0
        trough = int(np.nanargmin(v)) if np.isfinite(v).any() else 0
        summary.append(
            dict(
                source=i,
                target=j,
                component=k,
                events=int(hazard_num[:, q].sum()),
                source_frames=int(hazard_den[:, q].sum()),
                baseline_mean=float(bm[q]),
                phase_peak_degrees=-172.5 + 15 * peak if np.isfinite(v).any() else float("nan"),
                phase_peak_probability=float(v[peak]),
                phase_min_degrees=-172.5 + 15 * trough if np.isfinite(v).any() else float("nan"),
                phase_min_probability=float(v[trough]),
                minimum_participants_per_bin=int(arrays["observed_subjects"][q].min()),
                maximum_participants_per_bin=int(arrays["observed_subjects"][q].max()),
            )
        )
    write_csv(result / "phase_transition_probability.csv", rows)
    write_csv(result / "phase_transition_summary.csv", summary)
    arrays["phase_degrees"] = np.arange(-172.5, 180, 15)
    arrays["pairs"] = np.array(PAIRS)
    np.savez_compressed(result / "phase_transition_probability.npz", **arrays)
    after = [fingerprint(p) for p in files]
    assert before == after
    report = dict(
        status="PASS",
        seconds=time.time() - started,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_fingerprints=before,
        sources_unchanged=before == after,
        models_refitted=False,
        consensus_replay_exact=True,
        all_reference_events=all_events,
        zero_amplitude_coordinates=zero_amp,
        participants=1003,
        runs=2006,
        bootstrap_draws=2000,
        shift_count=32,
        shift_seed=SEED,
        bootstrap_seed=SEED + 1,
        shift_offsets=[100, 900],
        main_pairs=PAIRS[:3],
        probability_estimator="equal available runs within participant, then equal available participants in each phase bin",
        common_run_bin_cells=int(common.sum()),
        observed_run_bin_cells=int(np.isfinite(ratio(hazard_num, hazard_den)).sum()),
    )
    complete.write_text(json.dumps(report, indent=2))
    print("COMPLETE", json.dumps(stat), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test()
    if not args.self_test:
        if args.output is None:
            parser.error("--output required")
        main(args.output)
