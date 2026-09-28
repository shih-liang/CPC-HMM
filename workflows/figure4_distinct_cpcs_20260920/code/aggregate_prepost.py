"""Participant-paired amplitude and circular-phase comparisons; frozen protocol."""

from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np
from scipy.stats import false_discovery_control

SOURCE = Path(
    os.path.join(
        os.environ.get("HCP_DERIVATIVES", "/configure/HCP_DERIVATIVES"),
        "wave_rsn_revision_20260906/data",
    )
)
LAGS = np.arange(-5, 6)
BEFORE, AFTER = slice(0, 5), slice(5, 10)
N_PERM, SEED = 19999, 20260920


def mean_valid(a, axis=0):
    ok = np.isfinite(a)
    n = ok.sum(axis)
    return np.divide(
        np.where(ok, a, 0).sum(axis),
        n,
        out=np.full(n.shape, np.nan, dtype=np.result_type(a.dtype, float)),
        where=n > 0,
    )


def unit(a):
    """Normalize defined complex values to unit length; near-zero values become NaN."""
    return np.divide(a, np.abs(a), out=np.full(a.shape, np.nan + 0j), where=np.abs(a) > 1e-12)


def grouped(a, keys):
    ok = np.isfinite(a)
    total = np.zeros((144, 11), dtype=a.dtype)
    count = np.zeros((144, 11), int)
    np.add.at(total, keys, np.where(ok, a, 0))
    np.add.at(count, keys, ok)
    return np.divide(
        total, count, out=np.full(total.shape, np.nan, dtype=total.dtype), where=count > 0
    )


def fingerprint(path):
    stat = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(65536))
        f.seek(max(0, stat.st_size - 65536))
        h.update(f.read(65536))
    return dict(
        path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns, edge_sha256=h.hexdigest()
    )


def paired_pvalues(amplitude_difference, phase_difference):
    """Test participant-paired amplitude and complex phase-vector differences.

    Arrays are (participants, directed transitions); missing pairs are NaN.
    Shared sign flips preserve dependence across transitions. Small samples use
    exact enumeration. Phase tests concern the complex difference, not angles
    subtracted linearly. Multiplicity correction is applied by the caller.
    """
    ad, zd = amplitude_difference, phase_difference
    na, nz = np.isfinite(ad).sum(0), np.isfinite(zd).sum(0)
    da, dz = np.nan_to_num(ad), np.nan_to_num(zd)
    observed_a, observed_z = np.abs(da.sum(0)), np.abs(dz.sum(0))
    extreme_a, extreme_z = np.zeros(ad.shape[1], int), np.zeros(zd.shape[1], int)
    # Common participant signs retain dependence across components and transition types.
    joined = np.concatenate([da, dz.real, dz.imag], axis=1)
    rng = np.random.default_rng(SEED)
    ntypes = ad.shape[1]
    for start in range(0, N_PERM, 500):
        signs = rng.integers(0, 2, size=(min(500, N_PERM - start), len(ad)), dtype=np.int8) * 2 - 1
        sums = signs.astype(float) @ joined
        sa = np.abs(sums[:, :ntypes])
        sz = np.hypot(sums[:, ntypes : 2 * ntypes], sums[:, 2 * ntypes :])
        extreme_a += (sa >= observed_a[None, :] - 1e-12 * np.maximum(1, observed_a)).sum(0)
        extreme_z += (sz >= observed_z[None, :] - 1e-12 * np.maximum(1, observed_z)).sum(0)
    pa, pz = (extreme_a + 1) / (N_PERM + 1), (extreme_z + 1) / (N_PERM + 1)
    exact_a, exact_z = np.zeros(ntypes, bool), np.zeros(ntypes, bool)
    for values, counts, pvalues, exact in [(ad, na, pa, exact_a), (zd, nz, pz, exact_z)]:
        for q in range(ntypes):
            n = int(counts[q])
            if n < 2:
                pvalues[q] = 1.0
                continue
            if n <= 14:
                v = values[np.isfinite(values[:, q]), q]
                signs = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1) * 2 - 1
                observed = abs(v.sum())
                null = abs(signs @ v)
                pvalues[q] = np.mean(null >= observed - 1e-12 * max(1, observed))
                exact[q] = True
    return pa, pz, na, nz, exact_a, exact_z


def self_test():
    z = unit(np.exp(1j * np.deg2rad([179.0, -179.0])))
    assert abs(abs(np.angle(z.mean())) - np.pi) < 1e-12
    a = np.array([[1.0, np.nan], [3.0, 2.0]])
    assert np.allclose(mean_valid(a), [2, 2])
    # Exact paired sign flips agree with SciPy for a scalar contrast and vector phase contrast.
    from scipy.stats import permutation_test

    a = np.array([0.2, 0.5, -0.1, 0.7])
    z = np.array([0.2 + 0.3j, -0.1 + 0.4j, 0.6 - 0.2j, 0.1 + 0.1j])
    signs = ((np.arange(16)[:, None] >> np.arange(4)) & 1) * 2 - 1
    for difference in [a, z]:
        exact = np.mean(abs(signs @ difference) >= abs(difference.sum()) - 1e-12)
        scipy_result = permutation_test(
            (difference,),
            lambda x: abs(x.sum()),
            permutation_type="samples",
            alternative="greater",
            n_resamples=np.inf,
        )
        assert abs(exact - scipy_result.pvalue) < 1e-12
    rotation = np.exp(0.83j)
    assert np.allclose(abs(signs @ (z * rotation)), abs(signs @ z))
    print("SELF_TEST PASS", flush=True)


def main(output, reference):
    start = time.time()
    output.mkdir(parents=True, exist_ok=True)
    result = output / "results"
    result.mkdir(exist_ok=True)
    assert not (result / "complete.json").exists(), (
        "Completed run already exists; do not overwrite."
    )
    ref = np.load(reference)
    chosen = ref["chosen"]
    paths = [SOURCE / "cpca_scores_200.npy", SOURCE / "hmm_alpha_robust_seed2.npy"]
    before = [fingerprint(p) for p in paths]
    scores, alpha = [np.load(p, mmap_mode="r") for p in paths]
    assert scores.shape == (4012, 1200, 200) and alpha.shape == (4012, 1200, 12)
    amps = np.full((1003, 144, 11), np.nan)
    phases = np.full_like(amps, np.nan + 0j, dtype=complex)
    counts = np.zeros(144, int)
    for subject in range(1003):
        run_a, run_z = [], []
        for scan in [subject * 4 + 2, subject * 4 + 3]:
            c = np.asarray(scores[scan, :, :30], complex)
            y = np.asarray(alpha[scan]).argmax(1)
            events = np.flatnonzero(y[1:] != y[:-1]) + 1
            events = events[(events >= 105) & (events < 1095)]
            keys = y[events - 1] * 12 + y[events]
            counts += np.bincount(keys, minlength=144)
            windows = c[events[:, None] + LAGS, chosen[keys, None]]
            normalized = np.abs(windows) / np.abs(c[100:1100]).mean(0)[chosen[keys], None]
            run_a.append(grouped(normalized, keys))
            run_z.append(grouped(unit(windows), keys))
        amps[subject] = mean_valid(np.stack(run_a))
        phases[subject] = mean_valid(np.stack(run_z))
        if subject % 250 == 0:
            print("SUBJECT", subject + 1, "seconds", round(time.time() - start), flush=True)
    keys = np.array([i * 12 + j for i in range(12) for j in range(12) if i != j])
    assert np.array_equal(counts, ref["window_counts"])
    replay_a = mean_valid(amps[:, keys])
    replay_z = mean_valid(unit(phases[:, keys]))
    error_a = np.nanmax(np.abs(replay_a - ref["amplitude_reference"]))
    error_z = np.nanmax(np.abs(np.angle(replay_z * np.exp(-1j * ref["phase_reference"]))))
    assert np.array_equal(np.isfinite(amps[:, keys]).sum(0), ref["participant_reference"])
    a = np.stack([mean_valid(amps[:, keys, period], axis=2) for period in [BEFORE, AFTER]], axis=-1)
    z = unit(
        np.stack(
            [mean_valid(phases[:, keys, period], axis=2) for period in [BEFORE, AFTER]], axis=-1
        )
    )
    valid_a = np.isfinite(a).all(axis=-1)
    valid_z = np.isfinite(z).all(axis=-1)
    a = np.where(valid_a[:, :, None], a, np.nan)
    z = np.where(valid_z[:, :, None], z, np.nan + 0j)
    da, dz = a[:, :, 1] - a[:, :, 0], z[:, :, 1] - z[:, :, 0]
    pa, pz, na, nz, exact_a, exact_z = paired_pvalues(da, dz)
    qa, qz = np.split(false_discovery_control(np.r_[pa, pz], method="by"), 2)
    means_a = mean_valid(a)
    means_z = mean_valid(z)
    phase_angle = np.angle(means_z)
    phase_resultant = np.abs(means_z)
    np.savez_compressed(
        output / "participant_prepost_private.npz", amplitude=a, phase_unit=z, transition_keys=keys
    )
    np.savez_compressed(
        result / "prepost_summary.npz",
        transition_keys=keys,
        CPC=chosen[keys] + 1,
        amplitude_mean=means_a,
        amplitude_sd=np.nanstd(a, axis=0, ddof=1),
        amplitude_participants=na,
        phase_mean_degrees=np.rad2deg(phase_angle),
        phase_resultant=phase_resultant,
        phase_circular_sd_degrees=np.rad2deg(np.sqrt(-2 * np.log(np.minimum(phase_resultant, 1)))),
        phase_participants=nz,
        amplitude_p=pa,
        phase_p=pz,
        amplitude_q=qa,
        phase_q=qz,
        amplitude_significant=qa < 0.05,
        phase_significant=qz < 0.05,
        window_counts=counts[keys],
        before_TR=LAGS[BEFORE],
        after_TR=LAGS[AFTER],
    )
    rows = []
    for row, key in enumerate(keys):
        rows.append(
            dict(
                source=int(key // 12 + 1),
                target=int(key % 12 + 1),
                CPC=int(chosen[key] + 1),
                events=int(counts[key]),
                amplitude_participants=int(na[row]),
                phase_participants=int(nz[row]),
                amplitude_before=float(means_a[row, 0]),
                amplitude_after=float(means_a[row, 1]),
                amplitude_difference=float(mean_valid(da[:, row])),
                phase_before_degrees=float(np.rad2deg(phase_angle[row, 0])),
                phase_after_degrees=float(np.rad2deg(phase_angle[row, 1])),
                phase_vector_difference=float(abs(mean_valid(dz[:, row]))),
                amplitude_p=float(pa[row]),
                amplitude_q_BY=float(qa[row]),
                amplitude_exact=bool(exact_a[row]),
                phase_p=float(pz[row]),
                phase_q_BY=float(qz[row]),
                phase_exact=bool(exact_z[row]),
            )
        )
    with (result / "prepost_statistics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert before == [fingerprint(p) for p in paths]
    complete = dict(
        status="PASS",
        source_fingerprints=before,
        source_preserved=True,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        participants=1003,
        runs=2006,
        complete_events=int(counts.sum()),
        directions=132,
        period_frames=5,
        before_TR=[-5, -4, -3, -2, -1],
        after_TR=[0, 1, 2, 3, 4],
        permutations=N_PERM,
        seed=SEED,
        correction="BY, all 264 tests jointly",
        amplitude_significant=int((qa < 0.05).sum()),
        phase_significant=int((qz < 0.05).sum()),
        amplitude_replay_max_error=float(error_a),
        phase_replay_max_error_radians=float(error_z),
        undefined_period_phase_count=int(np.sum(np.isfinite(phases[:, keys, 0]) & ~valid_z)),
        seconds=time.time() - start,
    )
    (result / "complete.json").write_text(json.dumps(complete, indent=2))
    print(json.dumps(complete, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test()
    if not args.self_test:
        if args.output is None or args.reference is None:
            parser.error("--output and --reference required")
        main(args.output, args.reference)
