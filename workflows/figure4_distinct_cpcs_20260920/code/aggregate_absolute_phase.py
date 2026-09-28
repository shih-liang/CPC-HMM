"""Original-reference event phases: circular participant means and SDs."""

import os  # Public release: configurable data roots.
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np

SOURCE = Path(
    os.path.join(
        os.environ.get("HCP_DERIVATIVES", "/configure/HCP_DERIVATIVES"),
        "wave_rsn_revision_20260906/data",
    )
)
LAGS = np.arange(-5, 6)


def divide(a, b):
    return np.divide(
        a,
        b,
        out=np.full(
            np.broadcast_shapes(np.shape(a), np.shape(b)), np.nan, dtype=np.result_type(a, float)
        ),
        where=np.asarray(b) > 0,
    )


def summarize(z):
    """Return circular mean, circular SD (radians), resultant length and valid count.

    Zero-length/undefined complex vectors are omitted before averaging directions.
    """
    ok = np.isfinite(z) & (abs(z) > 1e-12)
    units = divide(z, abs(z))
    avg = divide(np.where(ok, units, 0).sum(0), ok.sum(0))
    r = np.minimum(abs(avg), 1)
    return np.angle(avg), np.sqrt(-2 * np.log(np.maximum(r, 1e-300))), r, ok.sum(0)


def self_test():
    z = np.exp(1j * np.deg2rad([179, -179]))
    m, sd, r, n = summarize(z)
    assert abs(abs(m) - np.pi) < 1e-10 and abs(sd - np.deg2rad(1)) < 1e-5
    m2, sd2, _, _ = summarize(z * np.exp(0.4j))
    assert abs(np.angle(np.exp(1j * (m2 - m - 0.4)))) < 1e-10 and abs(sd - sd2) < 1e-10
    assert n == 2 and r > 0.999
    print("SELF_TEST PASS", flush=True)


def fingerprint(path):
    s = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(65536))
        f.seek(s.st_size - 65536)
        h.update(f.read(65536))
    return dict(path=str(path), size=s.st_size, mtime_ns=s.st_mtime_ns, edge_sha256=h.hexdigest())


def main(out):
    start = time.time()
    out.mkdir(parents=True, exist_ok=True)
    result = out / "results"
    result.mkdir(exist_ok=True)
    assert not (result / "absolute_phase_complete.json").exists(), "Completed output exists."
    paths = [SOURCE / "cpca_scores_200.npy", SOURCE / "hmm_alpha_robust_seed2.npy"]
    before = [fingerprint(p) for p in paths]
    scores, alpha = [np.load(p, mmap_mode="r") for p in paths]
    scans = np.sort(np.r_[np.arange(1003) * 4 + 2, np.arange(1003) * 4 + 3])
    shape = (144, 11, 30)
    unitsum = np.zeros(shape, complex)
    n = np.zeros(shape, int)
    within = np.zeros(shape)
    undefined = np.zeros(shape, int)
    counts = np.zeros(144, int)
    runs = []
    for pos, scan in enumerate(scans):
        c = np.asarray(scores[scan, :, :30], complex)
        y = np.asarray(alpha[scan]).argmax(1)
        t = np.flatnonzero(y[1:] != y[:-1]) + 1
        t = t[(t >= 105) & (t < 1095)]
        key = y[t - 1] * 12 + y[t]
        cnt = np.bincount(key, minlength=144)
        counts += cnt
        windows = c[t[:, None] + LAGS]
        valid = abs(windows) > 1e-12
        unit = divide(windows, abs(windows))
        sums = np.zeros(shape, complex)
        denom = np.zeros(shape, int)
        np.add.at(sums, key, np.where(valid, unit, 0))
        np.add.at(denom, key, valid)
        runs.append(divide(sums, denom))
        if len(runs) == 2:
            z = np.stack(runs)
            ok = np.isfinite(z)
            participant = divide(np.where(ok, z, 0).sum(0), ok.sum(0))
            defined = np.isfinite(participant) & (abs(participant) > 1e-12)
            u = divide(participant, abs(participant))
            unitsum += np.where(defined, u, 0)
            within += np.where(defined, abs(participant), 0)
            n += defined
            undefined += np.isfinite(participant) & ~defined
            runs = []
        if pos % 500 == 0:
            print("RUN", pos + 1, len(scans), "seconds", round(time.time() - start), flush=True)
    avg = divide(unitsum, n)
    r = np.minimum(abs(avg), 1)
    mean = np.angle(avg)
    sd = np.sqrt(-2 * np.log(np.maximum(r, 1e-300)))
    sd[n < 2] = np.nan
    mean[(n == 0) | (r <= 1e-12)] = np.nan
    assert counts.sum() == 223131 and not runs
    np.savez_compressed(
        result / "absolute_phase_profiles.npz",
        absolute_phase_mean=mean,
        absolute_phase_sd=sd,
        absolute_phase_participants=n,
        absolute_phase_resultant=r,
        within_participant_resultant=divide(within, n),
        undefined_participant_means=undefined,
        window_counts=counts,
        lags_TR=LAGS,
    )
    assert before == [fingerprint(p) for p in paths]
    (result / "absolute_phase_complete.json").write_text(
        json.dumps(
            dict(
                status="PASS",
                source_fingerprints=before,
                source_preserved=True,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                participants=1003,
                runs=2006,
                events=int(counts.sum()),
                phase_origin="Original fixed CPC basis; no per-event rotation",
                group_mean="Circular mean of participant mean directions",
                SD="Circular SD across participant mean directions",
                undefined_participant_means=int(undefined.sum()),
                numpy=np.__version__,
                seconds=time.time() - start,
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
