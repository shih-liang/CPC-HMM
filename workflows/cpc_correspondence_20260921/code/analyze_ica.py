"""Bounded CPC30 to ICA50 linear analysis. Never mutates source files."""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
from pathlib import Path
import numpy as np
import json
import csv
import time
import hashlib

P = Path(
    os.path.join(
        os.environ.get("HCP_DERIVATIVES", "/configure/HCP_DERIVATIVES"),
        "wave_rsn_revision_20260906",
    )
)
OUT = Path(
    os.path.join(
        os.environ.get("HCP_DERIVATIVES", "/configure/HCP_DERIVATIVES"),
        "cpc_correspondence_20260921_01",
    )
)
OUT.mkdir(exist_ok=True)
assert not (OUT / "complete.json").exists(), "Reuse completed run; do not overwrite"
paths = [
    P / "data/cpca_scores_200.npy",
    Path(
        os.path.join(
            os.environ.get("HCP_ICA_ROOT", "/configure/HCP_ICA_ROOT"), "X_ICA50_zscore.npy"
        )
    ),
    P / "data/hmm_alpha_robust_seed2.npy",
]


def fp(p):
    s = p.stat()
    return dict(path=str(p), size=s.st_size, mtime_ns=s.st_mtime_ns)


before = [fp(p) for p in paths]
start = time.time()
c = np.load(paths[0], mmap_mode="r")
y = np.load(paths[1], mmap_mode="r").reshape(4012, 1200, 50)
alpha = np.load(paths[2], mmap_mode="r")
assert c.shape == (4012, 1200, 200) and alpha.shape == (4012, 1200, 12)


def features(scan):
    z = np.asarray(c[scan, :, :30], dtype=np.complex128)
    return np.c_[z.real, z.imag]


def blank():
    return dict(
        n=0,
        sx=np.zeros(60),
        sy=np.zeros(50),
        xx=np.zeros((60, 60)),
        xy=np.zeros((60, 50)),
        yy=np.zeros(50),
    )


def add(s, x, z):
    s["n"] += len(x)
    s["sx"] += x.sum(0)
    s["sy"] += z.sum(0)
    s["xx"] += x.T @ x
    s["xy"] += x.T @ z
    s["yy"] += (z * z).sum(0)


def moments(s):
    mx = s["sx"] / s["n"]
    my = s["sy"] / s["n"]
    return (
        mx,
        my,
        s["xx"] / s["n"] - np.outer(mx, mx),
        s["xy"] / s["n"] - np.outer(mx, my),
        s["yy"] / s["n"] - my * my,
    )


full = blank()
train = blank()
val = blank()
state_n = np.zeros(12, int)
state_x = np.zeros((12, 60))
state_xx = np.zeros((12, 60, 60))
for u in range(1003):
    scan = 4 * u
    x = features(scan)
    z = np.asarray(y[scan], float)
    add(full, x, z)
    add(train, x[100:1100], z[100:1100])
    labels = np.asarray(alpha[scan, 100:1100]).argmax(1)
    for k in range(12):
        a = x[100:1100][labels == k]
        state_n[k] += len(a)
        state_x[k] += a.sum(0)
        state_xx[k] += a.T @ a
    add(val, features(scan + 1)[100:1100], np.asarray(y[scan + 1, 100:1100], float))
    if u % 200 == 0:
        print("TRAIN/VALID", u, round(time.time() - start), flush=True)
mx, my, xx, xy, yy = moments(train)
sd = np.sqrt(np.diag(xx))
G = xx / sd[:, None] / sd[None, :]
Q = xy / sd[:, None]
lams = np.array([0, 0.0001, 0.001, 0.01, 0.1, 1.0])
weights = np.array([np.linalg.solve(G + l * np.eye(60), Q) for l in lams])
vmx, vmy, vxx, vxy, vyy = moments(val)
vG = vxx / sd[:, None] / sd[None, :]
vQ = vxy / sd[:, None]
errs = []
for w in weights:
    bias = (vmx - mx) / sd @ w + my - vmy
    errs.append(
        float(
            np.mean(
                vyy + np.einsum("ij,ij->j", w, vG @ w) - 2 * np.sum(w * vQ, axis=0) + bias * bias
            )
        )
    )
best = int(np.argmin(errs))
W = weights[best]
pairs = np.c_[np.arange(30), np.arange(30) + 30]
single = np.stack([np.linalg.solve(G[np.ix_(ix, ix)], Q[ix]) for ix in pairs])
np.savez(
    OUT / "linear_model.npz",
    mean=mx,
    std=sd,
    intercept=my,
    weight=W,
    ridge=lams[best],
    single_weights=single,
    pairs=pairs,
)


def cov_audit(v):
    rr = v[:30, :30]
    ii = v[30:, 30:]
    ri = v[:30, 30:]
    H = rr + ii + 1j * (ri - ri.T)
    pseudo = rr - ii + 1j * (ri + ri.T)
    scale = np.sqrt(np.outer(np.diag(H).real, np.diag(H).real))
    h = H / scale
    np.fill_diagonal(h, 0)
    block = np.zeros_like(v)
    for ix in pairs:
        block[np.ix_(ix, ix)] = v[np.ix_(ix, ix)]
    return dict(
        max_normalized_complex_offdiagonal=float(abs(h).max()),
        real_between_CPC_fraction=float(np.linalg.norm(v - block) / np.linalg.norm(v)),
        pseudocovariance_fraction=float(np.linalg.norm(pseudo) / np.linalg.norm(H)),
    )


audit = {
    "full_training": cov_audit(moments(full)[2]),
    "core_training": cov_audit(xx),
    "state_conditioned": [],
}
for k in range(12):
    mu = state_x[k] / state_n[k]
    v = state_xx[k] / state_n[k] - np.outer(mu, mu)
    audit["state_conditioned"].append(dict(state=k + 1, n=int(state_n[k]), **cov_audit(v)))
(OUT / "covariance_audit.json").write_text(json.dumps(audit, indent=2))
run_r = []
run_r2 = []
ctrl_r = []
ctrl_r2 = []
coupling = []
fc_r = []
fc_obs = np.zeros((50, 50))
fc_pred = np.zeros((50, 50))
state_obs = np.zeros((12, 50))
state_pred = np.zeros((12, 50))
state_counts = np.zeros(12, int)


def corr_cov(v):
    return v / np.sqrt(np.outer(np.diag(v), np.diag(v)))


upper = np.triu_indices(50, 1)
for u in range(1003):
    for run in [2, 3]:
        scan = 4 * u + run
        x = (features(scan)[100:1100] - mx) / sd
        z = np.asarray(y[scan, 100:1100], float)
        pred = x @ W + my
        xc = x - x.mean(0)
        zc = z - z.mean(0)
        pc = pred - pred.mean(0)
        vx = xc.T @ xc / len(x)
        cross = xc.T @ zc / len(x)
        vz = (zc * zc).mean(0)
        vp = (pc * pc).mean(0)
        r = (zc * pc).mean(0) / np.sqrt(vz * vp)
        r2 = 1 - np.mean((z - pred) ** 2, axis=0) / vz
        shifted = np.roll(pred, 317, axis=0)
        sc = shifted - shifted.mean(0)
        cr = (zc * sc).mean(0) / np.sqrt(vz * vp)
        cr2 = 1 - np.mean((z - shifted) ** 2, axis=0) / vz
        single_r = []
        for k, ix in enumerate(pairs):
            w = single[k]
            cv = np.sum(w * cross[ix], axis=0)
            pv = np.einsum("ij,ij->j", w, vx[np.ix_(ix, ix)] @ w)
            single_r.append(cv / np.sqrt(np.maximum(pv * vz, 1e-20)))
        run_r.append(r)
        run_r2.append(r2)
        ctrl_r.append(cr)
        ctrl_r2.append(cr2)
        coupling.append(single_r)
        ofc = corr_cov(zc.T @ zc / len(z))
        pfc = corr_cov(pc.T @ pc / len(z))
        fc_obs += ofc
        fc_pred += pfc
        fc_r.append(np.corrcoef(ofc[upper], pfc[upper])[0, 1])
        if u == 0 and run == 2:
            np.savez(
                OUT / "fixed_example.npz",
                observed=z[200:400],
                predicted=pred[200:400],
                cpc=np.asarray(c[scan, 300:500, :30]),
                frames=np.arange(300, 500),
                ICA=np.array([1, 10, 20]),
            )
    if u % 200 == 0:
        print("TEST", u, round(time.time() - start), flush=True)


def paired(a):
    return np.asarray(a).reshape(1003, 2, *np.asarray(a).shape[1:]).mean(1)


r, r2, cr, cr2, cp, fr = map(paired, [run_r, run_r2, ctrl_r, ctrl_r2, coupling, fc_r])
np.savez_compressed(
    OUT / "participant_results.npz", r=r, r2=r2, shift_r=cr, shift_r2=cr2, single_cpc_r=cp, fc_r=fr
)
np.savez(
    OUT / "figure_arrays.npz",
    single_r_mean=cp.mean(0),
    single_r_sd=cp.std(0, ddof=1),
    r_mean=r.mean(0),
    r_sd=r.std(0, ddof=1),
    r2_mean=r2.mean(0),
    r2_sd=r2.std(0, ddof=1),
    shift_r_mean=cr.mean(0),
    shift_r_sd=cr.std(0, ddof=1),
    shift_r2_mean=cr2.mean(0),
    shift_r2_sd=cr2.std(0, ddof=1),
    fc_observed=fc_obs / 2006,
    fc_predicted=fc_pred / 2006,
)
with (OUT / "ica_summary.csv").open("w") as f:
    wr = csv.writer(f)
    wr.writerow(["ICA", "r_mean", "r_sd", "R2_mean", "R2_sd", "shift_r_mean", "shift_R2_mean"])
    for k in range(50):
        wr.writerow(
            [
                k + 1,
                r[:, k].mean(),
                r[:, k].std(ddof=1),
                r2[:, k].mean(),
                r2[:, k].std(ddof=1),
                cr[:, k].mean(),
                cr2[:, k].mean(),
            ]
        )
assert before == [fp(p) for p in paths]
assert np.isfinite(cp).all() and np.max(abs(cp)) <= 1 + 1e-10
meta = dict(
    status="COMPLETE",
    seconds=time.time() - start,
    participants=1003,
    test_runs=2006,
    CPC=30,
    real_features=60,
    ICA=50,
    ridge=float(lams[best]),
    validation_penalties=lams.tolist(),
    validation_mse=errs,
    mean_network_r=float(r.mean()),
    mean_network_R2=float(r2.mean()),
    mean_shift_r=float(cr.mean()),
    fc_r_mean=float(fr.mean()),
    fc_r_sd=float(fr.std(ddof=1)),
    source_fingerprints=before,
    sources_unchanged=True,
)
(OUT / "complete.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta), flush=True)
