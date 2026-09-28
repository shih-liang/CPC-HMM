"""Fit additional regularized EM HMM initializations for reliability analysis.

Reads ICA50 data and writes fit-specific posteriors; executes on import.
"""

import os
import time
import json
import argparse

args = argparse.ArgumentParser()
args.add_argument("--seed", type=int, default=20260906)
args.add_argument("--tag", default="robust")
args.add_argument("--iterations", type=int, default=100)
args.add_argument("--acquisition", type=int, default=0)
args = args.parse_args()
os.environ.setdefault("NUMBA_NUM_THREADS", "8")
from pathlib import Path
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from sklearn.cluster import MiniBatchKMeans
from numba import njit, prange

P = Path(
    os.path.join(
        os.environ.get("HCP_DERIVATIVES", "/configure/HCP_DERIVATIVES"),
        "wave_rsn_revision_20260906/revision_20260910",
    )
)
for directory in (P / "data", P / "results"):
    directory.mkdir(parents=True, exist_ok=True)
H = Path(
    os.path.join(os.environ.get("HCP_ICA_ROOT", "/configure/HCP_ICA_ROOT"), "X_ICA50_zscore.npy")
)
rng = np.random.default_rng(args.seed)
K = 12
V = 50
L = 1200
start = time.time()
raw = np.load(H, mmap_mode="r").reshape(4012, L, V)
x = np.asarray(raw[(np.arange(1003) * 4 + args.acquisition)], dtype=np.float64).reshape(-1, V)
N = len(x)


@njit(parallel=True)
def fb(B, A, pi):
    S, T, K = B.shape
    G = np.empty_like(B)
    XX = np.zeros((S, K, K))
    LL = np.zeros(S)
    for s in prange(S):
        a = np.zeros((T, K))
        scale = np.zeros(T)
        for k in range(K):
            a[0, k] = pi[k] * B[s, 0, k]
        scale[0] = a[0].sum()
        a[0] /= scale[0]
        for t in range(1, T):
            for j in range(K):
                v = 0.0
                for i in range(K):
                    v += a[t - 1, i] * A[i, j]
                a[t, j] = v * B[s, t, j]
            scale[t] = a[t].sum()
            a[t] /= scale[t]
        b = np.ones((T, K))
        for t in range(T - 2, -1, -1):
            for i in range(K):
                v = 0.0
                for j in range(K):
                    z = A[i, j] * B[s, t + 1, j] * b[t + 1, j] / scale[t + 1]
                    v += z
                    XX[s, i, j] += a[t, i] * z
                b[t, i] = v
        for t in range(T):
            den = 0.0
            for k in range(K):
                G[s, t, k] = a[t, k] * b[t, k]
                den += G[s, t, k]
            G[s, t] /= den
            LL[s] += np.log(scale[t])
    return G, XX, LL


def emissions(data, means, covs):
    out = np.empty((len(data), K))
    for k in range(K):
        chol = cholesky(covs[k], lower=True)
        w = solve_triangular(chol, (data - means[k]).T, lower=True, check_finite=False)
        out[:, k] = -0.5 * (
            np.sum(w * w, axis=0) + 2 * np.log(np.diag(chol)).sum() + V * np.log(2 * np.pi)
        )
    return out


def infer(data, means, covs, A, pi):
    e = emissions(data, means, covs)
    shift = e.max(1)
    B = np.exp(e - shift[:, None]).reshape(-1, L, K)
    g, xx, ll = fb(B, A, pi)
    return g.reshape(-1, K), xx.sum(0), ll + shift.reshape(-1, L).sum(1)


model = P / f"data/training_hmm_{args.tag}.npz"
histpath = P / f"results/hmm_{args.tag}_training_history.json"
if model.exists():
    m = np.load(model)
    means = m["means"]
    covs = m["covariances"]
    A = m["transition"]
    pi = m["initial"]
    history = json.loads(histpath.read_text())
else:
    means = np.array(
        [
            x[(int(i) // 1200) * 1200 + 100 : (int(i) // 1200) * 1200 + 200].mean(0)
            for i in rng.integers(0, N, 12)
        ]
    )
    shared = np.cov(x, rowvar=False) + 0.05 * np.eye(V)
    covs = np.repeat(shared[None], K, axis=0)
    A = np.full((K, K), 0.1 / (K - 1))
    np.fill_diagonal(A, 0.9)
    pi = np.ones(K) / K
    history = []
for it in range(len(history), args.iterations):
    g, xx, ll = infer(x, means, covs, A, pi)
    score = float(ll.sum() / N)
    nk = g.sum(0)
    means = (g.T @ x) / nk[:, None]
    for k in range(K):
        c = x - means[k]
        covs[k] = ((c * g[:, k, None]).T @ c) / nk[k] + 0.01 * np.eye(V)
    A = xx + 1e-3
    A /= A.sum(1, keepdims=True)
    pi = g.reshape(-1, L, K)[:, 0].mean(0)
    pi = np.maximum(pi, 1e-10)
    pi /= pi.sum()
    history.append(
        dict(
            iteration=it + 1,
            mean_loglik=score,
            min_state_occupancy=float(nk.min() / N),
            seconds=time.time() - start,
        )
    )
    print(history[-1], flush=True)
    np.savez(
        model,
        means=means,
        covariances=covs,
        transition=A,
        initial=pi,
        training_scans=(np.arange(1003) * 4 + args.acquisition),
    )
    histpath.write_text(json.dumps(history, indent=2))
    if it >= 49 and abs(history[-1]["mean_loglik"] - history[-2]["mean_loglik"]) < 1e-5:
        break
alpha = np.lib.format.open_memmap(
    P / f"data/hmm_alpha_{args.tag}.npy", mode="w+", dtype="float32", shape=(4012, L, K)
)
scores = []
for startscan in range(0, 4012, 20):
    stop = min(startscan + 20, 4012)
    data = np.asarray(raw[startscan:stop], dtype=np.float64).reshape(-1, V)
    g, xx, ll = infer(data, means, covs, A, pi)
    alpha[startscan:stop] = g.reshape(stop - startscan, L, K)
    scores.extend(ll.tolist())
    if startscan % 200 == 0:
        print("inference", stop, "seconds", round(time.time() - start), flush=True)
alpha.flush()
np.save(P / f"results/hmm_{args.tag}_loglik_per_scan.npy", scores)
(P / f"results/hmm_{args.tag}_metadata.json").write_text(
    json.dumps(
        {
            "training": ["REST1_LR", "REST1_RL", "REST2_LR", "REST2_RL"][args.acquisition],
            "n_states": K,
            "seed": args.seed,
            "covariance_floor": 0.01,
            "algorithm": "scaled full-sequence Baum-Welch, separate scans",
            "initialization": "Training scan-segment means and shared covariance",
            "training_iterations": len(history) if "history" in globals() else "resumed",
            "seconds": time.time() - start,
        },
        indent=2,
    )
)
print("COMPLETE", flush=True)
