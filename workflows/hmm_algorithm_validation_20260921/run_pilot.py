"""Run the bounded EM versus Variational Bayes comparison.

Our Gaussian EM and Variational Bayes implementations; executes on import.
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
import json
import time
from pathlib import Path
import numpy as np
from scipy.stats import multivariate_normal
from gaussian_vb_port import priors, update, log_emission, infer

P = Path(__file__).parent
R = P / "pilot"
R.mkdir(exist_ok=True)
raw = np.load(
    os.path.join(os.environ.get("HCP_ICA_ROOT", "/configure/HCP_ICA_ROOT"), "X_ICA50_zscore.npy"),
    mmap_mode="r",
).reshape(4012, 1200, 50)
idx = np.arange(100) * 4
x = np.asarray(raw[idx], dtype="float64").reshape(-1, 50)
y = np.asarray(raw[np.sort(np.r_[idx + 2, idx + 3])], dtype="float64").reshape(-1, 50)
K = 12
D = 50
L = 1200
starts = np.arange(0, len(x), L)
p = priors(x, K)
seed_file = R / "seeds.json"
if os.environ.get("SEEDS"):
    seeds = [int(s) for s in os.environ["SEEDS"].split(",")]
elif seed_file.exists():
    seeds = json.loads(seed_file.read_text())
else:
    seeds = np.random.default_rng().choice(2**32, size=6, replace=False).tolist()
seed_file.write_text(json.dumps(seeds))
for method in os.environ.get("METHODS", "em,vb").split(","):
    for seed in seeds:
        dest = R / f"{method}_{seed}.npz"
        if dest.exists() and dest.stat().st_size > 0:
            continue
        rng = np.random.default_rng(seed)
        t0 = time.time()
        means = np.array([x[i * L + 100 : i * L + 200].mean(0) for i in rng.integers(0, 100, K)])
        cov = np.repeat((np.cov(x, rowvar=False) + 0.05 * np.eye(D))[None], K, axis=0)
        A = np.full((K, K), 0.1 / (K - 1))
        np.fill_diagonal(A, 0.9)
        pi = np.ones(K) / K
        m = dict(
            mean=means,
            mean_cov=np.zeros_like(cov),
            nu=np.full(K, D + 10.0),
            rate=cov * (D + 10.0),
            A=A,
            pi=pi,
        )
        hist = []
        prev = None
        for it in range(500):
            if method == "em":
                loge = np.column_stack(
                    [multivariate_normal.logpdf(x, mean=means[k], cov=cov[k]) for k in range(K)]
                )
            else:
                loge = log_emission(x, m)
            g, xi, ll = infer(loge, A, pi, L)
            change = float(np.mean(abs(g - prev))) if prev is not None else None
            hist.append(dict(iteration=it + 1, score=ll / len(x), posterior_change=change))
            if method == "em":
                nk = g.sum(0)
                means = g.T @ x / nk[:, None]
                for k in range(K):
                    c = x - means[k]
                    cov[k] = (c * g[:, k, None]).T @ c / nk[k] + 0.01 * np.eye(D)
                A = xi + 0.001
                A /= A.sum(1)[:, None]
                pi = np.maximum(g[starts].mean(0), 1e-10)
                pi /= pi.sum()
            else:
                m = update(x, g, xi, starts, m, p)
                A = m["A"]
                pi = m["pi"]
            if it % 10 == 0:
                print(
                    method, seed, it + 1, ll / len(x), change, round(time.time() - t0), flush=True
                )
            # Common posterior-change criterion, retain likelihood and mark cap.
            if it >= 49 and change is not None and change < 1e-5:
                break
            prev = g

        def emission(z):
            return (
                np.column_stack(
                    [multivariate_normal.logpdf(z, mean=means[k], cov=cov[k]) for k in range(K)]
                )
                if method == "em"
                else log_emission(z, m)
            )

        gt, _, _ = infer(emission(x), A, pi, L)
        gy, _, _ = infer(emission(y), A, pi, L)
        np.savez_compressed(
            dest,
            train=gt.reshape(100, L, K)[:, 100:1100:10].astype("float32"),
            test=gy.reshape(200, L, K)[:, 100:1100].astype("float32"),
            A=A,
            pi=pi,
            means=means if method == "em" else m["mean"],
        )
        (R / f"{method}_{seed}.json").write_text(
            json.dumps(
                dict(
                    history=hist,
                    seconds=time.time() - t0,
                    converged=bool(change is not None and change < 1e-5),
                    scope="100-subject batch equations pilot; matched segment initialization; NOT PNAS stochastic/init reproduction",
                ),
                indent=2,
            )
        )
        print("SAVED", dest, flush=True)
