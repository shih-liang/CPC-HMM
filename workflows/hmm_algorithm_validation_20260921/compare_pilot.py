"""Match archived EM/VB states on training data and compare held-out labels."""

from pathlib import Path
import json
import itertools
import csv
import numpy as np
from scipy.optimize import linear_sum_assignment

P = Path(__file__).parent / "pilot"
files = sorted(
    f for f in P.glob("*.npz") if f.stat().st_size > 0 and f.with_suffix(".json").exists()
)
models = {f.stem: np.load(f) for f in files}


def mapping(a, b):
    a = a.reshape(-1, 12).astype(float)
    b = b.reshape(-1, 12).astype(float)
    a -= a.mean(0)
    b -= b.mean(0)
    c = a.T @ b / np.maximum(np.sqrt((a * a).sum(0)[:, None] * (b * b).sum(0)), 1e-30)
    rr, cc = linear_sum_assignment(-c)
    return cc[np.argsort(rr)]


def ari(a, b):
    c = np.bincount((a.ravel() * 12 + b.ravel()), minlength=144).reshape(12, 12)

    def comb(z):
        return np.sum(z * (z - 1) / 2)

    n = c.sum()
    nij = comb(c)
    r = comb(c.sum(1))
    q = comb(c.sum(0))
    expected = r * q / (n * (n - 1) / 2)
    return float((nij - expected) / (0.5 * (r + q) - expected))


rows = []
state_rows = []
for a, b in itertools.combinations(models, 2):
    ma, mb = models[a], models[b]
    perm = mapping(ma["train"], mb["train"])
    aa = ma["test"].reshape(-1, 12).astype(float)
    bb = mb["test"][..., perm].reshape(-1, 12).astype(float)
    aa -= aa.mean(0)
    bb -= bb.mean(0)
    corr = (aa * bb).sum(0) / np.sqrt((aa * aa).sum(0) * (bb * bb).sum(0))
    for k, v in enumerate(corr):
        state_rows.append(
            dict(a=a, b=b, state_a=k + 1, state_b=int(perm[k] + 1), posterior_correlation=float(v))
        )
    agreement = (ma["test"].argmax(2) == mb["test"][..., perm].argmax(2)).reshape(100, -1).mean(1)
    rng = np.random.default_rng(917)
    boot = agreement[rng.integers(0, 100, (2000, 100))].mean(1)
    rows.append(
        dict(
            a=a,
            b=b,
            group=a[:2] if a[:2] == b[:2] else "cross",
            agreement=float(agreement.mean()),
            mean_posterior_correlation=float(corr.mean()),
            fraction_states_correlation_gt_075=float(np.mean(corr > 0.75)),
            ari=ari(ma["test"].argmax(2), mb["test"].argmax(2)),
            transition_rmse=float(np.sqrt(np.mean((ma["A"] - mb["A"][np.ix_(perm, perm)]) ** 2))),
            subject_sd=float(agreement.std(ddof=1)),
            ci_low=float(np.quantile(boot, 0.025)),
            ci_high=float(np.quantile(boot, 0.975)),
        )
    )
if rows:
    with (P / "pairwise.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)
if state_rows:
    with (P / "matched_states.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=state_rows[0])
        w.writeheader()
        w.writerows(state_rows)
s = {}
for group in ["em", "vb", "cross"]:
    vals = [r["agreement"] for r in rows if r["group"] == group]
    if vals:
        s[group] = dict(
            n_pairs=len(vals),
            mean=float(np.mean(vals)),
            sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
            min=min(vals),
            max=max(vals),
        )
for method in ["em", "vb"]:
    names = [k for k in models if k.startswith(method)]
    if len(names) == 6:
        ref = models[names[0]]
        labels = [
            models[n]["test"][..., mapping(ref["train"], models[n]["train"])].argmax(2)
            for n in names
        ]
        s[method]["six_fit_unanimity"] = float(np.all(np.array(labels) == labels[0], axis=0).mean())
s["convergence"] = {k: json.loads((P / (k + ".json")).read_text())["converged"] for k in models}
s["completed_models"] = list(models)
s["complete"] = len(models) == 12
(P / "summary.json").write_text(json.dumps(s, indent=2))
print(json.dumps(s, indent=2))
