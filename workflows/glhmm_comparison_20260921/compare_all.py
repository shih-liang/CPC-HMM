"""Compare archived EM, Variational Bayes and official GLHMM labels after matching.

Reads existing model archives and writes comparison tables; executes on import.
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "2"
from pathlib import Path
import json
import itertools
import csv
import numpy as np
from scipy.optimize import linear_sum_assignment

P = Path(__file__).parent
R = P / "results"
old = P.parent / "hmm_algorithm_validation_20260921/pilot"
models = {}
for root in [old, R]:
    for f in sorted(root.glob("*.npz")):
        if f.with_suffix(".json").exists():
            with np.load(f) as a:
                models[f.stem] = {k: a[k] for k in ["train", "test", "A", "pi", "means"]}
assert len(models) == 18, "Require all18 completed fits"


def mapping(a, b):
    a = a.reshape(-1, 12).astype(float)
    b = b.reshape(-1, 12).astype(float)
    a -= a.mean(0)
    b -= b.mean(0)
    c = a.T @ b / np.maximum(np.sqrt((a * a).sum(0)[:, None] * (b * b).sum(0)), 1e-30)
    rr, cc = linear_sum_assignment(-c)
    return cc[np.argsort(rr)]


def ari(a, b):
    c = np.bincount(a.ravel() * 12 + b.ravel(), minlength=144).reshape(12, 12)

    def choose(v):
        return np.sum(v * (v - 1) / 2)

    n = c.sum()
    s = choose(c)
    r = choose(c.sum(1))
    q = choose(c.sum(0))
    ex = r * q / (n * (n - 1) / 2)
    return float((s - ex) / (0.5 * (r + q) - ex))


rows = []
states = []
for a, b in itertools.combinations(sorted(models), 2):
    ma, mb = models[a], models[b]
    perm = mapping(ma["train"], mb["train"])
    aa = ma["test"].reshape(-1, 12).astype(float)
    bb = mb["test"][..., perm].reshape(-1, 12).astype(float)
    aa -= aa.mean(0)
    bb -= bb.mean(0)
    corr = (aa * bb).sum(0) / np.sqrt((aa * aa).sum(0) * (bb * bb).sum(0))
    la = ma["test"].argmax(2)
    lb = mb["test"][..., perm].argmax(2)
    agree = (la == lb).reshape(100, 2000).mean(1)
    boot = agree[np.random.default_rng(917).integers(0, 100, (2000, 100))].mean(1)
    group = "–".join(sorted([a.split("_")[0], b.split("_")[0]]))
    rows.append(
        dict(
            a=a,
            b=b,
            group=group,
            agreement=float(agree.mean()),
            posterior_correlation=float(corr.mean()),
            ari=ari(la, lb),
            ci_low=float(np.quantile(boot, 0.025)),
            ci_high=float(np.quantile(boot, 0.975)),
        )
    )
    for k, c in enumerate(corr):
        states.append(
            dict(a=a, b=b, state_a=k + 1, state_b=int(perm[k] + 1), posterior_correlation=float(c))
        )
for name, data in [("pairwise.csv", rows), ("matched_states.csv", states)]:
    with (R / name).open("w") as f:
        w = csv.DictWriter(f, fieldnames=data[0])
        w.writeheader()
        w.writerows(data)
summary = {}
for group in sorted({r["group"] for r in rows}):
    rr = [r for r in rows if r["group"] == group]
    d = {"pairs": len(rr)}
    for key in ["agreement", "posterior_correlation", "ari"]:
        v = np.array([r[key] for r in rr])
        d[key] = {
            "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)),
            "min": float(v.min()),
            "max": float(v.max()),
        }
    summary[group] = d
unanimity = {}
for method in ["em", "vb", "glhmm"]:
    names = sorted(n for n in models if n.startswith(method + "_"))
    assert len(names) == 6
    ref = models[names[0]]
    labels = [
        models[n]["test"][..., mapping(ref["train"], models[n]["train"])].argmax(2) for n in names
    ]
    mask = np.all(np.array(labels) == labels[0], axis=0)
    count = int(mask.sum())
    unanimity[method] = dict(
        frames=count, total_frames=int(mask.size), fraction=count / mask.size, reference=names[0]
    )
result = dict(groups=summary, six_fit_unanimity=unanimity, models=list(models))
(R / "comparison.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
