"""Match HMM fits and summarize agreement and decoding on common frames.

Uses existing fit outputs; executes on import.
"""

import os  # Public release: configurable data roots.
from pathlib import Path
import numpy as np
import pandas as pd
import json
import itertools
import time
from scipy.optimize import linear_sum_assignment

P = Path(
    os.path.join(
        os.environ.get("HCP_DERIVATIVES", "/configure/HCP_DERIVATIVES"),
        "wave_rsn_revision_20260906",
    )
)
Q = P / "revision_20260910"
R = Q / "results"
tags = [
    "robust_seed2",
    "robust",
    "consensus_seed3",
    "consensus_seed4",
    "consensus_seed5",
    "consensus_seed6",
]
include_acq = (R / "hmm_acquisition_rest1rl_metadata.json").exists()
new = ["robust", "consensus_seed3", "consensus_seed4", "consensus_seed6"] + (
    ["acquisition_rest1rl"] if include_acq else []
)
alltags = tags + (["acquisition_rest1rl"] if include_acq else [])
for t in new:
    if not (R / f"hmm_{t}_metadata.json").exists():
        raise SystemExit("NOT READY " + t)
train = np.arange(1003) * 4
test = np.sort(np.r_[train + 2, train + 3])
K = 12
target = np.load(P / "data/hmm_alpha_robust_seed2.npy", mmap_mode="r")
tr = np.asarray(target[train, 100:1100:10], dtype="float64").reshape(-1, K)


def correlation(a, b):
    a = a - a.mean(0)
    b = b - b.mean(0)
    return (a.T @ b) / np.sqrt((a * a).sum(0)[:, None] * (b * b).sum(0)[None, :])


arrays = []
labels = []
mappings = {}
alignment = []
quality = []
for tag in alltags:
    root = P if tag in ["robust_seed2", "consensus_seed5"] else Q
    a = np.load(root / f"data/hmm_alpha_{tag}.npy", mmap_mode="r")
    other = np.asarray(a[train, 100:1100:10], dtype="float64").reshape(-1, K)
    co = correlation(tr, other)
    rr, cc = linear_sum_assignment(-co)
    perm = cc[np.argsort(rr)]
    mappings[tag] = perm.tolist()
    for k in range(K):
        alignment.append(
            dict(
                fit=tag,
                target_state=k + 1,
                matched_native_state=int(perm[k]) + 1,
                training_posterior_correlation=float(co[k, perm[k]]),
            )
        )
    ev = np.asarray(a[test, 100:1100], dtype="float32")[..., perm]
    assert np.all(np.isfinite(ev)) and np.max(abs(ev.sum(2) - 1)) < 2e-5
    arrays.append(ev)
    labels.append(ev.argmax(2))
    hist = json.loads((root / f"results/hmm_{tag}_training_history.json").read_text())
    meta = json.loads((root / f"results/hmm_{tag}_metadata.json").read_text())
    delta = hist[-1]["mean_loglik"] - hist[-2]["mean_loglik"]
    quality.append(
        dict(
            fit=tag,
            seed=meta["seed"],
            iterations=len(hist),
            final_delta=delta,
            tolerance_met=abs(delta) < 1e-5,
            final_training_iteration_loglik=hist[-1]["mean_loglik"],
        )
    )
pd.DataFrame(quality).to_csv(R / "updated_hmm_fit_quality.csv", index=False)
pd.DataFrame(alignment).to_csv(R / "updated_hmm_alignment.csv", index=False)
(R / "updated_hmm_mappings.json").write_text(json.dumps(mappings, indent=2))
pairrows = []
partrows = []
statepairs = []
pairs = list(itertools.combinations(range(6), 2)) + ([(0, 6)] if include_acq else [])
for i, j in pairs:
    a = arrays[i].reshape(1003, 2000, K).astype("float64")
    b = arrays[j].reshape(1003, 2000, K).astype("float64")
    ac = a - a.mean(1, keepdims=True)
    bc = b - b.mean(1, keepdims=True)
    r = (ac * bc).sum(1) / np.sqrt((ac * ac).sum(1) * (bc * bc).sum(1))
    r = np.nan_to_num(r)
    agree = (labels[i] == labels[j]).reshape(1003, 2000).mean(1)
    pc = correlation(a.reshape(-1, K), b.reshape(-1, K)).diagonal()
    pairrows.append(
        dict(
            fit_a=(tags + ["acquisition_rest1rl"])[i],
            fit_b=(tags + ["acquisition_rest1rl"])[j],
            comparison="acquisition" if j == 6 else "initialization",
            argmax_agreement=float(agree.mean()),
            mean_pooled_posterior_correlation=float(pc.mean()),
            minimum_pooled_posterior_correlation=float(pc.min()),
        )
    )
    for k in range(K):
        statepairs.append(
            dict(
                fit_a=(tags + ["acquisition_rest1rl"])[i],
                fit_b=(tags + ["acquisition_rest1rl"])[j],
                target_state=k + 1,
                pooled_posterior_correlation=float(pc[k]),
            )
        )
    for s in range(1003):
        partrows.append(
            dict(
                fit_a=(tags + ["acquisition_rest1rl"])[i],
                fit_b=(tags + ["acquisition_rest1rl"])[j],
                comparison="acquisition" if j == 6 else "initialization",
                participant_index=s,
                argmax_agreement=float(agree[s]),
                mean_posterior_correlation=float(r[s].mean()),
            )
        )
pd.DataFrame(pairrows).to_csv(R / "hmm_reliability_pairs.csv", index=False)
pd.DataFrame(partrows).to_csv(R / "hmm_reliability_participants.csv", index=False)
pd.DataFrame(statepairs).to_csv(R / "hmm_reliability_states.csv", index=False)
truth = labels[0]
pred = np.load(
    P
    / "data/lower_components_agreement_20260909/complex_30_agreement_accuracy_test_predictions.npy",
    mmap_mode="r",
)
guess = pred.argmax(2)
correct = guess == truth
mse = ((pred - arrays[0]) ** 2).mean(2)
majority = int(tr.mean(0).argmax())
rng = np.random.default_rng(20260910)
boot = rng.integers(0, 1003, (2000, 1003))
summary = []
state = []
participants = []
masks = {"all": np.ones_like(truth, dtype=bool)}
for n in range(2, 7):
    masks[f"fits{n}"] = np.logical_and.reduce([label == truth for label in labels[1:n]])
for name, mask in masks.items():
    num = mask.reshape(1003, 2000).sum(1)
    hit = (correct & mask).reshape(1003, 2000).sum(1)
    bs = hit[boot].sum(1) / num[boot].sum(1)
    mn = (mse * mask).reshape(1003, 2000).sum(1)
    row = dict(
        subset=name,
        hmm_fits=0 if name == "all" else int(name[4:]),
        n_frames=int(num.sum()),
        coverage=float(num.sum() / truth.size),
        accuracy=float(hit.sum() / num.sum()),
        ci_low=float(np.quantile(bs, 0.025)),
        ci_high=float(np.quantile(bs, 0.975)),
        mse=float(mn.sum() / num.sum()),
        training_majority_accuracy=float(((truth == majority) & mask).sum() / num.sum()),
    )
    recalls = []
    for k in range(K):
        support = int((truth == k).sum())
        selected = int(((truth == k) & mask).sum())
        hits = int(((truth == k) & mask & correct).sum())
        recalls.append(hits / selected if selected else np.nan)
        state.append(
            dict(
                subset=name,
                target_state=k + 1,
                all_support=support,
                n_frames=selected,
                n_correct=hits,
                retention=selected / support,
                recall=hits / selected if selected else np.nan,
            )
        )
    row["present_states"] = int(np.sum(np.isfinite(recalls)))
    row["balanced_accuracy_present_states"] = float(np.nanmean(recalls))
    summary.append(row)
    for s in range(1003):
        participants.append(
            dict(
                subset=name,
                participant_index=s,
                n_frames=int(num[s]),
                n_correct=int(hit[s]),
                mse_sum=float(mn[s]),
            )
        )
pd.DataFrame(summary).to_csv(R / "CPCA30_updated_HMM_fits_accuracy.csv", index=False)
pd.DataFrame(state).to_csv(R / "updated_consensus_state_metrics.csv", index=False)
pd.DataFrame(participants).to_csv(R / "updated_consensus_participant_metrics.csv", index=False)
np.savez_compressed(Q / "data/updated_consensus_masks.npz", test_scan_indices=test, **masks)
df = pd.DataFrame(state)
base = df[df.subset == "fits2"].sort_values("target_state")
last = df[df.subset == "fits6"].sort_values("target_state")
valid = last.n_frames.to_numpy() > 0
n2 = base.n_frames.to_numpy()
r2 = base.recall.to_numpy()
n6 = last.n_frames.to_numpy()
r6 = last.recall.to_numpy()
a2 = float((n2 * r2).sum() / n2.sum())
a2common = float((n2[valid] * r2[valid]).sum() / n2[valid].sum())
std = float((n2[valid] * r6[valid]).sum() / n2[valid].sum())
a6 = float((n6[valid] * r6[valid]).sum() / n6[valid].sum())
dec = dict(
    base_accuracy=a2,
    base_common_state_accuracy=a2common,
    state_standardized_final_accuracy=std,
    final_accuracy=a6,
    unsupported_state_term=a2common - a2,
    within_state_recall_term=std - a2common,
    state_prevalence_term=a6 - std,
    states_with_support=int(valid.sum()),
)
assert (
    abs(
        sum(
            dec[k]
            for k in ["unsupported_state_term", "within_state_recall_term", "state_prevalence_term"]
        )
        - (a6 - a2)
    )
    < 1e-10
)
(R / "updated_consensus_decomposition.json").write_text(json.dumps(dec, indent=2))
pr = pd.DataFrame(partrows)
statistics = []
for metric in ["argmax_agreement", "mean_posterior_correlation"]:
    for kind in ["initialization", "acquisition"] if include_acq else ["initialization"]:
        vals = pr[pr.comparison == kind].groupby("participant_index")[metric].mean().to_numpy()
        bs = vals[boot].mean(1)
        statistics.append(
            dict(
                method="HMM",
                comparison=kind,
                metric=metric,
                value=float(vals.mean()),
                ci_low=float(np.quantile(bs, 0.025)),
                ci_high=float(np.quantile(bs, 0.975)),
                n_participants=1003,
                n_fit_pairs=15 if kind == "initialization" else 1,
            )
        )
cp = pd.read_csv(R / "cpca_test_score_reproducibility.csv")
for n in [3, 10, 30, 50]:
    vals = (
        cp[cp.component <= n]
        .groupby("participant_index")
        .phase_aligned_score_correlation.mean()
        .to_numpy()
    )
    bs = vals[boot].mean(1)
    statistics.append(
        dict(
            method="CPCA",
            comparison="acquisition",
            metric=f"mean_score_correlation_top{n}",
            value=float(vals.mean()),
            ci_low=float(np.quantile(bs, 0.025)),
            ci_high=float(np.quantile(bs, 0.975)),
            n_participants=1003,
            n_fit_pairs=1,
        )
    )
pd.DataFrame(statistics).to_csv(R / "reliability_bootstrap_summary.csv", index=False)
(R / "reliability_evaluation_complete.json").write_text(
    json.dumps(
        {
            "complete": include_acq,
            "consensus_complete": True,
            "acquisition_evaluated": include_acq,
            "target": "robust_seed2",
            "all_frame_accuracy": float(correct.mean()),
            "updated_consensus": summary[-1],
            "decomposition": dec,
            "note": "Acquisition comparison uses same seed, two distinct REST1 training acquisitions, common REST2 observations; alignment uses only REST1_LR.",
        },
        indent=2,
    )
)
print(json.dumps(summary, indent=2))
print(json.dumps(statistics, indent=2))
