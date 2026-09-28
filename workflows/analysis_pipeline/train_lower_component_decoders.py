"""Fit CPC-count decoders with 50-frame windows and two checkpoint selectors.

Requires the archived training protocol JSON. Writes checkpoints and predictions.
This historical training program executes on import.
"""

import os  # Public release: configurable data roots.
from pathlib import Path
import argparse
import time
import json
import gc
import numpy as np
import pandas as pd
import torch
from torch import nn

parser = argparse.ArgumentParser()
parser.add_argument("--counts", required=True)
parser.add_argument("--device", required=True)
args = parser.parse_args()
P = Path(
    os.path.join(
        os.environ.get("HCP_DERIVATIVES", "/configure/HCP_DERIVATIVES"),
        "wave_rsn_revision_20260906",
    )
)
R = P / "results/lower_components_agreement_20260909"
D = P / "data/lower_components_agreement_20260909"
D.mkdir(parents=True, exist_ok=True)
R.mkdir(parents=True, exist_ok=True)
protocol_path = R / "protocol.json"
if not protocol_path.exists():
    protocol_path.write_bytes((Path(__file__).with_name("rank_protocol.json")).read_bytes())
protocol = json.loads(protocol_path.read_text())
counts = [int(x) for x in args.counts.split(",")]
assert set(counts) <= set(protocol["counts"])
dev = args.device
seed = protocol["seed"]
torch.set_num_threads(4)
torch.set_float32_matmul_precision("high")
began = time.time()
S = 4012
L = 1200
F = 400
K = 12
W = 50
batch = 1024
tag = protocol["target"]
train = np.arange(1003) * 4
val = train + 1
test = np.sort(np.r_[train + 2, train + 3])
ti = (train[:, None] * L + np.arange(100, 1100, 4)).ravel()
vi = (val[:, None] * L + np.arange(100, 1100, 10)).ravel()
ei = (test[:, None] * L + np.arange(100, 1100)).ravel()
lags = torch.arange(-49, 1, device=dev)
alpha = np.load(P / f"data/hmm_alpha_{tag}.npy", mmap_mode="r")
other = np.load(P / "data/hmm_alpha_robust.npy", mmap_mode="r")
from scipy.optimize import linear_sum_assignment

alignment_path = P / "results/hmm_initialization_stability.json"
if alignment_path.exists():
    perm = np.array(json.loads(alignment_path.read_text())["seed2_column_order_to_seed1"])
else:
    first = np.asarray(other[train, 100:1100:10]).reshape(-1, 12)
    second = np.asarray(alpha[train, 100:1100:10]).reshape(-1, 12)
    corr = np.corrcoef(first.T, second.T)[:12, 12:]
    rows, columns = linear_sum_assignment(-corr)
    perm = columns[np.argsort(rows)]
inv = np.argsort(perm)
vagree = other[val, 100:1100:10].argmax(-1) == inv[alpha[val, 100:1100:10].argmax(-1)]
VM = torch.tensor(vagree.ravel(), device=dev)
vn_agree = int(vagree.sum())
assert vn_agree > 0
# Use the training-derived label mapping; no archived test-mask file is required.
agree = other[test, 100:1100].argmax(-1) == inv[alpha[test, 100:1100].argmax(-1)]
Y = torch.from_numpy(np.array(alpha).reshape(-1, K)).to(dev)
raw = np.load(P / "data/cpca_scores_200.npy", mmap_mode="r")


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(F * W, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, K),
        )

    def forward(self, x):
        return self.net(x)


selectors = ["all_mse", "agreement_accuracy"]
for count in counts:
    name = f"complex_{count}"
    started = time.time()
    z = np.zeros((S, L, F), dtype="float32")
    for ids in np.array_split(np.arange(S), 101):
        c = raw[ids, :, :count]
        z[ids, :, :count] = c.real
        z[ids, :, 200 : 200 + count] = c.imag
    t = np.asarray(z[train, 51:1100:5], dtype="float64")
    mean = t.mean((0, 1))
    sd = t.std((0, 1))
    sd[sd < 1e-6] = 1
    del t
    z -= mean.astype("float32")
    z /= sd.astype("float32")
    X = torch.from_numpy(z.reshape(-1, F)).to(dev)
    del z
    best = {"all_mse": float("inf"), "agreement_accuracy": (float("inf"), float("inf"))}
    history = []
    for lr, wd in protocol["configuration_grid"]:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = Decoder().to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        local_mse = float("inf")
        local_acc = -float("inf")
        bad = 0

        def forward(indices):
            ix = torch.as_tensor(indices, device=dev, dtype=torch.long)
            return model(X[(ix[:, None] + lags).reshape(-1)].reshape(len(ix), -1)), Y[ix]  # noqa: F821 - Bound before calls; deleted only after the training/evaluation loop.

        for ep in range(protocol["maximum_epochs_per_configuration"]):
            model.train()
            order = rng.permutation(ti)
            for start in range(0, len(order), batch):
                logits, target = forward(order[start : start + batch])
                loss = -(target * torch.log_softmax(logits, 1)).sum(1).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            model.eval()
            vsum = 0.0
            ag_sum = 0.0
            ag_correct = 0
            all_correct = 0
            with torch.no_grad():
                for start in range(0, len(vi), batch):
                    logits, target = forward(vi[start : start + batch])
                    pred = torch.softmax(logits, 1)
                    err = torch.square(pred - target).sum(1)
                    correct = pred.argmax(1) == target.argmax(1)
                    keep = VM[start : start + batch]
                    vsum += float(err.sum().item())
                    ag_sum += float(err[keep].sum().item())
                    ag_correct += int(correct[keep].sum().item())
                    all_correct += int(correct.sum().item())
            vmse = vsum / (len(vi) * K)
            agmse = ag_sum / (vn_agree * K)
            agacc = ag_correct / vn_agree
            record = {
                "component_count": count,
                "learning_rate": lr,
                "weight_decay": wd,
                "epoch": ep + 1,
                "validation_mse": vmse,
                "validation_agreement_mse": agmse,
                "validation_agreement_accuracy": agacc,
                "validation_all_accuracy": all_correct / len(vi),
            }
            history.append(record)
            metrics = {"all_mse": vmse, "agreement_accuracy": (-agacc, agmse)}
            for selector in selectors:
                if metrics[selector] < best[selector]:
                    best[selector] = metrics[selector]
                    torch.save(
                        {
                            "state_dict": {
                                key: value.detach().cpu()
                                for key, value in model.state_dict().items()
                            },
                            "feature_mean": torch.tensor(mean),
                            "feature_std": torch.tensor(sd),
                            "target": tag,
                            "seed": seed,
                            "input_slots": F,
                            "selection_rule": selector,
                            **record,
                        },
                        D / f"{name}_{selector}.pt",
                    )
            im = vmse < local_mse - 1e-5
            ia = agacc > local_acc + 1e-4
            if im:
                local_mse = vmse
            if ia:
                local_acc = agacc
            bad = 0 if im or ia else bad + 1
            if bad >= 4:
                break
        print(
            "CONFIG",
            count,
            lr,
            wd,
            "epochs",
            ep + 1,
            "best_validation_mse",
            local_mse,
            "best_agreement_accuracy",
            local_acc,
            "seconds",
            round(time.time() - started),
            flush=True,
        )
        del model, opt
        gc.collect()
        torch.cuda.empty_cache()
    pd.DataFrame(history).to_csv(R / f"{name}_validation_history.csv", index=False)
    selections = {}
    for selector in selectors:
        ck = torch.load(D / f"{name}_{selector}.pt", weights_only=True, map_location=dev)
        model = Decoder().to(dev)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        predfile = D / f"{name}_{selector}_test_predictions.npy"
        prediction = np.lib.format.open_memmap(
            predfile, mode="w+", dtype="float32", shape=(len(test), 1000, K)
        )
        with torch.no_grad():
            flat = prediction.reshape(-1, K)
            for start in range(0, len(ei), batch):
                ix = torch.tensor(ei[start : start + batch], device=dev)
                logits = model(X[(ix[:, None] + lags).reshape(-1)].reshape(len(ix), -1))
                flat[start : start + batch] = torch.softmax(logits, 1).cpu().numpy()
        prediction.flush()
        rows = []
        confusion = {s: np.zeros((K, K), dtype=np.int64) for s in ["all", "agree", "disagree"]}
        for j, scan in enumerate(test):
            a = np.asarray(alpha[scan, 100:1100], dtype="float64")
            pp = np.asarray(prediction[j], dtype="float64")
            pp = np.maximum(pp, 1e-8)
            pp /= pp.sum(1, keepdims=True)
            target = a.argmax(1)
            guess = pp.argmax(1)
            correct = target == guess
            mse = np.square(pp - a).mean(1)
            row = {
                "subject_index": int(scan // 4),
                "scan": int(scan),
                "component_count": count,
                "selector": selector,
            }
            for subset, mask in [
                ("all", np.ones(1000, dtype=bool)),
                ("agree", agree[j]),
                ("disagree", ~agree[j]),
            ]:
                row[subset + "_n"] = int(mask.sum())
                row[subset + "_correct"] = int(correct[mask].sum())
                row[subset + "_mse_sum"] = float(mse[mask].sum())
                np.add.at(confusion[subset], (target[mask], guess[mask]), 1)
            assert row["all_n"] == row["agree_n"] + row["disagree_n"]
            assert row["all_correct"] == row["agree_correct"] + row["disagree_correct"]
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(R / f"{name}_{selector}_scan_metrics.csv", index=False)
        np.savez(R / f"{name}_{selector}_confusion.npz", **confusion)
        selection = {
            key: ck[key]
            for key in [
                "component_count",
                "target",
                "seed",
                "input_slots",
                "selection_rule",
                "learning_rate",
                "weight_decay",
                "epoch",
                "validation_mse",
                "validation_agreement_mse",
                "validation_agreement_accuracy",
                "validation_all_accuracy",
            ]
        }
        selection["test_predictions"] = str(predfile)
        (R / f"{name}_{selector}_selection.json").write_text(json.dumps(selection, indent=2))
        selections[selector] = selection
        print(
            "SCORED",
            count,
            selector,
            {
                s: round(100 * df[s + "_correct"].sum() / df[s + "_n"].sum(), 4)
                for s in ["all", "agree", "disagree"]
            },
            flush=True,
        )
        del model, prediction
        gc.collect()
        torch.cuda.empty_cache()
    (R / f"complex_{count}_complete.json").write_text(
        json.dumps(
            {
                "complete": True,
                "component_count": count,
                "device": dev,
                "seconds": time.time() - started,
                "selections": selections,
            },
            indent=2,
        )
    )
    del X
    gc.collect()
    torch.cuda.empty_cache()
(R / ("job_" + dev.replace(":", "") + "_complete.json")).write_text(
    json.dumps(
        {
            "complete": True,
            "counts": counts,
            "device": dev,
            "seconds": time.time() - began,
            "validation_agreement_frames": vn_agree,
            "validation_frames": len(vi),
        },
        indent=2,
    )
)
print("JOB COMPLETE", dev, "seconds", round(time.time() - began), flush=True)
