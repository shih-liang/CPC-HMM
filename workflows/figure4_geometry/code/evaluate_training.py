"""Read-only replay of selected decoders to diagnose generalization differences."""

from pathlib import Path
import argparse
import gc
import importlib.util
import json
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--implementation", type=Path, default=Path(__file__).with_name("direct_decoding.py")
)
parser.add_argument("--device", default="cuda:1")
args = parser.parse_args()
spec = importlib.util.spec_from_file_location("decoding", args.implementation)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
torch.set_num_threads(4)
torch.set_float32_matmul_precision("high")
alpha = np.load(base.SOURCE / "data/hmm_alpha_robust_seed2.npy", mmap_mode="r")
Y = torch.from_numpy(np.array(alpha).reshape(-1, base.STATES)).to(args.device)
lags = torch.arange(-49, 1, device=args.device)
rows = []
for name in base.CONDITIONS:
    complete = json.loads((args.output / name / "complete.json").read_text())
    assert complete["status"] == "PASS"
    ck = torch.load(args.output / name / "selected.pt", weights_only=True, map_location=args.device)
    X, width, mean, sd = base.load_features(name, args.output, args.device)
    mean_error = float(abs(mean - ck["feature_mean"].cpu().numpy()).max())
    std_error = float(abs(sd - ck["feature_std"].cpu().numpy()).max())
    assert max(mean_error, std_error) < 1e-7
    model = base.Decoder().to(args.device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    result = dict(
        condition=name,
        normalization_mean_max_error=mean_error,
        normalization_std_max_error=std_error,
    )
    with torch.no_grad():
        for split, scans, stride in [("training", base.TRAIN, 4), ("validation", base.VALID, 10)]:
            indices = (scans[:, None] * base.T + np.arange(100, 1100, stride)).ravel()
            correct = 0
            mse = 0.0
            for start in range(0, len(indices), 1024):
                ix = torch.tensor(indices[start : start + 1024], device=args.device)
                v = X[ix[:, None] + lags]
                if width == 400:
                    x = v.reshape(len(ix), -1)
                else:
                    z = torch.zeros((len(ix), base.W, base.F), device=args.device)
                    z[..., :width] = v[..., :width]
                    z[..., 400 : 400 + width] = v[..., width:]
                    x = z.reshape(len(ix), -1)
                p = torch.softmax(model(x), 1)
                y = Y[ix]
                correct += int((p.argmax(1) == y.argmax(1)).sum().item())
                mse += float(torch.square(p - y).sum().item())
            result[split + "_accuracy"] = correct / len(indices)
            result[split + "_mse"] = mse / (len(indices) * base.STATES)
        result["validation_mse_replay_error"] = abs(result["validation_mse"] - ck["validation_mse"])
        assert result["validation_mse_replay_error"] < 5e-6
    rows.append(result)
    print(json.dumps(result), flush=True)
    del X, model, ck, x, v
    gc.collect()
    torch.cuda.empty_cache()
base.write_csv(args.output / "export" / "training_validation_replay.csv", rows)
