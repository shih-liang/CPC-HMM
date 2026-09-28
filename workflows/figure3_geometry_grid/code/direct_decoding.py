"""Matched CPC/geometric-coordinate decoding; all individual data stay on server."""

from pathlib import Path
import argparse
import csv
import gc
import hashlib
import json
import os
import time
import numpy as np
import torch
from torch import nn

SOURCE = Path(
    os.path.join(
        os.environ.get("HCP_DERIVATIVES", "/configure/HCP_DERIVATIVES"),
        "wave_rsn_revision_20260906",
    )
)
RAW = Path(
    os.path.join(
        os.environ.get("HCP_CORTICAL_ROOT", "/configure/HCP_CORTICAL_ROOT"),
        "input_hmm_order/group_fs4_concat_z_hmm_order.npy",
    )
)
S, T, N, W, F, STATES = 4012, 1200, 1003, 50, 800, 12
CONDITIONS = {
    "cpc30": ("cpc", 30),
    "geo15": ("geometry", 15),
    "geo50": ("geometry", 50),
    "geo57": ("geometry", 57),
    "geo200": ("geometry", 200),
}
TRAIN = np.arange(N) * 4
VALID = TRAIN + 1
TEST = np.sort(np.r_[TRAIN + 2, TRAIN + 3])


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def fingerprint(path):
    size = path.stat().st_size
    h = hashlib.sha256()
    with path.open("rb") as f:
        offsets = (
            [0]
            if size <= 32 * 2**20
            else sorted(set([0, size // 4, size // 2, 3 * size // 4, max(0, size - 2**20)]))
        )
        for offset in offsets:
            f.seek(offset)
            h.update(f.read() if len(offsets) == 1 else f.read(2**20))
    return dict(
        path=str(path),
        bytes=size,
        mtime_ns=path.stat().st_mtime_ns,
        sha256=h.hexdigest(),
        method="full" if len(offsets) == 1 else "five 1-MiB samples",
    )


def analytic(x):
    """Form the analytic field along the time axis of an individual run."""
    h = torch.zeros(T, device=x.device)
    h[0] = h[T // 2] = 1
    h[1 : T // 2] = 2
    return torch.fft.ifft(torch.fft.fft(x, dim=-2) * h[:, None], dim=-2)


def project(out, device):
    """Project analytic cortical runs onto separately orthogonalized hemisphere bases.

    Writes coordinate arrays and validation metadata in the requested output tree.
    """
    torch.set_float32_matmul_precision("highest")
    score_path = out / "native_geometry_scores.npy"
    assert not score_path.exists(), "Do not overwrite an existing projection."
    paths = [
        RAW,
        SOURCE / "data/surfaces_and_eigenmodes.npz",
        SOURCE / "data/training_bases.npz",
        SOURCE / "data/cpca_scores_200.npy",
        SOURCE / "data/hmm_alpha_robust_seed2.npy",
    ]
    before = [fingerprint(p) for p in paths]
    surface = np.load(paths[1])
    basis = np.load(paths[2])
    indices = np.r_[surface["L_indices"], len(surface["L_pial_vertices"]) + surface["R_indices"]]
    assert np.array_equal(indices, basis["vertex_indices"])
    assert np.array_equal(basis["training_scans"], TRAIN)
    Q = np.zeros((4801, 400), dtype=np.float64)
    offset = 0
    qr_error = {}
    for j, hemi in enumerate(["L", "R"]):
        E = surface[hemi + "_eigenmodes"]
        q, _ = np.linalg.qr(E, mode="reduced")
        Q[offset : offset + len(E), j * 200 : (j + 1) * 200] = q
        qr_error[hemi] = float(abs(q.T @ q - np.eye(200)).max())
        offset += len(E)
    assert max(qr_error.values()) < 1e-10 and offset == 4801
    q = torch.tensor(Q, dtype=torch.float32, device=device)
    raw = np.load(RAW, mmap_mode="r").reshape(S, T, 5124)
    ix = torch.tensor(indices, dtype=torch.long, device=device)
    a = torch.from_numpy(np.array(raw[0:1])).to(device).index_select(-1, ix)
    a -= a.mean(-2, keepdim=True)
    direct = analytic(a) @ q.to(torch.complex64)
    projected = analytic(a @ q)
    commutation = float(
        (torch.linalg.vector_norm(direct - projected) / torch.linalg.vector_norm(direct)).item()
    )
    assert commutation < 1e-5, commutation
    U = basis["complex_vectors"][:, :30]
    predicted = projected.cpu().numpy()[0] @ (Q.T @ U)
    old = np.load(SOURCE / "data/geometry_scores_200.npy", mmap_mode="r")[0, :, :30]
    reference_error = float(np.linalg.norm(predicted - old) / np.linalg.norm(old))
    assert reference_error < 1e-4, reference_error
    scores = np.lib.format.open_memmap(score_path, mode="w+", dtype="complex64", shape=(S, T, 400))
    start = time.time()
    for first in range(0, S, 8):
        stop = min(first + 8, S)
        a = torch.from_numpy(np.array(raw[first:stop])).to(device).index_select(-1, ix)
        a -= a.mean(-2, keepdim=True)
        scores[first:stop] = analytic(a @ q).cpu().numpy()
        if first % 160 == 0:
            print(
                json.dumps(
                    dict(
                        stage="projection",
                        scans=stop,
                        total=S,
                        seconds=round(time.time() - start, 1),
                    )
                ),
                flush=True,
            )
    scores.flush()
    after = [fingerprint(p) for p in paths]
    assert before == after
    write_json(
        out / "projection_validation.json",
        dict(
            status="PASS",
            sources=before,
            source_fingerprints_unchanged=True,
            qr_orthogonality=qr_error,
            hilbert_projection_relative_error=commutation,
            archived_projected_CPC_relative_error=reference_error,
            coordinate_definition="Z @ blockdiag(Q_L,Q_R); 200 modes per hemisphere",
            shape=list(scores.shape),
            seconds=time.time() - start,
        ),
    )
    print("PROJECTION COMPLETE", flush=True)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(F * W, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, STATES),
        )

    def forward(self, x):
        return self.net(x)


def load_features(name, out, device):
    """Load and training-standardize features for one CPC/geometric condition."""
    family, count = CONDITIONS[name]
    path = (
        SOURCE / "data/cpca_scores_200.npy"
        if family == "cpc"
        else out / "native_geometry_scores.npy"
    )
    raw = np.load(path, mmap_mode="r")
    columns = (
        np.arange(count) if family == "cpc" else np.r_[np.arange(count), 200 + np.arange(count)]
    )
    width = len(columns)
    mean = np.zeros(2 * width)
    second = mean.copy()
    sample_n = 0
    for scan in TRAIN:
        c = np.asarray(raw[scan, 51:1100:5])[:, columns]
        x = np.concatenate([c.real, c.imag], axis=-1).astype(float)
        mean += x.sum(0)
        second += np.square(x).sum(0)
        sample_n += len(x)
    mean /= sample_n
    sd = np.sqrt(np.maximum(second / sample_n - mean**2, 0))
    sd[sd < 1e-6] = 1
    X = torch.empty((S * T, 2 * width), device=device, dtype=torch.float32)
    mean32, sd32 = mean.astype("float32"), sd.astype("float32")
    for first in range(0, S, 8):
        stop = min(first + 8, S)
        c = np.asarray(raw[first:stop])[..., columns]
        x = (np.concatenate([c.real, c.imag], axis=-1) - mean32) / sd32
        assert np.isfinite(x).all()
        X[first * T : stop * T].copy_(torch.from_numpy(x.reshape(-1, 2 * width)))
    return X, width, mean, sd


def fit(names, out, device):
    """Fit 50-frame, 800-slot decoders and select checkpoints by validation posterior MSE.

    Training, validation and test use separate acquisitions of the same people.
    """
    if any(name != "cpc30" for name in names):
        assert json.loads((out / "projection_validation.json").read_text())["status"] == "PASS"
    torch.set_float32_matmul_precision("high")
    alpha = np.load(SOURCE / "data/hmm_alpha_robust_seed2.npy", mmap_mode="r")
    Y = torch.from_numpy(np.array(alpha).reshape(-1, STATES)).to(device)
    tidx = (TRAIN[:, None] * T + np.arange(100, 1100, 4)).ravel()
    vidx = (VALID[:, None] * T + np.arange(100, 1100, 10)).ravel()
    eidx = (TEST[:, None] * T + np.arange(100, 1100)).ravel()
    lags = torch.arange(-49, 1, device=device)
    assert tidx.min() % T == 100 and np.all(tidx % T >= 100)
    for name in names:
        folder = out / name
        folder.mkdir()
        started = time.time()
        X, width, mean, sd = load_features(name, out, device)
        print(
            json.dumps(
                dict(
                    stage="features_loaded",
                    condition=name,
                    complex_coordinates=width,
                    seconds=round(time.time() - started, 1),
                )
            ),
            flush=True,
        )

        def window(indices):
            """Gather within-run 50-frame windows and pad real/imaginary slots.

            X is initialized for this condition and deleted only after all calls finish.
            """
            ix = torch.as_tensor(indices, device=device, dtype=torch.long)
            values = X[ix[:, None] + lags]  # noqa: F821 - Bound before calls; deleted only after the training/evaluation loop.
            if width == 400:
                return values.reshape(len(ix), -1), Y[ix]
            z = torch.zeros((len(ix), W, F), device=device)
            z[..., :width] = values[..., :width]
            z[..., 400 : 400 + width] = values[..., width:]
            return z.reshape(len(ix), -1), Y[ix]

        best = float("inf")
        history = []
        for lr, wd in [(0.001, 0.0001), (0.001, 0.001), (0.0003, 0.0001), (0.0003, 0.001)]:
            torch.manual_seed(20260908)
            rng = np.random.default_rng(20260908)
            model = Decoder().to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            local_best = float("inf")
            bad = 0
            for epoch in range(1, 31):
                model.train()
                order = rng.permutation(tidx)
                for start in range(0, len(order), 1024):
                    x, y = window(order[start : start + 1024])
                    logits = model(x)
                    loss = -(y * torch.log_softmax(logits, 1)).sum(1).mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                model.eval()
                mse_sum = 0.0
                correct = 0
                with torch.no_grad():
                    for start in range(0, len(vidx), 1024):
                        x, y = window(vidx[start : start + 1024])
                        p = torch.softmax(model(x), 1)
                        mse_sum += float(torch.square(p - y).sum().item())
                        correct += int((p.argmax(1) == y.argmax(1)).sum().item())
                mse = mse_sum / (len(vidx) * STATES)
                assert np.isfinite(mse)
                row = dict(
                    condition=name,
                    learning_rate=lr,
                    weight_decay=wd,
                    epoch=epoch,
                    validation_mse=mse,
                    validation_accuracy=correct / len(vidx),
                    seconds=time.time() - started,
                )
                history.append(row)
                if mse < best:
                    best = mse
                    torch.save(
                        dict(
                            state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                            feature_mean=torch.tensor(mean),
                            feature_std=torch.tensor(sd),
                            input_slots=F,
                            complex_coordinates=width,
                            seed=20260908,
                            **row,
                        ),
                        folder / "selected.pt",
                    )
                improved = mse < local_best - 1e-5
                if improved:
                    local_best = mse
                bad = 0 if improved else bad + 1
                if epoch == 1 or epoch % 5 == 0 or bad >= 4:
                    print(json.dumps(row), flush=True)
                if bad >= 4:
                    break
            write_csv(folder / "validation_history.csv", history)
            del model, opt
            gc.collect()
            torch.cuda.empty_cache()
        ck = torch.load(folder / "selected.pt", weights_only=True, map_location=device)
        model = Decoder().to(device)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        predictions = np.lib.format.open_memmap(
            folder / "test_predictions.npy",
            mode="w+",
            dtype="float32",
            shape=(len(TEST), 1000, STATES),
        )
        with torch.no_grad():
            for start in range(0, len(eidx), 1024):
                x, y = window(eidx[start : start + 1024])
                p = torch.softmax(model(x), 1)
                assert torch.isfinite(p).all()
                predictions.reshape(-1, STATES)[start : start + 1024] = p.cpu().numpy()
        predictions.flush()
        confusion = np.zeros((STATES, STATES), dtype=np.int64)
        rows = []
        probability_error = 0.0
        for j, scan in enumerate(TEST):
            truth = np.asarray(alpha[scan, 100:1100], dtype=float)
            p = np.asarray(predictions[j], dtype=float)
            probability_error = max(probability_error, float(abs(p.sum(1) - 1).max()))
            a, b = truth.argmax(1), p.argmax(1)
            np.add.at(confusion, (a, b), 1)
            rows.append(
                dict(
                    subject_index=int(scan // 4),
                    scan=int(scan),
                    frames=1000,
                    correct=int((a == b).sum()),
                    accuracy=float((a == b).mean()),
                    mse=float(np.square(p - truth).mean()),
                )
            )
        assert probability_error < 1e-5 and confusion.sum() == 2006000
        write_csv(folder / "scan_metrics.csv", rows)
        np.save(folder / "confusion.npy", confusion)
        selected = {
            k: v for k, v in ck.items() if k not in ["state_dict", "feature_mean", "feature_std"]
        }
        write_json(
            folder / "complete.json",
            dict(
                status="PASS",
                selection=selected,
                seconds=time.time() - started,
                probability_sum_max_error=probability_error,
                evaluation_frames=int(confusion.sum()),
                evaluation_participants=N,
            ),
        )
        print(
            json.dumps(
                dict(
                    stage="complete",
                    condition=name,
                    accuracy=float(np.trace(confusion) / confusion.sum()),
                    seconds=time.time() - started,
                )
            ),
            flush=True,
        )
        del X, model, predictions, ck
        gc.collect()
        torch.cuda.empty_cache()


def summarize(out):
    rng = np.random.default_rng(20260918)
    boot = rng.integers(0, N, size=(2000, N))
    result = []
    per_subject = {}
    histories = []
    selections = []
    export = out / "export"
    export.mkdir(exist_ok=True)
    for name, (family, count) in CONDITIONS.items():
        folder = out / name
        complete = json.loads((folder / "complete.json").read_text())
        assert complete["status"] == "PASS"
        rows = list(csv.DictReader((folder / "scan_metrics.csv").open()))
        assert np.array_equal([int(r["scan"]) for r in rows], TEST)
        accuracy = np.array([float(r["accuracy"]) for r in rows]).reshape(N, 2).mean(1)
        mse = np.array([float(r["mse"]) for r in rows]).reshape(N, 2).mean(1)
        per_subject[name] = accuracy
        cf = np.load(folder / "confusion.npy")
        interval = np.quantile(accuracy[boot].mean(1), [0.025, 0.975])
        selection = complete["selection"]
        selections.append(selection)
        result.append(
            dict(
                condition=name,
                representation=family,
                complex_coordinates=count if family == "cpc" else 2 * count,
                modes_per_hemisphere=0 if family == "cpc" else count,
                participants=N,
                frames=int(cf.sum()),
                accuracy_mean=float(accuracy.mean()),
                accuracy_sd=float(accuracy.std(ddof=1)),
                accuracy_ci_low=float(interval[0]),
                accuracy_ci_high=float(interval[1]),
                balanced_accuracy=float(np.mean(np.diag(cf) / cf.sum(1))),
                posterior_mse=float(mse.mean()),
                validation_mse=selection["validation_mse"],
                epoch=selection["epoch"],
                learning_rate=selection["learning_rate"],
                weight_decay=selection["weight_decay"],
            )
        )
        histories.extend(list(csv.DictReader((folder / "validation_history.csv").open())))
    differences = []
    for name in [key for key in CONDITIONS if key != "cpc30"]:
        d = per_subject[name] - per_subject["cpc30"]
        ci = np.quantile(d[boot].mean(1), [0.025, 0.975])
        differences.append(
            dict(
                geometry_condition=name,
                reference="cpc30",
                accuracy_difference_pp=float(100 * d.mean()),
                difference_ci_low_pp=float(100 * ci[0]),
                difference_ci_high_pp=float(100 * ci[1]),
            )
        )
    validation = json.loads((out / "projection_validation.json").read_text())
    for record in validation["sources"]:
        assert fingerprint(Path(record["path"])) == record
    write_csv(export / "direct_decoding_summary.csv", result)
    write_csv(export / "paired_accuracy_differences.csv", differences)
    write_csv(export / "validation_histories.csv", histories)
    write_json(
        export / "decoding_validation.json",
        dict(
            status="PASS",
            projection=validation,
            selections=selections,
            source_fingerprints_unchanged=True,
            statistics="2000 paired participant bootstraps; both REST2 runs together",
            plot_error_bars="Between-participant SD after averaging both REST2 scans",
            scope="Offline same-participant acquisition transfer; fixed robust_seed2 target; all evaluation frames",
            family_dependence_controlled=False,
            training_uncertainty_estimated=False,
            script_fingerprint=fingerprint(Path(__file__)),
        ),
    )
    print(json.dumps(dict(results=result, paired_differences=differences), indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["project", "fit", "summarize"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(exist_ok=True, parents=True)
    write_json(
        args.output / ("status_" + args.stage + "_" + args.device.replace(":", "") + ".json"),
        dict(stage=args.stage, pid=os.getpid(), started_at=time.time()),
    )
    if args.stage == "project":
        project(args.output, args.device)
    elif args.stage == "fit":
        fit(args.conditions.split(","), args.output, args.device)
    else:
        summarize(args.output)
