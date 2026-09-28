"""Compare independently fitted CPCA spatial modes and subspaces.

Reads full-cohort representations and writes reliability outputs; executes on import.
"""

import os  # Public release: configurable data roots.
from pathlib import Path
import numpy as np
import torch
import json
import time
import itertools
import pandas as pd
from scipy.optimize import linear_sum_assignment

P = Path(
    os.path.join(
        os.environ.get("HCP_DERIVATIVES", "/configure/HCP_DERIVATIVES"),
        "wave_rsn_revision_20260906",
    )
)
Q = P / "revision_20260910"
torch.set_num_threads(2)
dev = "cuda:0"
start = time.time()
b = np.load(P / "data/extended_training_bases.npz")
idx = b["vertex_indices"]
it = torch.tensor(idx, device=dev)
V = len(idx)
L = 1200
raw = np.load(
    os.path.join(
        os.environ.get("HCP_CORTICAL_ROOT", "/configure/HCP_CORTICAL_ROOT"),
        "input_hmm_order/group_fs4_concat_z_hmm_order.npy",
    ),
    mmap_mode="r",
).reshape(4012, L, 5124)
h = torch.zeros(L, device=dev)
h[0] = h[L // 2] = 1
h[1 : L // 2] = 2
names = ["REST1_LR", "REST1_RL", "REST2_LR", "REST2_RL"]


def get(ids):
    a = torch.from_numpy(np.array(raw[ids], copy=True)).to(dev).index_select(2, it)
    a -= a.mean(1, keepdim=True)
    return a, torch.fft.ifft(torch.fft.fft(a, dim=1) * h[None, :, None], dim=1)


for acq, name in enumerate(names):
    dest = Q / f"data/cpca_basis_{name}.npz"
    if dest.exists():
        continue
    if acq == 0:
        np.savez(
            dest,
            complex_vectors=b["complex_vectors"][:, :50],
            eigenvalues=b["complex_eigenvalues"],
            vertex_indices=idx,
        )
        continue
    C = torch.zeros((V, V), dtype=torch.complex64, device=dev)
    train = np.arange(1003) * 4 + acq
    for j in range(0, 1003, 8):
        a, z = get(train[j : j + 8])
        z = z.reshape(-1, V)
        C.addmm_(z.H, z)
        if j % 160 == 0:
            print("covariance", name, j, round(time.time() - start), flush=True)
    C = (C + C.H) / (2 * (1003 * L - 1))
    e, U = torch.linalg.eigh(C)
    e = e.flip(0)
    U = U.flip(1)[:, :50]
    for k in range(50):
        U[:, k] *= torch.exp(-1j * torch.angle(U[torch.argmax(abs(U[:, k])), k]))
    np.savez(dest, complex_vectors=U.cpu().numpy(), eigenvalues=e.cpu().numpy(), vertex_indices=idx)
    del C, U, e
    print("basis complete", name, round(time.time() - start), flush=True)
U = [np.load(Q / f"data/cpca_basis_{name}.npz")["complex_vectors"] for name in names]
rows = []
sub = []
for i, j in itertools.combinations(range(4), 2):
    ov = U[i].conj().T @ U[j]
    rr, cc = linear_sum_assignment(-abs(ov))
    perm = cc[np.argsort(rr)]
    for k in range(50):
        rows.append(
            dict(
                acquisition_a=names[i],
                acquisition_b=names[j],
                component_a=k + 1,
                component_b=int(perm[k]) + 1,
                complex_loading_overlap=float(abs(ov[k, perm[k]])),
                same_index_overlap=float(abs(ov[k, k])),
            )
        )
    for n in [3, 10, 30, 50]:
        sv = np.linalg.svd(ov[:n, :n], compute_uv=False)
        sub.append(
            dict(
                acquisition_a=names[i],
                acquisition_b=names[j],
                n_components=n,
                mean_squared_subspace_overlap=float(np.mean(sv**2)),
                minimum_subspace_cosine=float(sv.min()),
                mean_subspace_cosine=float(sv.mean()),
            )
        )
pd.DataFrame(rows).to_csv(Q / "results/cpca_acquisition_mode_overlaps.csv", index=False)
pd.DataFrame(sub).to_csv(Q / "results/cpca_acquisition_subspaces.csv", index=False)
ov = U[0].conj().T @ U[1]
rr, cc = linear_sum_assignment(-abs(ov))
perm = cc[np.argsort(rr)]
Un = U[1][:, perm] * np.exp(-1j * np.angle(ov[np.arange(50), perm]))[None, :]
Ut = torch.from_numpy(U[0]).to(dev)
Nt = torch.from_numpy(Un).to(dev)
test = np.sort(np.r_[np.arange(1003) * 4 + 2, np.arange(1003) * 4 + 3])
energy = []
corr = []
sums = np.zeros((3, 8, V), np.float64)
counts = np.zeros((3, 8), np.int64)
for j in range(0, len(test), 8):
    ids = test[j : j + 8]
    a, z = get(ids)
    a = a[:, 100:1100]
    z = z[:, 100:1100]
    c = z @ Ut
    d = z @ Nt
    norm = (z.abs() ** 2).sum((1, 2))
    ce = (c.abs() ** 2).sum(1)
    cs = ce.cumsum(1) / norm[:, None]
    cx = c - c.mean(1, keepdim=True)
    dx = d - d.mean(1, keepdim=True)
    cov = (cx.conj() * dx).sum(1)
    sd = torch.sqrt((cx.abs() ** 2).sum(1) * (dx.abs() ** 2).sum(1))
    rho = (cov / sd).real.cpu().numpy()
    cc = c.cpu().numpy()
    aa = a.cpu().numpy()
    for l, scan in enumerate(ids):
        for k in [3, 10, 30, 50]:
            energy.append(
                dict(
                    scan_index=int(scan),
                    participant_index=int(scan // 4),
                    rank=k,
                    analytic_variance_fraction=float(cs[l, k - 1]),
                )
            )
        for k in range(50):
            corr.append(
                dict(
                    scan_index=int(scan),
                    participant_index=int(scan // 4),
                    component=k + 1,
                    phase_aligned_score_correlation=float(rho[l, k]),
                )
            )
        for k in range(3):
            phase = np.angle(cc[l, :, k])
            ib = np.floor((phase + np.pi) / (2 * np.pi) * 8).astype(int).clip(0, 7)
            for t in range(8):
                mask = ib == t
                sums[k, t] += aa[l, mask].sum(0)
                counts[k, t] += mask.sum()
    if j % 160 == 0:
        print("test scores", j, round(time.time() - start), flush=True)
pd.DataFrame(energy).to_csv(Q / "results/cpca_test_variance_participants.csv", index=False)
pd.DataFrame(corr).to_csv(Q / "results/cpca_test_score_reproducibility.csv", index=False)
np.savez(
    Q / "results/empirical_phase_conditioned_maps.npz",
    mean_maps=sums / counts[:, :, None],
    counts=counts,
    phase_bin_centers=-np.pi + (np.arange(8) + 0.5) * 2 * np.pi / 8,
    components=np.arange(1, 4),
)
rng = np.random.default_rng(20260910)
ers = []
df = pd.DataFrame(energy)
for rank in [3, 10, 30, 50]:
    vals = (
        df[df["rank"] == rank].groupby("participant_index").analytic_variance_fraction.mean().values
    )
    boot = np.array([rng.choice(vals, len(vals), replace=True).mean() for _ in range(2000)])
    ers.append(
        dict(
            rank=rank,
            mean=float(vals.mean()),
            ci_low=float(np.quantile(boot, 0.025)),
            ci_high=float(np.quantile(boot, 0.975)),
            n_participants=len(vals),
        )
    )
pd.DataFrame(ers).to_csv(Q / "results/cpca_test_variance_summary.csv", index=False)
(Q / "results/cpca_reliability_complete.json").write_text(
    json.dumps(
        {
            "seconds": time.time() - start,
            "complete": True,
            "test_phase_maps": "Observed mean cortical BOLD by training-derived CPCA coefficient phase; descriptive conditional averages, not a propagation test.",
        },
        indent=2,
    )
)
print("COMPLETE", round(time.time() - start), flush=True)
