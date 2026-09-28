"""Fit REST1_LR CPCA/PCA bases and project all runs. Executes on import.

Reads preprocessed cortical arrays; writes bases, scores and metadata.
C = Z @ U and reconstructed analytic activity = C @ U.conj().T.
Use a fresh derivative tree: score memmaps are opened for writing.
"""

import os
import time
import json

os.environ.setdefault("OMP_NUM_THREADS", "4")
from pathlib import Path
import numpy as np
import torch

P = Path(
    os.path.join(
        os.environ.get("HCP_DERIVATIVES", "/configure/HCP_DERIVATIVES"),
        "wave_rsn_revision_20260906",
    )
)
for directory in (P / "data", P / "results"):
    directory.mkdir(parents=True, exist_ok=True)
R = Path(os.environ.get("HCP_CORTICAL_ROOT", "/configure/HCP_CORTICAL_ROOT"))
torch.set_num_threads(4)
dev = torch.device(os.environ.get("HCP_DEVICE", "cuda:1"))
torch.manual_seed(20260906)
x = np.load(R / "input_hmm_order/group_fs4_concat_z_hmm_order.npy", mmap_mode="r")
assert x.shape == (4814400, 5124)
E = Path(
    os.path.join(
        os.environ.get("HCP_DERIVATIVES", "/configure/HCP_DERIVATIVES"),
        "wmy/geometry/Eigenmodes_fs4",
    )
)
idxs = [np.load(E / f"{h}.fs4_idx.npy") + i * 2562 for i, h in enumerate(["L", "R"])]
idx = np.concatenate(idxs)
V = len(idx)
length = 1200
hs = torch.zeros(length, device=dev)
hs[0] = 1
hs[1 : length // 2] = 2
hs[length // 2] = 1
start = time.time()
n = 0
C = torch.zeros((V, V), dtype=torch.complex64, device=dev)
D = torch.zeros((V, V), dtype=torch.float32, device=dev)


def get(scan):
    """Return one demeaned (1200, retained_vertices) cortical run on the device."""
    a = np.asarray(x[scan * length : (scan + 1) * length, idx], dtype=np.float32).copy()
    # Saved inputs are already per-run standardized; remove residual numerical mean only.
    a -= a.mean(0, keepdims=True)
    return torch.from_numpy(a).to(dev)


def analytic(a):
    """Form the analytic signal along time within this run; never concatenate runs."""
    return torch.fft.ifft(torch.fft.fft(a, dim=0) * hs[:, None], dim=0)


if not (P / "data/training_bases.npz").exists():
    for subject in range(1003):
        a = get(subject * 4)
        z = analytic(a)
        C.addmm_(z.H, z)
        D.addmm_(a.T, a)
        n += length
        if subject % 50 == 0:
            print(
                "covariance subjects",
                subject + 1,
                "seconds",
                round(time.time() - start),
                flush=True,
            )
    C = (C + C.H) / (2 * (n - 1))
    D = (D + D.T) / (2 * (n - 1))
    print("eigendecomposition", round(time.time() - start), flush=True)
    ec, uc = torch.linalg.eigh(C)
    er, ur = torch.linalg.eigh(D)
    ec = ec.flip(0)
    uc = uc.flip(1)[:, :50]
    er = er.flip(0)
    ur = ur.flip(1)[:, :100]
    # Fix each component's arbitrary global phase/sign with its largest loading.
    for j in range(50):
        uc[:, j] *= torch.exp(-1j * torch.angle(uc[torch.argmax(abs(uc[:, j])), j]))
    for j in range(100):
        if ur[torch.argmax(abs(ur[:, j])), j] < 0:
            ur[:, j] *= -1
    np.savez(
        P / "data/training_bases.npz",
        complex_vectors=uc.cpu().numpy(),
        complex_eigenvalues=ec.cpu().numpy(),
        real_vectors=ur.cpu().numpy(),
        real_eigenvalues=er.cpu().numpy(),
        vertex_indices=idx,
        training_scans=np.arange(1003) * 4,
    )
else:
    b = np.load(P / "data/training_bases.npz")
    uc = torch.from_numpy(b["complex_vectors"]).to(dev)
    ur = torch.from_numpy(b["real_vectors"]).to(dev)
del C, D
geo = []
for m in [10, 50, 200]:
    vv = np.zeros((V, 50), dtype=np.complex64)
    offset = 0
    for hi, h in enumerate(["L", "R"]):
        ph = np.loadtxt(E / f"eigenmodes_fs4_{h}.txt")
        q, _ = np.linalg.qr(ph[:, :m], mode="reduced")
        size = len(ph)
        u = uc[offset : offset + size].cpu().numpy()
        vv[offset : offset + size] = q @ (q.T @ u)
        offset += size
    geo.append(torch.from_numpy(vv).to(dev))
combined = torch.cat([uc] + geo, dim=1)
cp = np.lib.format.open_memmap(
    P / "data/cpca_scores.npy", mode="w+", dtype="complex64", shape=(4012, 1200, 50)
)
rp = np.lib.format.open_memmap(
    P / "data/real_pca_scores.npy", mode="w+", dtype="float32", shape=(4012, 1200, 100)
)
gs = [
    np.lib.format.open_memmap(
        P / f"data/geometry_scores_{m}.npy", mode="w+", dtype="complex64", shape=(4012, 1200, 50)
    )
    for m in [10, 50, 200]
]
for scan in range(4012):
    a = get(scan)
    z = analytic(a)
    scores = (z @ combined).cpu().numpy()
    cp[scan] = scores[:, :50]
    rp[scan] = (a @ ur).cpu().numpy()
    for j, g in enumerate(gs):
        g[scan] = scores[:, 50 * (j + 1) : 50 * (j + 2)]
    if scan % 100 == 0:
        print("projection scans", scan + 1, "seconds", round(time.time() - start), flush=True)
cp.flush()
rp.flush()
for g in gs:
    g.flush()
meta = {
    "training": "REST1_LR only, 1003 participants",
    "hilbert": "FFT separately within each 1200-frame scan",
    "input": "ICA-FIX BOLD; fifth-order 0.01–0.10 Hz filter, TR 0.72 s, odd padding 33; standardized after fs4 resampling",
    "vertices": V,
    "complex_components": 50,
    "real_components": 100,
    "geometry_modes_per_hemisphere": [10, 50, 200],
    "seconds": time.time() - start,
    "device": str(dev),
}
(P / "results/representation_metadata.json").write_text(json.dumps(meta, indent=2))
print("COMPLETE", meta, flush=True)
