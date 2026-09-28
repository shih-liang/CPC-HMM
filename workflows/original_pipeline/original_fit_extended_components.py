"""Extend the fixed REST1_LR basis to 200 complex components.

Writes full-cohort score arrays; this historical script executes on import.
"""

import os  # Public release: configurable data roots.
from pathlib import Path
import numpy as np
import torch
import json
import time

P = Path(
    os.path.join(
        os.environ.get("HCP_DERIVATIVES", "/configure/HCP_DERIVATIVES"),
        "wave_rsn_revision_20260906",
    )
)
torch.set_num_threads(4)
dev = os.environ.get("HCP_DEVICE", "cuda:1")
raw = np.load(
    os.path.join(
        os.environ.get("HCP_CORTICAL_ROOT", "/configure/HCP_CORTICAL_ROOT"),
        "input_hmm_order/group_fs4_concat_z_hmm_order.npy",
    ),
    mmap_mode="r",
).reshape(4012, 1200, 5124)
old = np.load(P / "data/training_bases.npz")
idx = old["vertex_indices"]
it = torch.tensor(idx, device=dev)
V = len(idx)
L = 1200
start = time.time()
h = torch.zeros(L, device=dev)
h[0] = h[L // 2] = 1
h[1 : L // 2] = 2
(P / "results/extended_component_protocol.json").write_text(
    json.dumps(
        {
            "purpose": "User-requested exploratory component-count comparison",
            "complex_components": [50, 100, 200],
            "real_components": [100, 200, 400],
            "fit": "REST1_LR only; per-scan Hilbert transform; same masked cortical inputs",
            "validation": "REST1_RL, select hyperparameters and component count",
            "evaluation": "REST2_LR/RL already inspected in previous analyses; retrospective comparison, not untouched confirmation",
            "fixed_target": json.loads((P / "results/selected_hmm.json").read_text())["tag"],
            "decoder": "Cartesian complex coordinates; common 400 input slots, 50-frame past window, hidden128/64; real PCA rank controls",
            "grid": "Learning rate 0.001 or 0.0003; weight decay 0.0001 or 0.001; maximum30epochs, validation early stopping; same budget across counts",
        },
        indent=2,
    )
)


def get(ids):
    a = torch.from_numpy(np.array(raw[ids], copy=True)).to(dev).index_select(2, it)
    a -= a.mean(1, keepdim=True)
    return a


def analytic(a):
    return torch.fft.ifft(torch.fft.fft(a, dim=1) * h[None, :, None], dim=1)


basis = P / "data/extended_training_bases.npz"
if not basis.exists():
    C = torch.zeros((V, V), dtype=torch.complex64, device=dev)
    D = torch.zeros((V, V), device=dev)
    train = np.arange(1003) * 4
    for j in range(0, 1003, 8):
        a = get(train[j : j + 8])
        z = analytic(a).reshape(-1, V)
        af = a.reshape(-1, V)
        C.addmm_(z.H, z)
        D.addmm_(af.T, af)
        if j % 80 == 0:
            print(
                "training covariance",
                min(j + 8, 1003),
                "seconds",
                round(time.time() - start),
                flush=True,
            )
    C = (C + C.H) / (2 * (1003 * L - 1))
    D = (D + D.T) / (2 * (1003 * L - 1))
    ec, U = torch.linalg.eigh(C)
    er, R = torch.linalg.eigh(D)
    ec = ec.flip(0)
    er = er.flip(0)
    U = U.flip(1)[:, :200]
    R = R.flip(1)[:, :400]
    for k in range(200):
        U[:, k] *= torch.exp(-1j * torch.angle(U[torch.argmax(abs(U[:, k])), k]))
    for k in range(400):
        if R[torch.argmax(abs(R[:, k])), k] < 0:
            R[:, k] *= -1
    np.savez(
        basis,
        complex_vectors=U.cpu().numpy(),
        real_vectors=R.cpu().numpy(),
        complex_eigenvalues=ec.cpu().numpy(),
        real_eigenvalues=er.cpu().numpy(),
        vertex_indices=idx,
        training_scans=train,
    )
    np.save(P / "data/extended_training_complex_covariance.npy", C.cpu().numpy())
    np.save(P / "data/extended_training_real_covariance.npy", D.cpu().numpy())
    del C, D
else:
    b = np.load(basis)
    U = torch.from_numpy(b["complex_vectors"]).to(dev)
    R = torch.from_numpy(b["real_vectors"]).to(dev)
a = np.load(basis)
overlap = abs(old["complex_vectors"].conj().T @ a["complex_vectors"][:, :50])
sing = np.linalg.svd(
    old["complex_vectors"].conj().T @ a["complex_vectors"][:, :50], compute_uv=False
)
(P / "results/extended_basis_agreement.json").write_text(
    json.dumps(
        {
            "minimum_top50_subspace_singular_value": float(sing.min()),
            "mean_same_index_loading_overlap": float(np.diag(overlap).mean()),
            "minimum_same_index_loading_overlap": float(np.diag(overlap).min()),
        },
        indent=2,
    )
)
print("basis saved", round(time.time() - start), flush=True)
cp = np.lib.format.open_memmap(
    P / "data/cpca_scores_200.npy", mode="w+", dtype="complex64", shape=(4012, L, 200)
)
rp = np.lib.format.open_memmap(
    P / "data/real_pca_scores_400.npy", mode="w+", dtype="float32", shape=(4012, L, 400)
)
for j in range(0, 4012, 8):
    stop = min(j + 8, 4012)
    a = get(slice(j, stop))
    z = analytic(a)
    cp[j:stop] = (z @ U).cpu().numpy()
    rp[j:stop] = (a @ R).cpu().numpy()
    if j % 160 == 0:
        print("projection", stop, "seconds", round(time.time() - start), flush=True)
cp.flush()
rp.flush()
b = np.load(basis)
ec = b["complex_eigenvalues"]
er = b["real_eigenvalues"]
meta = {
    "complex_variance_fractions": {str(k): float(ec[:k].sum() / ec.sum()) for k in [50, 100, 200]},
    "real_variance_fractions": {str(k): float(er[:k].sum() / er.sum()) for k in [100, 200, 400]},
    "seconds": time.time() - start,
    "complete": True,
}
(P / "results/extended_representation_metadata.json").write_text(json.dumps(meta, indent=2))
print("COMPLETE", meta, flush=True)
