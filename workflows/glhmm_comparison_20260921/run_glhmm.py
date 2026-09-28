"""Run one seed of the official GLHMM Gaussian-state comparison.

model_beta="no" disables regression; this is not a beta-enabled model.
Requires official GLHMM and ICA50 data; executes on import.
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["NUMBA_NUM_THREADS"] = "4"
import sys
import json
import time
import pickle
import hashlib
import copy
import subprocess
from pathlib import Path
import numpy as np
import scipy
import numba

P = Path(__file__).resolve().parent
if os.environ.get("GLHMM_SOURCE"):
    sys.path.insert(0, os.environ["GLHMM_SOURCE"])
from glhmm.glhmm import glhmm


def train_seed(model, x, indices, seed, options):
    """Seed legacy and default_rng calls during training; restore the RNG factory afterward."""
    original = np.random.default_rng
    calls = []

    def factory(value=None):
        if value is not None:
            return original(value)
        calls.append([seed, len(calls)])
        return original(np.random.SeedSequence(calls[-1]))

    np.random.seed(seed)
    np.random.default_rng = factory
    try:
        result = model.train(Y=x, indices=indices, options=copy.deepcopy(options))
    finally:
        np.random.default_rng = original
    return result, calls


def indices(n, l):
    """Return half-open sequence boundaries for n independent runs of l samples."""
    return np.column_stack((np.arange(n) * l, (np.arange(n) + 1) * l))


def source_provenance(core):
    """Describe the loaded code without treating an expected revision as observed."""
    core = Path(core).resolve()
    expected = "9a5c59acfc7956eb3b579fc92e54d108708a5d02"
    actual = None
    dirty = None
    # Only accept a checkout that tracks this exact loaded source file.
    try:

        def git(*args):
            return subprocess.check_output(
                ["git", "-C", str(core.parent), *args],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=10,
            ).strip()

        git("ls-files", "--error-unmatch", core.name)
        actual = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain", "--untracked-files=all"))
    except (OSError, subprocess.SubprocessError):
        actual = None
        dirty = None
    return dict(
        expected_commit=expected,
        actual_commit=actual,
        checkout_dirty=dirty,
        expected_checkout_verified=actual == expected and dirty is False,
        core_sha256=hashlib.sha256(core.read_bytes()).hexdigest(),
    )


source = source_provenance(sys.modules[glhmm.__module__].__file__)
seed = int(sys.argv[1]) if len(sys.argv) > 1 else int(np.random.default_rng().integers(2**32))
R = P / "results"
R.mkdir(exist_ok=True)
meta = R / f"glhmm_{seed}.json"
if meta.exists():
    raise SystemExit("Already complete; preserving " + str(meta))
t0 = time.time()
raw = np.load(
    os.path.join(os.environ.get("HCP_ICA_ROOT", "/configure/HCP_ICA_ROOT"), "X_ICA50_zscore.npy"),
    mmap_mode="r",
).reshape(4012, 1200, 50)
idx = np.arange(100) * 4
x = np.asarray(raw[idx], dtype=float).reshape(-1, 50)
y = np.asarray(raw[np.sort(np.r_[idx + 2, idx + 3])], dtype=float).reshape(-1, 50)
opts = dict(
    cyc=500,
    initrep=5,
    initcyc=10,
    tol=1e-4,
    cyc_to_go_under_th=10,
    deactivate_states=False,
    verbose=True,
)
m = glhmm(K=12, covtype="full", model_mean="state", model_beta="no", dirichlet_diag=10)
(_, _, fe), calls = train_seed(m, x, indices(100, 1200), seed, opts)
with (R / f"glhmm_{seed}.pkl").open("wb") as f:
    pickle.dump(m, f)
g, _, _ = m.decode(None, x, indices(100, 1200))
h, _, _ = m.decode(None, y, indices(200, 1200))
for a in [g, h]:
    assert np.isfinite(a).all() and np.allclose(a.sum(1), 1)
np.savez_compressed(
    R / f"glhmm_{seed}.npz",
    train=g.reshape(100, 1200, 12)[:, 100:1100:10].astype("float32"),
    test=h.reshape(200, 1200, 12)[:, 100:1100].astype("float32"),
    A=m.P,
    pi=m.Pi,
    means=np.array([v["Mu"] for v in m.mean]),
    active=m.active_states,
)
metadata = dict(
    seed=seed,
    options=opts,
    iterations=len(fe),
    hit_cap=len(fe) >= 500,
    free_energy=fe.tolist(),
    rng_calls=calls,
    active_states=m.active_states.tolist(),
    seconds=time.time() - t0,
    python=sys.version,
    numpy=np.__version__,
    scipy=scipy.__version__,
    numba=numba.__version__,
    **source,
)
meta.write_text(json.dumps(metadata, indent=2))
print("SAVED", meta, flush=True)
