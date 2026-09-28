"""Fit only the two missing geometric-coordinate counts with the existing recipe."""

from pathlib import Path
import argparse
import json
import torch
import direct_decoding as base

ADDITIONS = {"geo100": ("geometry", 100), "geo150": ("geometry", 150)}
parser = argparse.ArgumentParser()
parser.add_argument("--condition", choices=list(ADDITIONS), required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--device", required=True)
args = parser.parse_args()
assert args.output.is_dir() and not (args.output / args.condition).exists()
assert json.loads((args.output / "projection_validation.json").read_text())["status"] == "PASS"
sources = [
    args.output / "native_geometry_scores.npy",
    args.output / "projection_validation.json",
    base.SOURCE / "data/hmm_alpha_robust_seed2.npy",
    Path(base.__file__),
    Path(__file__),
    Path(__file__).resolve().parents[1] / "GRID_EXTENSION_PROTOCOL.md",
]
before = [base.fingerprint(p) for p in sources]
base.CONDITIONS.update(ADDITIONS)
torch.set_num_threads(4)
base.fit([args.condition], args.output, args.device)
assert before == [base.fingerprint(p) for p in sources]
base.write_json(
    args.output / args.condition / "fitting_validation.json",
    dict(
        status="PASS",
        condition=args.condition,
        complex_coordinates=2 * ADDITIONS[args.condition][1],
        fitting_recipe="Unchanged direct_decoding.fit function",
        sources=before,
        source_fingerprints_unchanged=True,
        device=args.device,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
    ),
)
print(
    json.dumps(
        {"status": "PASS", "condition": args.condition, "source_fingerprints_unchanged": True}
    ),
    flush=True,
)
