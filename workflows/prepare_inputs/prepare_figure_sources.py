"""Assemble the figure-source directory from completed study analyses."""

import argparse
from pathlib import Path
import shutil


def prepare(derivatives, output):
    base = derivatives / "wave_rsn_revision_20260906"
    revision = base / "revision_20260910"
    files = {
        "figure_data/cpca_basis_REST1_LR.npz": revision / "data/cpca_basis_REST1_LR.npz",
        "figure_data/surfaces_and_eigenmodes.npz": base / "data/surfaces_and_eigenmodes.npz",
        "figure_data/geometry_incremental_energy.npz": base / "results/geometry_incremental_energy.npz",
        "source_data/Figure2_geometry_summary.csv": base / "results/Figure2_geometry_summary.csv",
        "source_data/original_cpca_component_properties.csv": base / "results/cpca_component_properties.csv",
    }
    for name in ("cpca_test_variance_summary.csv", "cpca_acquisition_mode_overlaps.csv",
                 "cpca_acquisition_subspaces.csv", "cpca_test_variance_participants.csv",
                 "cpca_test_score_reproducibility.csv"):
        files[f"source_data/{name}"] = revision / "results" / name
    for source in files.values():
        if not source.is_file():
            raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    for target, source in files.items():
        destination = output / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derivatives", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.derivatives, args.output)
