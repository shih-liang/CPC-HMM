"""Prepare aligned cortical and official HCP ICA50 arrays without fitting ICA.

Use an explicit subject list to reproduce an existing cohort. Output directories
must be new, so interrupted or completed inputs cannot be silently overwritten.
"""

import argparse
import csv
from pathlib import Path
import shutil
import subprocess
import tempfile

import nibabel as nib
import numpy as np
from scipy.signal import butter, sosfiltfilt

RUNS = ("REST1_LR", "REST1_RL", "REST2_LR", "REST2_RL")


def standardize(x):
    """Match HMM_input.py: float32, population SD, independently within a run."""
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Non-finite input values")
    sd = x.std(axis=0, keepdims=True)
    sd[sd == 0] = 1
    return (x - x.mean(axis=0, keepdims=True)) / sd


def metric(path):
    """Read a GIFTI metric as time/maps by vertices."""
    arrays = nib.load(path).darrays
    return np.stack([a.data for a in arrays])


def command(*args):
    subprocess.run([str(a) for a in args], check=True)


def resample(wb, source, target, templates, hemi):
    command(
        wb, "-metric-resample", source,
        templates / f"fs_LR-deformed_to-fsaverage.{hemi}.sphere.32k_fs_LR.surf.gii",
        templates / f"fsaverage4_std_sphere.{hemi}.3k_fsavg_{hemi}.surf.gii",
        "ADAP_BARY_AREA", target, "-area-metrics",
        templates / f"fs_LR.{hemi}.midthickness_va_avg.32k_fs_LR.shape.gii",
        templates / f"fsaverage4.{hemi}.midthickness_va_avg.3k_fsavg_{hemi}.shape.gii",
    )


def filter_cortical(data):
    """Apply the study's fifth-order zero-phase 0.01–0.10 Hz filter.

    Actual inputs match filtering without pre-filter z-scoring or detrending;
    z-scoring follows fsaverage4 resampling. Bolt et al. (2022), BOLD_WAVES
    norm_filter.py, instead z-scores before filtering. Both use TR 0.72 s,
    order 5 and SOS forward/backward filtering with 33-frame odd padding.
    """
    data = np.asarray(data, dtype=np.float64)
    if not np.isfinite(data).all():
        raise ValueError("Non-finite cortical data")
    sos = butter(5, [0.01, 0.10], btype="bandpass", fs=1 / 0.72, output="sos")
    return sosfiltfilt(sos, data, axis=0, padtype="odd", padlen=33)


def cortical_run(source, templates, wb, tmp):
    """Smooth, filter in time and resample one ICA-FIX CIFTI acquisition."""
    smooth = tmp / "smooth.dtseries.nii"
    command(wb, "-cifti-smoothing", source, "2.12", "0", "COLUMN", smooth,
            "-left-surface", templates / "S900.L.midthickness_MSMAll.32k_fs_LR.surf.gii",
            "-right-surface", templates / "S900.R.midthickness_MSMAll.32k_fs_LR.surf.gii")
    image = nib.load(smooth)
    series = image.header.get_axis(0)
    if image.shape[0] != 1200 or not np.isclose(series.step, 0.72):
        raise ValueError(f"Expected 1200 frames at TR=0.72 s: {source}")
    data = image.get_fdata(dtype=np.float64)
    if not np.isfinite(data).all():
        raise ValueError(f"Non-finite CIFTI: {source}")
    data = filter_cortical(data)
    filtered = tmp / "filtered.dtseries.nii"
    nib.save(nib.Cifti2Image(data.astype(np.float32), header=image.header,
                           nifti_header=image.nifti_header), filtered)
    left, right = tmp / "L.func.gii", tmp / "R.func.gii"
    command(wb, "-cifti-separate", filtered, "COLUMN", "-metric", "CORTEX_LEFT", left,
            "-metric", "CORTEX_RIGHT", right)
    parts = []
    for hemi, path in zip(("L", "R"), (left, right)):
        target = tmp / f"fs4.{hemi}.func.gii"
        resample(wb, path, target, templates, hemi)
        part = metric(target)
        if part.shape != (1200, 2562):
            raise ValueError(f"Unexpected resampled shape: {part.shape}")
        parts.append(part)
    return np.concatenate(parts, axis=1)


def prepare(args):
    subjects = args.subjects.read_text().split()
    if not subjects or len(subjects) != len(set(subjects)) or not all(s.isdigit() for s in subjects):
        raise ValueError("Subject file must contain unique numeric HCP IDs in the intended order")
    if args.geometry_reference not in subjects:
        raise ValueError("--geometry-reference must be included in the subject list")
    rows = []
    for subject in subjects:
        ica = Path(args.ica_pattern.format(subject=subject))
        if not ica.is_file():
            raise FileNotFoundError(ica)
        for run in RUNS:
            source = Path(args.cortical_pattern.format(subject=subject, run=run, hemi="L"))
            if not source.is_file():
                raise FileNotFoundError(source)
            if args.cortical_kind == "fs4":
                right = Path(args.cortical_pattern.format(subject=subject, run=run, hemi="R"))
                if right == source or not right.is_file():
                    raise ValueError("fs4 pattern must contain {hemi} and resolve both hemispheres")
            rows.append((subject, run, str(source), str(ica)))
    if args.output.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    if args.cortical_kind == "cifti":
        if args.templates is None:
            raise ValueError("CIFTI conversion requires --templates")
        if shutil.which(args.wb_command) is None:
            raise FileNotFoundError(args.wb_command)
    args.output.mkdir(parents=True)
    cortical = args.output / "cortical/input_hmm_order"
    cortical.mkdir(parents=True)
    ica_root = args.output / "ica"
    ica_root.mkdir()
    mask_dir = args.output / "geometry_masks"
    mask_dir.mkdir()
    (mask_dir / "reference_subject.txt").write_text(args.geometry_reference + "\n")
    nrun = len(rows)
    bold = np.lib.format.open_memmap(cortical / "group_fs4_concat_z_hmm_order.npy", mode="w+",
                                    dtype="float32", shape=(nrun * 1200, 5124))
    ica_out = np.lib.format.open_memmap(ica_root / "X_ICA50_zscore.npy", mode="w+",
                                       dtype="float32", shape=(nrun * 1200, 50))
    for i, subject in enumerate(subjects):
        x = np.loadtxt(args.ica_pattern.format(subject=subject)).astype(np.float32)
        if x.shape != (4800, 50):
            raise ValueError(f"ICA50 for {subject}: expected (4800, 50), got {x.shape}")
        for j, run in enumerate(RUNS):
            scan = 4 * i + j
            sl = slice(scan * 1200, (scan + 1) * 1200)
            ica_out[sl] = standardize(x[j * 1200:(j + 1) * 1200])
            if args.cortical_kind == "fs4":
                parts = [metric(args.cortical_pattern.format(subject=subject, run=run, hemi=h))
                         for h in ("L", "R")]
                if any(part.shape != (1200, 2562) for part in parts):
                    raise ValueError(f"Unexpected fs4 metric shape for {subject} {run}")
                data = np.concatenate(parts, axis=1)
            else:
                with tempfile.TemporaryDirectory(dir=args.output) as folder:
                    data = cortical_run(rows[scan][2], args.templates, args.wb_command, Path(folder))
            # Preserve the original nonzero support before within-run standardization.
            if subject == args.geometry_reference and run in RUNS[:2]:
                if not np.isfinite(data).all():
                    raise ValueError("Non-finite reference functional data")
                for h, part in zip(("L", "R"), np.split(data, 2, axis=1)):
                    np.save(mask_dir / f"{run}.{h}.mask.npy", np.any(part != 0, axis=0))
            bold[sl] = standardize(data)
            print(f"Prepared {scan + 1}/{nrun}: {subject} {run}", flush=True)
    bold.flush()
    ica_out.flush()
    np.save(ica_root / "T_ICA50.npy", np.full(nrun, 1200, dtype=np.int32))
    with (args.output / "run_order.tsv").open("w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(("subject", "run", "cortical_source", "ica_source"))
        writer.writerows(rows)
    (args.output / "subjects.txt").write_text("\n".join(subjects) + "\n")
    (args.output / "conversion_settings.txt").write_text(
        "\n".join(f"{k}={v}" for k, v in vars(args).items()) + "\ncomplete=true\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-reference", default="100206",
                        help="Reference participant for nonzero-support geometry masks")
    parser.add_argument("--subjects", type=Path, required=True)
    parser.add_argument("--ica-pattern", required=True, help="Official ICA50 text path with {subject}")
    parser.add_argument("--cortical-pattern", required=True, help="Path with {subject}, {run}, and for fs4 {hemi}")
    parser.add_argument("--cortical-kind", choices=("cifti", "fs4"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--templates", type=Path)
    parser.add_argument("--wb-command", default="wb_command")
    prepare(parser.parse_args())
