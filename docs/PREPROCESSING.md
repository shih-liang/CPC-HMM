# Convert HCP downloads into analysis inputs

## Sources and order

Use the HCP ICA-FIX `*_Atlas_hp2000_clean.dtseries.nii` resting-state scans and the official HCP1200 MSMAll `NodeTimeseries_3T_HCP1200_MSMAll_ICAd50_ts2` release. Each subject's ICA50 text file has 4,800 rows and 50 columns. The study uses four contiguous 1,200-frame blocks in REST1_LR, REST1_RL, REST2_LR, REST2_RL order. Supply a plain-text subject list, one ID per line, in the desired order. The published full-cohort analyses require the recorded 1,003 participants; the converter permits smaller verification subsets.

Missing files, unexpected dimensions and non-finite inputs cause errors rather than silently dropping runs. Outputs require a new directory. Only a completed conversion writes `conversion_settings.txt` with `complete=true`; an interrupted directory must not be passed to analysis.

## From ICA-FIX CIFTI

Install `requirements-analysis.txt` and Connectome Workbench (`wb_command` on PATH, or supply `--wb-command`). Obtain the anatomical templates referenced below from the original providers, for example the [BOLD_WAVES templates directory](https://github.com/tsb46/BOLD_WAVES/tree/96e91ddccdacee4f6a0faa623d28528937fc9313/templates), respecting their data terms.

```bash
python workflows/prepare_inputs/prepare_hcp.py \
  --subjects /path/to/subjects.txt \
  --ica-pattern '/path/to/3T_HCP1200_MSMAll_d50_ts2/{subject}.txt' \
  --cortical-kind cifti \
  --cortical-pattern '/path/to/HCP/{subject}/MNINonLinear/Results/rfMRI_{run}/rfMRI_{run}_Atlas_hp2000_clean.dtseries.nii' \
  --templates /path/to/templates \
  --output /path/to/new-inputs
```

File patterns are quoted so the shell preserves the placeholders. Adapt paths to your downloaded layout; no specific absolute directory is required.

The converter processes one scan at a time:

1. Smooth on the supplied S900 left/right midthickness surfaces with cortical sigma 2.12 mm (about 5 mm FWHM), volume sigma 0.
2. Apply fifth-order Butterworth bandpass filtering at 0.01–0.10 Hz, TR 0.72 s, using SciPy SOS forward/backward filtering with odd padding of 33 frames. No pre-filter standardization, additional detrending or global signal regression is performed.
3. Extract the two cortical metrics and resample each to fsaverage4 using `ADAP_BARY_AREA` and the source/target vertex-area metrics.
4. Concatenate left then right vertices (2,562 each), standardize within this run, and write it in the shared subject/run order.
5. Independently standardize each 1,200-frame ICA50 block, without additional bandpass filtering or ICA fitting.

Required templates, for each `H=L,R`:

- `S900.H.midthickness_MSMAll.32k_fs_LR.surf.gii`
- `fs_LR-deformed_to-fsaverage.H.sphere.32k_fs_LR.surf.gii`
- `fsaverage4_std_sphere.H.3k_fsavg_H.surf.gii`
- `fs_LR.H.midthickness_va_avg.32k_fs_LR.shape.gii`
- `fsaverage4.H.midthickness_va_avg.3k_fsavg_H.shape.gii`

The output contains `cortical/input_hmm_order/group_fs4_concat_z_hmm_order.npy`, `ica/X_ICA50_zscore.npy`, `ica/T_ICA50.npy`, a common subject list and a run-order table. Set `HCP_CORTICAL_ROOT=/path/to/new-inputs/cortical` and `HCP_ICA_ROOT=/path/to/new-inputs/ica`.

## From existing filtered fsaverage4 metrics

To assemble the recorded preprocessed metrics without filtering twice:

```bash
python workflows/prepare_inputs/prepare_hcp.py \
  --subjects /path/to/subjects.txt \
  --ica-pattern '/path/to/3T_HCP1200_MSMAll_d50_ts2/{subject}.txt' \
  --cortical-kind fs4 \
  --cortical-pattern '/path/to/fs4/{subject}_{run}_Atlas_hp2000_clean_smooth_filt_resamp.{hemi}.func.gii' \
  --output /path/to/new-inputs
```

This path reads already filtered/resampled inputs and performs only within-run standardization and ordered assembly. Both hemispheres are required for every run.

## Generate geometric eigenmodes

Provide the white, pial and sphere surfaces as local files.

Manually download and extract the [fsaverage4 surface archive used by Nilearn 0.12.1](https://osf.io/28uma/download). Place these six files in `/path/to/templates/fsaverage4`:

- `white_left.gii.gz`, `white_right.gii.gz`
- `pial_left.gii.gz`, `pial_right.gii.gz`
- `sphere_left.gii.gz`, `sphere_right.gii.gz`

Templates must preserve the vertex/face ordering of the functional resampling sphere. The generator checks this ordering and stops on a mismatch. Do not substitute a reindexed template without also updating the functional resampling inputs.

The target registration spheres and area metrics are distributed in [HCPpipelines resample_fsaverage](https://github.com/Washington-University/HCPpipelines/tree/master/global/templates/standard_mesh_atlases/resample_fsaverage). Place the required files directly in the `--templates` directory used above.

`prepare_hcp.py` writes `geometry_masks/REST1_{LR,RL}.{L,R}.mask.npy` from the reference participant's filtered/resampled signals, before standardization. Each vertex is retained if any time point is nonzero. The default reference is `100206`; it must be in the subject list. `--geometry-reference ID` explicitly selects another participant when needed. `geometry_masks/reference_subject.txt` records the selection. Do not derive this mask from standardized arrays, because standardization removes constant nonzero signals.

```bash
python workflows/prepare_inputs/generate_geometry.py \
  --surface-dir /path/to/templates/fsaverage4 \
  --templates /path/to/templates \
  --mask-dir /path/to/new-inputs/geometry_masks \
  --output /path/to/analysis-derivatives/wmy/geometry/Eigenmodes_fs4
```

The generator uses the recovered study procedure:

1. Average corresponding white and pial coordinates on fsaverage4 (2,562 vertices per hemisphere).
2. Check face ordering against the fsaverage4 sphere and the HCP target registration sphere.
3. Require identical REST1_LR/RL functional-support masks. Retain only triangles whose three vertices survive the mask, and require a connected mesh without isolated vertices.
4. Apply `lapy.Solver(tria).eigs(k=200)` separately to each hemisphere, with no explicit boundary-condition override. Include the constant mode.
5. Save `eigenvalues_fs4_{L,R}.txt`, `eigenmodes_fs4_{L,R}.txt`, `{L,R}.fs4_mask.txt` and `{L,R}.fs4_idx.npy`. Eigenmode rows follow ascending retained-vertex indices; columns follow ascending eigenvalues.

This follows the finite-element geometric eigenmode approach of [Pang et al. (2023)](https://doi.org/10.1038/s41586-023-06098-1), whose [reference implementation](https://github.com/NSBLab/BrainEigenmodes) uses fsLR32k midthickness surfaces and supplied cortical masks. Here the eigensystem is calculated directly on the masked fsaverage4 midpoint surface. The public dependency specifies LaPy 1.0.1; the historical environment version has not been established.

For already generated eigenmodes, `prepare_geometry.py` remains an optional format converter (`--modes-pattern`, `--values-pattern`, `--mask-pattern`, `--output`); it is not needed after `generate_geometry.py`.

## Verification

For one participant, filtering and resampling of all four archived smoothed acquisitions matched the stored outputs with standardized RMSE below 8 × 10⁻¹¹. Starting from the two available REST1 ICA-FIX files, the complete conversion matched the stored cortical arrays with maximum absolute difference at most 1.2 × 10⁻⁷. ICA50 conversion matched all four runs exactly. These checks include the position of standardization relative to filtering/resampling. The filter uses the actual study settings; its code comment records the literature comparison. The full 1,003-participant models were not refitted for this update.

See [ANALYSIS.md](ANALYSIS.md) for subsequent fitting and [README.md](../README.md#preparing-the-figure-source-directory) for automated figure-source assembly.
