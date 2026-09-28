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

## Geometric eigenmode format conversion

The study's supplied geometric eigenmodes are distinct from the CPCA basis and HCP ICA50 maps. The following program converts full or masked fsaverage4 mode arrays (`.txt`, `.npy`, or metric `.gii`) to the files used by the analysis:

```bash
python workflows/prepare_inputs/prepare_geometry.py \
  --modes-pattern '/path/to/modes_{hemi}.txt' \
  --values-pattern '/path/to/eigenvalues_{hemi}.txt' \
  --mask-pattern '/path/to/mask_{hemi}.txt' \
  --output /path/to/analysis-derivatives/wmy/geometry/Eigenmodes_fs4
```

Masks must be binary arrays of 2,562 vertices. Eigenmode rows must be in full fsaverage4 order or in the ascending retained-vertex order of that mask; columns and supplied eigenvalues must follow ascending geometric eigenvalue order. The first 200 modes, including the constant mode, are retained. The recorded masks retain 2,395 left and 2,406 right vertices.

This is format conversion, not a substitute derivation of the geometric modes. The original mesh/eigensolver and mapping used to obtain the supplied geometric modes have not yet been independently recovered. Do not identify newly computed modes as numerically identical to these study inputs.

## Verification

For one participant, filtering and resampling of all four archived smoothed acquisitions matched the stored outputs with standardized RMSE below 8 × 10⁻¹¹. Starting from the two available REST1 ICA-FIX files, the complete conversion matched the stored cortical arrays with maximum absolute difference at most 1.2 × 10⁻⁷. ICA50 conversion matched all four runs exactly. These checks include the position of standardization relative to filtering/resampling. The filter uses the actual study settings; its code comment records the literature comparison. The full 1,003-participant models were not refitted for this update.

See [ANALYSIS.md](ANALYSIS.md) for subsequent fitting and [README.md](../README.md#preparing-the-figure-source-directory) for automated figure-source assembly.
