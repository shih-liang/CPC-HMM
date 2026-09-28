# CPC–HMM: travelling waves and resting-state network states

Code for analysing CPC travelling-wave coordinates, ICA networks and HMM states, and generating main Figures 1–4 and Supplementary Figures 1–3.

**Download data → preprocess → analyse → plot.** Accuracy, retained variance, consensus coverage and transition counts are computed from the supplied data and fitted models. They are not required to match previously reported values.

## 1. Download data

Obtain these inputs under their providers' access terms:

| Input | Required files |
| --- | --- |
| HCP resting-state ICA-FIX data | `rfMRI_{run}_Atlas_hp2000_clean.dtseries.nii`, for REST1_LR, REST1_RL, REST2_LR and REST2_RL |
| HCP-provided group ICA50 time courses | `NodeTimeseries_3T_HCP1200_MSMAll_ICAd50_ts2`; one 4,800 × 50 text file per participant |
| HCP-provided ICA50 spatial maps | `groupICA_3T_HCP1200_MSMAll_d50.ica/melodic_IC.dscalar.nii` |
| Cortical templates | S900 surfaces, registration spheres and vertex-area metrics listed in [PREPROCESSING.md](docs/PREPROCESSING.md) |
| FreeSurfer surfaces | `fsaverage4/surf/{lh,rh}.{inflated,pial,sphere}` |
| Geometric eigenmodes and masks | At least 200 ordered modes per hemisphere, eigenvalues, and binary fsaverage4 cortical masks |

Here “downloaded data” means **HCP-preprocessed ICA-FIX data**. This repository does not rerun HCP preprocessing or estimate the official ICA50 decomposition.

**Remaining input dependency:** geometric eigenmode derivation from cortical meshes is not included. `prepare_geometry.py` converts supplied modes and masks; it does not generate them. These files must be provided separately to run the complete workflow. HCP downloads alone are therefore not yet sufficient for end-to-end reproduction.

Prepare `subjects.txt`, one participant ID per line. The current full-cohort analyses require 1,003 participants with all four 1,200-frame acquisitions. Use the same list and order for cortical and ICA data. The original cohort list and participant data are not bundled; changing participants changes the analysis. Conversion supports smaller subsets, but full-cohort fitting and figure analyses retain the study dimensions.

## 2. Preprocess

### Install

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
```

Install Connectome Workbench separately and make `wb_command` available. Fitting uses PyTorch; select an available device below. Full-cohort arrays and decoder training require substantial RAM and, when using CUDA, GPU memory. For plotting already generated inputs, `requirements-figures.txt` is sufficient.

### Convert cortical and ICA inputs

Replace paths in these commands with your download locations:

```bash
python workflows/prepare_inputs/prepare_hcp.py \
  --subjects /path/to/subjects.txt \
  --ica-pattern '/path/to/ICA50/{subject}.txt' \
  --cortical-kind cifti \
  --cortical-pattern '/path/to/HCP/{subject}/MNINonLinear/Results/rfMRI_{run}/rfMRI_{run}_Atlas_hp2000_clean.dtseries.nii' \
  --templates /path/to/templates \
  --output /path/to/new-inputs

python workflows/prepare_inputs/prepare_geometry.py \
  --modes-pattern '/path/to/modes_{hemi}.txt' \
  --values-pattern '/path/to/eigenvalues_{hemi}.txt' \
  --mask-pattern '/path/to/mask_{hemi}.txt' \
  --output /path/to/analysis-derivatives/wmy/geometry/Eigenmodes_fs4
```

Actual cortical preprocessing is: surface smoothing with sigma 2.12 mm (volume sigma 0); fifth-order Butterworth 0.01–0.10 Hz filtering, TR 0.72 s, SOS forward/backward with 33-frame odd padding; `ADAP_BARY_AREA` resampling to fsaverage4; then within-run standardization. No pre-filter standardization, additional detrending or added global signal regression is performed. The literature's different standardization order is noted in the filter function. ICA50 time courses are standardized separately within each run, without additional filtering or ICA fitting.

Outputs are ordered participant × `[REST1_LR, REST1_RL, REST2_LR, REST2_RL]`. A completed conversion writes `conversion_settings.txt` with `complete=true`. See [PREPROCESSING.md](docs/PREPROCESSING.md) for required templates and the alternative entry point for existing filtered fsaverage4 metrics.

### Configure paths

Use a new derivative directory and set these variables in the shell used for subsequent commands:

```bash
export HCP_DERIVATIVES=/path/to/analysis-derivatives
export HCP_CORTICAL_ROOT=/path/to/new-inputs/cortical
export HCP_ICA_ROOT=/path/to/new-inputs/ica
export HCP_ICA_SPATIAL_FILE=/path/to/melodic_IC.dscalar.nii
export SUBJECTS_DIR=/path/to/freesurfer/subjects
export HCP_SOURCE_PACKAGE=/path/to/new-figure-sources
export HCP_DEVICE=cuda:0
```

`HCP_DERIVATIVES` contains generated analyses and supplied geometry. `HCP_SOURCE_PACKAGE` will be assembled from analysis outputs in step 4; it is not another HCP download. Alternatively, configure the data roots in `paths.local.json` using [paths.example.json](paths.example.json), and launch an individual program through `run.py --paths paths.local.json --script …`.

## 3. Analyse

REST1_LR fits the CPCA basis and HMMs. REST1_RL selects decoder checkpoints. REST2_LR/RL provides evaluation data from the same participants. HMM labels are matched using training data. This tests transfer between acquisitions, not generalization to unseen participants.

### Fit representations and HMMs

Run from the repository root, in order:

```bash
python workflows/analysis_pipeline/fit_representations.py
python workflows/analysis_pipeline/fit_extended_components.py
python workflows/analysis_pipeline/prepare_spatial.py
python workflows/analysis_pipeline/geometry_analysis.py

for tag in robust robust_seed2 consensus_seed5; do
  python workflows/analysis_pipeline/fit_hmm_robust.py --tag "$tag" --iterations 500
done
for tag in robust consensus_seed3 consensus_seed4 consensus_seed6; do
  python workflows/analysis_pipeline/fit_hmm_reliability.py --tag "$tag" --iterations 500
done

python workflows/analysis_pipeline/train_lower_component_decoders.py \
  --counts 3,5,10,20,30,40,50 --device "$HCP_DEVICE"
python workflows/analysis_pipeline/cpca_reliability.py
python workflows/analysis_pipeline/evaluate_reliability.py
```

Each new HMM fit samples a random seed and records it for resuming that fit. `--seed` is optional. Tags such as `robust_seed2` are file identifiers, not prescribed seed values. There are seven fits above: six for the consensus analysis and an additional reference used in two-fit decoder checkpoint selection. Never create additional fits by copying one model. The iteration limit is a cap; inspect the generated training histories for convergence.

The first 30 complex coordinates are used for state analyses; extended projections also support component-count comparisons. Six-fit consensus coverage and decoding accuracy are measured after fitting. Missing support is an undefined estimate, not a requirement to recreate a historical result.

### Generate figure-specific results

Follow the commands in [ANALYSIS.md](docs/ANALYSIS.md):

| Figure | Analysis |
| --- | --- |
| 1 | Training/test component variance and single-CPC fields |
| 2 | State-associated amplitude/phase, decoding versus CPC count, HMM/CPCA reproducibility |
| 3 | Geometric expansion, activity variance, decoding with 30/100/200/300/400 bilateral geometric coordinates |
| 4 | All 132 possible directed non-self transitions; training-selected CPCs; amplitude, phase and participant-level pre/post statistics |
| Supplementary 1 | Spatial amplitude and phase for CPC1–30 |
| Supplementary 2–3 | ICA50/HMM12 descriptions, CPC–ICA correspondence and state-associated cortical maps |

Not every possible transition must occur. Unobserved transitions or unsupported estimates remain missing rather than being assigned a result. Structural and numerical checks validate dimensions, ordering, probability normalization and internal consistency; no historical accuracy or event count is an acceptance criterion.

### Optional HMM algorithm comparisons

The comparison uses the first 100 participants. Run EM and Variational Bayes with six randomly sampled seeds:

```bash
python workflows/hmm_algorithm_validation_20260921/run_pilot.py
python workflows/hmm_algorithm_validation_20260921/compare_pilot.py
python workflows/hmm_algorithm_validation_20260921/plot_comparison.py
```

For official GLHMM, install `requirements-glhmm.txt`, run six fits using the seed list saved at `workflows/hmm_algorithm_validation_20260921/pilot/seeds.json`, then compare and plot. See the [GLHMM instructions](workflows/glhmm_comparison_20260921/README.md). Matching uses training outputs; held-out agreement measures consistency between methods, not accuracy against known biological states.

## 4. Plot

Assemble the spatial figure sources after completing the shared analyses:

```bash
python workflows/prepare_inputs/prepare_figure_sources.py \
  --derivatives "$HCP_DERIVATIVES" --output "$HCP_SOURCE_PACKAGE"
```

Generate and place each figure's analysis outputs using [ANALYSIS.md](docs/ANALYSIS.md). The required files and their locations are listed in [FIGURE_INPUTS.md](docs/FIGURE_INPUTS.md). Then render:

```bash
python workflows/cpca_figure1/code/make_figure.py --root workflows/cpca_figure1
python workflows/main_results_figures_v4/code/plot_figures.py
python workflows/figure3_geometry_grid/code/make_figure.py
python workflows/figure4_distinct_cpcs_20260920/code/plot_figure4.py
python workflows/supplementary_atlases_20260921/code/make_figures.py
python workflows/supplementary_figures_2_3_20260921/code/make_supplements.py
```

Figures are written beside their workflows. PDF text remains editable. Reuse of an old output directory can mix different fits; use a fresh checkout/output tree for a new analysis. Some fitting scripts execute on import and overwrite score arrays, so run them as programs rather than importing them.

## Verification and license

Input conversion has been checked against stored HCP scan outputs; numerical routines and selected workflow paths have targeted checks. These checks do not constitute a complete new 1,003-participant fit and figure run.

Project-owned code is MIT. The HMM-MAR-derived comparison component retains GPL-3.0. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Data and trained models are not distributed, and code licenses do not grant data access or redistribution rights.
