# Generate the figure inputs

This repository includes the retained analysis programs as well as the renderers. The study source is ICA-FIX-denoised resting-state fMRI data downloaded from HCP. The programs below use prepared cortical BOLD arrays and the HCP-provided ICA50 time courses, together with cortical surfaces and geometric eigenmodes. ICA50 refers to the HCP1200 MSMAll 50-component group-ICA release; the study does not refit these components. They do not rerun HCP preprocessing or ICA-FIX denoising. Conversion and organization of downloaded inputs are provided in `workflows/prepare_inputs/`; start with [PREPROCESSING.md](PREPROCESSING.md). Geometric eigenmodes are generated from manually supplied local surface templates and functional-support masks by `generate_geometry.py`; optional format conversion is also included. The HMM input `X_ICA50_zscore.npy` contains the supplied within-scan standardized ICA50 time courses; the included converter reproduces the recovered per-run standardization. Neither the time series nor trained models are distributed here.

Install `requirements-analysis.txt`. Set `HCP_DERIVATIVES`, `HCP_CORTICAL_ROOT`, `HCP_ICA_ROOT`, and, for spatial extraction, `HCP_ICA_SPATIAL_FILE`. Set `SUBJECTS_DIR` to your FreeSurfer subjects directory; its default is `/opt/freesurfer/subjects`. The spatial extractor reads `fsaverage4/surf/` there. `HCP_DEVICE` selects the device for CPCA fitting (default `cuda:1`) and reliability (default `cuda:0`); decoder scripts expose `--device`.

## Shared inputs and ordering

The retained study code expects 1,003 participants × four acquisitions × 1,200 frames. Acquisition order within each participant is REST1_LR, REST1_RL, REST2_LR, REST2_RL. Cortical BOLD is `HCP_CORTICAL_ROOT/input_hmm_order/group_fs4_concat_z_hmm_order.npy`, shape `(4814400, 5124)` before vertex exclusion. ICA input is `HCP_ICA_ROOT/X_ICA50_zscore.npy`, reshaped `(4012, 1200, 50)` in the identical order. Geometric inputs are under `HCP_DERIVATIVES/wmy/geometry/Eigenmodes_fs4`.

Below, `BASE` means `HCP_DERIVATIVES/wave_rsn_revision_20260906`; `REV` means `BASE/revision_20260910`. Use a fresh derivative tree with `data/` and `results/` folders under both `BASE` and `REV`. Some original programs execute on import and open writable arrays: execute them as scripts, not as importable libraries. Full-cohort fitting is computationally expensive and is not launched by installation.

REST1_LR fits the basis and decoder; REST1_RL selects decoder checkpoints; REST2_LR/RL evaluates them in the same participants. This is acquisition transfer. The six-HMM agreement mask is an evaluation subset, not a new training target.

## Upstream representations, HMM and decoders

Programs are in `workflows/analysis_pipeline/`. These programs implement the study’s CPCA, Gaussian EM HMM, decoding and reliability analyses:

| Order | Program | Generated inputs |
|---|---|---|
| 1 | `fit_representations.py` | REST1_LR CPCA/PCA basis and projections under `BASE/data` |
| 2 | `fit_extended_components.py` | Extended training covariance, basis and `cpca_scores_200.npy`; the first 30 coefficients are used by current state analyses |
| 3 | `prepare_spatial.py` | `surfaces_and_eigenmodes.npz` from supplied geometry and FreeSurfer surfaces |
| 4 | `geometry_analysis.py` | Per-CPC geometric energy increments and spatial component properties |
| 5 | `fit_hmm_robust.py --seed … --tag … --iterations …` | Original reference HMM posteriors and parameters under `BASE` |
| 6 | `fit_hmm_reliability.py --seed … --tag … --iterations …` | Additional HMM fits under `REV` |
| 7 | `train_lower_component_decoders.py --counts 3,5,10,20,30,40,50 --device cuda:0` | CPC-count checkpoints and held-out predictions; bundled `rank_protocol.json` supplies the recorded training settings |
| 8 | `cpca_reliability.py` | Acquisition-specific CPC bases, overlap, subspace and variance statistics under `REV` |
| 9 | `evaluate_reliability.py` | Training-derived HMM matching, six-fit masks, decoding and reliability statistics under `REV` |

The matching evaluator reads `robust_seed2` and `consensus_seed5` from `BASE`, and `robust`, `consensus_seed3`, `consensus_seed4`, `consensus_seed6` from `REV`. The original `robust` posterior under `BASE` is also required for the decoder's original two-fit validation criterion. Preserve these distinct fit versions when replaying the study. Do not replace all tags with copies of one fit. Each new fit samples and saves a random initialization seed; `--seed` is optional. Use the commands in README to produce all seven required fit files. The tag names identify outputs and do not specify numeric seeds.

The decoder computes the two-fit evaluation agreement mask directly from the posteriors using its training-derived matching. Extended CPCA reads the target label from the bundled `rank_protocol.json` for descriptive metadata only; CPCA fitting does not depend on an HMM target.

Cohort dimensions and input consistency are checked. Accuracy, common-frame coverage, component selection and transition counts are computed from the current fits, with no required historical values.

## Figure 1 and Supplementary Figure 1

`cpca_figure1/code/prepare_sources.py --source SOURCE_PACKAGE --root workflows/cpca_figure1` prepares the group summary inputs. `component_variance.py --root workflows/cpca_figure1` generates participant-level component and cumulative variance summaries used for the SD error bars. Then run `make_figure.py`.

Run `workflows/prepare_inputs/prepare_figure_sources.py --derivatives HCP_DERIVATIVES --output SOURCE_PACKAGE` to assemble the source directory. Its layout consists of `figure_data/cpca_basis_REST1_LR.npz` (from `REV/data`), `figure_data/surfaces_and_eigenmodes.npz` (from `BASE/data`), and `source_data/` containing the `cpca_*` reliability CSVs from `REV/results`. The assembler also copies the component-property and geometric-summary files. Figure 1 reads its basis directly from `REV/data/cpca_basis_REST1_LR.npz`.

Supplementary Figure 1 reads the same basis and surfaces. Set `HCP_SOURCE_PACKAGE` to this source package or place these two NPZ files in its `source_data/` directory.

## Figure 2

Run, with the same new analysis output directory:

```bash
python workflows/main_results_figures_v4/code/aggregate.py --output /path/to/fig2-analysis
python workflows/main_results_figures_v4/code/add_comparisons.py --output /path/to/fig2-analysis
python workflows/main_results_figures_v4/code/prepare_inputs.py --analysis /path/to/fig2-analysis --revision /path/to/REV --derivatives /path/to/BASE --output workflows/main_results_figures_v4/source_data
python workflows/main_results_figures_v4/code/plot_figures.py
```

The first two programs generate state profiles, confusion, component-count accuracy and their participant summaries. `prepare_inputs.py` transfers those outputs, adds the CPC/HMM reproducibility tables and generates the fixed example from scan 2, frames 300–499. It does not select a new example based on accuracy.

## Figure 3

Programs are in `workflows/figure3_geometry_grid/code/`:

1. `geometry_activity.py --source /path/to/BASE --output /path/to/activity` generates covariance sufficient statistics and the retained-variance CSV/JSON files.
2. `direct_decoding.py --stage project --output /path/to/HCP_DERIVATIVES/wave_geometry_decoder_20260918_01/results --device cuda:0` generates native geometric coefficients.
3. Run the same program with `--stage fit --conditions geo15,geo50,geo200` and then `--stage summarize --conditions geo15,geo50,geo200` to generate the initial decoder predictions and summaries. These counts are per hemisphere.
4. For the 100 and 150 modes-per-hemisphere additions, create `HCP_DERIVATIVES/wave_geometry_decoder_grid_20260918_01/results`, with links to `native_geometry_scores.npy` and `projection_validation.json` from the projection output. Run `fit_additional.py --condition geo100 --output … --device cuda:0` and repeat for `geo150`.
5. `evaluate_consensus.py --output /path/to/fig3-evaluation` evaluates all five counts on the same six-fit common frames and writes `export/`.
6. Copy the activity CSV/JSON files and the evaluation `export/` files to `figure3_geometry_grid/source_data/`. Add the basis, surfaces and `BASE/results/geometry_incremental_energy.npz`. Copy the generated `BASE/results/Figure2_geometry_summary.csv` as `original_geometry_summary.csv`. Then run `make_figure.py`.

`evaluate_training.py` retains the training-set evaluation of saved decoders. The five plotted bilateral coordinate counts are 30, 100, 200, 300 and 400. Participant accuracy averages the available run-specific consensus accuracies; a run without consensus frames is undefined. Participants with no supported run are excluded from this summary, with the supported count saved in the output. `GRID_EXTENSION_PROTOCOL.md` records the decoder settings and the common-state evaluation definition.

## Figure 4

All analysis scripts now live beside the current renderer, in `workflows/figure4_distinct_cpcs_20260920/code/`:

```bash
python workflows/figure4_distinct_cpcs_20260920/code/aggregate_all.py --output /path/to/fig4-profiles
python workflows/figure4_distinct_cpcs_20260920/code/aggregate_absolute_phase.py --output /path/to/fig4-profiles
python workflows/figure4_distinct_cpcs_20260920/code/prepare_reference.py --profiles /path/to/fig4-profiles/results/all_transition_profiles.npz --phases /path/to/fig4-profiles/results/absolute_phase_profiles.npz --out /path/to/reference.npz
python workflows/figure4_distinct_cpcs_20260920/code/aggregate_prepost.py --output /path/to/fig4-prepost --reference /path/to/reference.npz
```

Copy `all_transition_profiles.npz`, `absolute_phase_profiles.npz`, and `prepost_summary.npz` from those results directories into the current figure's `results/`, then run `plot_figure4.py`. Separate profile and pre/post output directories avoid completion-file collisions. The calculations cover all 132 directed non-self transitions; CPC selection uses training events. Pre/post significance uses participant-paired permutations with joint multiple-comparison correction.

## Supplementary Figures 2–3

`cpc_correspondence_20260921/code/analyze_ica.py` generates CPC–ICA mapping, functional-connectivity arrays and fixed examples under `HCP_DERIVATIVES/cpc_correspondence_20260921_01`. Copy its `figure_arrays.npz`, `fixed_example.npz` and `complete.json` to `supplementary_figures_2_3_20260921/source_data/`.

Generate the state-map inputs without the unrelated transition-control analysis:

```bash
python workflows/cpc_correspondence_20260921/code/state_maps.py --scores /path/to/BASE/data/cpca_scores_200.npy --posterior /path/to/BASE/data/hmm_alpha_robust_seed2.npy --basis /path/to/REV/data/cpca_basis_REST1_LR.npz --bold /path/to/cortical/input_hmm_order/group_fs4_concat_z_hmm_order.npy --out /path/to/state-maps
```

Copy its three outputs (`state_associated_cortical_maps.npz`, `state_map_by_run.csv`, `archived_state_map_summary.csv`) to the supplement's `source_data/`. The last filename is retained for compatibility; its values are generated from this run. Also copy Figure 2's `state_profiles.npz`, the cortical geometry NPZ, and the reference HMM parameters as `HMM_reference.npz`. Supply the HCP ICA50 spatial map `groupICA_3T_HCP1200_MSMAll_d50.ica/melodic_IC.dscalar.nii` as `melodic_IC.dscalar.nii` and the left/right `S900.*.midthickness_MSMAll.32k_fs_LR.surf.gii` files. Run `make_supplements.py`.

State reconstruction uses `Re(C Uᴴ)`. Maps first average frames within each state/run, then runs within each participant, then participants; spatial correlation and reconstruction R² remain distinct metrics.

## Verification scope

Execution checks cover full-cohort CPCA fitting and coordinate generation, one reference HMM refit, seven CPC decoder fits, five geometric-coordinate decoder fits, transition and correspondence analyses, and all main and supplementary renderers. Six-initialization agreement and optional EM/Variational Bayes/GLHMM comparison statistics were rerun using saved posterior/model files; all of those models were not retrained. Raw ICA-FIX preprocessing was checked on available reference scans rather than the full cohort. Recorded seeds were used for numerical replay; routine runs may use new random seeds. These checks therefore do not establish a completely fresh download-to-all-models run.
