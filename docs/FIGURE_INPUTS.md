# Required figure inputs

Generate inputs using the programs and placement instructions in [ANALYSIS.md](ANALYSIS.md), then run the renderers. Subject order, masks and state matching must remain consistent throughout.

## Main Fig. 1

Directory: `workflows/cpca_figure1/`. Individual CPC1–6; needs source_data plus the basis, scores, geometry and original example run under the configured data roots.

## Main Fig. 2

Directory: `workflows/main_results_figures_v4/`. Needs the complete source_data folder containing state profiles, accuracy, confusion and reproducibility summaries.

## Main Fig. 3

Directory: `workflows/figure3_wave_transitions/`. Run its `prepare_inputs.py` on the CPC scores and HMM posteriors to generate all inputs in one new `source_data/` directory:

- `transition_summary.npz`: HMM transition matrix, selected CPCs, pre/post means and significance.
- `phase_conditioned_probability.npz`: observed probability within phase bins.
- `examples.npz` and `individual_event_selection.csv`: three individual trajectories and actual event identities.
- `displacement_plot_inputs.npz` and `violin_statistics.csv`: participant-weighted densities, means, quartiles and whiskers.
- `transition_coverage.csv`: training/evaluation counts and any selection fallback.

The renderer reads only these locally generated files. No old transition-profile or cumulative-distribution directory is needed.

## Main Fig. 4

Directory: `workflows/figure4_geometry/`. Needs source_data containing spatial geometry, incremental energy, activity statistics and five-count decoding tables.

## Supplementary Fig. 1

Directory: `workflows/supplementary_atlases_20260921/`. Needs source_data/cpca_basis_REST1_LR.npz and source_data/surfaces_and_eigenmodes.npz.

## Supplementary Figs. 2–3

Directory: `workflows/supplementary_figures_2_3_20260921/`. Needs the complete source_data bundle: ICA/surface files, state maps, HMM parameters, correspondence arrays and fixed examples.

## Data roots

- HCP_DERIVATIVES: existing analysis derivative tree, needed by Figure 1.
- HCP_CORTICAL_ROOT: cortical BOLD arrays prepared from HCP-downloaded ICA-FIX-denoised resting-state fMRI, needed by Figure 1.
- HCP_SOURCE_PACKAGE: prepared directory assembled from analysis outputs, not an HCP-provided package. Figure 4 and Supplement 1 use it when local source copies are absent; Figure 1 preparation accepts it through `--source`. The assembler is `workflows/prepare_inputs/prepare_figure_sources.py`; its inputs come from `analysis_pipeline` as described in ANALYSIS.md.
- HCP_ICA_ROOT: directory containing X_ICA50_zscore.npy for HMM fitting.

The HMM comparison reshapes ICA data as (4012, 1200, 50), with scan order participant × [REST1_LR, REST1_RL, REST2_LR, REST2_RL]. The first 100 participants form the recorded comparison. No HCP arrays or example subject traces are shipped here.
