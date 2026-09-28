# Required figure inputs

Run the renderers on copies of the existing figure-source bundles. This release does not recompute full-cohort summaries. Subject order, masks, state matching and array content must remain those used for the current figures.

## Main Fig. 1

Directory: `workflows/cpca_figure1/`. Individual CPC1–6; needs source_data plus the basis, scores, geometry and original example run under the configured data roots.

## Main Fig. 2

Directory: `workflows/main_results_figures_v4/`. Needs the complete source_data folder containing state profiles, accuracy, confusion and reproducibility summaries.

## Main Fig. 3

Directory: `workflows/figure3_geometry_grid/`. Needs source_data containing spatial geometry, incremental energy, activity statistics and five-count decoding tables.

## Main Fig. 4

Directory: `workflows/figure4_distinct_cpcs_20260920/`. Needs results/all_transition_profiles.npz, results/absolute_phase_profiles.npz and results/prepost_summary.npz. All three now belong in this one directory.

## Supplementary Fig. 1

Directory: `workflows/supplementary_atlases_20260921/`. Needs source_data/cpca_basis_REST1_LR.npz and source_data/surfaces_and_eigenmodes.npz.

## Supplementary Figs. 2–3

Directory: `workflows/supplementary_figures_2_3_20260921/`. Needs the complete source_data bundle: ICA/surface files, state maps, HMM parameters, correspondence arrays and fixed examples.

## Data roots

- HCP_DERIVATIVES: existing analysis derivative tree, needed by Figure 1.
- HCP_CORTICAL_ROOT: preprocessed cortical BOLD root, needed by Figure 1.
- HCP_SOURCE_PACKAGE: optional existing spatial/figure source package used by Figures 3 and Supplement 1 when local source copies are absent.
- HCP_ICA_ROOT: directory containing X_ICA50_zscore.npy for HMM fitting.

The HMM comparison reshapes ICA data as (4012, 1200, 50), with scan order participant × [REST1_LR, REST1_RL, REST2_LR, REST2_RL]. The first 100 participants form the recorded comparison. No HCP arrays or example subject traces are shipped here.

Figure 4's numerical summaries were previously located in a sibling phase-reference directory. Place the two NPZ summaries listed above directly in the current Figure 4 results folder; the obsolete sibling directory is no longer required.
