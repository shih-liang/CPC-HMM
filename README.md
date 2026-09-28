# CPC–HMM: travelling waves and resting-state network states

Analysis and plotting code for studying the correspondence between complex principal components (CPCs), ICA networks and HMM states in resting-state fMRI. The repository includes the programs that generate the inputs for main Figures 1–4 and Supplementary Figures 1–3, together with comparisons of EM, Variational Bayes and official GLHMM implementations.

## Workflow

**Preprocessed cortical BOLD and ICA time series → CPCA/HMM fitting → decoding and statistical analysis → figure inputs → figures.**

| Analysis | Programs | Outputs used by |
| --- | --- | --- |
| CPCA fitting, projection, spatial preparation and reproducibility | [original_pipeline](workflows/original_pipeline/) | Figures 1–3; Supplementary Figure 1 |
| HMM fitting, state matching and CPC-count decoder training | [original_pipeline](workflows/original_pipeline/) | Figures 2–4; Supplementary Figures 2–3 |
| State-associated amplitude/phase, decoding accuracy and reproducibility summaries | [main_results_figures_v4](workflows/main_results_figures_v4/) | Figure 2 |
| Geometric expansion, retained variance and geometric-coordinate decoding | [figure3_geometry_grid](workflows/figure3_geometry_grid/) | Figure 3 |
| All 132 directed state transitions, phase profiles and paired pre/post statistics | [figure4_distinct_cpcs_20260920](workflows/figure4_distinct_cpcs_20260920/) | Figure 4 |
| CPC–ICA correspondence and observed/reconstructed state maps | [cpc_correspondence_20260921](workflows/cpc_correspondence_20260921/) | Supplementary Figures 2–3 |

Start with [analysis order, commands and output placement](docs/ANALYSIS.md). The [figure input reference](docs/FIGURE_INPUTS.md) lists the files expected by each renderer. Dated directory names preserve the study's input/output conventions.

## Installation

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
```

For plotting existing figure inputs only, install `requirements-figures.txt` instead. For the official GLHMM comparison, additionally install:

```bash
python -m pip install -r requirements-glhmm.txt
```

The recorded packaging runtime was Python 3.14.7; this is not a tested compatibility matrix for every fitting dependency. CPCA and decoder fitting use PyTorch; GPU requirements and device options are described in [ANALYSIS.md](docs/ANALYSIS.md).

## Data and configuration

The study uses **ICA-FIX-denoised resting-state fMRI data downloaded from the Human Connectome Project (HCP)**. Cortical BOLD arrays are prepared from these data. The ICA50 spatial maps and corresponding time courses come from the HCP-provided 50-component group-ICA release (HCP1200 MSMAll); this study does not refit ICA50. The HMM scripts read the supplied within-scan standardized time courses from the local array `X_ICA50_zscore.npy`. HCP preprocessing and ICA-FIX denoising are not rerun by this repository. Time series, trained weights, spatial source files and prepared figure-data bundles are not included.

The study uses 1,003 participants, four acquisitions per participant and 1,200 frames per acquisition. Scan order is participant × `[REST1_LR, REST1_RL, REST2_LR, REST2_RL]`. Training uses REST1_LR, decoder checkpoint selection uses REST1_RL, and evaluation uses REST2_LR/RL of the same participants.

Copy `paths.example.json` to `paths.local.json` and configure the roots needed for your workflow:

| Setting | Input |
| --- | --- |
| `HCP_DERIVATIVES` | Analysis derivative tree containing the study's named data/result directories |
| `HCP_CORTICAL_ROOT` | Preprocessed cortical BOLD arrays |
| `HCP_ICA_ROOT` | Directory containing `X_ICA50_zscore.npy` |
| `HCP_SOURCE_PACKAGE` | Spatial and summary source bundle, when used by a renderer |

Spatial extraction also requires the `HCP_ICA_SPATIAL_FILE` environment variable. Array shapes, spatial inputs and model-version conventions are detailed in [ANALYSIS.md](docs/ANALYSIS.md).

The launcher applies a path configuration to one script:

```bash
python run.py --paths paths.local.json --script workflows/main_results_figures_v4/code/aggregate.py -- --output /path/to/new-fig2-analysis
```

It does not execute the whole pipeline. Use fresh analysis output directories: some fitting scripts write large arrays and execute on import. Follow the documented stage order before rendering.

## Render the figures

After generating and placing the required inputs, run these commands from the repository root. Apply the path configuration through `run.py` where required, or set the corresponding environment variables.

| Figure | Command |
| --- | --- |
| Main Figure 1 | `python workflows/cpca_figure1/code/make_figure.py --root workflows/cpca_figure1` |
| Main Figure 2 | `python workflows/main_results_figures_v4/code/plot_figures.py` |
| Main Figure 3 | `python workflows/figure3_geometry_grid/code/make_figure.py` |
| Main Figure 4 | `python workflows/figure4_distinct_cpcs_20260920/code/plot_figure4.py` |
| Supplementary Figure 1 | `python workflows/supplementary_atlases_20260921/code/make_figures.py` |
| Supplementary Figures 2–3 | `python workflows/supplementary_figures_2_3_20260921/code/make_supplements.py` |

Renderers save figures beside their workflows. PDF export retains editable text.

## HMM implementation comparisons

- [EM and Variational Bayes](workflows/hmm_algorithm_validation_20260921/README.md): fitting, state matching, statistical comparison and plotting for the recorded 100-participant subset. Our Variational Bayes implementation provides Gaussian HMM inference in Python based on HMM-MAR.
- [Official GLHMM](workflows/glhmm_comparison_20260921/README.md): fitting with the pinned upstream package and joint three-method comparison. The included configuration uses `model_mean="state"`, full covariance and `model_beta="no"`.

State matching uses training outputs. Held-out label agreement measures consistency between fitted models; it is not accuracy against known biological state labels.

## Reproduction scope

The release includes analysis and plotting programs, not a bundled dataset or a single-command raw-data pipeline. Some analyses retain the original cohort dimensions and replay assertions for the recorded fits. Replacing those fits can change consensus masks, event counts and results; see [ANALYSIS.md](docs/ANALYSIS.md).

Syntax, style and selected numerical routines were checked during code preparation. The full 1,003-participant pipeline was not refitted during packaging.

## License

Project-owned code is MIT. The HMM-MAR-derived comparison component retains GPL-3.0. Attribution and dependency licenses are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Code licenses do not grant access or redistribution rights to the study data.
