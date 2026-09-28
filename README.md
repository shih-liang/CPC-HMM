# CPC–HMM: travelling waves and resting-state network states

Analysis and plotting code for studying the correspondence between complex principal components (CPCs), ICA networks and HMM states in resting-state fMRI. The repository includes analysis programs used to generate the inputs for main Figures 1–4 and Supplementary Figures 1–3, together with comparisons of EM, Variational Bayes and official GLHMM implementations.

## Workflow

**HCP ICA-FIX CIFTI and official ICA50 time series → input conversion → CPCA/HMM fitting → decoding and statistical analysis → figure inputs → figures.**

| Analysis | Programs | Outputs used by |
| --- | --- | --- |
| HCP CIFTI/ICA50 conversion and figure-source assembly | [prepare_inputs](workflows/prepare_inputs/) | Shared analysis inputs and spatial figure sources |
| CPCA fitting, projection, spatial preparation and reproducibility | [analysis_pipeline](workflows/analysis_pipeline/) | Figures 1–3; Supplementary Figure 1 |
| HMM fitting, state matching and CPC-count decoder training | [analysis_pipeline](workflows/analysis_pipeline/) | Figures 2–4; Supplementary Figures 2–3 |
| State-associated amplitude/phase, decoding accuracy and reproducibility summaries | [main_results_figures_v4](workflows/main_results_figures_v4/) | Figure 2 |
| Geometric expansion, retained variance and geometric-coordinate decoding | [figure3_geometry_grid](workflows/figure3_geometry_grid/) | Figure 3 |
| All 132 directed state transitions, phase profiles and paired pre/post statistics | [figure4_distinct_cpcs_20260920](workflows/figure4_distinct_cpcs_20260920/) | Figure 4 |
| CPC–ICA correspondence and observed/reconstructed state maps | [cpc_correspondence_20260921](workflows/cpc_correspondence_20260921/) | Supplementary Figures 2–3 |

Start with [analysis order, commands and output placement](docs/ANALYSIS.md). The [figure input reference](docs/FIGURE_INPUTS.md) lists the files expected by each renderer. The `analysis_pipeline` directory contains our study’s CPCA, Gaussian EM HMM and decoder analyses. Dated data directories preserve the study’s input/output conventions.

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

Copy `paths.example.json` to `paths.local.json`, remove entries not needed for your workflow, and replace all remaining placeholders with absolute paths:

| Setting | Directory contents |
| --- | --- |
| `HCP_DERIVATIVES` | Root for derived analysis outputs and supplied geometric eigenmodes; contains `wave_rsn_revision_20260906/` and `wmy/geometry/Eigenmodes_fs4/` |
| `HCP_CORTICAL_ROOT` | Prepared cortical BOLD array at `input_hmm_order/group_fs4_concat_z_hmm_order.npy` |
| `HCP_ICA_ROOT` | Prepared HCP group-ICA50 time courses at `X_ICA50_zscore.npy` |
| `HCP_SOURCE_PACKAGE` | Prepared figure-source directory containing the CPCA spatial basis, cortical surfaces, geometric eigenmodes and analysis summaries; layout below |

### Preparing downloaded data

The repository provides the conversion programs in `workflows/prepare_inputs/`:

- `prepare_hcp.py` converts downloaded ICA-FIX CIFTI scans to cortical arrays and official HCP ICA50 text files to within-run standardized time courses. Both use one explicit subject list and the same four-run order. It can also assemble existing filtered fsaverage4 GIFTI files.
- `prepare_geometry.py` converts supplied fsaverage4 geometric eigenmodes and binary cortical masks into the indexed arrays read by the analyses. It does not estimate geometric eigenmodes from BOLD data.
- `prepare_figure_sources.py` assembles the spatial arrays and summaries produced by the analyses into `HCP_SOURCE_PACKAGE`.

CIFTI conversion uses Connectome Workbench for cortical smoothing (sigma 2.12 mm; volume sigma 0) and `ADAP_BARY_AREA` resampling. Temporal preprocessing applies a fifth-order 0.01–0.10 Hz Butterworth filter at TR 0.72 s, using SOS forward/backward filtering with 33-frame odd padding and no added detrending or global signal regression. Standardization is performed after resampling, within each run. See [input conversion commands and verification](docs/PREPROCESSING.md).

Obtain HCP data and anatomical templates under their providers’ access terms. The conversion code is included; the scans, templates and geometric mode arrays are not bundled. The conversion parameters were checked against archived scan outputs; the difference from the literature’s pre-filter standardization is noted in the filter function.

### Preparing the figure-source directory

`HCP_SOURCE_PACKAGE` is a directory assembled from our analysis outputs, **not an HCP download or an additional official dataset**. Let `BASE = HCP_DERIVATIVES/wave_rsn_revision_20260906` and `REV = BASE/revision_20260910`. After the generating analyses finish, assemble the directory with:

```bash
python workflows/prepare_inputs/prepare_figure_sources.py --derivatives /path/to/analysis-derivatives --output /path/to/prepared-figure-sources
```

The program copies the following files:

| Destination relative to `HCP_SOURCE_PACKAGE` | Generated file | Generating script in `workflows/analysis_pipeline/` |
| --- | --- | --- |
| `figure_data/cpca_basis_REST1_LR.npz` | `REV/data/cpca_basis_REST1_LR.npz` | `cpca_reliability.py`, using the extended REST1_LR basis |
| `figure_data/surfaces_and_eigenmodes.npz` | `BASE/data/surfaces_and_eigenmodes.npz` | `prepare_spatial.py`, packaging supplied surfaces and eigenmodes |
| `figure_data/geometry_incremental_energy.npz` | `BASE/results/geometry_incremental_energy.npz` | `geometry_analysis.py` |
| `source_data/Figure2_geometry_summary.csv` | `BASE/results/Figure2_geometry_summary.csv` | `geometry_analysis.py`; filename retained although the current figure is Figure 3 |
| `source_data/cpca_*.csv` | `REV/results/cpca_*.csv` | `cpca_reliability.py` |
| `source_data/original_cpca_component_properties.csv` | `BASE/results/cpca_component_properties.csv` | `geometry_analysis.py`; rename when copying |

Figure 3 and Supplementary Figure 1 can use their workflow-local `source_data/` files instead. Figure 1’s preparation script accepts this directory through `--source`; its other analysis inputs remain under the configured derivative roots. Follow [ANALYSIS.md](docs/ANALYSIS.md) for placement and execution order.

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

The release includes analysis and plotting programs, not a bundled dataset. Input conversion starts from HCP-preprocessed downloads; it does not rerun HCP ICA-FIX denoising. Some analyses retain the original cohort dimensions and replay assertions for the recorded fits. Replacing those fits can change consensus masks, event counts and results; see [ANALYSIS.md](docs/ANALYSIS.md).

Syntax, style and selected numerical routines were checked during code preparation. The full 1,003-participant pipeline was not refitted during packaging.

## License

Project-owned code is MIT. The HMM-MAR-derived comparison component retains GPL-3.0. Attribution and dependency licenses are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Code licenses do not grant access or redistribution rights to the study data.
