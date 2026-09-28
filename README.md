# Current manuscript figures and HMM method comparisons

This code-only release contains the analysis programs that generate the current main Figures 1–4 and Supplementary Figures 1–3, their drawing programs, and the EM / Variational Bayes / official GLHMM comparisons. See [analysis order and output mapping](docs/ANALYSIS.md). Input data and fitted models are supplied separately.

## Install

From the repository root (recorded Python runtime: 3.14.7):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-figures.txt
```

Fitting official GLHMM separately requires `requirements-glhmm.txt` and substantial compute.

## Render the current figures

**The arrays, trained weights and figure-source bundles are not included.** Supply the existing inputs described below in a disposable working copy. Renderers write PDFs/SVGs/PNGs and provenance next to their workflow. The code deliberately retains the historical directory names to preserve input/output conventions; these names do not denote multiple delivered figure versions.

| Figure | Command from repository root |
| --- | --- |
| Main Fig. 1 | `python workflows/cpca_figure1/code/make_figure.py --root workflows/cpca_figure1` |
| Main Fig. 2 | `python workflows/main_results_figures_v4/code/plot_figures.py` |
| Main Fig. 3 | `python workflows/figure3_geometry_grid/code/make_figure.py` |
| Main Fig. 4 | `python workflows/figure4_distinct_cpcs_20260920/code/plot_figure4.py` |
| Supplementary Fig. 1 | `python workflows/supplementary_atlases_20260921/code/make_figures.py` |
| Supplementary Figs. 2–3 | `python workflows/supplementary_figures_2_3_20260921/code/make_supplements.py` |

To generate figure inputs, install `requirements-analysis.txt` and follow [ANALYSIS.md](docs/ANALYSIS.md). Figure 1 requires configured full-data paths; other renderers use generated source bundles. Copy paths.example.json to paths.local.json, fill only needed roots and remove unused placeholders. Example:

```bash
python run.py --paths paths.local.json --script workflows/cpca_figure1/code/make_figure.py -- --root workflows/cpca_figure1
```

Use a fresh output copy; the launcher configures environment variables but does not guarantee overwrite protection. Some HMM scripts execute on import; do not import them merely to inspect their options.

## HMM comparisons

- [EM and Variational Bayes](workflows/hmm_algorithm_validation_20260921/README.md): fit/evaluate the recorded 100-participant comparison.
- [Official GLHMM](workflows/glhmm_comparison_20260921/README.md): pinned upstream implementation, repeated fits and joint three-method comparison. The retained configuration is `model_beta="no"`; it does not claim to include a beta-enabled experiment.

Our Variational Bayes implementation is a Python implementation of Gaussian HMM inference based on HMM-MAR. Label correspondence, repeatability, convergence and runtime are separate outcomes. Preserved accuracy values and figure data are not recalculated by this cleanup.

## Inputs and license

- [Required figure inputs](docs/FIGURE_INPUTS.md)

Use `pip install -r requirements-dev.txt`, `ruff check .` and `ruff format --check .` for maintenance. Project-owned code is MIT; the HMM-MAR-derived comparison folder remains GPL-3.0. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Data are not covered by the code license. Authors should add the confirmed manuscript citation and repository/archive DOI before public deposition.
