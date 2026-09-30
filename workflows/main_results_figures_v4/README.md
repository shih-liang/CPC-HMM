# Main Fig. 2

Needs the complete source_data folder containing state profiles, accuracy, confusion and reproducibility summaries.

Panel a, "Network state fitting example", compares HMM posteriors and CPC decoder outputs over the same interval. Its time axis is below the CPC heatmap; no agreement strip is displayed.

From the repository root:

```bash
python workflows/main_results_figures_v4/code/plot_figures.py
```

Use prepared inputs in a disposable copy. The retained helper modules support only this current figure or supplement layout; they are not separate figure entrypoints. See the [root README](../../README.md) and [input guide](../../docs/FIGURE_INPUTS.md).
