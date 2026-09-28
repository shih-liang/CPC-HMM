# EM and Variational Bayes Gaussian HMM comparison

This GPL-3.0 component contains our Python implementation of Gaussian HMM inference using Variational Bayes, based on HMM-MAR, and the matched EM comparison. The numerical and fit settings are unchanged.

## Scripts

- gaussian_vb_port.py: Gaussian prior/update and sequence inference functions.
- run_pilot.py: fit EM/VB to the first 100 participants, six seeds each; needs HCP_ICA_ROOT/X_ICA50_zscore.npy.
- compare_pilot.py: training-derived state matching and held-out comparisons.
- plot_comparison.py: render comparison summaries.

The fit driver reads METHODS (default em,vb) and SEEDS (optional comma-separated integers; by default six random seeds saved in `pilot/seeds.json`) from the environment. It writes pilot/ next to itself. Run in a fresh copy.

After fitting, from the repository root:

```bash
python workflows/hmm_algorithm_validation_20260921/compare_pilot.py
python workflows/hmm_algorithm_validation_20260921/plot_comparison.py
```

The official GLHMM comparison consumes this directory's pilot/ outputs. See the [GLHMM README](../glhmm_comparison_20260921/README.md).
