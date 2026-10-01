# EM and Variational Bayes Gaussian HMM comparison

This GPL-3.0 component contains our Python implementation of Gaussian HMM inference using Variational Bayes, based on HMM-MAR, and the matched EM comparison. The numerical and fit settings are unchanged.

## Scripts

- gaussian_vb_port.py: Gaussian prior/update and sequence inference functions.
- run_pilot.py: fit EM/VB to the first 100 participants, six seeds each; needs HCP_ICA_ROOT/X_ICA50_zscore.npy.
- compare_pilot.py: training-derived state matching and held-out comparisons.
- plot_comparison.py: render comparison summaries.

The fit driver reads METHODS (default em,vb) and SEEDS (optional comma-separated integers; by default six random seeds saved in `pilot/seeds.json`) from the environment. It writes pilot/ next to itself. Run in a fresh copy.

Both methods use 12 states and full covariance. They start from the same training-scan segments for each seed: 12 scan indices are sampled with replacement, and zero-based frames 100–199 of each selected scan initialize a state mean. This sampling can give two states the same initial mean. The common covariance initializer adds 0.05 to the diagonal; subsequent EM covariance updates add 0.01. Fitting stops when the mean absolute posterior change is below `1e-5` after at least 50 iterations, or at 500 iterations. The saved `converged` field distinguishes these outcomes.

After fitting, from the repository root:

```bash
python workflows/hmm_algorithm_validation_20260921/compare_pilot.py
python workflows/hmm_algorithm_validation_20260921/plot_comparison.py
```

The official GLHMM comparison consumes this directory's pilot/ outputs. See the [GLHMM README](../glhmm_comparison_20260921/README.md).
