# Official GLHMM and joint HMM comparisons

Install the pinned official source using requirements-glhmm.txt. This runner uses model_mean="state", full covariance and model_beta="no". GLHMM_SOURCE can point to an explicit checkout; otherwise the installed module is used. The runner records the loaded source hash, expected commit, actual checkout commit (when available), and checkout modification status. An unavailable commit remains null; it is not reported as the pinned version.

With HCP_ICA_ROOT configured, run run_glhmm.py with a single seed from 20260906 through 20260911. It writes results/ next to itself.

After all six official fits and all 12 EM/VB fits exist:

```bash
python workflows/glhmm_comparison_20260921/compare_all.py
python workflows/glhmm_comparison_20260921/plot_results.py
```

compare_all.py reads this directory's results/ and ../hmm_algorithm_validation_20260921/pilot/. These arrays are not included. Matching uses training outputs; evaluation uses the held-out acquisitions of the same 100 participants. Convergence and wall time are not universal method rankings. Use a new copy/output tree for refitting; the package does not start remote jobs.

The comparison accepts new six-seed results without requiring historical consensus counts.
