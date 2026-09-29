# Figure 3: wave dynamics and network-state transitions

This is the six-panel figure: **a**, HMM transition probabilities; **b**, three individual CPC trajectories; **c**, violin plots of changes in CPC coefficients; **d**, amplitude before/after transitions; **e**, circular phase before/after transitions; **f**, transition probability conditional on source state and CPC phase.

## Generate inputs

First generate the fixed-basis CPC scores and HMM posteriors with the shared analysis pipeline. From the repository root:

```bash
python workflows/figure3_wave_transitions/code/prepare_inputs.py \
  --scores "$HCP_DERIVATIVES/wave_rsn_revision_20260906/data/cpca_scores_200.npy" \
  --posterior "$HCP_DERIVATIVES/wave_rsn_revision_20260906/data/hmm_alpha_robust_seed2.npy" \
  --output workflows/figure3_wave_transitions/source_data

python workflows/figure3_wave_transitions/code/plot_figure3.py
```

The source-data directory must be new. Rendering also accepts `--source /path/to/source_data --output /path/to/figures`. It writes PDF, SVG and PNG; PDF fonts are embedded TrueType and SVG text remains editable. No datasets, templates, fitted models or individual traces are downloaded or bundled.

## Definitions

- Inputs are complex scores `(4*N, 1200, >=30)` and HMM posteriors `(4*N, 1200, 12)`, ordered participant × REST1_LR, REST1_RL, REST2_LR, REST2_RL. The first 30 CPCs are used. Non-finite scores, invalid posterior probabilities and inconsistent shapes are rejected.
- REST1_LR selects a CPC for each directed state pair using the standardized mean complex change from two frames before to two frames after transitions. If a pair has no training events, CPC1 is the display fallback, recorded in `transition_coverage.csv`.
- REST2_LR/RL supplies all evaluated intervals. Panel a includes self-transitions in each row's denominator and hides the diagonal.
- Panel b uses the three illustration anchors in `example_anchors.csv`. Indices are zero-based; CPC numbers are one-based. The CPC2 illustration was selected by requiring one label change across its 17-frame window: eight source-state frames followed by nine target-state frames. These are illustrative events, not a representative sample. For changed HMM fits, the nearest actual transition in each anchored run is used and recorded; the original label pattern is not imposed on a new fit. If that run has none, its example is explicitly empty. Use `--examples` with a three-row CSV for a different cohort order.
- Panel c measures the Euclidean distance between consecutive vectors of 30 original-scale complex CPC coefficients: `D(t) = sqrt(sum_k |c_k(t) - c_k(t-1)|^2)`. Each coefficient is the complex temporal coefficient of one CPC; its modulus gives amplitude and its angle gives phase. The distance therefore reflects both amplitude and phase changes across all 30 CPCs. Each interval spans one TR (0.72 s); the two endpoint labels determine whether a state transition occurred. It covers edges from frames 100→101 through 1098→1099 without event matching, standardization or step ratios. Each participant with support has equal weight within each condition. The density uses 0.2-unit bins and common Gaussian smoothing with SD 0.30; means and quartiles use the unsmoothed observations. Boxes span Q1–Q3, lines mark medians, and offset diamonds mark means. Whiskers reach the most extreme observations inside Q1−1.5×IQR and Q3+1.5×IQR; the violin still includes observations outside those whiskers. No confidence intervals or cumulative-distribution panel are drawn.
- Panels d/e average events within run, runs within participant, then participants. Amplitude is divided by the CPC's mean amplitude in that run's core window. Phase uses the fixed CPC basis and circular averages. Complete event windows use lags −5…+5 TR; pre/post averages use −5…−1 and 0…+4 TR (TR=0.72 s).
- Pre/post significance uses shared participant sign flips (19,999 draws, default seed 20260920), exact enumeration for up to 14 supported participants, and joint Benjamini–Yekutieli correction across 264 contrasts. `--seed` controls this resampling, not HMM initialization. These tests do not account for HCP family clusters. Phase-vector differences can reflect direction and/or concentration.
- Panel f estimates the observed next-frame probability within 24 current-frame phase bins. Denominators include staying and switching outcomes. Runs are averaged within participants, then supported participants are averaged. Undefined phases and unsupported bins remain missing. No circularly shifted control is computed.
- All 132 directed non-self pairs retain their cells. Missing estimates stay blank. Counts, selected CPCs, means, quartiles, probabilities and significance are computed from the supplied fit, not prescribed by this code.

`prepare_inputs.py` generates every input used by the renderer, including the consecutive-vector difference summaries and individual event windows. `summary_statistics.py` contains the shared numerical functions; it is not a separate program. There is no dependency on the removed transition renderer or its old intermediate folders.

The distributions describe changes in CPC coefficients obtained from filtered BOLD data sampled at 0.72 s. They do not establish mathematical continuity or a causal effect of state switching.
