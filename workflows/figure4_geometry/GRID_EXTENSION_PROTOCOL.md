# Figure 4: 30,100,200,300,400 bilateral geometric coordinates

User-requested extension, specified on 2026-09-18 before fitting the two new conditions. The earlier four-point results are known and remain archived; this is an explicit extension rather than a claim that all five conditions were chosen before those results.

## Conditions and reuse

- Display exactly 30,100,200,300,400 bilateral complex geometric coordinates per frame, corresponding to 15,50,100,150,200 eigenmodes per hemisphere.
- Reuse completed models and predictions for 30,100,400 coordinates. Fit only the missing 200- and 300-coordinate decoders, using the existing native geometric score array. Do not repeat the cortical projection or refit any HMM/CPCA basis.
- Preserve the 114-coordinate result as the prior variance-matched condition in archived source tables. It is not an accuracy point in the new Figure 4d. The variance comparison in panel c remains unchanged.
- CPC30 decoding is not plotted in Figure 4d.

## Fitting rules for the two added conditions

Reuse the exact existing `direct_decoding.py` fitting function. Only the condition-to-column mapping is extended to 100 and 150 modes per hemisphere. Training uses REST1_LR and validation uses REST1_RL; the fixed target posterior is `robust_seed2`. The input is a 50-frame past window with train-derived Cartesian normalization, 800 padded real slots per frame, hidden widths 128/64, ReLU and dropout 0.1. Use the same seed 20260908, AdamW learning-rate/weight-decay grid (0.001/0.0003 crossed with 0.0001/0.001), batch 1024, maximum 30 epochs, early stopping after four epochs without validation-MSE improvement exceeding 1e-5, and checkpoint selection by validation posterior MSE only.

No test-accuracy tuning or repeated seed selection. All requested conditions are retained regardless of accuracy or whether the curve is monotonic. Effective input parameter counts vary with coordinate count, as in the previous comparison.

## Common-state evaluation and statistics

- Use the identical six-HMM unanimous-label definition and training-derived permutations from `CONSENSUS_EVALUATION_PROTOCOL.md`.
- All five conditions must use the same common-state REST2 frames selected from the current six fits, 1,003 participants and two evaluation acquisitions per participant.
- Figure and primary manuscript text use the participant mean: calculate run-specific common-state accuracies, average the two runs within each participant, then summarize across participants. Error bars are between-participant SD.
- Pooled accuracy is a separate secondary estimator, reported explicitly in source tables. It is not substituted for the participant mean in the figure or its accompanying main paragraph.
- Use the existing 2,000 participant-bootstrap resamples, seed 20260918, keeping both runs together. No frame-level inferential tests. Family dependence and model-fitting uncertainty remain outside these intervals.
- Compute all five geometric-coordinate results and the CPC30 reference from the current model outputs. Do not infer an information ceiling or a universal minimum dimension from the sampled points.

## Isolation and checks

Write new models and results only under `/configure/HCP_DERIVATIVES/wave_geometry_decoder_grid_20260918_01`. The existing native score array and projection validation are linked read only into the new results directory. Existing models and source runs remain untouched. Check GPU availability before launch. Record input fingerprints before and after fitting and evaluation, all optimizer histories, checkpoint selections, prediction normalization, frame support and source-code versions. Transfer only aggregate scalar summaries and validation metadata locally; scores, posteriors, predictions, checkpoints and participant metrics remain on the data server.
