# Figure 3 correction: six-HMM unanimous network states

Specified 2026-09-18 before calculating the corrected geometric-decoder accuracies.

## Target and scope

- Use the existing updated six-fit definition from `revision_20260910`, with fit tags `robust_seed2`, `robust`, `consensus_seed3`, `consensus_seed4`, `consensus_seed5`, `consensus_seed6`.
- Reuse the existing state permutations derived only from REST1_LR posterior correlations. After alignment, a frame is included only when all six hard labels are identical. Use that common label as the decoding target. Do not use majority voting, confidence thresholds, the acquisition-fit control, or decoder predictions to determine eligibility.
- Rebuild this mask from the six posterior arrays and require exact equality with the archived `updated_consensus_masks.npz` fits6 mask and scan order. Expected REST2 support: 582,874 of 2,006,000 core frames, or 29.0565304%.
- Use both REST2 acquisitions, frames 100:1100, for all 1,003 participants. All four geometric representations use exactly the same selected frames and labels.
- Reuse saved predictions for geometry15, geometry50, geometry57 and geometry200 per hemisphere (30,100,114,400 bilateral complex coordinates). Model fitting and validation remain unchanged. This matches the existing consensus analysis, which selects the evaluation frames rather than retraining the decoder on consensus frames.
- Plot only geometric decoding results in Figure 3d. CPC30 decoding belongs in Figure 2. Re-evaluate the archived CPC30 predictions only as a reproducibility check of the existing six-fit target definition; do not plot that benchmark.

## Estimands and uncertainty

- Per-run accuracy is the number of correct common-state predictions divided by the number of six-fit unanimous frames in that run. Average the two run accuracies within each participant, then report the participant mean and SD. This is the figure's point and error bar, consistent with the previous Figure 3 display.
- Report pooled correct/selected-frame accuracy separately. With variable agreement coverage, it need not equal the participant mean. Preserve this distinction when reproducing the historical 77.8909% pooled CPC30 result.
- Use 2,000 participant bootstrap resamples with seed 20260918, keeping each participant's two runs together. Provide intervals for both participant-mean and pooled accuracy. Report aggregate state support and per-state recall.
- Require nonzero selected support in every run, as established by the initial mask audit (minimum 76 frames per run). If this check fails, stop and report the discrepancy.
- No frame-level inferential tests. These intervals do not incorporate family dependence or refitting uncertainty. Agreement-conditioned accuracy is not an all-frame estimate.

## Preservation and checks

- Keep all prior models, predictions, HMM fits, masks and figure archives read only. Write new results to `/configure/HCP_DERIVATIVES/wave_geometry_consensus_20260918_01`.
- Require finite normalized predictions, exact scan ordering and mask agreement, the expected consensus support, and reproduction of the archived CPC30 pooled accuracy within 1e-12.
- Reproduce the four geometric all-frame accuracies from saved predictions to confirm they are the same models used in the preceding figure.
- Record input fingerprints before and after evaluation. Only aggregate scalar tables and validation metadata are exported locally; masks, posteriors, predictions and participant metrics remain on the server.
