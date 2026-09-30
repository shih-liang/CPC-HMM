# Main Fig. 1

Individual CPC1–6; needs source_data plus the basis, scores, geometry and original example run under the configured data roots.

Panel d shows only held-out REST2 estimates: cumulative variance on the left and individual component contributions on the right. Both display participant means ± SD after averaging the two runs within each participant. Acquisition names are specified in the figure legend rather than repeated inside panel d. Row percentages in panels a-c continue to describe the fitted training modes.

From the repository root:

```bash
python workflows/cpca_figure1/code/make_figure.py --root workflows/cpca_figure1
```

Use prepared inputs in a disposable copy. The retained helper modules support only this current figure or supplement layout; they are not separate figure entrypoints. See the [root README](../../README.md) and [input guide](../../docs/FIGURE_INPUTS.md).
