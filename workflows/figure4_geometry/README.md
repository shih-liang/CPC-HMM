# Main Fig. 4

Needs source_data containing spatial geometry, incremental energy, activity statistics and five-count decoding tables.

Panel a displays cortical surfaces and representative geometric eigenmodes. Panel b shows individual geometric-mode contributions; panel c shows cumulative reconstruction curves for every CPC. Curve colours identify CPC number; the black dashed line is the equal-weight mean across CPC1-30. Panels d and e show activity variance and state decoding. The five-panel figure is 7.4 × 8.28 inches.

Panels b and c use each CPC's full spatial energy as the denominator. The renderer obtains the cumulative curves by summing the existing QR energy contributions; no additional analysis inputs are required. Counts in b and c denote geometric modes per hemisphere. Counts in d and e denote bilateral complex coefficients per frame.

From the repository root:

```bash
python workflows/figure4_geometry/code/make_figure.py
```

Use prepared inputs in a disposable copy. The retained helper modules support only this current figure or supplement layout; they are not separate figure entrypoints. See the [root README](../../README.md) and [input guide](../../docs/FIGURE_INPUTS.md).
