# License scope and attribution
The root MIT license applies to project-owned release code. It does not relicense third-party work or data.

- `workflows/hmm_algorithm_validation_20260921/` is a separately marked GPL-3.0 comparison component, containing the historical HMM-MAR-derived Python core and its comparison scripts. Its LICENSE is retained from HMM-MAR. It contains our Python implementation of Gaussian HMM inference using Variational Bayes, based on HMM-MAR.
- HMM-MAR: Diego Vidaurre and contributors, https://github.com/OHBA-analysis/HMM-MAR, commit `023905b18be11981c7549bbc6ff6e88bd3aaff1e`. License copy: `third_party/HMM-MAR/LICENSE`.
- Official GLHMM: Diego Vidaurre and contributors, https://github.com/vidaurre/glhmm, version 1.1.2, commit `9a5c59acfc7956eb3b579fc92e54d108708a5d02`. License copy: `third_party/glhmm/LICENSE`. Official code is NOT vendored; install the pinned upstream implementation separately. Do not replace it with our Variational Bayes implementation.
- HCP observations, atlas surfaces, ICA maps, subject information and their derivatives are not licensed under this code license. Obtain data through their original providers and comply with applicable access/use terms. No such arrays are included in this code-only archive.
- Do not place an MIT header on upstream GPL files. Consult these notices when extracting subdirectories.

- The Butterworth filter settings match Bolt et al. (2022), https://doi.org/10.1038/s41593-022-01118-1, and the public BOLD_WAVES `preprocess/norm_filter.py` at commit `96e91ddccdacee4f6a0faa623d28528937fc9313`. Our study standardizes after resampling, whereas the cited code also standardizes before filtering. This release uses our conversion implementation and SciPy routines; the upstream repository and its anatomical templates are not redistributed.
