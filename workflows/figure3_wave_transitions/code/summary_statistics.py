"""Participant-level pre/post tests and weighted step-length summaries."""

import numpy as np


def mean_valid(values, axis=0):
    """Average finite values without turning an unsupported estimate into zero."""
    valid = np.isfinite(values)
    count = valid.sum(axis=axis)
    return np.divide(
        np.where(valid, values, 0).sum(axis=axis),
        count,
        out=np.full(count.shape, np.nan, dtype=np.result_type(values.dtype, float)),
        where=count > 0,
    )


def unit(values):
    """Return defined complex directions; zero-amplitude phases are missing."""
    return np.divide(
        values, abs(values), out=np.full(values.shape, np.nan + 0j), where=abs(values) > 1e-12
    )


def paired_pvalues(ad, zd, permutations=19999, seed=20260920):
    """Shared participant sign flips for amplitude and complex phase differences.

    Small samples use exact enumeration. These tests use participants, not
    temporally adjacent events, as the resampling unit. They do not model HCP
    family clusters. Unsupported contrasts receive p=1 for joint correction.
    """
    na, nz = np.isfinite(ad).sum(0), np.isfinite(zd).sum(0)
    da, dz = np.nan_to_num(ad), np.nan_to_num(zd)
    observed_a, observed_z = abs(da.sum(0)), abs(dz.sum(0))
    extreme_a, extreme_z = np.zeros(ad.shape[1], int), np.zeros(zd.shape[1], int)
    joined = np.concatenate([da, dz.real, dz.imag], axis=1)
    rng = np.random.default_rng(seed)
    ntypes = ad.shape[1]
    for start in range(0, permutations, 500):
        signs = rng.integers(0, 2, (min(500, permutations - start), len(ad)), dtype=np.int8) * 2 - 1
        sums = signs.astype(float) @ joined
        sa = abs(sums[:, :ntypes])
        sz = np.hypot(sums[:, ntypes : 2 * ntypes], sums[:, 2 * ntypes :])
        extreme_a += (sa >= observed_a[None, :] - 1e-12 * np.maximum(1, observed_a)).sum(0)
        extreme_z += (sz >= observed_z[None, :] - 1e-12 * np.maximum(1, observed_z)).sum(0)
    pa, pz = (extreme_a + 1) / (permutations + 1), (extreme_z + 1) / (permutations + 1)
    for values, counts, pvalues in [(ad, na, pa), (zd, nz, pz)]:
        for q, n in enumerate(counts):
            if n < 2:
                pvalues[q] = 1.0
            elif n <= 14:
                v = values[np.isfinite(values[:, q]), q]
                signs = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1) * 2 - 1
                pvalues[q] = np.mean(abs(signs @ v) >= abs(v.sum()) - 1e-12 * max(1, abs(v.sum())))
    return pa, pz, na, nz


def step_summary(distance, mask, name):
    """Summarize raw steps, giving each supported participant equal total weight."""
    counts = mask.sum(axis=1)
    supported = int(np.count_nonzero(counts))
    row = dict(condition=name, intervals=int(counts.sum()), participants=supported)
    fields = ["mean", "minimum", "q25", "median", "q75", "maximum", "whisker_low", "whisker_high"]
    if not supported:
        return {**row, **dict.fromkeys(fields, np.nan)}
    values = distance[mask]
    per_person = np.divide(1.0, counts, out=np.zeros(len(counts)), where=counts > 0) / supported
    weights = np.repeat(per_person, counts)
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    q1, median, q3 = values[np.searchsorted(cumulative, [0.25, 0.5, 0.75])]
    iqr = q3 - q1
    low = values[np.searchsorted(values, q1 - 1.5 * iqr, side="left")]
    high = values[np.searchsorted(values, q3 + 1.5 * iqr, side="right") - 1]
    return dict(
        **row,
        mean=float(np.sum(values * weights)),
        minimum=float(values[0]),
        q25=float(q1),
        median=float(median),
        q75=float(q3),
        maximum=float(values[-1]),
        whisker_low=float(low),
        whisker_high=float(high),
    )
