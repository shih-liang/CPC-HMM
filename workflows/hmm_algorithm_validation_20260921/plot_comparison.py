"""Plot the EM/Variational Bayes comparison from archived result summaries."""

from pathlib import Path
import csv
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = Path(__file__).parent
R = P / "pilot"
s = json.loads((R / "summary.json").read_text())
assert s["complete"], "Do not plot incomplete comparisons"
r = list(csv.DictReader((R / "pairwise.csv").open()))
plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
f, axs = plt.subplots(1, 3, figsize=(11, 3.8), layout="constrained")
for ax, key, title, scale in zip(
    axs,
    ["agreement", "mean_posterior_correlation", "ari"],
    [
        "a  State assignment agreement",
        "b  Posterior time-course correlation",
        "c  Adjusted Rand index",
    ],
    [100, 1, 1],
):
    for j, g in enumerate(["em", "vb", "cross_different", "cross_same"]):
        v = np.array(
            [
                float(z[key]) * scale
                for z in r
                if (
                    z["group"] == g
                    if g in ["em", "vb"]
                    else z["group"] == "cross"
                    and (
                        (z["a"][3:] == z["b"][3:])
                        if g == "cross_same"
                        else z["a"][3:] != z["b"][3:]
                    )
                )
            ]
        )
        offset = np.linspace(-0.16, 0.16, len(v))
        ax.scatter(
            j + offset, v, s=12, alpha=0.7, color=["#317496", "#cc7c30", "#68568d", "#279680"][j]
        )
        ax.errorbar(j, v.mean(), yerr=v.std(ddof=1), fmt="o", color="black", capsize=4, ms=4, lw=1)
    ax.set_xticks(range(4), ["EM–EM", "VB–VB", "EM–VB\ndifferent init.", "EM–VB\nsame init."])
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_ylim(min(0, min(float(z[key]) * scale for z in r) - 0.03 * scale), scale)
    ax.set_ylabel(
        "Agreement (%)"
        if scale == 100
        else ("Correlation" if key.endswith("correlation") else "ARI")
    )
f.suptitle(
    "100 participants · matched initialization · batch EM and Variational Bayes updates",
    fontsize=10,
)
f.savefig(P / "pilot_comparison.pdf")
f.savefig(P / "pilot_comparison.png", dpi=220)
