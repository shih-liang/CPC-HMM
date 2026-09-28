"""Plot archived method-comparison summaries; requires generated result tables."""

from pathlib import Path
import json
import csv
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = Path(__file__).parent
R = P / "results"
s = json.loads((R / "comparison.json").read_text())
rows = list(csv.DictReader((R / "pairwise.csv").open()))
plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
fig, ax = plt.subplots(1, 3, figsize=(11, 3.7))
methods = ["em", "vb", "glhmm"]
names = ["Original EM", "Variational Bayes", "Native GLHMM"]
colors = ["#777777", "#bd7a39", "#187e99"]
for i, (m, c) in enumerate(zip(methods, colors)):
    v = [float(r["agreement"]) * 100 for r in rows if r["group"] == m + "–" + m]
    ax[0].scatter(i + np.linspace(-0.12, 0.12, len(v)), v, s=16, color=c, alpha=0.75)
    ax[0].errorbar(
        i, np.mean(v), yerr=np.std(v, ddof=1), fmt="_", markersize=16, capsize=5, color="black"
    )
    u = s["six_fit_unanimity"][m]
    ax[1].bar(i, u["fraction"] * 100, color=c, width=0.6)
    ax[1].text(
        i,
        u["fraction"] * 100 + 1,
        f"{u['fraction'] * 100:.2f}%\n{u['frames']:,} frames",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    ax[2].plot([], [], color=c, label=names[i])
    for meta in sorted(R.glob(m + "_*.json")) if m == "glhmm" else []:
        fe = np.array(json.loads(meta.read_text())["free_energy"])
        ax[2].plot(np.arange(len(fe)) + 1, fe - fe[-1], color=c, alpha=0.55, lw=1)
for a in ax[:2]:
    a.set_xticks(range(3), names, rotation=15)
    a.set_ylim(0, 105)
ax[0].set_ylabel("Matched hard-state agreement (%)")
ax[0].set_title("a  Pairwise reproducibility (15 pairs)", loc="left", fontsize=10)
ax[1].set_ylabel("All six fits agree (%)")
ax[1].set_title("b  Six-fit unanimous frames", loc="left", fontsize=10)
ax[2].set_title("c  Native GLHMM convergence", loc="left", fontsize=10)
ax[2].set_xlabel("Training iteration")
ax[2].set_ylabel("Free energy minus final value")
fig.text(
    0.02,
    0.02,
    "100 participants · 200,000 held-out frames · Training-derived state matching · Bars in a: mean ± SD across fit pairs",
    fontsize=8,
)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(P / "glhmm_comparison.png", dpi=200)
fig.savefig(P / "glhmm_comparison.pdf")
plt.close(fig)
