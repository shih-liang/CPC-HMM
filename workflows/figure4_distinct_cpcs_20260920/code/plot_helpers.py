"""Drawing helpers used by the current Figure 4; no historical figure entrypoint."""

from pathlib import Path


import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hcp_main_figure_matplotlib")

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib.patches import FancyArrowPatch, Circle

from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]

BLUE, ORANGE, GRAY, PURPLE, DARK = "#247B94", "#CA7951", "#929AA1", "#7155A3", "#25333D"

STATE = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#B279A2",
    "#E45756",
    "#72B7B2",
    "#1D6996",
    "#EDAD08",
    "#CC503E",
    "#94346E",
    "#0F8554",
    "#6F4070",
]

EXAMPLES = [(1, 9), (7, 1), (11, 5)]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7,
        "axes.labelsize": 6.5,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.55,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "text.color": DARK,
        "axes.labelcolor": DARK,
        "axes.edgecolor": DARK,
        "xtick.color": DARK,
        "ytick.color": DARK,
    }
)


def band(ax, x, mean, sd, color, probability=False, **kwargs):
    low, high = mean - sd, mean + sd
    if probability:
        low = np.maximum(low, 0)
        high = np.minimum(high, 100 if kwargs.pop("percent", False) else 1)
    ax.fill_between(x, low, high, color=color, alpha=0.17, lw=0)
    ax.plot(x, mean, color=color, lw=1.05, **kwargs)


def key_of(i, j):
    return (i - 1) * 12 + j - 1


def phase_curve(data, key):
    k = int(data["chosen"][key])
    canonical = data["absolute_phase_mean"][key, :, k]
    mean = np.unwrap(canonical)
    mean -= 2 * np.pi * np.round((mean[4] - canonical[4]) / (2 * np.pi))
    return np.rad2deg(mean), np.rad2deg(data["absolute_phase_sd"][key, :, k])


def graph(fig, prob):
    ax = fig.add_axes([0.052, 0.690, 0.377, 0.267])
    angles = np.pi / 2 - np.arange(12) * 2 * np.pi / 12
    xy = np.c_[np.cos(angles), np.sin(angles)]
    for i, j in sorted(
        ((i, j) for i in range(12) for j in range(12) if i != j), key=lambda p: prob[p]
    ):
        p = prob[i, j] * 100
        if p <= 0:
            continue
        ax.add_patch(
            FancyArrowPatch(
                xy[i],
                xy[j],
                connectionstyle="arc3,rad=.12",
                arrowstyle="-|>",
                mutation_scale=3.2 + 0.65 * p,
                shrinkA=7.8,
                shrinkB=7.8,
                lw=0.15 + 0.5 * p,
                color=BLUE,
                alpha=0.12 + 0.85 * p / 5,
                zorder=1,
            )
        )
    for i, (x, y) in enumerate(xy):
        ax.add_patch(
            Circle((x, y), 0.095, facecolor=STATE[i], edgecolor="white", lw=0.65, zorder=3)
        )
        ax.text(
            x,
            y,
            str(i + 1),
            ha="center",
            va="center",
            color="white",
            size=6.5,
            weight="bold",
            zorder=4,
        )
    ax.set(xlim=(-1.18, 1.18), ylim=(-1.18, 1.18), aspect="equal")
    ax.axis("off")
    fig.legend(
        handles=[Line2D([], [], color=BLUE, lw=0.15 + 0.5 * p, label=f"{p}%") for p in [1, 3, 5]],
        loc="center",
        bbox_to_anchor=(0.239, 0.674),
        ncol=3,
        frameon=False,
        fontsize=5.9,
        handlelength=1.5,
        columnspacing=1.1,
        title="Edge width: transition probability",
        title_fontsize=6,
    )


def matrix(fig, prob):
    ax = fig.add_axes([0.574, 0.708, 0.314, 0.253])
    values = prob * 100
    values = values.copy()
    np.fill_diagonal(values, np.nan)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("white")
    m = ax.pcolormesh(
        np.arange(13) - 0.5,
        np.arange(13) - 0.5,
        values,
        cmap=cmap,
        vmin=0,
        vmax=5,
        linewidth=0.15,
        edgecolor="white",
        rasterized=False,
        antialiased=False,
    )
    ax.set(
        xlim=(-0.5, 11.5),
        ylim=(11.5, -0.5),
        aspect="equal",
        xticks=np.arange(12),
        yticks=np.arange(12),
        xticklabels=np.arange(1, 13),
        yticklabels=np.arange(1, 13),
        xlabel="Next state",
        ylabel="Current state",
    )
    ax.tick_params(length=0, pad=2, labelsize=5.8)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(m, cax=fig.add_axes([0.915, 0.715, 0.011, 0.235]), ticks=[0, 1, 2, 3, 4, 5])
    cb.set_label("Transition probability (%)", fontsize=6, labelpad=3)
    cb.ax.tick_params(labelsize=5.7, length=2)
    cb.outline.set_linewidth(0.4)
    cb.solids.set_rasterized(False)
    cb.solids.set_edgecolor("face")
