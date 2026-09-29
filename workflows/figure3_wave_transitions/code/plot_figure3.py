"""Render the six-panel Figure 3 from locally generated analysis inputs."""

import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--source", type=Path, default=Path(__file__).resolve().parents[1] / "source_data"
)
parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1])
args = parser.parse_args()
ROOT = args.output
ROOT.mkdir(parents=True, exist_ok=True)
source_dir = args.source
data = np.load(source_dir / "transition_summary.npz")
paired = data
ex = np.load(source_dir / "examples.npz")
with (source_dir / "individual_event_selection.csv").open() as handle:
    examples = list(csv.DictReader(handle))
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
W, H = 180, 164
fig = plt.figure(figsize=(W / 25.4, H / 25.4))


def ax(x, y, w, h):
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])


def text(x, y, s, **kw):
    return fig.text(x / W, 1 - y / H, s, va="top", **kw)


def heading(x, y, letter, title):
    text(x, y, letter, weight="bold", fontsize=10)
    text(x + 5, y + 0.4, title, weight="bold", fontsize=8)


heading(4, 3, "a", "State transitions")
aa = ax(11, 14, 32, 39)
p = data["transition_probability"] * 100
assert p.shape == (12, 12)
offdiag = p[~np.eye(12, dtype=bool)]
upper = float(np.nanmax(offdiag)) if np.isfinite(offdiag).any() else 1.0
im = aa.imshow(
    np.ma.masked_invalid(np.where(np.eye(12, dtype=bool), np.nan, p)),
    cmap="YlOrRd",
    vmin=0,
    vmax=max(upper, 0.01),
    aspect="auto",
)
aa.set(
    xticks=[0, 3, 7, 11],
    xticklabels=[1, 4, 8, 12],
    yticks=[0, 3, 7, 11],
    yticklabels=[1, 4, 8, 12],
    xlabel="Target state",
    ylabel="Source state",
)
cb = fig.colorbar(im, cax=ax(11, 64, 32, 1.7), orientation="horizontal")
cb.set_label("Next-frame probability (%)", labelpad=1, fontsize=6)
cb.ax.tick_params(length=2, pad=1)

heading(50, 3, "b", "Individual wave trajectories")
blue, orange = "#236e91", "#c96e3d"
assert ex["coefficients"].shape == (3, 17, 30) and ex["labels"].shape == (3, 17)
for j in range(3):
    a = ax(57 + j * 26, 15, 19, 36)
    if not ex["valid"][j]:
        a.text(
            0.5,
            0.5,
            "No transitions\nin example run",
            ha="center",
            va="center",
            transform=a.transAxes,
            fontsize=6,
        )
        a.set_axis_off()
        continue
    z = ex["coefficients"][j]
    lab = ex["labels"][j]
    v = z[:, j]
    source, target = int(examples[j]["source"]), int(examples[j]["target"])
    assert source != target and lab[7] == source and lab[8] == target
    a.plot(v.real, v.imag, color="#777777", lw=0.7, zorder=0)
    for value, color in [(source, blue), (target, orange)]:
        mask = lab == value
        a.scatter(v.real[mask], v.imag[mask], s=12, color=color, zorder=2)
    other = (lab != source) & (lab != target)
    a.scatter(
        v.real[other], v.imag[other], s=10, facecolor="white", edgecolor="black", linewidth=0.5
    )
    a.plot(v[7:9].real, v[7:9].imag, color="black", lw=1.4)
    a.scatter(v[8].real, v[8].imag, s=36, marker="*", color="black", zorder=4)
    a.set(title=f"{source} → {target} · CPC{j + 1}")
    if j == 0:
        a.set_ylabel("Imaginary part", fontsize=6)
    a.yaxis.labelpad = 0
    a.set_aspect("equal", adjustable="datalim")
    a.locator_params(axis="both", nbins=3)
handles = [
    Line2D([], [], marker="o", ls="", color=blue, label="Source state"),
    Line2D([], [], marker="o", ls="", color=orange, label="Target state"),
    Line2D([], [], marker="o", ls="", markerfacecolor="white", color="black", label="Other states"),
    Line2D([], [], marker="*", ls="", color="black", label="State transition"),
]
text(92, 56, "Real part", ha="center", fontsize=6.5)
fig.legend(
    handles=handles,
    loc="center",
    bbox_to_anchor=(92 / W, 1 - 65 / H),
    frameon=False,
    ncol=2,
    fontsize=6,
    handletextpad=0.3,
    columnspacing=0.9,
    markerscale=0.7,
)

heading(136, 3, "c", "Changes in CPC\ncoefficients")
inputs = np.load(source_dir / "displacement_plot_inputs.npz")
with (source_dir / "violin_statistics.csv").open() as handle:
    rows = {row["condition"]: row for row in csv.DictReader(handle)}
violin_stats = [
    {key: float(value) for key, value in rows[name].items() if key != "condition"}
    for name in ["nontransition", "transition"]
]
means = np.array([row["mean"] for row in violin_stats])
intervals = np.array([row["intervals"] for row in violin_stats])
# Each participant contributes a unit-area density in each condition.
# Exact histograms include every eligible edge; common Gaussian smoothing SD 0.30 a.u.
edges = inputs["bin_edges"]
centres = (edges[:-1] + edges[1:]) / 2
histograms = inputs["participant_histograms"]
assert histograms.ndim == 3 and histograms.shape[1] == 2
np.testing.assert_array_equal(histograms.sum(axis=(0, 2)), intervals)
kernel = np.exp(-0.5 * (np.arange(-7, 8) / 1.5) ** 2)
kernel /= kernel.sum()
densities = []
for j in range(2):
    counts = histograms[:, j].sum(1)
    valid = counts > 0
    density = np.zeros(len(centres))
    if valid.any():
        density = np.convolve(
            (histograms[valid, j] / counts[valid, None]).mean(0) / 0.2, kernel, mode="same"
        )
        density /= np.trapezoid(density, centres)
    densities.append(density)
densities = np.array(densities)
np.testing.assert_allclose(np.trapezoid(densities, centres, axis=1), intervals > 0)
a = ax(143, 16, 32, 38)
width = 0.36 * densities / densities.max()
boxes = []
positions = []
for j, color in enumerate([blue, orange]):
    row = violin_stats[j]
    if row["intervals"] == 0:
        continue
    inside = (centres > row["minimum"]) & (centres < row["maximum"])
    observed = np.r_[row["minimum"], centres[inside], row["maximum"]]
    body = np.r_[0, width[j, inside], 0]
    a.fill_betweenx(observed, j - body, j + body, color=color, alpha=0.45, lw=0.8, edgecolor=color)
    boxes.append(
        dict(
            q1=row["q25"],
            med=row["median"],
            q3=row["q75"],
            whislo=row["whisker_low"],
            whishi=row["whisker_high"],
            fliers=[],
        )
    )
    positions.append(j)
    # Offset the mean horizontally so that it cannot obscure the median line.
    a.plot(
        j + 0.20,
        means[j],
        marker="D",
        ls="",
        color="black",
        ms=4,
        markeredgecolor="white",
        markeredgewidth=0.5,
        zorder=5,
    )
a.bxp(
    boxes,
    positions=positions,
    widths=0.16,
    showfliers=False,
    showmeans=False,
    patch_artist=True,
    manage_ticks=False,
    zorder=3,
    boxprops=dict(facecolor="white", edgecolor="black", linewidth=0.8),
    medianprops=dict(color="black", linewidth=1.2),
    whiskerprops=dict(color="black", linewidth=0.8),
    capprops=dict(color="black", linewidth=0.8),
)
ymax = max(10.0, 10 * np.ceil(float(np.nanmax([row["maximum"] for row in violin_stats])) / 10))
a.set(
    xticks=[0, 1],
    xticklabels=["No state\ntransition", "State\ntransition"],
    xlim=(-0.55, 1.55),
    ylim=(0, ymax),
    ylabel="Difference between frames (a.u.)",
)
a.locator_params(axis="y", nbins=5)
a.tick_params(axis="x", length=0, pad=3, labelsize=6)
a.set_ylabel("Difference between frames (a.u.)", fontsize=6, labelpad=1)
violin_handles = [
    Line2D([], [], marker="D", ls="", color="black", markersize=3, label="Mean"),
    Line2D([], [], color="black", lw=1.2, label="Median"),
    Patch(facecolor="white", edgecolor="black", linewidth=0.8, label="25–75%"),
]
fig.legend(
    handles=violin_handles,
    loc="center",
    bbox_to_anchor=(159 / W, 1 - 66 / H),
    frameon=False,
    ncol=1,
    fontsize=5.5,
    handlelength=1.2,
    handletextpad=0.4,
    labelspacing=0.3,
)

keys = paired["transition_keys"]
assert len(keys) == 132
prob = np.load(source_dir / "phase_conditioned_probability.npz")
assert np.array_equal(keys, prob["transition_keys"])
assert np.array_equal(paired["CPC"], prob["CPC"])
for j, (values, cmap, norm, title, unit, significant) in enumerate(
    [
        (
            paired["amplitude_mean"],
            "YlOrRd",
            Normalize(0.75, 1.85),
            "Amplitude",
            "Amplitude / run mean",
            paired["amplitude_significant"],
        ),
        (
            paired["phase_mean_degrees"],
            "twilight_shifted",
            Normalize(-180, 180),
            "Phase",
            "Phase (°)",
            paired["phase_significant"],
        ),
        (
            prob["probability_percent"],
            "YlOrRd",
            Normalize(0, 25),
            "Transition probability",
            "Probability (%)",
            None,
        ),
    ]
):
    x = 16 + j * 56
    heading(x - 12, 80, "def"[j], title)
    a = ax(x, 94, 47, 47)
    samples = values.shape[1]
    packed = np.full((12, 12, samples), np.nan)
    for n, key in enumerate(keys):
        packed[key // 12, key % 12] = values[n]
    # Keep all 132 cells; estimates without observations remain blank.
    for target in range(12):
        im = a.pcolormesh(
            np.linspace(target + 0.12, target + 0.88, samples + 1),
            np.arange(13),
            packed[:, target, :],
            cmap=cmap,
            norm=norm,
            edgecolors="face",
            linewidth=0.1,
        )
        a.hlines(np.arange(13), target + 0.12, target + 0.88, color="white", lw=0.25)
    if significant is not None:
        for n, key in enumerate(keys):
            if significant[n]:
                rgb = plt.get_cmap(cmap)(norm(values[n, 1]))[:3]
                color = "black" if np.dot(rgb, [0.2126, 0.7152, 0.0722]) > 0.45 else "white"
                a.text(
                    key % 12 + 0.69,
                    key // 12 + 0.62,
                    "*",
                    color=color,
                    fontsize=5.2,
                    ha="center",
                    va="center",
                )
    a.set(
        xlim=(0, 12),
        ylim=(12, 0),
        xticks=np.arange(12) + 0.5,
        xticklabels=np.arange(1, 13),
        yticks=np.arange(12) + 0.5,
        yticklabels=np.arange(1, 13) if j == 0 else [],
    )
    a.tick_params(length=0, pad=1.5, labelsize=5.5)
    a.xaxis.tick_top()
    if j == 0:
        a.set_ylabel("Source state", labelpad=2)
    for spine in a.spines.values():
        spine.set_visible(False)
    text(
        x + 23.5,
        142.5,
        "Before | After" if j < 2 else "Phase: −180° → +180°",
        ha="center",
        fontsize=6,
    )
    cb = fig.colorbar(
        im,
        cax=ax(x, 148, 47, 1.8),
        orientation="horizontal",
        ticks=([0.75, 1, 1.5, 1.85] if j == 0 else [-180, 0, 180] if j == 1 else [0, 10, 20, 25]),
    )
    cb.ax.tick_params(length=1.5, pad=1.4, labelsize=5.5)
    cb.set_label(unit, labelpad=1, fontsize=6)
text(95.5, 87.5, "Target state", ha="center", fontsize=6.5)

for ext in ["pdf", "svg", "png"]:
    fig.savefig(ROOT / f"Figure_3_wave_dynamics_and_network_state_transitions.{ext}", dpi=350)
print(f"Wrote six-panel Figure 3 to {ROOT}")
