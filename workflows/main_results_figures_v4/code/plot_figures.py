"""Publication figures from archived results and verified new aggregations."""

from pathlib import Path
import csv
import hashlib
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hcp_main_figure_matplotlib")
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "source_data"
BLUE = "#247B94"
ORANGE = "#CA7951"
GRAY = "#939AA0"
DARK = "#25333D"
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
PAIRS = [(1, 9, 1), (7, 1, 1), (11, 5, 2), (4, 11, 4), (5, 6, 3), (9, 7, 1)]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7.5,
        "axes.labelsize": 7.2,
        "axes.titlesize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.15,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "text.color": DARK,
        "axes.labelcolor": DARK,
        "axes.edgecolor": DARK,
        "xtick.color": DARK,
        "ytick.color": DARK,
    }
)


def read(name):
    with (DATA / name).open() as f:
        return list(csv.DictReader(f))


def one(rows, **keys):
    result = [r for r in rows if all(str(r[k]) == str(v) for k, v in keys.items())]
    assert len(result) == 1, keys
    return result[0]


def heading(fig, letter, title, x, y, size=8.7):
    fig.text(x, y, letter, fontweight="bold", fontsize=size + 1.3, va="bottom")
    fig.text(x + 0.024, y, title, fontweight="bold", fontsize=size, va="bottom")


def matrix(ax, a, cmap="magma", vmin=0, vmax=1, norm=None):
    a = np.asarray(a)
    ny, nx = a.shape
    kw = dict(cmap=cmap, edgecolors="face", linewidth=0.18, antialiased=False, rasterized=False)
    if norm is None:
        kw.update(vmin=vmin, vmax=vmax)
    else:
        kw["norm"] = norm
    m = ax.pcolormesh(np.arange(nx + 1) - 0.5, np.arange(ny + 1) - 0.5, a, **kw)
    ax.set(xlim=(-0.5, nx - 0.5), ylim=(ny - 0.5, -0.5))
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    return m


def cbar(fig, m, rect, label, ticks=None, horizontal=False):
    ax = fig.add_axes(rect)
    c = fig.colorbar(m, cax=ax, orientation="horizontal" if horizontal else "vertical", ticks=ticks)
    c.set_label(label, fontsize=6.5, labelpad=3)
    c.ax.tick_params(labelsize=6, length=2)
    c.outline.set_linewidth(0.4)
    if c.solids is not None:
        c.solids.set_rasterized(False)
        c.solids.set_edgecolor("face")
        c.solids.set_linewidth(0.18)
    return c


def save(fig, name):
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(ROOT / f"{name}.{ext}", dpi=300)
    # Record text positions for basic clipping checks; visual review remains required.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = []
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text().strip():
            b = text.get_window_extent(renderer)
            boxes.append(dict(text=text.get_text(), bounds=[float(v) for v in b.bounds]))
    (ROOT / "provenance" / f"{name}_text_boxes.json").write_text(
        json.dumps(dict(figure_pixels=list(fig.bbox.size), text=boxes), indent=2)
    )
    plt.close(fig)
    print("SAVED", name, flush=True)


def figure2():
    fig = plt.figure(figsize=(180 / 25.4, 200 / 25.4))
    square_axes = []
    stat = json.loads((DATA / "decoding_summary.json").read_text())
    profiles = np.load(DATA / "state_profiles.npz")
    example = read("Figure3_example_posteriors.csv")
    heading(fig, "a", "Network state fitting example", 0.065, 0.973, size=8)
    example_heading = list(fig.texts)
    example_axes = []
    for idx, stem in enumerate(["HMM", "CPCA30"]):
        ax = fig.add_axes([0.075, 0.878 - idx * 0.063, 0.815, 0.048])
        example_axes.append(ax)
        ax.set_facecolor(plt.get_cmap("magma")(0))
        values = np.array([[float(r[f"{stem}_state{k}"]) for r in example] for k in range(1, 13)])
        m = matrix(ax, values)
        ax.set(yticks=[0, 5, 11], yticklabels=["1", "6", "12"], xticks=[])
        ax.set_ylabel("HMM state" if idx == 0 else "CPC output", labelpad=3, fontsize=6)
        if idx == 1:
            ax.set(
                xticks=[-0.5, 49.5, 99.5, 149.5, 199.5],
                xticklabels=["0", "36", "72", "108", "144"],
            )
            ax.set_xlabel("Time within interval (s)", labelpad=2, fontsize=6.4)
    example_axes.append(cbar(fig, m, [0.909, 0.815, 0.012, 0.111], "Probability", [0, 0.5, 1]).ax)

    heading(fig, "b", "State-associated amplitude of all 30 CPCs", 0.065, 0.725, size=8)
    ax = fig.add_axes([0.075, 0.548, 0.815, 0.155])
    m = matrix(
        ax,
        profiles["six_fit_amplitude"],
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=0.75, vcenter=1, vmax=1.85),
    )
    ax.set(
        yticks=np.arange(12),
        yticklabels=np.arange(1, 13),
        xticks=np.arange(30),
        xticklabels=np.arange(1, 31),
        ylabel="HMM state",
        xlabel="CPC",
    )
    ax.tick_params(labelsize=6)
    ax.xaxis.label.set_size(6.5)
    ax.yaxis.label.set_size(6.5)
    cbar(fig, m, [0.909, 0.548, 0.012, 0.155], "Amplitude / run mean", [0.75, 1, 1.5, 1.85])

    heading(fig, "c", "State-associated phase of all 30 CPCs", 0.065, 0.486, size=8)
    z = profiles["six_fit_phase"]
    xx, yy = np.meshgrid(np.arange(1, 31), np.arange(1, 13))
    ax = fig.add_axes([0.075, 0.324, 0.815, 0.143])
    dots = ax.scatter(
        xx,
        yy,
        c=np.degrees(np.angle(z)),
        s=38 * abs(z),
        cmap="twilight",
        vmin=-180,
        vmax=180,
        linewidths=0,
    )
    ax.set(
        xlim=(0.5, 30.5),
        ylim=(12.6, 0.4),
        xticks=np.arange(1, 31),
        yticks=np.arange(1, 13),
        xlabel="CPC",
        ylabel="HMM state",
    )
    ax.tick_params(labelsize=6, length=0)
    ax.xaxis.label.set_size(6.5)
    ax.yaxis.label.set_size(6.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar(fig, dots, [0.909, 0.324, 0.012, 0.143], "Circular mean phase (°)", [-180, 0, 180])
    handles = [
        Line2D([], [], ls="", marker="o", color=DARK, markersize=np.sqrt(38 * r), label=f"{r:.1f}")
        for r in [0.2, 0.5, 0.8]
    ]
    fig.text(0.665, 0.493, "Phase concentration (R)", fontsize=5.6, color="#5C6770")
    phase_legend = fig.legend(
        handles=handles,
        ncol=3,
        loc="center",
        bbox_to_anchor=(0.777, 0.479),
        frameon=False,
        fontsize=5.3,
        handletextpad=0.2,
        columnspacing=0.7,
    )
    # Square data frames form the bottom row after the amplitude and phase panels.
    heading(fig, "d", "Prediction accuracy", 0.060, 0.260, size=7.7)
    fig.text(0.075, 0.237, "Participant mean ± SD", fontsize=5.9, color="#5C6770")
    ranks = [r for r in read("cpc_count_accuracy.csv") if r["subset"] == "six_fit"]
    x = np.array([int(r["components"]) for r in ranks])
    y = np.array([float(r["participant_mean"]) * 100 for r in ranks])
    sd = np.array([float(r["participant_sd"]) * 100 for r in ranks])
    ax = fig.add_axes([0.075, 0.060, 0.185, 0.1665])
    square_axes.append(("d_accuracy", ax))
    ax.axvline(30, color="#C3C9CD", ls=":", lw=0.7)
    ax.plot(x, y, "o-", color=BLUE, ms=2.6, lw=1.0, zorder=2)
    ax.scatter([30], [y[4]], s=18, color=ORANGE, zorder=3)
    bars = ax.errorbar(
        x, y, yerr=sd, fmt="none", ecolor=DARK, elinewidth=1.0, capsize=2.8, capthick=1.0, zorder=4
    )
    ax.set(
        xlim=(0, 53),
        ylim=(15, 95),
        xticks=[0, 10, 20, 30, 40, 50],
        yticks=[20, 40, 60, 80],
        xlabel="Complex CPCs",
        ylabel="Accuracy (%)",
    )
    ax.tick_params(axis="x", labelsize=5.5)
    ax.xaxis.label.set_size(6.3)
    ax.yaxis.label.set_size(6.3)
    segments = [v.tolist() for v in bars.lines[2][0].get_segments()]
    (ROOT / "provenance/accuracy_errorbars.json").write_text(
        json.dumps(
            dict(
                estimator="mean of participant accuracies; participant accuracy averages available REST2 run accuracies",
                interval="sample SD across 1003 participants (ddof=1)",
                components=x.tolist(),
                mean_percent=y.tolist(),
                sd_percentage_points=sd.tolist(),
                drawn_errorbar_segments=segments,
                capsize_points=2.8,
                linewidth_points=1.0,
            ),
            indent=2,
        )
    )

    heading(fig, "e", "State decoding", 0.305, 0.260, size=7.7)
    rows = read("confusion.csv")
    cm = np.array([float(r["row_fraction"]) for r in rows]).reshape(12, 12)
    counts = np.array([int(one(rows, state=k, predicted_state=1)["support"]) for k in range(1, 13)])
    ax = fig.add_axes([0.323, 0.060, 0.170, 0.153])
    square_axes.append(("e_confusion", ax))
    m = matrix(ax, cm * 100, cmap="Blues", vmax=100)
    ax.set(
        xticks=np.arange(12),
        yticks=np.arange(12),
        xticklabels=np.arange(1, 13),
        yticklabels=np.arange(1, 13),
        xlabel="Decoded state",
        ylabel="HMM state",
    )
    ax.xaxis.set_label_position("top")
    ax.tick_params(labelsize=4.9)
    ax.xaxis.label.set_size(6)
    ax.yaxis.label.set_size(6)
    for k in range(12):
        ax.text(
            k,
            k,
            f"{100 * cm[k, k]:.0f}",
            ha="center",
            va="center",
            fontsize=4.6,
            color="white" if cm[k, k] > 0.55 else DARK,
        )
    ax = fig.add_axes([0.507, 0.060, 0.027, 0.153])
    ax.barh(np.arange(12), counts / 1000, color="#7F8A92", height=0.65)
    ax.set(ylim=(11.5, -0.5), yticks=[], xticks=[0, 100], xlabel="n / 10³")
    ax.tick_params(labelsize=4.7)
    ax.xaxis.label.set_size(5.3)
    ax.spines["left"].set_visible(False)
    cb = cbar(fig, m, [0.338, 0.024, 0.140, 0.005], "", [0, 50, 100], True)
    cb.ax.tick_params(labelsize=5)
    cb.set_label("", fontsize=5.1, labelpad=1)

    heading(fig, "f", "Reproducibility across independent fits", 0.560, 0.260, size=7.7)
    fig.text(
        0.578, 0.237, "All distinct fit pairs; black bars: mean ± SD", fontsize=5.5, color="#5C6770"
    )
    cpc_pairs = [r for r in read("cpca_acquisition_subspaces.csv") if r["n_components"] == "30"]
    hmm_pairs = [
        r for r in read("hmm_reliability_pairs.csv") if r["comparison"] == "initialization"
    ]
    fit_summary = []
    fit_points = []
    specifications = [
        (
            "CPC30",
            cpc_pairs,
            "mean_squared_subspace_overlap",
            "Subspace overlap (%)",
            "6 acquisition pairs",
            0.601,
            BLUE,
        ),
        (
            "HMM",
            hmm_pairs,
            "argmax_agreement",
            "Time points assigned\nto different states (%)",
            "15 initialization pairs",
            0.824,
            ORANGE,
        ),
    ]
    for name, rows, key, ylabel, xlabel, left, color in specifications:
        values = np.array([float(r[key]) * 100 for r in rows])
        if name == "HMM":
            values = 100 - values
        metric = key if name == "CPC30" else "argmax_disagreement"
        mean = float(values.mean())
        spread = float(values.std(ddof=1))
        assert len(values) == (6 if name == "CPC30" else 15)
        ax = fig.add_axes([left, 0.066, 0.151, 0.1359])
        square_axes.append(("f_" + name + "_pairs", ax))
        ax.axhline(100 if name == "CPC30" else 50, color="#ADB5BB", ls=":", lw=0.7, zorder=1)
        dotx = np.linspace(-0.22, 0.22, len(values))
        ax.scatter(dotx, values, s=12, color=color, edgecolors="white", linewidths=0.25, zorder=3)
        ax.errorbar(
            0.60,
            mean,
            yerr=spread,
            fmt="D",
            ms=3.0,
            color=DARK,
            lw=1.0,
            capsize=3,
            capthick=1.0,
            zorder=4,
        )
        ax.text(
            0.03,
            0.79 if name == "CPC30" else 0.13,
            f"{mean:.1f} ± {spread:.1f}%",
            transform=ax.transAxes,
            fontsize=6.2,
            color=color,
        )
        if name == "HMM":
            fit_labels = {
                "robust_seed2": "S2",
                "robust": "S1",
                **{f"consensus_seed{k}": f"S{k}" for k in range(3, 7)},
            }
            worst = int(values.argmax())
            ax.scatter(
                [dotx[worst]],
                [values[worst]],
                s=24,
                facecolors="none",
                edgecolors=DARK,
                linewidths=0.7,
                zorder=5,
            )
            ax.annotate(
                f"{fit_labels[rows[worst]['fit_a']]}–{fit_labels[rows[worst]['fit_b']]}: {values[worst]:.1f}%",
                xy=(dotx[worst], values[worst]),
                xytext=(0.03, 0.93),
                textcoords="axes fraction",
                fontsize=5.8,
                color=DARK,
                arrowprops=dict(arrowstyle="-", color="#77838B", lw=0.6),
            )
        ax.set(
            xlim=(-0.38, 0.84),
            ylim=(0, 105),
            xticks=[0, 0.60],
            xticklabels=["Pairs", "Mean"],
            yticks=[0, 50, 100] if name == "CPC30" else [0, 25, 50],
            xlabel=xlabel,
            ylabel=ylabel,
            title=name if name == "CPC30" else "HMM (matched states)",
        )
        ax.xaxis.label.set_size(5.4)
        ax.yaxis.label.set_size(6.0)
        ax.tick_params(labelsize=5.6)
        ax.title.set_fontsize(6.9)
        fit_summary.append(
            dict(
                method=name,
                comparison="acquisition" if name == "CPC30" else "initialization",
                metric=metric,
                source_metric=key,
                transform="100*x" if name == "CPC30" else "100*(1-x)",
                pairs=len(values),
                mean_percent=mean,
                sd_percentage_points=spread,
                minimum_percent=float(values.min()),
                maximum_percent=float(values.max()),
                diagonal_included=False,
                errorbar_low=mean - spread,
                errorbar_high=mean + spread,
            )
        )
        for row, value in zip(rows, values):
            fit_points.append(
                dict(
                    method=name,
                    fit_a=row["acquisition_a" if name == "CPC30" else "fit_a"],
                    fit_b=row["acquisition_b" if name == "CPC30" else "fit_b"],
                    metric=metric,
                    value_percent=float(value),
                )
            )
    (ROOT / "source_data/reproducibility_pair_summary.json").write_text(
        json.dumps(fit_summary, indent=2)
    )
    with (ROOT / "source_data/reproducibility_pair_points.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fit_points[0]))
        writer.writeheader()
        writer.writerows(fit_points)

    # Remove 12 mm of blank space around panel a, preserving all data-frame sizes.
    height_mm = 188
    vertical_scale = 200 / height_mm
    fig.set_figheight(height_mm / 25.4)
    for ax in fig.axes:
        box = ax.get_position()
        shift = 5.6 / 200 if ax in example_axes else 0
        ax.set_position(
            [box.x0, (box.y0 - shift) * vertical_scale, box.width, box.height * vertical_scale]
        )
    for text in fig.texts:
        x, y = text.get_position()
        shift = 12 / 200 if text in example_heading else 0
        text.set_position((x, (y - shift) * vertical_scale))
    phase_legend.set_bbox_to_anchor((0.777, 0.479 * vertical_scale))

    geometry = []
    for name, ax in square_axes:
        box = ax.get_position()
        width = box.width * 180
        height = box.height * height_mm
        geometry.append(
            dict(panel=name, width_mm=width, height_mm=height, aspect_ratio=width / height)
        )
    (ROOT / "provenance/figure2_layout.json").write_text(
        json.dumps(
            dict(
                size_mm=[180, height_mm],
                panel_order=[
                    "a_example",
                    "b_amplitude",
                    "c_phase",
                    "d_accuracy",
                    "e_confusion",
                    "f_reproducibility",
                ],
                bottom_axes=geometry,
            ),
            indent=2,
        )
    )
    save(fig, "Figure_2_CPC_states_and_reproducibility")


if __name__ == "__main__":
    assert json.loads((DATA / "aggregation_complete.json").read_text())["status"] == "PASS"
    assert json.loads((DATA / "comparison_complete.json").read_text())["status"] == "PASS"
    (ROOT / "provenance").mkdir(exist_ok=True)
    figure2()
    (ROOT / "provenance/figure_generation.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "figure": "Figure_2_CPC_states_and_reproducibility",
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "pdf_fonttype": 42,
                "svg_fonttype": "none",
            },
            indent=2,
        )
    )
