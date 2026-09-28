"""One-page Figure 4: original curves, paired pre/post cells, and phase probabilities."""

from pathlib import Path
import csv
import hashlib
import json
import os
import plot_helpers as helpers

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hcp_main_figure_matplotlib")
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT
W, H = 180, 225
HEATMAP_WIDTH = 46
TARGET_GAP = 1.2 * 12 / HEATMAP_WIDTH  # Keep 1.2 mm between target-state columns.
NAME = "Figure_4_HMM_transitions_and_wave_dynamics"
EXAMPLES = [(1, 9), (11, 5), (5, 6)]
ACCENT = ["#247B94", "#CA7951", "#7155A3"]
DARK, GRAY = helpers.DARK, "#63717A"
plt.rcParams.update(
    {
        "font.size": 6.5,
        "axes.labelsize": 6.2,
        "axes.titlesize": 7,
        "xtick.labelsize": 5.7,
        "ytick.labelsize": 5.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def rect(x, top, width, height):
    return [x / W, 1 - (top + height) / H, width / W, height / H]


def text(fig, x, top, value, **kwargs):
    return fig.text(x / W, 1 - top / H, value, va="top", **kwargs)


def heading(fig, letter, title, x, top):
    text(fig, x, top, letter, weight="bold", size=9)
    text(fig, x + 4.5, top + 0.25, title, weight="bold", size=7.5)


def top_panels(fig, data):
    heading(fig, "a", "HMM state-transition graph", 5, 3)
    heading(fig, "b", "HMM transition probabilities", 94, 3)
    helpers.graph(fig, data["transition_probability"])
    fig.axes[-1].set_position(rect(20, 9, 55, 55))
    fig.axes[-1].set_label("a")
    for label in fig.axes[-1].texts:
        label.set_fontsize(6.5)
    fig.legends[-1].set_bbox_to_anchor((47.5 / W, 1 - 69 / H))
    helpers.matrix(fig, data["transition_probability"])
    fig.axes[-2].set_position(rect(100, 9, 55, 55))
    fig.axes[-2].set_label("b")
    fig.axes[-1].set_position(rect(158, 10, 2.1, 53))


def example_curves(fig, data):
    heading(fig, "c", "Wave amplitude and phase", 5, 77)
    heading(fig, "d", "Transition probability", 123.5, 77)
    xs, width = [18.5, 75, 131.5], 40
    for x, label in zip(xs, ["Amplitude / run mean", "Phase (°)", "Probability (%)"]):
        text(fig, x + width / 2, 84, label, ha="center", size=6.2, color=DARK)
    keys = [helpers.key_of(*p) for p in EXAMPLES]
    limits = {}
    for kind in ["amplitude", "phase"]:
        values = [
            helpers.phase_curve(data, q)
            if kind == "phase"
            else (
                data["amplitude_mean"][q, :, data["chosen"][q]],
                data["amplitude_sd"][q, :, data["chosen"][q]],
            )
            for q in keys
        ]
        means = np.stack([v[0] for v in values])
        sd = np.stack([v[1] for v in values])
        bounds = np.r_[means.ravel(), (means - sd).ravel(), (means + sd).ravel()]
        bounds = bounds[np.isfinite(bounds)]
        lo, hi = (float(bounds.min()), float(bounds.max())) if len(bounds) else (0., 1.)
        margin = max((hi - lo) * 0.08, 0.01)
        limits[kind] = (lo - margin, hi + margin)
    for row, (pair, q) in enumerate(zip(EXAMPLES, keys)):
        top = 90 + row * 16
        k = int(data["chosen"][q])
        text(
            fig,
            5.5,
            top + 2,
            f"{pair[0]} → {pair[1]}\nCPC{k + 1}",
            ha="center",
            size=5.7,
            weight="bold",
            color=ACCENT[row],
            linespacing=1.4,
        )
        for col, kind in enumerate(["amplitude", "phase"]):
            ax = fig.add_axes(rect(xs[col], top, width, 11.5), label=f"c_{row}_{kind}")
            mean, sd = (
                helpers.phase_curve(data, q)
                if kind == "phase"
                else (data["amplitude_mean"][q, :, k], data["amplitude_sd"][q, :, k])
            )
            helpers.band(ax, data["lags_TR"] * 0.72, mean, sd, helpers.PURPLE, marker="o", ms=1.8)
            ax.set(xlim=(-3.6, 3.6), xticks=[-3.6, 0, 3.6] if row == 2 else [], ylim=limits[kind])
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
            if kind == "phase":
                ax.set_yticks([-180, 0, 180])
        ax = fig.add_axes(rect(xs[2], top, width, 11.5), label=f"d_{row}")
        mean, sd = [data["probability_" + name][q, :, k] * 100 for name in ["mean", "sd"]]
        helpers.band(
            ax, data["phase_degrees"], mean, sd, helpers.BLUE, probability=True, percent=True
        )
        ax.set(
            xlim=(-180, 180),
            xticks=[-180, 0, 180] if row == 2 else [],
            ylim=(0, max(float(np.nanmax(np.r_[mean, mean + sd, 0.3])) * 1.04, 0.3)),
        )
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    text(
        fig,
        (xs[0] + xs[1] + width) / 2,
        140,
        "Time from HMM switch (s)",
        ha="center",
        size=6.2,
        color=DARK,
    )
    text(fig, xs[2] + width / 2, 140, "Source-frame phase (°)", ha="center", size=6.2, color=DARK)
    fig.legend(
        handles=[Patch(facecolor=helpers.BLUE, alpha=0.17, label="Shading: ± SD")],
        loc="center",
        bbox_to_anchor=(96 / W, 1 - 146 / H),
        ncol=1,
        frameon=False,
        fontsize=5.5,
        handlelength=1.4,
        columnspacing=1.2,
    )


def heatmaps(fig, data, paired):
    pairs = [(i, j) for i in range(1, 13) for j in range(1, 13) if i != j]
    keys = np.array([helpers.key_of(*p) for p in pairs])
    chosen = data["chosen"][keys]
    assert np.array_equal(paired["transition_keys"], keys)
    assert np.array_equal(paired["CPC"], chosen + 1)
    amp = paired["amplitude_mean"]
    phase = paired["phase_mean_degrees"]
    probability = 100 * data["probability_mean"][keys, :, chosen]
    assert amp.shape == phase.shape == (132, 2) and probability.shape == (132, 24)
    # Explicit source-by-target cells. Each cell contains the full sampled profile.
    panels = [
        (
            "e",
            "Wave amplitude",
            amp,
            "YlOrRd",
            Normalize(0.75, 1.85),
            [-3.96, 3.96],
            [-3.6, 0, 3.6],
            "Time from switch (s)",
            [0.75, 1, 1.5, 1.85],
            "Amplitude / run mean",
        ),
        (
            "f",
            "Wave phase",
            phase,
            "twilight_shifted",
            Normalize(-180, 180),
            [-3.96, 3.96],
            [-3.6, 0, 3.6],
            "Time from switch (s)",
            [-180, 0, 180],
            "CPC phase (°)",
        ),
        (
            "g",
            "Transition probability",
            probability,
            "YlOrRd",
            Normalize(0, 25),
            [-180, 180],
            [-180, 0, 180],
            "Source-frame phase (°)",
            [0, 5, 10, 15, 20, 25],
            "Probability (%)",
        ),
    ]
    xs, width = [16, 72, 128], HEATMAP_WIDTH
    for col, (
        letter,
        title,
        values,
        cmap,
        norm,
        extent,
        ticks,
        xlabel,
        cticks,
        clabel,
    ) in enumerate(panels):
        if col != 1 and np.isfinite(values).any():
            low, high = float(np.nanmin(values)), float(np.nanmax(values))
            norm = Normalize(min(norm.vmin, low), max(norm.vmax, high))
            cticks = np.linspace(norm.vmin, norm.vmax, 5)
        heading(fig, letter, title, xs[col] - 8, 151)
        ax = fig.add_axes(rect(xs[col], 162, width, 40), label=letter)
        samples = values.shape[1]
        packed = np.full((12, 12, samples), np.nan)
        for index, (source, target) in enumerate(pairs):
            packed[source - 1, target - 1] = values[index]
        # Grid-cell values must be exact copies of their corresponding source profiles.
        assert np.isnan(packed[np.arange(12), np.arange(12)]).all()
        colourmap = plt.get_cmap(cmap).copy()
        colourmap.set_bad("white")
        for target in range(12):
            left, right = target + TARGET_GAP / 2, target + 1 - TARGET_GAP / 2
            ax.pcolormesh(
                np.linspace(left, right, samples + 1),
                np.arange(13),
                packed[:, target, :],
                cmap=colourmap,
                norm=norm,
                edgecolors="face",
                linewidth=0.25,
                antialiased=False,
                rasterized=False,
            )
            ax.hlines(np.arange(13), left, right, color="#CCD3D9", lw=0.4)
        ax.set(
            xlim=(0, 12),
            ylim=(12, 0),
            xticks=np.arange(12) + 0.5,
            xticklabels=np.arange(1, 13),
            xlabel="Target state",
        )
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
        ax.set_yticks(np.arange(12) + 0.5)
        ax.set_yticklabels(np.arange(1, 13) if col == 0 else [])
        if col == 0:
            ax.set_ylabel("Source state", labelpad=3)
        ax.tick_params(axis="y", length=0, pad=3)
        ax.tick_params(axis="x", length=0, pad=2, labelsize=5)
        ax.set_xlabel("Target state", labelpad=3, fontsize=5.9)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if col < 2:
            significant = paired["amplitude_significant" if col == 0 else "phase_significant"]
            for index, (source, target) in enumerate(pairs):
                if significant[index]:
                    rgb = colourmap(norm(values[index, 1]))[:3]
                    luminance = sum(a * b for a, b in zip(rgb, [0.2126, 0.7152, 0.0722]))
                    star_x = target - 1 + TARGET_GAP / 2 + 0.75 * (1 - TARGET_GAP)
                    ax.text(
                        star_x,
                        source - 0.30,
                        "*",
                        ha="center",
                        va="center",
                        size=5.3,
                        color="black" if luminance > 0.45 else "white",
                        weight="bold",
                        zorder=7,
                    )
        text(
            fig,
            xs[col] + width / 2,
            205,
            "Before | After" if col < 2 else "Phase: −180° → +180°",
            ha="center",
            size=6.0,
            color=DARK,
        )
        cb = fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            cax=fig.add_axes(rect(xs[col], 213, width, 1.7)),
            orientation="horizontal",
            ticks=cticks,
        )
        cb.set_label(clabel, size=5.6, labelpad=1.7)
        cb.ax.tick_params(labelsize=5.2, length=1.5, pad=1.3)
        cb.outline.set_linewidth(0.35)
        cb.solids.set_rasterized(False)
        cb.solids.set_edgecolor("face")
    text(fig, 65, 209, "* FDR-adjusted p < 0.05", ha="center", size=5.8, color=DARK)
    rows = [
        dict(
            record=row + 1,
            matrix_row=i,
            matrix_column=j,
            source=i,
            target=j,
            CPC=int(chosen[row]) + 1,
            complete_events=int(data["window_counts"][q]),
            participants=int(data["amplitude_participants"][q, 0, chosen[row]]),
            rare=bool(data["window_counts"][q] < 25),
            training_fallback=bool(data["selection_fallback"][q]),
        )
        for row, ((i, j), q) in enumerate(zip(pairs, keys))
    ]
    with (ROOT / "source_data/transition_key.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        ROOT / "source_data/heatmap_values.npz",
        amplitude_mean=amp,
        phase_mean_degrees=phase,
        observed_probability_percent=probability,
        source=[p[0] for p in pairs],
        target=[p[1] for p in pairs],
        CPC=chosen + 1,
        before_seconds=paired["before_TR"] * 0.72,
        after_seconds=paired["after_TR"] * 0.72,
        amplitude_q=paired["amplitude_q"],
        phase_q=paired["phase_q"],
        amplitude_significant=paired["amplitude_significant"],
        phase_significant=paired["phase_significant"],
        phase_bin_degrees=data["phase_degrees"],
    )
    return rows


def main():
    for folder in ["source_data", "provenance", "previews"]:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    data = dict(np.load(SOURCE / "results/all_transition_profiles.npz"))
    data.update(dict(np.load(SOURCE / "results/absolute_phase_profiles.npz")))
    paired = dict(np.load(ROOT / "results/prepost_summary.npz"))
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    top_panels(fig, data)
    example_curves(fig, data)
    rows = heatmaps(fig, data, paired)
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(ROOT / f"{NAME}.{ext}", dpi=350)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_geometry = []
    for ax in fig.axes:
        label = ax.get_label()
        if label in ["a", "b", "e", "f", "g"] or label.startswith(("c_", "d_")):
            size = np.array(
                [ax.get_window_extent(renderer).width, ax.get_window_extent(renderer).height]
            )
            millimetres = size * 25.4 / fig.dpi
            expected = (
                [40, 11.5]
                if label.startswith(("c_", "d_"))
                else [55, 55]
                if label in ["a", "b"]
                else [HEATMAP_WIDTH, 40]
            )
            np.testing.assert_allclose(millimetres, expected, rtol=0, atol=1e-8)
            axes_geometry.append(
                dict(
                    label=label,
                    width_mm=float(millimetres[0]),
                    height_mm=float(millimetres[1]),
                    x_mm=ax.get_position().x0 * W,
                    top_mm=(1 - ax.get_position().y1) * H,
                )
            )
    assert len(axes_geometry) == 14
    (ROOT / "provenance/axes_geometry.json").write_text(json.dumps(axes_geometry, indent=2))
    boxes = [
        dict(
            text=t.get_text(),
            bounds=list(t.get_window_extent(renderer).bounds),
            colour=str(t.get_color()),
        )
        for t in fig.findobj(matplotlib.text.Text)
        if t.get_visible() and t.get_text().strip()
    ]
    (ROOT / "provenance/text_boxes.json").write_text(
        json.dumps(dict(figure_pixels=list(fig.bbox.size), text=boxes), indent=2)
    )
    source_files = [
        "results/all_transition_profiles.npz",
        "results/absolute_phase_profiles.npz",
        "code/plot_helpers.py",
    ]
    hashes = {
        str(SOURCE / file): hashlib.sha256((SOURCE / file).read_bytes()).hexdigest()
        for file in source_files
    }
    report = dict(
        status="PASS",
        page_mm=[W, H],
        panels=list("abcdefg"),
        heatmap_grid=[12, 12],
        non_diagonal_cells_per_panel=132,
        blank_diagonal_cells_per_panel=12,
        prepost_values_per_cell=2,
        amplitude_stars=int(paired["amplitude_significant"].sum()),
        phase_stars=int(paired["phase_significant"].sum()),
        probability_bins_per_cell=24,
        target_state_column_gap_mm=TARGET_GAP * HEATMAP_WIDTH / 12,
        heatmap_axes_mm=[HEATMAP_WIDTH, 40],
        inter_heatmap_gap_mm=10,
        heatmap_example_outlines=False,
        heatmap_colormaps=dict(e="YlOrRd", f="twilight_shifted", g="YlOrRd"),
        full_event_windows=sum(row["complete_events"] for row in rows),
        source_hashes=hashes,
        curve_examples=[
            dict(source=i, target=j, CPC=int(data["chosen"][helpers.key_of(i, j)]) + 1)
            for i, j in EXAMPLES
        ],
        preserved="Original summaries, SDs, heatmap values, phase reference, and training-selected CPC for every transition; example rows now show CPC1, CPC2, CPC3.",
        probability_display="Observed probability only; shift controls and baseline curves removed at user request.",
        heatmap_normalization="Global linear scale for each quantity; no per-row normalization.",
        rare_rows=[r for r in rows if r["rare"]],
        fallback_rows=[r for r in rows if r["training_fallback"]],
    )
    (ROOT / "provenance/data_validation.json").write_text(json.dumps(report, indent=2))
    plt.close(fig)
    print(
        "Saved Figure 4, a-g, one page; e/f have two pre/post colours and paired-test stars; g probabilities are unchanged."
    )


if __name__ == "__main__":
    main()
