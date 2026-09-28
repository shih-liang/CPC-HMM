"""Correspondence figures from fixed-basis CPC30 and archived reference HMM."""

from pathlib import Path
import csv
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.cm import ScalarMappable
from make_atlas import Cortex

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "source_data"
BLUE = "#246B8A"
ORANGE = "#C16F35"
DARK = "#253441"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "savefig.facecolor": "white",
    }
)


def save(fig, name):
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(ROOT / f"{name}.{ext}", dpi=350 if ext != "png" else 220)
    plt.close(fig)
    print("SAVED", name, flush=True)


def read(name):
    with (D / name).open() as f:
        return list(csv.DictReader(f))


def panel(fig, x, y, text):
    fig.text(x, y, text, fontsize=8, weight="bold")


def ticks(ax, n):
    ax.set(xticks=[1, 10, 20, 30, 40, 50] if n == 50 else [1, 5, 10, 15, 20, 25, 30])


def ica():
    a = np.load(D / "figure_arrays.npz")
    ex = np.load(D / "fixed_example.npz")
    meta = json.loads((D / "complete.json").read_text())
    fig = plt.figure(figsize=(7.4, 8.5))
    fig.text(
        0.065, 0.974, "CPC30 correspondence with ICA network activity", fontsize=11, weight="bold"
    )
    panel(fig, 0.065, 0.938, "a  Individual CPC–ICA correspondence in REST2")
    ax = fig.add_axes([0.075, 0.625, 0.82, 0.298])
    m = a["single_r_mean"]
    lim = max(0.7, float(np.ceil(m.max() * 10) / 10))
    im = ax.pcolormesh(
        np.arange(51) + 0.5,
        np.arange(31) + 0.5,
        m,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim),
    )
    ax.set(
        xlim=(0.5, 50.5),
        ylim=(30.5, 0.5),
        xlabel="ICA component",
        ylabel="CPC",
        yticks=[1, 5, 10, 15, 20, 25, 30],
    )
    ax.set_xticks(np.arange(1, 51))
    ax.set_yticks(np.arange(1, 31))
    ax.tick_params(labelsize=5, length=2, pad=1)
    cb = fig.colorbar(im, cax=fig.add_axes([0.92, 0.655, 0.013, 0.23]))
    cb.set_label("Test correlation r", fontsize=6.5)
    panel(fig, 0.065, 0.562, "b  Recovery of all 50 ICA time courses from CPC30")
    ax = fig.add_axes([0.075, 0.405, 0.82, 0.14])
    x = np.arange(1, 51)
    for key, color, offset, label in [("r", BLUE, -0.13, "Pearson r"), ("r2", ORANGE, 0.13, "R²")]:
        ax.errorbar(
            x + offset,
            a[key + "_mean"],
            yerr=a[key + "_sd"],
            fmt="o",
            ms=2.2,
            color=color,
            elinewidth=0.5,
            capsize=1,
            label=label,
        )
    ax.plot(x, a["shift_r_mean"], color="#636363", lw=0.7, ls="--", label="Shift control r")
    ax.axhline(0, color="#555555", lw=0.5)
    ax.set(xlim=(0.2, 50.8), ylim=(-0.16, 1), xlabel="ICA component", ylabel="Test performance")
    ticks(ax, 50)
    ax.legend(
        frameon=False,
        ncol=3,
        loc="lower left",
        bbox_to_anchor=(0, 1.00),
        fontsize=6,
        borderaxespad=0,
    )
    panel(fig, 0.065, 0.338, "c  Observed and CPC30-estimated network activity")
    for row, k in enumerate([0, 9, 19]):
        ax = fig.add_axes([0.075, 0.24 - row * 0.087, 0.49, 0.067])
        t = (ex["frames"] - ex["frames"][0]) * 0.72
        ax.plot(t, ex["observed"][:, k], color=DARK, lw=0.7, label="Observed")
        ax.plot(t, ex["predicted"][:, k], color=ORANGE, lw=0.8, label="CPC30")
        ax.set(xlim=(0, t[-1]), ylabel=f"IC{k + 1}", xticks=[0, 36, 72, 108, 144])
        ax.tick_params(length=2, pad=1)
        if row < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time in example (s)")
        if row == 0:
            ax.legend(
                frameon=False, ncol=2, fontsize=5.8, loc="upper right", bbox_to_anchor=(1, 1.42)
            )
    panel(fig, 0.64, 0.338, "d  ICA functional connectivity")
    for idx, key in enumerate(["fc_observed", "fc_predicted"]):
        ax = fig.add_axes([0.655 + idx * 0.185, 0.148, 0.14, 0.139])
        v = a[key].copy()
        np.fill_diagonal(v, np.nan)
        cm = plt.get_cmap("RdBu_r").copy()
        cm.set_bad("white")
        im = ax.pcolormesh(
            np.arange(51) + 0.5,
            np.arange(51) + 0.5,
            np.ma.masked_invalid(v),
            cmap=cm,
            vmin=-1,
            vmax=1,
        )
        ax.set(xlim=(0.5, 50.5), ylim=(50.5, 0.5), aspect="equal", xticks=[1, 50], yticks=[1, 50])
        ax.set_title(["Observed", "CPC30"][idx], fontsize=7, pad=3)
        ax.tick_params(length=2, pad=1)
    cb = fig.colorbar(
        im, cax=fig.add_axes([0.69, 0.11, 0.24, 0.007]), orientation="horizontal", ticks=[-1, 0, 1]
    )
    cb.set_label("ICA correlation", fontsize=6)
    cb.ax.tick_params(pad=1, length=2)
    fig.text(
        0.655,
        0.056,
        f"FC similarity: {meta['fc_r_mean']:.3f} ± {meta['fc_r_sd']:.3f}",
        fontsize=6.3,
    )
    fig.text(
        0.065,
        0.014,
        "REST2 LR + RL · n = 1,003 · mean ± SD across participants · instantaneous linear mapping fitted in REST1",
        fontsize=6,
    )
    save(fig, "Figure_CPC30_ICA50_correspondence")


def state():
    p = np.load(D / "state_profiles.npz")
    amp = p["all_reference_amplitude"]
    z = p["all_reference_phase"]
    assert amp.shape == z.shape == (12, 30)
    maps = np.load(D / "state_associated_cortical_maps.npz")
    brain = Cortex(np.load(D / "surfaces_and_eigenmodes.npz"))
    rows = read("state_map_by_run.csv")
    values = np.full((1003, 2, 12, 2), np.nan)
    for r in rows:
        values[int(r["subject"]), int(r["scan"]) % 4 - 2, int(r["state"]) - 1] = [
            float(r["spatial_r"]),
            float(r["reconstruction_r2"]),
        ]

    def mean_nan(a, axis):
        n = np.isfinite(a).sum(axis)
        return np.divide(np.nansum(a, axis), n, out=np.full(n.shape, np.nan), where=n > 0)

    subject = mean_nan(values, 1)
    means = mean_nan(subject, 0)
    sd = np.nanstd(subject, axis=0, ddof=1)
    counts = np.isfinite(subject[:, :, 0]).sum(0)
    archived = read("archived_state_map_summary.csv")
    for r in archived:
        if r["metric"] in ["spatial_r", "reconstruction_r2"]:
            k = ["spatial_r", "reconstruction_r2"].index(r["metric"])
            assert abs(means[int(r["state"]) - 1, k] - float(r["mean"])) < 1e-10
    np.savez(
        D / "state_spatial_participant_metrics.npz",
        participant=subject,
        mean=means,
        sd=sd,
        n=counts,
    )
    fig = plt.figure(figsize=(7.4, 9.2))
    fig.text(
        0.065, 0.975, "CPC30 correspondence with HMM state identity", fontsize=11, weight="bold"
    )
    panel(fig, 0.065, 0.940, "a  State-associated CPC amplitude")
    ax = fig.add_axes([0.075, 0.745, 0.82, 0.18])
    im = ax.pcolormesh(
        np.arange(31) + 0.5,
        np.arange(13) + 0.5,
        amp,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(
            vmin=min(0.6, float(amp.min())), vcenter=1, vmax=max(1.6, float(amp.max()))
        ),
    )
    ax.set(xlim=(0.5, 30.5), ylim=(12.5, 0.5), yticks=np.arange(1, 13), ylabel="HMM state")
    ticks(ax, 30)
    cb = fig.colorbar(im, cax=fig.add_axes([0.92, 0.755, 0.013, 0.16]))
    cb.set_label("Amplitude / run mean", fontsize=6.5)
    panel(fig, 0.065, 0.695, "b  State-associated CPC phase")
    ax = fig.add_axes([0.075, 0.50, 0.82, 0.18])
    xx, yy = np.meshgrid(np.arange(1, 31), np.arange(1, 13))
    strength = abs(z)
    sc = ax.scatter(
        xx.ravel(),
        yy.ravel(),
        c=np.angle(z).ravel(),
        s=(3 + 65 * strength).ravel(),
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
        edgecolors="none",
    )
    ax.set(
        xlim=(0.5, 30.5),
        ylim=(12.5, 0.5),
        yticks=np.arange(1, 13),
        xlabel="CPC",
        ylabel="HMM state",
    )
    ticks(ax, 30)
    cb = fig.colorbar(sc, cax=fig.add_axes([0.92, 0.51, 0.013, 0.16]), ticks=[-np.pi, 0, np.pi])
    cb.ax.set_yticklabels(["−π", "0", "π"])
    cb.set_label("Coefficient phase", fontsize=6.5)
    # Point area encodes resultant length; label the scale without adding a paragraph.
    for i, r in enumerate([0.1, 0.5, 0.9]):
        ax.scatter([], [], s=3 + 65 * r, c="black", label=f"{r:.1f}")
    ax.legend(
        title="Phase concentration",
        frameon=False,
        ncol=3,
        loc="upper right",
        bbox_to_anchor=(1, -0.16),
        fontsize=5.7,
        title_fontsize=5.7,
        handletextpad=0.15,
        columnspacing=0.7,
    )
    panel(fig, 0.065, 0.418, "c  Observed and CPC30-reconstructed state maps")
    for s in range(12):
        row, col = divmod(s, 3)
        x = 0.065 + col * 0.218
        y = 0.393 - row * 0.085
        fig.text(x, y, f"State {s + 1}", fontsize=6.5, weight="bold")
        for j, key in enumerate(["state_observed", "state_wave"]):
            brain.quartet(
                fig,
                [x + 0.013, y - 0.030 - j * 0.027, 0.190, 0.027],
                maps[key][s],
                "RdBu_r",
                -1.25,
                1.25,
            )
            fig.text(x, y - 0.016 - j * 0.027, ["O", "C"][j], fontsize=5.3, va="center")
    panel(fig, 0.747, 0.418, "d  Spatial correspondence")
    ax = fig.add_axes([0.79, 0.10, 0.165, 0.295])
    for k, color, offset, label in [(0, BLUE, -0.13, "Spatial r"), (1, ORANGE, 0.13, "R²")]:
        ax.errorbar(
            means[:, k],
            np.arange(1, 13) + offset,
            xerr=sd[:, k],
            fmt="o",
            color=color,
            ms=2.4,
            elinewidth=0.65,
            capsize=1.5,
            label=label,
        )
    ax.axvline(0, color="#555555", lw=0.5)
    ax.set(
        ylim=(12.6, 0.4),
        yticks=np.arange(1, 13),
        xlim=(min(-0.05, float(np.nanmin(means - sd)) - 0.03), 1),
        xticks=[0, 0.5, 1],
        ylabel="HMM state",
    )
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.09), fontsize=6)
    cb = fig.colorbar(
        ScalarMappable(norm=Normalize(-1.25, 1.25), cmap="RdBu_r"),
        cax=fig.add_axes([0.245, 0.052, 0.28, 0.007]),
        orientation="horizontal",
        ticks=[-1.25, 0, 1.25],
    )
    cb.set_label("Mean BOLD (input units)", fontsize=6)
    cb.ax.tick_params(pad=1, length=2)
    fig.text(
        0.065,
        0.014,
        "REST2 LR + RL · all reference-state frames · O: observed; C: CPC30 · error bars: participant SD",
        fontsize=6,
    )
    save(fig, "Figure_CPC30_HMM12_correspondence")
