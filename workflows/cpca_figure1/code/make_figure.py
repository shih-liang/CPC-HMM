"""Individual CPC fields at recorded times; vector plots with editable text."""

import os  # Public release: configurable data roots.
from pathlib import Path
import argparse
import csv
import json
import hashlib
import time
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, QuadMesh
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.signal import hilbert

BASE = Path(
    os.path.join(
        os.environ.get("HCP_DERIVATIVES", "/configure/HCP_DERIVATIVES"),
        "wave_rsn_revision_20260906",
    )
)
BASIS = BASE / "revision_20260910/data/cpca_basis_REST1_LR.npz"
SCORES = BASE / "data/cpca_scores_200.npy"
RAW = Path(
    os.path.join(
        os.environ.get("HCP_CORTICAL_ROOT", "/configure/HCP_CORTICAL_ROOT"),
        "input_hmm_order/group_fs4_concat_z_hmm_order.npy",
    )
)
GEOMETRY = BASE / "data/surfaces_and_eigenmodes.npz"
SELECTED = list(range(6))
SNAP = np.array([360, 364, 368, 372])
WINDOW = np.arange(300, 500)
TR = 0.72
BLUE = "#286B8B"
ORANGE = "#C37840"
DARK = "#203444"
GREY = "#7B858D"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.65,
        "lines.linewidth": 1,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "text.usetex": False,
        "savefig.facecolor": "white",
        "axes.unicode_minus": False,
    }
)


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def fingerprint(path):
    s = path.stat()
    with path.open("rb") as f:
        x = f.read(65536)
        f.seek(max(0, s.st_size - 65536))
        x += f.read(65536)
    return dict(
        path=str(path),
        bytes=s.st_size,
        mtime_ns=s.st_mtime_ns,
        edge_sha256=hashlib.sha256(x).hexdigest(),
    )


class Cortex:
    def __init__(self, path):
        self.s = np.load(path)
        self.mesh = {}
        for h, side in [("L", -1), ("R", 1)]:
            v = self.s[h + "_inflated_vertices"]
            f = self.s[h + "_faces"]
            self.mesh[h] = (np.c_[side * v[:, 1], v[:, 2]], f, np.argsort(side * v[f, 0].mean(1)))
        v = self.s["L_pial_vertices"]
        f = self.s["L_faces"]
        keep = self.s["L_indices"]
        allowed = np.zeros(len(v), bool)
        allowed[keep] = True
        allowed &= v[:, 0] < -10
        candidate = np.flatnonzero(allowed)
        start = candidate[np.argmin(((v[candidate] - [-35, -75, 10]) ** 2).sum(1))]
        end = candidate[np.argmin(((v[candidate] - [-40, 45, 10]) ** 2).sum(1))]
        edges = np.unique(
            np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [0, 2]]]), axis=1), axis=0
        )
        edges = edges[allowed[edges].all(1)]
        w = np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1)
        graph = coo_matrix(
            (np.r_[w, w], (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
            shape=(len(v), len(v)),
        ).tocsr()
        dist, pred = dijkstra(graph, indices=start, return_predecessors=True)
        assert np.isfinite(dist[end]), "Fixed anatomical path is disconnected"
        route = [int(end)]
        while route[-1] != start:
            route.append(int(pred[route[-1]]))
        self.path = np.array(route[::-1])
        self.distance = dist[self.path]
        lookup = np.full(len(v), -1, int)
        lookup[keep] = np.arange(len(keep))
        self.retained_path = lookup[self.path]
        assert (self.retained_path >= 0).all() and np.all(np.diff(self.distance) > 0)
        self.path_coordinates = v[self.path]

    def draw(self, ax, field, h, limit, path=False, kind="contribution"):
        xy, faces, order = self.mesh[h]
        full = np.full(len(xy), np.nan)
        ids = self.s[h + "_indices"]
        full[ids] = field[: len(ids)] if h == "L" else field[-len(ids) :]
        values = full[faces]
        valid = np.isfinite(values).all(1)
        avg = np.zeros(len(faces))
        avg[valid] = (
            np.angle(np.exp(1j * values[valid]).mean(1))
            if kind == "phase"
            else values[valid].mean(1)
        )
        cmap = "twilight" if kind == "phase" else "magma" if kind == "amplitude" else "RdBu_r"
        low = 0 if kind == "amplitude" else -limit
        colors = plt.get_cmap(cmap)(Normalize(low, limit)(avg))
        colors[~valid] = [0.84, 0.85, 0.86, 1]
        ax.add_collection(
            PolyCollection(
                xy[faces][order],
                facecolors=colors[order],
                edgecolors="none",
                linewidths=0,
                antialiaseds=False,
                rasterized=False,
            )
        )
        if path:
            p = xy[self.path]
            ax.plot(p[:, 0], p[:, 1], color="#303536", lw=0.55, alpha=0.85)
            ax.plot(p[[0, -1], 0], p[[0, -1], 1], "o", color="#303536", ms=1.7)
        ax.set(
            xlim=(xy[:, 0].min() - 2, xy[:, 0].max() + 2),
            ylim=(xy[:, 1].min() - 2, xy[:, 1].max() + 2),
        )
        ax.set_aspect("equal")
        ax.axis("off")


def retain_vectors(fig):
    fig.canvas.draw()
    # Matplotlib rasterizes dense color bars by default; keep their cells vector too.
    for ax in fig.axes:
        for collection in ax.collections:
            collection.set_rasterized(False)
            if isinstance(collection, (PolyCollection, QuadMesh)):
                # Face-coloured sub-point edges close PDF anti-aliasing seams without a grid.
                collection.set_edgecolor("face")
                collection.set_linewidth(0.35)
                collection.set_antialiased(False)


def mode_rows(fig, rect, components, brain, fields, line_fields, limit, variance, bilateral=False):
    x, y, w, h = rect
    g = fig.add_gridspec(
        len(components),
        5,
        left=x,
        right=x + w,
        bottom=y,
        top=y + h,
        width_ratios=[1, 1, 1, 1, 1.65],
        hspace=0.10,
        wspace=0.08,
    )
    times = (WINDOW - 300) * TR
    for row, k in enumerate(components):
        for j, frame in enumerate(SNAP):
            if bilateral:
                sub = g[row, j].subgridspec(1, 2, wspace=0.005)
                axes = [fig.add_subplot(sub[0, hh]) for hh in range(2)]
                for ax, hemi in zip(axes, ["L", "R"]):
                    brain.draw(ax, fields[k, j], hemi, limit, path=j == 0 and hemi == "L")
                ax = axes[0]
            else:
                ax = fig.add_subplot(g[row, j])
                brain.draw(ax, fields[k, j], "L", limit, path=j == 0)
            if row == 0:
                ax.set_title(f"{(frame - 300) * TR:.2f} s", fontsize=7, pad=3, loc="center")
            if j == 0:
                ax.text(
                    -0.05,
                    0.80,
                    f"CPC{k + 1}",
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontweight="bold",
                    fontsize=7,
                )
                ax.text(
                    -0.05,
                    0.55,
                    f"{100 * variance[k]:.2f}%",
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=6,
                    color=GREY,
                )
        ax = fig.add_subplot(g[row, 4])
        z = line_fields[k]
        dd = brain.distance
        de = np.r_[
            dd[0] - (dd[1] - dd[0]) / 2, (dd[1:] + dd[:-1]) / 2, dd[-1] + (dd[-1] - dd[-2]) / 2
        ]
        ax.pcolormesh(
            np.r_[times - TR / 2, times[-1] + TR / 2],
            de,
            z.T,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            shading="flat",
            rasterized=False,
        )
        for frame in SNAP:
            ax.axvline((frame - 300) * TR, color="#272D30", alpha=0.65, lw=0.35)
        ax.set(
            xlim=(0, times[-1]),
            ylim=(dd[0], dd[-1]),
            xticks=[0, 72, 144],
            yticks=[0, round(dd[-1])],
        )
        ax.tick_params(length=1.7, pad=1.5, labelsize=5.5)
        ax.yaxis.tick_right()
        ax.set_yticklabels(["0", f"{dd[-1]:.0f}"] if row in [0, len(components) - 1] else [])
        if row == 0:
            ax.set_title("Along the marked cortical path", fontsize=6.5, pad=4)
        if row == len(components) - 1:
            ax.set_xlabel("Time in example (s)", fontsize=6, labelpad=2)
        else:
            ax.set_xticklabels([])
    fig.text(
        x + w + 0.043,
        y + h / 2,
        "Distance along path (mm)",
        rotation=90,
        fontsize=6,
        ha="center",
        va="center",
    )


def main(root):
    start_time = time.time()
    root.mkdir(exist_ok=True, parents=True)
    (root / "provenance").mkdir(exist_ok=True)
    (root / "provenance/render_validation.json").write_text(json.dumps(dict(status="RUNNING")))
    before = [fingerprint(p) for p in [BASIS, SCORES, RAW, GEOMETRY]]
    b = np.load(BASIS)
    u = b["complex_vectors"][:, :30].astype(np.complex128)
    idx = b["vertex_indices"]
    all_scores = np.load(SCORES, mmap_mode="r")
    c = np.asarray(all_scores[2, :, :30], dtype=np.complex128)
    x = np.asarray(np.load(RAW, mmap_mode="r")[2400:3600][:, idx], dtype=float)
    x -= x.mean(0)
    replay = hilbert(x, axis=0) @ u
    error = float(np.linalg.norm(replay - c) / np.linalg.norm(c))
    assert error < 5e-5, error
    orth = float(np.max(abs(u.conj().T @ u - np.eye(30))))
    assert orth < 2e-5, orth
    core = c[100:1100]
    rms = np.sqrt(
        (
            np.mean(core.real**2, axis=0) * np.sum(u.real**2, axis=0)
            + np.mean(core.imag**2, axis=0) * np.sum(u.imag**2, axis=0)
            + 2 * np.mean(core.real * core.imag, axis=0) * np.sum(u.real * u.imag, axis=0)
        )
        / len(idx)
    )
    fields = (
        np.real(c[SNAP, :, None] * u.conj().T[None, :, :]).transpose(1, 0, 2) / rms[:, None, None]
    )
    check = np.real(core[:, 0, None] * u[:, 0].conj()[None, :])
    assert np.isclose(np.sqrt(np.mean(check**2)), rms[0])
    limit = max(2, float(np.ceil(np.quantile(abs(fields), 0.995) * 2) / 2))
    brain = Cortex(GEOMETRY)
    line = (
        np.real(c[WINDOW, :, None] * u[brain.retained_path, :].conj().T[None, :, :]).transpose(
            1, 0, 2
        )
        / rms[:, None, None]
    )
    t = (WINDOW - 300) * TR
    data = root / "source_data"
    train = json.loads((data / "training_variance.json").read_text())
    test = read_csv(data / "cpca_cumulative_variance_distribution.csv")
    v = np.array(train["individual"])
    fig = plt.figure(figsize=(7.4, 8.6))
    fig.text(
        0.07,
        0.974,
        "Individual CPCA modes: spatial structure and evolution",
        fontsize=11,
        fontweight="bold",
        color=DARK,
    )
    fig.text(
        0.07,
        0.954,
        "Fixed REST1 LR basis · CPC1-6 · recorded REST2 LR example",
        fontsize=7,
        color=GREY,
    )
    grid = fig.add_gridspec(
        6,
        7,
        left=0.105,
        right=0.938,
        bottom=0.376,
        top=0.889,
        width_ratios=[1, 1, 1, 1, 1, 1, 1.6],
        hspace=0.08,
        wspace=0.055,
    )
    first = [grid[0, j].get_position(fig) for j in range(7)]
    fig.text(first[0].x0, 0.917, "a  Amplitude", fontsize=7.6, fontweight="bold", color=DARK)
    fig.text(first[1].x0, 0.917, "b  Phase", fontsize=7.6, fontweight="bold", color=DARK)
    fig.text(
        first[2].x0, 0.917, "c  Single-mode evolution", fontsize=8, fontweight="bold", color=DARK
    )
    fig.text(first[6].x0, 0.902, "Along the cortical path", fontsize=6.1, color=DARK)
    for row, k in enumerate(SELECTED):
        ax = fig.add_subplot(grid[row, 0])
        a = abs(u[:, k])
        brain.draw(ax, a / a.max(), "L", 1, kind="amplitude")
        ax.text(
            -0.065,
            0.72,
            f"CPC{k + 1}",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontweight="bold",
            fontsize=6.8,
        )
        ax.text(
            -0.065,
            0.44,
            f"{100 * v[k]:.2f}%",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=5.8,
            color=GREY,
        )
        ax = fig.add_subplot(grid[row, 1])
        brain.draw(ax, np.angle(u[:, k].conj()), "L", np.pi, kind="phase")
        for j, frame in enumerate(SNAP):
            ax = fig.add_subplot(grid[row, j + 2])
            brain.draw(ax, fields[k, j], "L", limit, path=j == 0)
            if row == 0:
                ax.set_title(f"{(frame - 300) * TR:.2f} s", fontsize=6.4, pad=4)
        ax = fig.add_subplot(grid[row, 6])
        dd = brain.distance
        de = np.r_[
            dd[0] - (dd[1] - dd[0]) / 2, (dd[1:] + dd[:-1]) / 2, dd[-1] + (dd[-1] - dd[-2]) / 2
        ]
        ax.pcolormesh(
            np.r_[t - TR / 2, t[-1] + TR / 2],
            de,
            line[k].T,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            shading="flat",
            rasterized=False,
        )
        for frame in SNAP:
            ax.axvline((frame - 300) * TR, color="#272D30", alpha=0.7, lw=0.35)
        ax.set(
            xlim=(0, t[-1]), ylim=(dd[0], dd[-1]), xticks=[0, 72, 144], yticks=[0, round(dd[-1])]
        )
        ax.yaxis.tick_right()
        ax.tick_params(length=1.5, pad=1.2, labelsize=5.3)
        ax.set_yticklabels(["0", f"{dd[-1]:.0f}"] if row in [0, 5] else [])
        if row == 5:
            ax.set_xlabel("Time (s)", fontsize=6, labelpad=2)
        else:
            ax.set_xticklabels([])
    fig.text(
        0.981, 0.632, "Distance along path (mm)", rotation=90, fontsize=6, va="center", ha="center"
    )
    for xpos, width, cmap, limits, ticks, ticklabels, title_ in [
        (first[0].x0, first[0].width * 0.80, "magma", (0, 1), [0, 1], None, "Amplitude / maximum"),
        (
            first[1].x0,
            first[1].width * 0.80,
            "twilight",
            (-np.pi, np.pi),
            [-np.pi, 0, np.pi],
            ["-π", "0", "π"],
            "Spatial phase",
        ),
        (
            0.394,
            0.365,
            "RdBu_r",
            (-limit, limit),
            [-limit, 0, limit],
            None,
            "Single-CPC contribution / component RMS",
        ),
    ]:
        cb = fig.colorbar(
            ScalarMappable(norm=Normalize(*limits), cmap=cmap),
            cax=fig.add_axes([xpos, 0.34, width, 0.007]),
            orientation="horizontal",
            ticks=ticks,
        )
        if ticklabels is not None:
            cb.ax.set_xticklabels(ticklabels)
        cb.ax.tick_params(labelsize=5.5, length=1.5, pad=1)
        cb.set_label(title_, fontsize=5.5, labelpad=2)
    fig.text(0.07, 0.295, "d  Variance retained", fontsize=8.5, fontweight="bold", color=DARK)
    fig.text(
        0.563,
        0.295,
        "Individual CPC contributions on REST2",
        fontsize=7.6,
        fontweight="bold",
        color=DARK,
    )
    ax = fig.add_axes([0.085, 0.096, 0.356, 0.177])
    rank = np.array([float(r["rank"]) for r in test])
    mean = np.array([float(r["mean"]) for r in test])
    ax.plot(range(1, 51), 100 * np.array(train["cumulative"]), color=BLUE, label="REST1 LR basis")
    cumulative_sd = np.array([float(r["sd"]) for r in test])
    interval = 100 * np.array([cumulative_sd, cumulative_sd])
    assert np.all(interval > 0)
    # Draw participant SDs above the mean markers at their actual data-scale extent.
    ax.errorbar(
        rank,
        100 * mean,
        yerr=interval,
        fmt="o",
        color=ORANGE,
        ms=2.2,
        mfc="white",
        mec=ORANGE,
        mew=0.8,
        ecolor=DARK,
        elinewidth=0.8,
        capsize=4,
        capthick=0.8,
        barsabove=True,
        zorder=5,
        label="REST2 mean ± SD",
    )
    ax.axvline(30, color=GREY, ls=":", lw=0.7)
    ax.set(
        xlim=(0, 51),
        ylim=(0, max(65, float(np.ceil(100 * (mean + cumulative_sd).max() / 5) * 5))),
        xlabel="Number of complex components",
        ylabel="Cumulative analytic variance (%)",
        xticks=[0, 10, 30, 50],
        yticks=[0, 20, 40, 60],
    )
    ax.text(
        0.09,
        0.29,
        "30 CPCs: 52.7%",
        transform=ax.transAxes,
        color=BLUE,
        fontsize=7.4,
        fontweight="bold",
    )
    rank30 = int(np.flatnonzero(rank == 30)[0])
    ax.text(
        0.09,
        0.17,
        f"REST2: {100 * mean[rank30]:.2f}%",
        transform=ax.transAxes,
        color=ORANGE,
        fontsize=6.8,
    )
    ax.text(
        0.09,
        0.06,
        f"SD: {100 * cumulative_sd[rank30]:.2f} percentage points",
        transform=ax.transAxes,
        color=DARK,
        fontsize=6,
    )
    ax.legend(frameon=False, fontsize=5.8, loc="upper left")
    ax = fig.add_axes([0.563, 0.096, 0.375, 0.177])
    colors = [ORANGE if k < 6 else BLUE for k in range(30)]
    component_stats = read_csv(data / "cpca_component_variance_distribution.csv")
    assert [int(row["component"]) for row in component_stats] == list(range(1, 31))
    component_mean = np.array([float(row["mean"]) for row in component_stats])
    component_sd = np.array([float(row["sd"]) for row in component_stats])
    component_error = 100 * np.array([component_sd, component_sd])
    assert np.all(component_sd > 0) and np.all(component_mean - component_sd > 0)
    ax.bar(
        np.arange(1, 31),
        100 * component_mean,
        color=colors,
        width=0.75,
        linewidth=0,
        yerr=component_error,
        error_kw=dict(ecolor=DARK, elinewidth=0.7, capsize=1.8, capthick=0.45),
    )
    ax.set_yscale("log")
    ax.set(
        xlim=(0.3, 30.7),
        ylim=(0.15, 30),
        xticks=[1, 5, 10, 15, 20, 25, 30],
        yticks=[0.2, 1, 5, 20],
        xlabel="CPC",
        ylabel="REST2 analytic variance (%)",
    )
    ax.set_yticklabels(["0.2", "1", "5", "20"])
    ax.minorticks_off()
    ax.text(
        0.98,
        0.91,
        "Mean ± SD (n = 1,003)",
        transform=ax.transAxes,
        ha="right",
        fontsize=6,
        color=DARK,
    )
    ax.text(
        0.98, 0.79, "Orange: CPC1-6", transform=ax.transAxes, ha="right", fontsize=6, color=ORANGE
    )
    ax.text(
        0.98,
        0.67,
        "Logarithmic vertical axis",
        transform=ax.transAxes,
        ha="right",
        fontsize=5.5,
        color=GREY,
    )
    fig.text(
        0.07,
        0.043,
        "Cortical views: left lateral. The same four recorded times are used for every CPC; black lines mark the fixed cortical path.",
        fontsize=5.8,
        color=GREY,
    )
    fig.text(
        0.07,
        0.025,
        "Each activity map contains one component. Row percentages refer to total training analytic covariance. All 30 modes appear in the supplement.",
        fontsize=5.8,
        color=GREY,
    )
    retain_vectors(fig)
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(
            root / f"Figure_1_CPCA_individual_modes.{ext}",
            dpi=300,
            metadata={"Creator": "Matplotlib; text retained"} if ext == "pdf" else None,
        )
        print("SAVED main", ext, flush=True)
    plt.close(fig)
    after = [fingerprint(p) for p in [BASIS, SCORES, RAW, GEOMETRY]]
    assert before == after
    metadata = dict(
        status="PASS",
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        layout="amplitude_phase_individual_evolution_variance",
        selection_source="user CPC1-6",
        vector_mesh_edge_overlap_points=0.35,
        seconds=time.time() - start_time,
        run_index=2,
        acquisition="REST2_LR",
        frames=WINDOW.tolist(),
        snapshot_frames=SNAP.tolist(),
        TR=TR,
        selected_components=[k + 1 for k in SELECTED],
        score_projection_relative_error=error,
        basis_orthogonality_max_error=orth,
        rest2_cumulative_variance=[
            dict(rank=int(k), mean_pct=float(m * 100), sd_pct=float(s * 100))
            for k, m, s in zip(rank, mean, cumulative_sd)
        ],
        errorbar_display=dict(
            statistic="Sample SD across participant means; ddof=1",
            summary_sha256=hashlib.sha256(
                (data / "cpca_cumulative_variance_distribution.csv").read_bytes()
            ).hexdigest(),
            cap_points=4,
            linewidth_points=0.8,
            marker_points=2.2,
            bars_above_markers=True,
            interval_scale_factor=1,
        ),
        component_variance_panel=dict(
            estimator="REST2 participant mean",
            components=30,
            errorbars="Sample SD across participant means; ddof=1",
            summary_sha256=hashlib.sha256(
                (data / "cpca_component_variance_distribution.csv").read_bytes()
            ).hexdigest(),
            cap_points=1.8,
            cap_thickness_points=0.45,
        ),
        display_limit=limit,
        snapshot_fraction_clipped=float(np.mean(abs(fields) > limit)),
        component_RMS=rms.tolist(),
        path_original_vertices=brain.path.tolist(),
        path_distance_mm=brain.distance.tolist(),
        path_endpoint_coordinates=brain.path_coordinates[[0, -1]].tolist(),
        source_preserved=before == after,
        input_fingerprints=before,
        interpretation="Single-mode reconstructions at recorded times; no 30-component sum; not an independent raw-BOLD propagation or neuronal-causality test.",
    )
    (root / "provenance/render_validation.json").write_text(json.dumps(metadata, indent=2))
    print(
        json.dumps(
            {
                k: metadata[k]
                for k in [
                    "status",
                    "seconds",
                    "score_projection_relative_error",
                    "display_limit",
                    "snapshot_fraction_clipped",
                ]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    main(a.root)
