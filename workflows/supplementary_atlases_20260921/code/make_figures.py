"""Full CPC/ICA/HMM atlases. Reuses the project's depth-sorted cortical rendering.
Run with the existing python. Text stays editable in PDF/SVG.
"""

import os  # Public release: configurable data roots.
from pathlib import Path
import json
import shutil
import hashlib
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "source_data"
SOURCE = Path(os.environ.get("HCP_SOURCE_PACKAGE", "/configure/HCP_SOURCE_PACKAGE"))
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 6,
        "axes.linewidth": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
        "savefig.facecolor": "white",
    }
)
VIEWS = [("L", -1), ("L", 1), ("R", -1), ("R", 1)]


def prepare():
    manifest = []
    for src, name in [
        (SOURCE / "figure_data/cpca_basis_REST1_LR.npz", "cpca_basis_REST1_LR.npz"),
        (SOURCE / "figure_data/surfaces_and_eigenmodes.npz", "surfaces_and_eigenmodes.npz"),
    ]:
        dst = DATA / name
        if not dst.exists():
            shutil.copy2(src, dst)

        def digest(p):
            return hashlib.sha256(p.read_bytes()).hexdigest()

        assert dst.is_file()
        if src.exists():
            assert digest(src) == digest(dst)
        manifest.append({"source": str(src), "copy": name, "sha256": digest(dst)})
    (ROOT / "provenance/local_sources.json").write_text(json.dumps(manifest, indent=2))


class Cortex:
    def __init__(self, s):
        self.s = s
        self.mesh = {}
        for h, side in VIEWS:
            v = s[h + "_inflated_vertices"]
            f = s[h + "_faces"]
            xy = np.c_[side * v[:, 1], v[:, 2]]
            self.mesh[h, side] = (xy, f, np.argsort(side * v[f, 0].mean(1)))

    def quartet(self, fig, rect, values, cmap, lo, hi, phase=False):
        x, y, w, h = rect
        offset = 0
        parts = {}
        for hemi in ["L", "R"]:
            n = len(self.s[hemi + "_indices"])
            parts[hemi] = values[offset : offset + n]
            offset += n
        assert offset == len(values)
        for j, (hemi, side) in enumerate(VIEWS):
            ax = fig.add_axes([x + j * w / 4, y, w / 4, h])
            xy, f, order = self.mesh[hemi, side]
            full = np.full(len(xy), np.nan)
            full[self.s[hemi + "_indices"]] = parts[hemi]
            v = full[f]
            valid = np.isfinite(v).all(1)
            avg = np.zeros(len(f))
            avg[valid] = np.angle(np.exp(1j * v[valid]).mean(1)) if phase else v[valid].mean(1)
            colors = plt.get_cmap(cmap)(Normalize(lo, hi)(avg))
            colors[~valid] = [0.84, 0.85, 0.86, 1]
            ax.add_collection(
                PolyCollection(
                    xy[f][order],
                    facecolors=colors[order],
                    edgecolors="face",
                    linewidths=0.12,
                    antialiaseds=False,
                    rasterized=True,
                )
            )
            ax.set(
                xlim=(xy[:, 0].min() - 2, xy[:, 0].max() + 2),
                ylim=(xy[:, 1].min() - 2, xy[:, 1].max() + 2),
            )
            ax.set_aspect("equal")
            ax.axis("off")


def bar(fig, rect, cmap, lo, hi, label, ticks, labels=None):
    cb = fig.colorbar(
        ScalarMappable(norm=Normalize(lo, hi), cmap=cmap),
        cax=fig.add_axes(rect),
        orientation="horizontal",
        ticks=ticks,
    )
    if labels:
        cb.ax.set_xticklabels(labels)
    cb.ax.tick_params(length=2, pad=1)
    cb.set_label(label, fontsize=6, labelpad=2)


def save(fig, name):
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(ROOT / f"{name}.{ext}", dpi=450 if ext != "png" else 220)
    plt.close(fig)
    print("SAVED", name, flush=True)


def cpc():
    b = np.load(DATA / "cpca_basis_REST1_LR.npz")
    psi = b["complex_vectors"][:, :30].conj()
    s = np.load(DATA / "surfaces_and_eigenmodes.npz")
    brain = Cortex(s)
    indices = np.r_[s["L_indices"], len(s["L_inflated_vertices"]) + s["R_indices"]]
    assert np.array_equal(indices, b["vertex_indices"])
    variance = 100 * b["eigenvalues"][:30] / b["eigenvalues"].sum()
    fig = plt.figure(figsize=(7.4, 9.5))
    fig.text(
        0.035,
        0.977,
        "Supplementary Fig. 1 | Spatial amplitude and phase of CPC1–30",
        fontsize=10,
        weight="bold",
    )
    fig.text(
        0.035,
        0.955,
        "REST1 LR basis · 4,801 cortical vertices · first 30 CPCs retain 52.7% of analytic variance",
        fontsize=6.7,
    )
    for k in range(30):
        row, col = divmod(k, 3)
        x = 0.035 + col * 0.326
        y = 0.934 - row * 0.085
        fig.text(x, y, f"CPC{k + 1}", weight="bold", fontsize=7)
        fig.text(x + 0.087, y, f"{variance[k]:.2f}%", fontsize=6)
        amp = np.abs(psi[:, k])
        amp /= amp.max()
        brain.quartet(fig, [x + 0.012, y - 0.034, 0.300, 0.031], amp, "magma", 0, 1)
        brain.quartet(
            fig,
            [x + 0.012, y - 0.068, 0.300, 0.031],
            np.angle(psi[:, k]),
            "twilight",
            -np.pi,
            np.pi,
            True,
        )
        fig.text(x, y - 0.019, "A", fontsize=5.3, va="center")
        fig.text(x, y - 0.053, "θ", fontsize=5.3, va="center")
    bar(fig, [0.10, 0.060, 0.28, 0.007], "magma", 0, 1, "Spatial amplitude / mode maximum", [0, 1])
    bar(
        fig,
        [0.60, 0.060, 0.28, 0.007],
        "twilight",
        -np.pi,
        np.pi,
        "Spatial phase",
        [-np.pi, 0, np.pi],
        ["−π", "0", "π"],
    )
    fig.text(
        0.035,
        0.014,
        "Each row: L lateral, L medial, R medial, R lateral. Percentages: individual training variance.",
        fontsize=6,
    )
    np.savetxt(
        DATA / "CPC30_training_variance.csv",
        np.c_[np.arange(1, 31), variance],
        delimiter=",",
        header="CPC,analytic_variance_percent",
        comments="",
    )
    save(fig, "Supplementary_Figure_1_CPC30")


if __name__ == "__main__":
    (ROOT / "provenance").mkdir(exist_ok=True)
    prepare()
    cpc()
