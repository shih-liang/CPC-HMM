"""Full CPC/ICA/HMM atlases. Reuses the project's depth-sorted cortical rendering.
Run with the existing python. Text stays editable in PDF/SVG.
"""

import os  # Public release: configurable data roots.
from pathlib import Path
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


def ica_hmm():
    import nibabel as nib

    image = nib.load(DATA / "melodic_IC.dscalar.nii")
    maps = np.asarray(image.dataobj)
    assert maps.shape == (50, 91282)
    surfaces = {}
    parts = []
    for name, sl, bm in image.header.get_axis(1).iter_structures():
        if name not in ["CIFTI_STRUCTURE_CORTEX_LEFT", "CIFTI_STRUCTURE_CORTEX_RIGHT"]:
            continue
        h = "L" if name.endswith("LEFT") else "R"
        surf = nib.load(DATA / f"S900.{h}.midthickness_MSMAll.32k_fs_LR.surf.gii")
        vertices, faces = surf.agg_data()
        assert len(vertices) == 32492
        surfaces[h + "_inflated_vertices"] = vertices
        surfaces[h + "_faces"] = faces
        surfaces[h + "_indices"] = bm.vertex
        parts.append(maps[:, sl])
    cortical = np.concatenate(parts, axis=1)
    brain = Cortex(surfaces)
    model = np.load(DATA / "HMM_reference.npz")
    mu = model["means"]
    cov = model["covariances"]
    A = model["transition"]
    pi = model["initial"]
    assert mu.shape == (12, 50) and cov.shape == (12, 50, 50)
    assert np.allclose(A.sum(1), 1) and np.isclose(pi.sum(), 1)
    assert np.allclose(cov, cov.transpose(0, 2, 1)) and np.linalg.eigvalsh(cov).min() > 0
    fig = plt.figure(figsize=(7.4, 10.8))
    fig.text(
        0.035,
        0.978,
        "Supplementary Fig. 2 | ICA50 components and the 12-state HMM",
        fontsize=10,
        weight="bold",
    )
    fig.text(0.035, 0.950, "a  All 50 ICA cortical maps", fontsize=8, weight="bold")
    for k in range(50):
        row, col = divmod(k, 5)
        x = 0.035 + col * 0.193
        y = 0.929 - row * 0.035
        fig.text(x, y, f"IC{k + 1}", fontsize=6.3, weight="bold")
        brain.quartet(
            fig,
            [x, y - 0.025, 0.184, 0.023],
            cortical[k] / np.max(abs(cortical[k])),
            "RdBu_r",
            -1,
            1,
        )
    bar(
        fig,
        [0.09, 0.582, 0.28, 0.006],
        "RdBu_r",
        -1,
        1,
        "ICA weight / cortical absolute maximum",
        [-1, 0, 1],
    )
    state_copy = DATA / "state_associated_cortical_maps.npz"
    state_maps = np.load(state_copy)["state_observed"]
    state_brain = Cortex(np.load(DATA / "surfaces_and_eigenmodes.npz"))
    fig.text(0.035, 0.540, "b  State-associated cortical activity", fontsize=8, weight="bold")
    for k in range(12):
        row, col = divmod(k, 6)
        x = 0.035 + col * 0.160
        y = 0.522 - row * 0.038
        fig.text(x, y, f"State {k + 1}", fontsize=6.3, weight="bold")
        state_brain.quartet(fig, [x, y - 0.026, 0.151, 0.024], state_maps[k], "RdBu_r", -1.25, 1.25)
    bar(
        fig,
        [0.65, 0.582, 0.25, 0.006],
        "RdBu_r",
        -1.25,
        1.25,
        "Mean BOLD (input units)",
        [-1.25, 0, 1.25],
    )
    fig.text(0.035, 0.438, "c  State means in ICA coordinates", fontsize=8, weight="bold")
    ax = fig.add_axes([0.06, 0.326, 0.89, 0.101])
    ax.pcolormesh(
        np.arange(51) + 0.5,
        np.arange(13) + 0.5,
        mu,
        cmap="RdBu_r",
        vmin=-1.5,
        vmax=1.5,
        rasterized=False,
    )
    ax.set(
        xlim=(0.5, 50.5),
        ylim=(12.5, 0.5),
        xticks=[1, 10, 20, 30, 40, 50],
        yticks=[1, 4, 8, 12],
        xlabel="ICA component",
        ylabel="State",
    )
    ax.tick_params(length=2, pad=1)
    bar(
        fig,
        [0.36, 0.293, 0.28, 0.006],
        "RdBu_r",
        -1.5,
        1.5,
        "Mean (standardized ICA units)",
        [-1.5, 0, 1.5],
    )
    fig.text(0.035, 0.248, "d  Full state covariance matrices", fontsize=8, weight="bold")
    for k in range(12):
        row, col = divmod(k, 6)
        x = 0.054 + col * 0.158
        y = 0.141 - row * 0.099
        ax = fig.add_axes([x, y, 0.117, 0.0825])
        ax.pcolormesh(
            np.arange(51) + 0.5, np.arange(51) + 0.5, cov[k], cmap="RdBu_r", vmin=-2.5, vmax=2.5
        )
        ax.set(xlim=(0.5, 50.5), ylim=(50.5, 0.5), aspect="equal", xticks=[1, 50], yticks=[1, 50])
        ax.tick_params(length=1.5, pad=1, labelsize=4.6)
        ax.set_title(f"State {k + 1}", fontsize=6.5, pad=3)
    bar(fig, [0.82, 0.269, 0.145, 0.006], "RdBu_r", -2.5, 2.5, "Covariance", [-2.5, 0, 2.5])
    fig.text(
        0.035,
        0.014,
        "Covariance axes: ICA components 1–50. All states retain the reference-HMM numbering.",
        fontsize=6,
    )
    np.savetxt(
        DATA / "HMM_state_means.csv",
        mu,
        delimiter=",",
        header=",".join(f"IC{k + 1}" for k in range(50)),
        comments="",
    )
    np.savetxt(
        DATA / "HMM_transition_probabilities.csv",
        A,
        delimiter=",",
        header=",".join(f"State{k + 1}" for k in range(12)),
        comments="",
    )
    np.savetxt(
        DATA / "HMM_initial_probabilities.csv",
        np.c_[np.arange(1, 13), pi],
        delimiter=",",
        header="state,initial_probability",
        comments="",
    )
    save(fig, "Supplementary_Figure_2_ICA50_HMM12_revised")
