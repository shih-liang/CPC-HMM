"""Figure 4: cortical geometry, activity retention and matched state decoding."""

import os  # Public release: configurable data roots.
from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, QuadMesh
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import FancyArrowPatch

DEFAULT_SOURCE = Path(os.environ.get("HCP_SOURCE_PACKAGE", "/configure/HCP_SOURCE_PACKAGE"))
ROOT = Path(__file__).resolve().parents[1]
BLUE, DARK, GREY = "#286B8B", "#203444", "#788690"
BAND_LABELS = ["1 (constant)", "2-10", "11-50", "51-200", "Uncaptured"]
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
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "text.usetex": False,
        "savefig.facecolor": "white",
    }
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def prepare_inputs(root, source):
    data = root / "source_data"
    data.mkdir(exist_ok=True, parents=True)
    entries = [
        ("figure_data/surfaces_and_eigenmodes.npz", "surfaces_and_eigenmodes.npz"),
        ("figure_data/cpca_basis_REST1_LR.npz", "cpca_basis_REST1_LR.npz"),
        ("figure_data/geometry_incremental_energy.npz", "geometry_incremental_energy.npz"),
        ("source_data/Figure2_geometry_summary.csv", "original_geometry_summary.csv"),
    ]
    manifest = []
    for relative, name in entries:
        original, local = source / relative, data / name
        if original.is_file():
            source_hash = sha(original)
            if not local.exists():
                shutil.copy2(original, local)
            assert sha(local) == source_hash, f"Source differs: {name}"
            manifest.append(dict(source=str(original), copy=name, sha256=source_hash))
        else:
            assert local.is_file(), f"Missing source: {local}"
            manifest.append(dict(source="Packaged source copy", copy=name, sha256=sha(local)))
    return data, manifest


def cortical_map(ax, surface, hemi, values=None):
    """Reuse the existing lateral-view, depth-sorted triangle rendering convention."""
    side = -1 if hemi == "L" else 1
    vertices = surface[hemi + ("_pial_vertices" if values is None else "_inflated_vertices")]
    faces = surface[hemi + "_faces"]
    xy = np.c_[side * vertices[:, 1], vertices[:, 2]]
    order = np.argsort(side * vertices[faces, 0].mean(axis=1))
    if values is None:
        triangles = vertices[faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-15)
        light = np.array([side * 0.8, -0.25, 0.55])
        light /= np.linalg.norm(light)
        intensity = 0.53 + 0.40 * np.maximum(normals @ light, 0)
        colors = np.c_[intensity, intensity, intensity, np.ones(len(faces))]
        collection = PolyCollection(
            xy[faces][order],
            facecolors=colors[order],
            edgecolors="#83909A",
            linewidths=0.065,
            antialiaseds=False,
            rasterized=False,
        )
        collection.set_gid("cortical_surface_mesh")
    else:
        full = np.full(len(vertices), np.nan)
        full[surface[hemi + "_indices"]] = values
        triangle_values = full[faces]
        valid = np.isfinite(triangle_values).all(axis=1)
        means = np.zeros(len(faces))
        means[valid] = triangle_values[valid].mean(axis=1)
        colors = plt.get_cmap("RdBu_r")(Normalize(-1, 1)(means))
        colors[~valid] = [0.84, 0.85, 0.86, 1]
        collection = PolyCollection(
            xy[faces][order],
            facecolors=colors[order],
            edgecolors="face",
            linewidths=0.35,
            antialiaseds=False,
            rasterized=False,
        )
    ax.add_collection(collection)
    ax.set(
        xlim=(xy[:, 0].min() - 2, xy[:, 0].max() + 2), ylim=(xy[:, 1].min() - 2, xy[:, 1].max() + 2)
    )
    ax.set_aspect("equal")
    ax.axis("off")


def retain_vectors(fig):
    fig.canvas.draw()
    for ax in fig.axes:
        for collection in ax.collections:
            collection.set_rasterized(False)
            if (
                isinstance(collection, (PolyCollection, QuadMesh))
                and collection.get_gid() != "cortical_surface_mesh"
            ):
                collection.set_edgecolor("face")
                collection.set_linewidth(0.35)
                collection.set_antialiased(False)


def main(root, source):
    (root / "provenance").mkdir(exist_ok=True, parents=True)
    data, sources = prepare_inputs(root, source)
    surface = np.load(data / "surfaces_and_eigenmodes.npz")
    basis = np.load(data / "cpca_basis_REST1_LR.npz")
    increment = np.load(data / "geometry_incremental_energy.npz")["vertex"][:, :30]
    cumulative = increment.cumsum(axis=0)
    mean = cumulative.mean(axis=1)
    assert increment.shape == (200, 30) and np.all(increment >= 0)
    assert np.all(cumulative <= 1 + 1e-7)
    with (data / "original_geometry_summary.csv").open() as f:
        for row in csv.DictReader(f):
            assert (
                abs(
                    mean[int(row["modes_per_hemisphere"]) - 1]
                    - float(row["unweighted_mean_captured_energy"])
                )
                < 1e-12
            )

    # Validate the archived result against the supplied maps and a fresh QR projection.
    # The repeated extended CPCA eigendecomposition differs from the archived basis at rounding precision.
    psi = basis["complex_vectors"][:, :30].conj().astype(np.complex128)
    direct = np.zeros_like(increment)
    grams, qr_errors, offset = {}, {}, 0
    for hemi in ["L", "R"]:
        E = surface[hemi + "_eigenmodes"]
        assert E.shape[1] == 200
        Q, _ = np.linalg.qr(E, mode="reduced")
        direct += abs(Q.T @ psi[offset : offset + len(E)]) ** 2
        G = E.T @ E
        norm = G / np.sqrt(np.outer(np.diag(G), np.diag(G)))
        grams[hemi] = float(abs(norm - np.eye(200)).max())
        qr_errors[hemi] = float(abs(Q.T @ Q - np.eye(200)).max())
        offset += len(E)
    assert offset == len(psi) == 4801
    direct /= np.sum(abs(psi) ** 2, axis=0)
    projection_error = float(abs(direct.cumsum(axis=0) - cumulative).max())
    assert projection_error < 5e-5, projection_error

    bands = np.vstack(
        [
            increment[0],
            increment[1:10].sum(axis=0),
            increment[10:50].sum(axis=0),
            increment[50:200].sum(axis=0),
            1 - cumulative[-1],
        ]
    )
    assert np.all(bands >= 0) and np.allclose(bands.sum(axis=0), 1, atol=1e-12)
    write_csv(
        data / "geometric_order_contributions.csv",
        [
            dict(CPC=k + 1, **{label: float(bands[j, k]) for j, label in enumerate(BAND_LABELS)})
            for k in range(30)
        ],
    )
    write_csv(
        data / "cumulative_spatial_energy.csv",
        [
            dict(
                modes_per_hemisphere=m,
                **{f"CPC{k + 1}": float(cumulative[m - 1, k]) for k in range(30)},
                equal_mean=float(mean[m - 1]),
            )
            for m in range(1, 201)
        ],
    )
    write_csv(
        data / "individual_geometric_mode_contributions.csv",
        [
            dict(
                geometric_order=m,
                **{f"CPC{k + 1}": float(increment[m - 1, k]) for k in range(30)},
                equal_mean=float(increment[m - 1].mean()),
            )
            for m in range(1, 201)
        ],
    )
    write_csv(
        data / "cpc_spatial_coverage_summary.csv",
        [
            dict(
                modes_per_hemisphere=m,
                total_geometric_coefficients=2 * m,
                cortical_vertices=4801,
                spatial_dimension_fraction=2 * m / 4801,
                equal_mean_spatial_energy=float(mean[m - 1]),
            )
            for m in [10, 50, 200]
        ],
    )

    with (data / "geometry_activity_retention.csv").open() as f:
        activity = list(csv.DictReader(f))
    activity_summary = json.loads((data / "geometry_activity_summary.json").read_text())
    assert activity_summary["status"] == "PASS"
    geo_activity = np.array([float(row["geometry_activity_variance_fraction"]) for row in activity])
    assert np.array_equal([int(row["CPCs"]) for row in activity], np.arange(1, 201))
    with (data / "cpca_cortical_variance.csv").open() as f:
        cortical_activity = np.array(
            [float(row["cortical_activity_variance_fraction"]) for row in csv.DictReader(f)]
        )
    with (data / "geometry_cortical_variance.csv").open() as f:
        geometry_cortical = np.array(
            [float(row["cortical_activity_variance_fraction"]) for row in csv.DictReader(f)]
        )
    assert cortical_activity.shape == (400,) and geometry_cortical.shape == (200,)
    assert np.allclose(
        cortical_activity[:50],
        basis["eigenvalues"][:50].cumsum() / basis["eigenvalues"].sum(),
        atol=1e-5,
    )
    assert np.all(cortical_activity[1::2] >= geometry_cortical - 1e-5)
    for name in [
        "geometry_activity_retention.csv",
        "geometry_activity_summary.json",
        "cpca_cortical_variance.csv",
        "geometry_cortical_variance.csv",
    ]:
        sources.append(
            dict(source="Archived group covariance export", copy=name, sha256=sha(data / name))
        )

    with (data / "direct_decoding_summary.csv").open() as f:
        decoding = {row["condition"]: row for row in csv.DictReader(f)}
    assert list(decoding) == ["geo15", "geo50", "geo100", "geo150", "geo200"]
    decoding_validation = json.loads((data / "consensus_validation.json").read_text())
    assert decoding_validation["status"] == "PASS"
    assert decoding_validation["source_fingerprints_unchanged"]
    for name in [
        "direct_decoding_summary.csv",
        "consensus_state_metrics.csv",
        "consensus_validation.json",
        "archived_CPC30_consensus_verification.csv",
        "grid_validation_histories.csv",
        "additional_fitting_validation.json",
    ]:
        sources.append(
            dict(
                source="Six-HMM common-state evaluation on data server",
                copy=name,
                sha256=sha(data / name),
            )
        )

    fig = plt.figure(figsize=(7.4, 9.0))
    fig.text(
        0.07,
        0.974,
        "Cortical geometry and network-state information",
        fontsize=10.6,
        fontweight="bold",
        color=DARK,
    )
    fig.text(
        0.07,
        0.954,
        "CPC spatial composition  |  Activity variance  |  Six-HMM common states",
        fontsize=7,
        color=GREY,
    )

    # a: actual cortical surfaces and actual geometric eigenmodes, plus the projection relation.
    fig.text(
        0.07,
        0.919,
        "a  A spatial basis defined by cortical geometry",
        fontsize=8.8,
        fontweight="bold",
        color=DARK,
    )
    orders = [1, 2, 3, 4, 5, 6, 10, 20, 50, 100, 150, 200]
    grid = fig.add_gridspec(
        3, 4, left=0.265, right=0.95, bottom=0.649, top=0.888, hspace=0.36, wspace=0.15
    )
    fig.text(0.085, 0.875, "Cortical surface", fontsize=6.8, color=DARK)
    for hemi, y in [("L", 0.757), ("R", 0.650)]:
        ax = fig.add_axes([0.080, y, 0.14, 0.106])
        cortical_map(ax, surface, hemi)
        ax.text(
            -0.05,
            0.5,
            "Left" if hemi == "L" else "Right",
            transform=ax.transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=6,
            color=GREY,
        )
    for position, order in enumerate(orders):
        pair = grid[position // 4, position % 4].subgridspec(1, 2, wspace=0.01)
        bounds = grid[position // 4, position % 4].get_position(fig)
        fig.text(
            (bounds.x0 + bounds.x1) / 2,
            bounds.y1 + 0.002,
            f"Mode {order}",
            ha="center",
            fontsize=6.5,
            color=DARK,
        )
        for j, hemi in enumerate(["L", "R"]):
            values = surface[hemi + "_eigenmodes"][:, order - 1].copy()
            values *= np.sign(values[np.argmax(abs(values))])
            values /= abs(values).max()
            cortical_map(fig.add_subplot(pair[0, j]), surface, hemi, values)
    arrow = FancyArrowPatch(
        (0.221, 0.771),
        (0.250, 0.771),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=8,
        lw=0.8,
        color=GREY,
    )
    fig.add_artist(arrow)
    cb = fig.colorbar(
        ScalarMappable(norm=Normalize(-1, 1), cmap="RdBu_r"),
        cax=fig.add_axes([0.495, 0.619, 0.24, 0.008]),
        orientation="horizontal",
        ticks=[-1, 0, 1],
    )
    cb.ax.tick_params(labelsize=5.5, length=1.8, pad=1)
    cb.set_label("Normalized geometric mode value", fontsize=6, labelpad=2)
    fig.text(0.105, 0.555, "CPC spatial map", fontsize=7.5, color=DARK, va="center")
    fig.add_artist(
        FancyArrowPatch(
            (0.268, 0.556),
            (0.345, 0.556),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=8,
            lw=0.8,
            color=GREY,
        )
    )
    fig.text(
        0.368, 0.555, "QR projection", fontsize=7.5, fontweight="bold", color=BLUE, va="center"
    )
    fig.text(0.368, 0.537, "Separately in each hemisphere", fontsize=5.8, color=GREY)
    fig.add_artist(
        FancyArrowPatch(
            (0.535, 0.556),
            (0.605, 0.556),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=8,
            lw=0.8,
            color=GREY,
        )
    )
    fig.text(
        0.637,
        0.554,
        r"$\psi_k(r)\ \approx\ \sum_{m=1}^{M} b_{mk}\,\phi_m(r)$",
        fontsize=10,
        color=DARK,
        va="center",
    )

    # b: each column is one ordered QR increment, summed across hemispheres.
    fig.text(
        0.07,
        0.493,
        "b  Individual geometric mode contributions to CPC1-30",
        fontsize=8.8,
        fontweight="bold",
        color=DARK,
    )
    ax = fig.add_axes([0.095, 0.332, 0.765, 0.136])
    mesh = ax.pcolormesh(
        np.arange(201) + 0.5,
        np.arange(31) + 0.5,
        100 * increment.T,
        cmap="magma",
        norm=LogNorm(vmin=0.0001, vmax=100),
        rasterized=False,
        shading="flat",
    )
    mesh.set_gid("mode_contribution_matrix")
    assert np.array_equal(np.asarray(mesh.get_array()), 100 * increment.T)
    ax.set(
        xlim=(0.5, 200.5),
        ylim=(30.5, 0.5),
        xticks=[1, 25, 50, 100, 150, 200],
        yticks=[1, 5, 10, 15, 20, 25, 30],
        xlabel="Geometric mode order (per hemisphere)",
        ylabel="CPC",
    )
    ax.tick_params(length=2)
    cb = fig.colorbar(
        mesh, cax=fig.add_axes([0.883, 0.332, 0.012, 0.136]), ticks=[0.0001, 0.01, 1, 100]
    )
    cb.ax.set_yticklabels(["0.0001", "0.01", "1", "100"])
    cb.ax.tick_params(labelsize=5.5, length=2)
    cb.minorticks_off()
    cb.set_label("Incremental spatial energy (%)", fontsize=6, labelpad=4)

    # c: same activity target and variance denominator for both representations.
    fig.text(
        0.07, 0.265, "c  Activity variance retained", fontsize=8.8, fontweight="bold", color=DARK
    )
    ax = fig.add_axes([0.095, 0.089, 0.374, 0.151])
    orange = "#B57036"
    ax.plot(np.arange(1, 401), 100 * cortical_activity, color=BLUE, lw=1.8, label="CPCs", zorder=4)
    ax.plot(
        2 * np.arange(1, 201),
        100 * geometry_cortical,
        color=orange,
        lw=1.5,
        label="Geometric modes",
    )
    matched_M = activity_summary["CPCA30_vs_geometry200"][
        "geometry_modes_per_hemisphere_to_match_CPC30"
    ]
    target = 100 * cortical_activity[29]
    ax.plot([30, 2 * matched_M], [target, target], ls="--", lw=0.8, color=GREY, zorder=2)
    ax.scatter([30], [target], s=18, color=BLUE, zorder=5)
    ax.scatter(
        [2 * matched_M, 400],
        100 * geometry_cortical[[matched_M - 1, -1]],
        s=18,
        color=orange,
        zorder=5,
    )
    ax.annotate(
        f"30 CPCs: {target:.1f}%",
        (30, target),
        xytext=(2, 17),
        textcoords="offset points",
        fontsize=6.2,
        color=BLUE,
        arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.6),
    )
    ax.annotate(
        f"{2 * matched_M} geometric\ncoefficients",
        (2 * matched_M, target),
        xytext=(6, -24),
        textcoords="offset points",
        fontsize=5.9,
        color=orange,
        arrowprops=dict(arrowstyle="-", color=orange, lw=0.6),
    )
    ax.annotate(
        f"{100 * geometry_cortical[-1]:.1f}%",
        (400, 100 * geometry_cortical[-1]),
        xytext=(-5, -17),
        ha="right",
        textcoords="offset points",
        fontsize=6.1,
        color=orange,
    )
    ax.set(
        xlim=(0, 407),
        ylim=(0, 100),
        xticks=[0, 100, 200, 300, 400],
        yticks=[0, 25, 50, 75, 100],
        ylabel="Cortical activity variance (%)",
    )
    ax.tick_params(length=2)
    ax.legend(fontsize=5.4, frameon=False, loc="upper right")

    # d: geometric decoders scored on the same six-HMM unanimous REST2 frames.
    fig.text(0.545, 0.265, "d  Network-state decoding", fontsize=8.8, fontweight="bold", color=DARK)
    ax = fig.add_axes([0.595, 0.089, 0.355, 0.151])
    geo_rows = list(decoding.values())
    ax.errorbar(
        [int(row["complex_coordinates"]) for row in geo_rows],
        [100 * float(row["accuracy_mean"]) for row in geo_rows],
        yerr=[100 * float(row["accuracy_sd"]) for row in geo_rows],
        fmt="o-",
        color=orange,
        lw=1.3,
        ms=3.7,
        capsize=2.5,
        elinewidth=0.8,
        label="Geometric modes",
    )
    for row in geo_rows:
        count = int(row["complex_coordinates"])
        ax.annotate(
            f"{100 * float(row['accuracy_mean']):.1f}%",
            (count, 100 * (float(row["accuracy_mean"]) + float(row["accuracy_sd"]))),
            xytext=(3 if count == 30 else -3 if count == 400 else 0, 5),
            textcoords="offset points",
            ha="left" if count == 30 else "right" if count == 400 else "center",
            fontsize=6.1,
            color=orange,
        )
    ax.set(
        xlim=(0, 420),
        ylim=(0, 100),
        xticks=[30, 100, 200, 300, 400],
        yticks=[40, 50, 60, 70, 80, 90],
        ylabel="State decoding accuracy (%)",
    )
    ax.tick_params(length=2)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], fontsize=5.4, frameon=False, loc="lower right")
    fig.text(
        0.535, 0.046, "Complex coordinates per frame (both hemispheres)", ha="center", fontsize=7
    )
    fig.text(
        0.07,
        0.024,
        f"d, Six-HMM unanimous frames: {int(geo_rows[0]['frames']):,} ({100 * float(geo_rows[0]['coverage']):.2f}%). REST2 LR + RL; 50-frame windows.",
        fontsize=5.8,
        color=GREY,
    )
    fig.text(
        0.07,
        0.009,
        f"Mean ± SD across {int(geo_rows[0]['participants']):,} participants with support. c, REST1_LR covariance.",
        fontsize=5.8,
        color=GREY,
    )

    retain_vectors(fig)
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(root / f"Figure_4_geometric_basis.{ext}", dpi=300)
    # Inspect text boxes to catch cropped labels before PDF review.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    out_of_bounds = []
    for text in fig.findobj(match=matplotlib.text.Text):
        if text.get_visible() and text.get_text().strip():
            box = text.get_window_extent(renderer=renderer)
            if (
                box.x0 < -1
                or box.y0 < -1
                or box.x1 > fig.bbox.width + 1
                or box.y1 > fig.bbox.height + 1
            ):
                out_of_bounds.append(text.get_text())
    assert not out_of_bounds, out_of_bounds
    plt.close(fig)

    validation = dict(
        status="PASS",
        sources=sources,
        script_sha256=sha(Path(__file__)),
        selected_geometric_modes=orders,
        displayed_CPCs=list(range(1, 31)),
        spatial_metric="Unweighted retained cortical vertices; separate hemisphere QR projections",
        basis_gram_max_normalized_offdiagonal=grams,
        qr_orthogonality_error=qr_errors,
        archived_vs_recomputed_cumulative_fraction_max_error=projection_error,
        individual_contribution_matrix_shape=list(increment.T.shape),
        individual_contribution_color_scale_percent=[0.0001, 100],
        main_curve_estimator="Fraction of the same full cortical analytic variance retained by each representation",
        activity_analysis_scope="REST1_LR training covariance; descriptive group result",
        activity_variance_percent={
            str(k): float(100 * geo_activity[k - 1]) for k in [3, 10, 30, 50]
        },
        same_denominator_comparison=activity_summary["CPCA30_vs_geometry200"],
        decoding_accuracy_percent={
            name: 100 * float(row["accuracy_mean"]) for name, row in decoding.items()
        },
        decoding_SD_percent={
            name: 100 * float(row["accuracy_sd"]) for name, row in decoding.items()
        },
        decoding_conditions=list(decoding),
        decoding_bilateral_coordinate_counts=[int(row["complex_coordinates"]) for row in geo_rows],
        decoding_scope="Geometric-coordinate decoders; identical six-HMM unanimous REST2 frames; participant mean ± SD",
        decoding_frame_count=int(geo_rows[0]["frames"]),
        decoding_coverage=float(geo_rows[0]["coverage"]),
        CPC30_decoding_displayed=False,
        retained_energy_percent={str(m): float(100 * mean[m - 1]) for m in [10, 50, 200]},
        no_cropped_text=True,
        figure_text="PDF TrueType fonts; SVG text nodes; no text-to-path conversion requested.",
    )
    for item in sources:
        original = Path(item["source"])
        if original.is_file():
            assert sha(original) == item["sha256"], f"Source changed: {original}"
    validation["source_hashes_unchanged"] = True
    (root / "provenance/numerical_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    main(args.root, args.source)
