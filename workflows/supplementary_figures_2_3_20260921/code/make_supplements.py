"""Re-layout retained panels without changing their underlying data."""

import matplotlib.pyplot as plt
import source_correspondence as src
import make_atlas as atlas

ROOT = src.ROOT


def save(fig, name):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(ROOT / f"{name}.{ext}", dpi=300 if ext != "png" else 220)
    plt.close(fig)
    print("SAVED", name, flush=True)


def combined():
    fig = plt.figure(figsize=(7.4, 9.5))
    original = plt.figure
    src.save = lambda *args: None
    plt.figure = lambda *args, **kwargs: fig
    try:
        src.ica()
        ia = list(fig.axes)
        it = list(fig.texts)
        src.state()
        sa = [a for a in fig.axes if a not in ia]
        st = [t for t in fig.texts if t not in it]
    finally:
        plt.figure = original
    ia[2].remove()
    for a in sa[:4]:
        a.remove()
    for t in it + st:
        t.remove()
    fig.text(
        0.065,
        0.973,
        "Supplementary Fig. 3 | CPC correspondence with ICA networks and HMM states",
        fontsize=9.3,
        weight="bold",
    )
    src.panel(fig, 0.065, 0.936, "a  Individual CPC–ICA correspondence")
    ia[0].set_position([0.075, 0.709, 0.82, 0.212])
    ia[1].set_position([0.92, 0.719, 0.013, 0.192])
    src.panel(fig, 0.065, 0.650, "b  Observed and estimated ICA activity")
    for j, a in enumerate(ia[3:6]):
        a.set_position([0.075, 0.583 - j * 0.061, 0.49, 0.044])
    src.panel(fig, 0.64, 0.650, "c  ICA functional connectivity")
    for j, a in enumerate(ia[6:8]):
        a.set_position([0.655 + j * 0.185, 0.482, 0.14, 0.13])
    ia[8].set_position([0.69, 0.460, 0.24, 0.007])
    fig.text(0.655, 0.414, it[5].get_text(), fontsize=6.3)
    src.panel(fig, 0.065, 0.386, "d  Observed and CPC30-reconstructed state maps")
    # Transform all cortex rectangles from their original rows to a compact four-row grid.
    for s in range(12):
        row, col = divmod(s, 3)
        oldy = 0.393 - row * 0.085
        newy = 0.359 - row * 0.073
        x = 0.065 + col * 0.218
        fig.text(x, newy, f"State {s + 1}", fontsize=6.5, weight="bold")
        for j in range(2):
            fig.text(x, newy - 0.014 - j * 0.024, ["O", "C"][j], fontsize=5.3, va="center")
            for a in sa[4 + s * 8 + j * 4 : 4 + s * 8 + j * 4 + 4]:
                p = a.get_position(original=True)
                a.set_position([p.x0, newy - 0.027 - j * 0.024, p.width, 0.024])
    src.panel(fig, 0.747, 0.386, "e  Spatial correspondence")
    sa[100].set_position([0.79, 0.099, 0.165, 0.263])
    sa[101].set_position([0.245, 0.053, 0.28, 0.007])
    fig.text(
        0.065,
        0.015,
        "REST2 LR + RL · n = 1,003 · O: observed; C: CPC30 · error bars: participant SD",
        fontsize=6,
    )
    save(fig, "Supplementary_Figure_3_CPC_correspondence")


def atlas_revised():
    box = []
    atlas.save = lambda fig, name: box.append(fig)
    atlas.ica_hmm()
    fig = box[0]
    for a in list(fig.axes):
        p = a.get_position(original=True)
        if p.y0 < 0.28:
            a.remove()
        else:
            a.set_position([p.x0, (p.y0 - 0.26) / 0.74, p.width, p.height / 0.74])
    for t in list(fig.texts):
        x, y = t.get_position()
        if y < 0.26:
            t.remove()
        else:
            t.set_position((x, (y - 0.26) / 0.74))
    fig.set_size_inches(7.4, 8.0)
    save(fig, "Supplementary_Figure_2_ICA50_HMM12")


if __name__ == "__main__":
    (ROOT / "provenance").mkdir(exist_ok=True)
    combined()
    atlas_revised()
