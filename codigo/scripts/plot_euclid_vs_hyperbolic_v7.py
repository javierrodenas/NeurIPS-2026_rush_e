#!/usr/bin/env python3
"""
v7: Euclid vs Poincare on CIFAR-100 + DINOv2-G.

Cleaner take: collapse the 100 fine centroids into 20 SUPERCLASS centroids
(one labelled point per superclass) and do the same 2D PCA on both panels.

  Left:  PCA on raw super-centroids (Euclidean).
  Right: PCA on phi-projected super-centroids, rescaled into the unit disk.

Same number of points (20), same projection method (PCA), only the metric
of the underlying space differs. Each panel is a fair 2D snapshot.
"""
import os, sys
sys.stdout.reconfigure(line_buffering=True)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/javi/Platonic")
CACHE = ROOT / "results/practical_tasks_cache"
OUT = ROOT / "results"

MODEL = "dinov2_g"
DATASET = "cifar100"
TARGET = 1 / np.sqrt(2)

COARSE = {0:[4,30,55,72,95],1:[1,32,67,73,91],2:[54,62,70,82,92],3:[9,10,16,28,61],
          4:[0,51,53,57,83],5:[22,39,40,86,87],6:[5,20,25,84,94],7:[6,7,14,18,24],
          8:[3,42,43,88,97],9:[12,17,37,68,76],10:[23,33,49,60,71],11:[15,19,21,31,38],
          12:[34,63,64,66,75],13:[26,45,77,79,99],14:[2,11,35,46,98],15:[27,29,44,78,93],
          16:[36,50,65,74,80],17:[47,52,56,59,96],18:[8,13,48,58,90],19:[41,69,81,85,89]}
COARSE_NAMES = ["aquatic mamm.","fish","flowers","food cont.","fruit & veg.",
                "h.h. elec.","h.h. furn.","insects","large carniv.","large outdoor",
                "large nat.","large omn.","medium mamm.","invertebrates","people",
                "reptiles","small mamm.","trees","vehicles 1","vehicles 2"]

# Super-super grouping for colouring: living vs man-made vs natural-outdoor
LIVING = {0,1,2,4,7,8,11,12,13,14,15,16,17}  # mammals, fish, plants, insects, people, reptiles
MANMADE = {3,5,6,18,19}                       # food cont, electronics, furniture, vehicles
OUTDOOR = {9,10}                               # large outdoor + large natural

fine_to_coarse = np.zeros(100, dtype=int)
for c, fs in COARSE.items():
    for f in fs: fine_to_coarse[f] = c


def super_color(c):
    if c in LIVING:
        return "#2ca02c"
    if c in MANMADE:
        return "#1f77b4"
    return "#8c564b"


def project(X, mu, p95, target=TARGET):
    s = 2 * np.arctanh(target) / max(p95, 1e-7)
    Xs = (X - mu) * s
    nrm = np.linalg.norm(Xs, axis=-1, keepdims=True).clip(min=1e-7)
    Y = np.tanh(nrm/2) / nrm * Xs
    cur = np.linalg.norm(Y, axis=-1, keepdims=True).clip(min=1e-7)
    return Y * np.minimum((1.0 - 1e-3)/cur, 1.0)


def main():
    d = np.load(CACHE / f"{MODEL}_{DATASET}_train.npz")
    X = d["features"].astype(np.float32); y = d["labels"].astype(np.int64)
    fine_cents = np.stack([X[y == c].mean(0) for c in range(100)])
    super_cents = np.stack([fine_cents[fine_to_coarse == c].mean(0)
                            for c in range(20)])

    # ---- LEFT: Euclidean PCA of super-centroids ----
    mu_e = super_cents.mean(0)
    Xc = super_cents - mu_e
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pca_e = Xc @ Vt[:2].T
    var_e = (S**2 / (S**2).sum())[:2]

    # ---- RIGHT: phi-project, then PCA in projected space, then rescale to disk ----
    mu_h = fine_cents.mean(0)
    p95 = np.percentile(np.linalg.norm(fine_cents - mu_h, axis=1), 95)
    super_h = project(super_cents, mu_h, p95)
    mu_h2 = super_h.mean(0)
    Xch = super_h - mu_h2
    Uh, Sh, Vth = np.linalg.svd(Xch, full_matrices=False)
    pca_h = Xch @ Vth[:2].T
    var_h = (Sh**2 / (Sh**2).sum())[:2]
    nrm = np.linalg.norm(pca_h, axis=1)
    pca_h_disk = pca_h * (0.92 / nrm.max())

    print(f"Variance explained — Euclidean PCA: {var_e[0]:.3f} {var_e[1]:.3f}  (sum {var_e.sum():.3f})")
    print(f"Variance explained — Hyperbolic PCA: {var_h[0]:.3f} {var_h[1]:.3f}  (sum {var_h.sum():.3f})")

    plt.rcParams.update({"font.family":"serif","font.size":11,
                         "figure.facecolor":"white","savefig.facecolor":"white",
                         "axes.facecolor":"white"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.0))

    def draw_panel(ax, coords, title, draw_disk=False, var=None):
        if draw_disk:
            theta = np.linspace(0, 2*np.pi, 200)
            ax.plot(np.cos(theta), np.sin(theta), color="#888888", linewidth=1.0)
            ax.fill(np.cos(theta), np.sin(theta), color="#fafafa", zorder=0)
        for c in range(20):
            color = super_color(c)
            ax.scatter(coords[c, 0], coords[c, 1],
                       c=color, s=120, alpha=0.95,
                       edgecolors="black", linewidth=0.6, zorder=3)
        # Label-placement: alternate left/right by sign(x)
        for c in range(20):
            xr, yr = coords[c]
            ha = "left" if xr >= 0 else "right"
            dx = 0.04 if ha == "left" else -0.04
            if not draw_disk:
                # axis-units offset proportional to coord range
                rng = max(np.abs(coords).max(), 1e-6)
                dx = 0.04 * rng if ha == "left" else -0.04 * rng
            ax.text(xr + dx, yr, COARSE_NAMES[c], fontsize=8.5,
                    ha=ha, va="center", color=super_color(c),
                    fontweight="bold", zorder=5)
        sub = title
        if var is not None:
            sub = f"{title}  (var explained: {var[0]*100:.0f}% + {var[1]*100:.0f}%)"
        ax.set_title(sub, fontsize=11.5)
        ax.set_aspect("equal", adjustable="box")
        if draw_disk:
            ax.set_xlim(-1.10, 1.10); ax.set_ylim(-1.10, 1.10)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ("top","right","bottom","left"):
                ax.spines[sp].set_visible(False)
        else:
            ax.set_xlabel("PC1", fontsize=10.5)
            ax.set_ylabel("PC2", fontsize=10.5)
            for sp in ("top","right"):
                ax.spines[sp].set_visible(False)
            # Pad x-range so labels fit
            xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
            ymin, ymax = coords[:, 1].min(), coords[:, 1].max()
            xpad = 0.18 * (xmax - xmin)
            ypad = 0.10 * (ymax - ymin)
            ax.set_xlim(xmin - xpad, xmax + xpad)
            ax.set_ylim(ymin - ypad, ymax + ypad)
        ax.grid(False)

    draw_panel(axes[0], pca_e,
               "(a) Euclidean: PCA of superclass centroids",
               var=var_e)
    draw_panel(axes[1], pca_h_disk,
               r"(b) Poincar" + "é" + r" disk: PCA of $\phi$-projected centroids",
               draw_disk=True, var=var_h)

    # Legend for super-super groups
    import matplotlib.lines as mlines
    handles = [
        mlines.Line2D([], [], marker="o", color="w",
                      markerfacecolor="#2ca02c", markeredgecolor="black",
                      markersize=10, label="Living"),
        mlines.Line2D([], [], marker="o", color="w",
                      markerfacecolor="#1f77b4", markeredgecolor="black",
                      markersize=10, label="Man-made"),
        mlines.Line2D([], [], marker="o", color="w",
                      markerfacecolor="#8c564b", markeredgecolor="black",
                      markersize=10, label="Outdoor scenes"),
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.01),
               ncol=3, frameon=False, fontsize=10.5, columnspacing=2.5)

    fig.suptitle(f"{MODEL.upper()} on CIFAR-100 -- 20 superclass centroids; "
                 r"$\delta = 0.07$, ORC $= +0.65$",
                 fontsize=11, y=0.96)
    plt.subplots_adjust(top=0.86, bottom=0.08, wspace=0.22)
    plt.savefig(OUT / "fig_euclid_vs_poincare.pdf", dpi=200, bbox_inches="tight")
    plt.savefig(OUT / "fig_euclid_vs_poincare.png", dpi=200, bbox_inches="tight")
    print(f"\nSaved {OUT}/fig_euclid_vs_poincare.pdf")


if __name__ == "__main__":
    main()
