#!/usr/bin/env python3
"""
Side-by-side visualisation of class centroids in Euclidean vs Poincare
space, for one (model, dataset) pair where the gain is concrete.

We pick CIFAR-100 + DINOv2-G:
  - 100 fine classes, 20 WordNet superclasses (colours).
  - 2D PCA on raw centroids -> Euclidean panel.
  - Apply our calibrated phi projection in 2D -> Poincare disk panel.

Saves figures/fig_euclid_vs_poincare.pdf.

Run on remote (where features are cached); output PNG/PDF can be
scp'd back.
"""
import os, sys
sys.stdout.reconfigure(line_buffering=True)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path("/home/javi/Platonic")
CACHE = ROOT / "results/practical_tasks_cache"
OUT = ROOT / "results"

MODEL = "dinov2_g"
DATASET = "cifar100"
TARGET = 1 / np.sqrt(2)

# CIFAR-100 fine -> coarse mapping (20 superclasses)
COARSE = {0:[4,30,55,72,95],1:[1,32,67,73,91],2:[54,62,70,82,92],3:[9,10,16,28,61],
          4:[0,51,53,57,83],5:[22,39,40,86,87],6:[5,20,25,84,94],7:[6,7,14,18,24],
          8:[3,42,43,88,97],9:[12,17,37,68,76],10:[23,33,49,60,71],11:[15,19,21,31,38],
          12:[34,63,64,66,75],13:[26,45,77,79,99],14:[2,11,35,46,98],15:[27,29,44,78,93],
          16:[36,50,65,74,80],17:[47,52,56,59,96],18:[8,13,48,58,90],19:[41,69,81,85,89]}
COARSE_NAMES = ["aquatic mamm.","fish","flowers","food cont.","fruit veg.",
                "househ. elec.","househ. furn.","insects","large carniv.","large outdoor",
                "large nat. out.","large omn.","medium mamm.","non-insect inv.","people",
                "reptiles","small mamm.","trees","veh. 1","veh. 2"]
fine_to_coarse = np.zeros(100, dtype=int)
for c, fs in COARSE.items():
    for f in fs: fine_to_coarse[f] = c


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
    n_classes = int(y.max()) + 1
    cents = np.stack([X[y == c].mean(0) for c in range(n_classes)])
    coarse = fine_to_coarse[np.arange(n_classes)]

    # ---- 2D PCA in original space ----
    mu_e = cents.mean(0)
    Xc = cents - mu_e
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pca2 = Xc @ Vt[:2].T  # n x 2

    # ---- Project to Poincare ball, then 2D PCA in projected space ----
    p95 = np.percentile(np.linalg.norm(cents - mu_e, axis=1), 95)
    cents_h = project(cents, mu_e, p95)
    mu_h = cents_h.mean(0)
    Xch = cents_h - mu_h
    Uh, Sh, Vth = np.linalg.svd(Xch, full_matrices=False)
    poin2 = Xch @ Vth[:2].T
    # Re-fit into a unit disk for visualisation: rescale so the MAX 2D-norm
    # lands at 0.92 (leave a margin so no point sits on the boundary).
    nrm2 = np.linalg.norm(poin2, axis=1)
    s2 = 0.92 / nrm2.max()
    poin2_disk = poin2 * s2

    # ---- Plot ----
    plt.rcParams.update({"font.family":"serif","font.size":10.5,
                         "figure.facecolor":"white","savefig.facecolor":"white",
                         "axes.facecolor":"white"})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.5))

    cmap = plt.cm.tab20
    colours = cmap(np.linspace(0, 1, 20))

    def add_super_labels(ax, coords):
        """Place superclass name at the median position of its centroids."""
        for c in range(20):
            mask = coarse == c
            if mask.sum() == 0: continue
            x_lab = float(np.median(coords[mask, 0]))
            y_lab = float(np.median(coords[mask, 1]))
            ax.text(x_lab, y_lab, COARSE_NAMES[c], fontsize=7.5,
                    ha="center", va="center",
                    color="#222222", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                              edgecolor=colours[c], linewidth=1.0, alpha=0.85),
                    zorder=5)

    # Euclidean panel
    ax = axes[0]
    for c in range(20):
        mask = coarse == c
        ax.scatter(pca2[mask,0], pca2[mask,1], c=[colours[c]], s=46,
                   alpha=0.65, edgecolors="black", linewidth=0.3, zorder=2)
    add_super_labels(ax, pca2)
    ax.set_title("(a) Euclidean: 2D PCA of centroids", fontsize=12)
    ax.set_xlabel("PC1", fontsize=10.5)
    ax.set_ylabel("PC2", fontsize=10.5)
    ax.set_aspect("equal", adjustable="box")
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    ax.grid(False)

    # Poincare disk panel
    ax = axes[1]
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="#888888", linewidth=1.0)
    ax.fill(np.cos(theta), np.sin(theta), color="#f8f8f8", zorder=0)
    for c in range(20):
        mask = coarse == c
        ax.scatter(poin2_disk[mask,0], poin2_disk[mask,1], c=[colours[c]], s=46,
                   alpha=0.65, edgecolors="black", linewidth=0.3, zorder=2)
    add_super_labels(ax, poin2_disk)
    ax.set_title(r"(b) Poincar" + "é" + r" ball: $\phi$-projected centroids", fontsize=12)
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ("top","right","bottom","left"): ax.spines[sp].set_visible(False)
    ax.grid(False)

    fig.suptitle(f"{MODEL.upper()} on {DATASET.upper()} -- 100 fine classes, 20 superclass labels; $\\delta = 0.07$, ORC $= +0.65$",
                 fontsize=11, y=0.98)
    plt.subplots_adjust(top=0.92, bottom=0.06, wspace=0.10)
    plt.savefig(OUT / "fig_euclid_vs_poincare.pdf", dpi=200, bbox_inches="tight")
    plt.savefig(OUT / "fig_euclid_vs_poincare.png", dpi=200, bbox_inches="tight")
    print(f"Saved {OUT}/fig_euclid_vs_poincare.pdf (and .png)")


if __name__ == "__main__":
    main()
