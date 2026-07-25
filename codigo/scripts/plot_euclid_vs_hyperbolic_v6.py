#!/usr/bin/env python3
"""
v6: Euclid vs Poincare on CIFAR-100 + DINOv2-G.

Spec from main.tex: "100 class centroids of CIFAR-100 ... coloured by the 20
superclasses. Left: 2D PCA in raw space. Right: same centroids after the
calibrated phi projection ... rescaled into the unit Poincare disk."

To make structure visible without 20 colour clashes:
  - Highlight 5 visually distinct superclasses with full colour + convex hull.
  - The remaining 15 superclasses are drawn in light gray (still visible as
    cloud, but not competing for attention).
"""
import os, sys
sys.stdout.reconfigure(line_buffering=True)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from scipy.spatial import ConvexHull

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
fine_to_coarse = np.zeros(100, dtype=int)
for c, fs in COARSE.items():
    for f in fs: fine_to_coarse[f] = c

# Five visually-distinct, semantically-distinct superclasses to highlight.
HIGHLIGHT = [17, 18, 14, 1, 2]  # trees, vehicles 1, people, fish, flowers
HIGHLIGHT_COLORS = {
    17: "#2ca02c",  # trees - green
    18: "#1f77b4",  # vehicles 1 - blue
    14: "#d62728",  # people - red
    1:  "#17becf",  # fish - cyan
    2:  "#9467bd",  # flowers - purple
}
GRAY = "#bbbbbb"


def project(X, mu, p95, target=TARGET):
    s = 2 * np.arctanh(target) / max(p95, 1e-7)
    Xs = (X - mu) * s
    nrm = np.linalg.norm(Xs, axis=-1, keepdims=True).clip(min=1e-7)
    Y = np.tanh(nrm/2) / nrm * Xs
    cur = np.linalg.norm(Y, axis=-1, keepdims=True).clip(min=1e-7)
    return Y * np.minimum((1.0 - 1e-3)/cur, 1.0)


def poincare_dist_matrix(X):
    n = len(X); D = np.zeros((n, n))
    x_sq = (X*X).sum(-1, keepdims=True)
    for i in range(n):
        d_sq = ((X[i] - X)**2).sum(-1)
        denom = ((1 - x_sq[i]) * (1 - x_sq.T[0])).clip(min=1e-7)
        arg = (1 + 2*d_sq / denom).clip(min=1+1e-7)
        D[i] = np.arccosh(arg)
    return D


def main():
    d = np.load(CACHE / f"{MODEL}_{DATASET}_train.npz")
    X = d["features"].astype(np.float32); y = d["labels"].astype(np.int64)
    n_classes = int(y.max()) + 1
    cents = np.stack([X[y == c].mean(0) for c in range(n_classes)])
    coarse = fine_to_coarse[np.arange(n_classes)]

    # ----- LEFT: Euclidean 2D PCA -----
    mu_e = cents.mean(0)
    Xc = cents - mu_e
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pca2 = Xc @ Vt[:2].T

    # ----- RIGHT: Hyperbolic-MDS on Poincare distance matrix -----
    p95 = np.percentile(np.linalg.norm(cents - mu_e, axis=1), 95)
    cents_h = project(cents, mu_e, p95)
    D_h = poincare_dist_matrix(cents_h)
    print(f"H dist range: [{D_h[D_h>0].min():.3f}, {D_h.max():.3f}]", flush=True)
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0,
              max_iter=2000, n_init=16, normalized_stress="auto")
    poin2 = mds.fit_transform(D_h)
    nrm2 = np.linalg.norm(poin2, axis=1)
    s2 = 0.92 / nrm2.max()
    poin2_disk = poin2 * s2

    plt.rcParams.update({"font.family":"serif","font.size":11,
                         "figure.facecolor":"white","savefig.facecolor":"white",
                         "axes.facecolor":"white"})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.7))

    def draw_panel(ax, coords, title, draw_disk=False):
        if draw_disk:
            theta = np.linspace(0, 2*np.pi, 200)
            ax.plot(np.cos(theta), np.sin(theta), color="#888888", linewidth=1.0)
            ax.fill(np.cos(theta), np.sin(theta), color="#fafafa", zorder=0)

        # First, draw all non-highlight superclasses in light gray
        not_high_mask = ~np.isin(coarse, HIGHLIGHT)
        ax.scatter(coords[not_high_mask, 0], coords[not_high_mask, 1],
                   c=GRAY, s=18, alpha=0.55,
                   edgecolors="white", linewidth=0.3, zorder=2)

        # Then highlight superclasses with full colour + convex hull + label
        for sup_id in HIGHLIGHT:
            mask = coarse == sup_id
            pts = coords[mask]
            color = HIGHLIGHT_COLORS[sup_id]
            # Convex hull (5 fine classes per super, often collinear → catch)
            try:
                hull = ConvexHull(pts)
                poly = pts[hull.vertices]
                poly = np.vstack([poly, poly[:1]])
                ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=0.18,
                        zorder=1)
                ax.plot(poly[:, 0], poly[:, 1], color=color, alpha=0.7,
                        linewidth=1.2, zorder=1)
            except Exception:
                pass
            ax.scatter(pts[:, 0], pts[:, 1], c=color, s=70, alpha=0.95,
                       edgecolors="black", linewidth=0.5, zorder=3)
            # Label at centroid
            cx, cy = pts.mean(0)
            ax.annotate(COARSE_NAMES[sup_id], (cx, cy),
                        xytext=(0, -14 if draw_disk else -16),
                        textcoords="offset points",
                        fontsize=9.5, ha="center", va="top",
                        color=color, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.18",
                                  facecolor="white",
                                  edgecolor=color, linewidth=0.8, alpha=0.95),
                        zorder=6)

        ax.set_title(title, fontsize=12)
        ax.set_aspect("equal", adjustable="box")
        if draw_disk:
            ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ("top","right","bottom","left"):
                ax.spines[sp].set_visible(False)
        else:
            ax.set_xlabel("PC1", fontsize=10.5)
            ax.set_ylabel("PC2", fontsize=10.5)
            for sp in ("top","right"):
                ax.spines[sp].set_visible(False)
        ax.grid(False)

    draw_panel(axes[0], pca2,
               "(a) Euclidean: PCA in raw feature space")
    draw_panel(axes[1], poin2_disk,
               r"(b) Poincar" + "é" + r" disk: hyperbolic-MDS on $d_\mathbb{B}(\phi(\cdot))$",
               draw_disk=True)

    fig.suptitle(f"{MODEL.upper()} on CIFAR-100 -- 100 fine class centroids; "
                 r"5 of 20 superclasses highlighted; $\delta = 0.07$, ORC $= +0.65$",
                 fontsize=11, y=0.97)
    plt.subplots_adjust(top=0.90, bottom=0.08, wspace=0.10)
    plt.savefig(OUT / "fig_euclid_vs_poincare.pdf", dpi=200, bbox_inches="tight")
    plt.savefig(OUT / "fig_euclid_vs_poincare.png", dpi=200, bbox_inches="tight")
    print(f"\nSaved {OUT}/fig_euclid_vs_poincare.pdf")


if __name__ == "__main__":
    main()
