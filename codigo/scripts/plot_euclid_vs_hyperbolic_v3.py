#!/usr/bin/env python3
"""
v3: Euclidean vs Poincare with hyperbolic-MDS for the right panel and
two annotated superclass pairs to show the metric expansion.

  Left:  2D PCA of raw centroids (Euclidean), with pair annotations of
         Euclidean centroid-to-centroid distance.
  Right: 2D MDS on the pairwise hyperbolic-distance matrix of the
         phi-projected centroids, rescaled into the unit disk. Same two
         pair annotations show the corresponding hyperbolic distance.

Two annotated pairs:
  (A) within-superclass: vehicles 1 <-> vehicles 2  (close in both)
  (B) across-superclass: trees <-> vehicles 1       (far in both, but
                                                     amplified in the
                                                     hyperbolic panel)
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


def poincare_dist_matrix(X):
    """Pairwise Poincare distance among rows of X (each in unit ball)."""
    n = len(X)
    D = np.zeros((n, n))
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

    # ----- LEFT PANEL: Euclidean 2D PCA -----
    mu_e = cents.mean(0)
    Xc = cents - mu_e
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pca2 = Xc @ Vt[:2].T

    # ----- RIGHT PANEL: Hyperbolic MDS -----
    p95 = np.percentile(np.linalg.norm(cents - mu_e, axis=1), 95)
    cents_h = project(cents, mu_e, p95)
    D_h = poincare_dist_matrix(cents_h)
    print(f"Hyperbolic dist range: [{D_h[D_h>0].min():.3f}, {D_h.max():.3f}]", flush=True)

    # 2D MDS on the hyperbolic distance matrix
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0,
              max_iter=400, n_init=4, normalized_stress="auto")
    poin2 = mds.fit_transform(D_h)
    nrm2 = np.linalg.norm(poin2, axis=1)
    s2 = 0.92 / nrm2.max()
    poin2_disk = poin2 * s2

    # Pairwise Euclidean centroid distances (for left panel annotations)
    D_e = np.sqrt(((cents[:,None,:] - cents[None,:,:])**2).sum(-1))

    # Median position per superclass
    def super_pos(coords):
        return np.array([coords[coarse == c].mean(0) for c in range(20)])

    # Hyperbolic distance between SUPERCLASS centroids in original feature space
    super_cents_h = np.stack([cents_h[coarse == c].mean(0) for c in range(20)])
    super_cents_h_norm = np.minimum(0.99/np.linalg.norm(super_cents_h,axis=1,keepdims=True).clip(min=1e-7),1.0) * super_cents_h
    D_super_h = poincare_dist_matrix(super_cents_h_norm)
    super_cents_e = np.stack([cents[coarse == c].mean(0) for c in range(20)])
    D_super_e = np.sqrt(((super_cents_e[:,None,:] - super_cents_e[None,:,:])**2).sum(-1))

    # Pairs to annotate
    PAIRS = [
        (18, 19, "veh. 1", "veh. 2"),     # within: vehicles 1 vs vehicles 2
        (17, 18, "trees", "veh. 1"),      # across: trees vs vehicles 1
    ]

    # ----- PLOT -----
    plt.rcParams.update({"font.family":"serif","font.size":10.5,
                         "figure.facecolor":"white","savefig.facecolor":"white",
                         "axes.facecolor":"white"})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.7))

    cmap = plt.cm.tab20
    colours = cmap(np.linspace(0, 1, 20))

    def add_super_labels(ax, super_pos):
        for c in range(20):
            x, y = super_pos[c]
            ax.text(x, y, COARSE_NAMES[c], fontsize=7.5,
                    ha="center", va="center", color="#222222", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                              edgecolor=colours[c], linewidth=1.0, alpha=0.9),
                    zorder=5)

    def annotate_pair(ax, super_pos, ci, cj, dist, label, offset_y=0):
        a = super_pos[ci]; b = super_pos[cj]
        ax.annotate("", xy=(b[0],b[1]), xytext=(a[0],a[1]),
                    arrowprops=dict(arrowstyle="<->", color="#444444",
                                    linewidth=1.4, alpha=0.85), zorder=4)
        mid = (a + b) / 2
        ax.text(mid[0], mid[1] + offset_y, f"{label}\n$d={dist:.2f}$", fontsize=8,
                ha="center", va="center", color="#444444",
                bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                          edgecolor="#444444", linewidth=0.8, alpha=0.95), zorder=6)

    # LEFT panel
    ax = axes[0]
    for c in range(20):
        mask = coarse == c
        ax.scatter(pca2[mask,0], pca2[mask,1], c=[colours[c]], s=42,
                   alpha=0.55, edgecolors="black", linewidth=0.3, zorder=2)
    spos_e = super_pos(pca2)
    add_super_labels(ax, spos_e)
    for (ci, cj, _, _), label in zip(PAIRS, ["within-super (veh.1$\\leftrightarrow$veh.2)",
                                              "across-super (trees$\\leftrightarrow$veh.1)"]):
        annotate_pair(ax, spos_e, ci, cj, D_super_e[ci, cj], label)
    ax.set_title("(a) Euclidean: PCA in raw feature space", fontsize=12)
    ax.set_xlabel("PC1", fontsize=10.5); ax.set_ylabel("PC2", fontsize=10.5)
    ax.set_aspect("equal", adjustable="box")
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    ax.grid(False)

    # RIGHT panel
    ax = axes[1]
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="#888888", linewidth=1.0)
    ax.fill(np.cos(theta), np.sin(theta), color="#f8f8f8", zorder=0)
    for c in range(20):
        mask = coarse == c
        ax.scatter(poin2_disk[mask,0], poin2_disk[mask,1], c=[colours[c]], s=42,
                   alpha=0.55, edgecolors="black", linewidth=0.3, zorder=2)
    spos_h = super_pos(poin2_disk)
    add_super_labels(ax, spos_h)
    for (ci, cj, _, _), label in zip(PAIRS, ["within-super (veh.1$\\leftrightarrow$veh.2)",
                                              "across-super (trees$\\leftrightarrow$veh.1)"]):
        annotate_pair(ax, spos_h, ci, cj, D_super_h[ci, cj], label)
    ax.set_title(r"(b) Poincar" + "é" + r" disk: hyperbolic-MDS on $d_\mathbb{B}(\phi(\cdot))$", fontsize=12)
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ("top","right","bottom","left"): ax.spines[sp].set_visible(False)
    ax.grid(False)

    fig.suptitle(f"{MODEL.upper()} on {DATASET.upper()} -- 100 fine classes, 20 superclass labels; $\\delta = 0.07$, ORC $= +0.65$",
                 fontsize=11, y=0.99)
    plt.subplots_adjust(top=0.92, bottom=0.06, wspace=0.10)
    plt.savefig(OUT / "fig_euclid_vs_poincare.pdf", dpi=200, bbox_inches="tight")
    plt.savefig(OUT / "fig_euclid_vs_poincare.png", dpi=200, bbox_inches="tight")
    print(f"Saved {OUT}/fig_euclid_vs_poincare.pdf")
    print(f"\nDistances:")
    for (ci, cj, ni, nj) in PAIRS:
        print(f"  {ni:>10s} <-> {nj:<10s}: Euclidean = {D_super_e[ci,cj]:.3f},  hyperbolic = {D_super_h[ci,cj]:.3f}")


if __name__ == "__main__":
    main()
