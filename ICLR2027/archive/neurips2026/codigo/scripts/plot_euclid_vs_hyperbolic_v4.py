#!/usr/bin/env python3
"""
v4: Euclid vs Poincare on CIFAR-10 + DINOv2-G.

CIFAR-10 has 10 classes with a clean binary superclass split:
    Vehicles: airplane, automobile, ship, truck
    Animals : bird, cat, deer, dog, frog, horse

10 points per panel, easy to spread in 2D, structure is visible.
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
DATASET = "cifar10"
TARGET = 1 / np.sqrt(2)

CIFAR10_NAMES = ["airplane","auto","bird","cat","deer","dog","frog","horse","ship","truck"]
# 0 = vehicle, 1 = animal
CIFAR10_SUPER = np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0])
SUPER_COLOR = {0: "#1f77b4", 1: "#d62728"}
SUPER_NAME = {0: "Vehicles", 1: "Animals"}


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

    # ----- LEFT: Euclidean 2D PCA -----
    mu_e = cents.mean(0)
    Xc = cents - mu_e
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pca2 = Xc @ Vt[:2].T

    # ----- RIGHT: Hyperbolic-MDS -----
    p95 = np.percentile(np.linalg.norm(cents - mu_e, axis=1), 95)
    cents_h = project(cents, mu_e, p95)
    D_h = poincare_dist_matrix(cents_h)
    print(f"Hyperbolic dist range across 10 classes: [{D_h[D_h>0].min():.3f}, {D_h.max():.3f}]", flush=True)
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0,
              max_iter=1000, n_init=8, normalized_stress="auto")
    poin2 = mds.fit_transform(D_h)
    nrm2 = np.linalg.norm(poin2, axis=1)
    s2 = 0.92 / nrm2.max()
    poin2_disk = poin2 * s2

    # Pairwise Euclidean centroid distances
    D_e = np.sqrt(((cents[:, None, :] - cents[None, :, :])**2).sum(-1))

    # Annotated pairs: within-vehicle and across (vehicle <-> animal)
    PAIRS = [
        (0, 1, "airplane", "auto"),    # within: airplane <-> auto
        (0, 3, "airplane", "cat"),     # across: airplane <-> cat
    ]

    plt.rcParams.update({"font.family":"serif","font.size":11,
                         "figure.facecolor":"white","savefig.facecolor":"white",
                         "axes.facecolor":"white"})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.7))

    def draw_panel(ax, coords, D_metric, title, draw_disk=False):
        if draw_disk:
            theta = np.linspace(0, 2*np.pi, 200)
            ax.plot(np.cos(theta), np.sin(theta), color="#888888", linewidth=1.0)
            ax.fill(np.cos(theta), np.sin(theta), color="#f8f8f8", zorder=0)
        # Scatter coloured by superclass
        for c in range(n_classes):
            sc = CIFAR10_SUPER[c]
            ax.scatter(coords[c,0], coords[c,1],
                       c=[SUPER_COLOR[sc]], s=180, alpha=0.85,
                       edgecolors="black", linewidth=0.7, zorder=3)
            # Class name label
            ax.text(coords[c,0], coords[c,1],
                    "  " + CIFAR10_NAMES[c],
                    fontsize=10, ha="left", va="center",
                    color="#222222", fontweight="bold", zorder=5)
        # Distance annotations
        annots = [
            ("within-super", "#444444"),
            ("across-super", "#000000"),
        ]
        for k, (ci, cj, _, _) in enumerate(PAIRS):
            a = coords[ci]; b = coords[cj]
            ax.annotate("", xy=(b[0],b[1]), xytext=(a[0],a[1]),
                        arrowprops=dict(arrowstyle="<->",
                                        color=annots[k][1],
                                        linewidth=1.5, alpha=0.85,
                                        connectionstyle="arc3,rad=0.0"),
                        zorder=4)
            mid = (a + b) / 2
            ax.text(mid[0], mid[1], f"{annots[k][0]}\n$d={D_metric[ci,cj]:.2f}$",
                    fontsize=9, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                              edgecolor=annots[k][1], linewidth=0.9, alpha=0.95),
                    zorder=6)
        ax.set_title(title, fontsize=12)
        ax.set_aspect("equal", adjustable="box")
        if draw_disk:
            ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ("top","right","bottom","left"): ax.spines[sp].set_visible(False)
        else:
            ax.set_xlabel("PC1", fontsize=10.5); ax.set_ylabel("PC2", fontsize=10.5)
            for sp in ("top","right"): ax.spines[sp].set_visible(False)
        ax.grid(False)

    draw_panel(axes[0], pca2, D_e,
               "(a) Euclidean: PCA in raw feature space")
    draw_panel(axes[1], poin2_disk, D_h,
               r"(b) Poincar" + "é" + r" disk: hyperbolic-MDS on $d_\mathbb{B}(\phi(\cdot))$",
               draw_disk=True)

    # Custom legend (super class colours)
    import matplotlib.lines as mlines
    handles = [mlines.Line2D([],[], marker="o", color="w",
                              markerfacecolor=SUPER_COLOR[s], markeredgecolor="black",
                              markersize=10, label=SUPER_NAME[s]) for s in [0, 1]]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.01),
               ncol=2, frameon=False, fontsize=11, columnspacing=2.5)

    fig.suptitle(f"{MODEL.upper()} on {DATASET.upper()} -- 10 classes, 2 superclasses; $\\delta = 0.025$, ORC $= +0.89$",
                 fontsize=11, y=0.96)
    plt.subplots_adjust(top=0.86, bottom=0.07, wspace=0.10)
    plt.savefig(OUT / "fig_euclid_vs_poincare.pdf", dpi=200, bbox_inches="tight")
    plt.savefig(OUT / "fig_euclid_vs_poincare.png", dpi=200, bbox_inches="tight")
    print(f"\nSaved {OUT}/fig_euclid_vs_poincare.pdf")
    print("Distances:")
    for (ci, cj, ni, nj) in PAIRS:
        print(f"  {ni:>10s} <-> {nj:<10s}: Euclidean = {D_e[ci,cj]:.3f},  hyperbolic = {D_h[ci,cj]:.3f}")
    print(f"  ratio across/within: Euclidean = {D_e[0,3]/D_e[0,1]:.2f}x, hyperbolic = {D_h[0,3]/D_h[0,1]:.2f}x")


if __name__ == "__main__":
    main()
