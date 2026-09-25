#!/usr/bin/env python3
"""
v5: Euclid vs Poincare on CIFAR-10 + DINOv2-G.

Two panels (PCA, hyperbolic-MDS) plus a third panel comparing average
within-superclass and across-superclass distances under each metric,
each normalised by the panel's own diameter so the bars are comparable.
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
CIFAR10_SUPER = np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0])
SUPER_COLOR = {0: "#1f77b4", 1: "#d62728"}
SUPER_NAME = {0: "Vehicles", 1: "Animals"}

LABEL_OFFSETS = {
    "airplane": (-0.02,  0.05, "right"),
    "auto":     ( 0.02, -0.05, "left"),
    "bird":     ( 0.02,  0.05, "left"),
    "cat":      ( 0.02, -0.05, "left"),
    "deer":     ( 0.02,  0.00, "left"),
    "dog":      (-0.02,  0.00, "right"),
    "frog":     ( 0.02,  0.00, "left"),
    "horse":    ( 0.00, -0.06, "center"),
    "ship":     (-0.02,  0.00, "right"),
    "truck":    ( 0.00,  0.06, "center"),
}


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
              max_iter=2000, n_init=16, normalized_stress="auto")
    poin2 = mds.fit_transform(D_h)
    nrm2 = np.linalg.norm(poin2, axis=1)
    s2 = 0.92 / nrm2.max()
    poin2_disk = poin2 * s2

    # Pairwise Euclidean centroid distances
    D_e = np.sqrt(((cents[:, None, :] - cents[None, :, :])**2).sum(-1))

    # Within / across superclass mean distances (use full feature-space metrics
    # for the bar chart, not the 2D embeddings, so the values reflect the real
    # geometry).
    sup = CIFAR10_SUPER
    within_mask = (sup[:, None] == sup[None, :]) & ~np.eye(n_classes, dtype=bool)
    across_mask = sup[:, None] != sup[None, :]
    de_within = D_e[within_mask].mean()
    de_across = D_e[across_mask].mean()
    dh_within = D_h[within_mask].mean()
    dh_across = D_h[across_mask].mean()
    # Normalise each metric by the within-distance so bar 1 is at 1.0 in both
    # panels and the across-bar shows the relative gap.
    e_norm = (de_within / de_within, de_across / de_within)
    h_norm = (dh_within / dh_within, dh_across / dh_within)
    print(f"Mean d (Euclidean):  within={de_within:.3f}  across={de_across:.3f}  ratio={de_across/de_within:.3f}")
    print(f"Mean d (Hyperbolic): within={dh_within:.3f}  across={dh_across:.3f}  ratio={dh_across/dh_within:.3f}")

    plt.rcParams.update({"font.family":"serif","font.size":11,
                         "figure.facecolor":"white","savefig.facecolor":"white",
                         "axes.facecolor":"white"})
    fig = plt.figure(figsize=(13.5, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.55], wspace=0.18)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    def draw_panel(ax, coords, title, draw_disk=False):
        if draw_disk:
            theta = np.linspace(0, 2*np.pi, 200)
            ax.plot(np.cos(theta), np.sin(theta), color="#888888", linewidth=1.0)
            ax.fill(np.cos(theta), np.sin(theta), color="#f8f8f8", zorder=0)
        # Polygon connecting same-superclass centroids to make clusters obvious
        for s_id in [0, 1]:
            idx = np.where(sup == s_id)[0]
            mu_xy = coords[idx].mean(0)
            order = np.argsort(np.arctan2(coords[idx, 1] - mu_xy[1],
                                          coords[idx, 0] - mu_xy[0]))
            poly = coords[idx][order]
            poly = np.vstack([poly, poly[:1]])
            ax.fill(poly[:, 0], poly[:, 1],
                    color=SUPER_COLOR[s_id], alpha=0.10, zorder=1)
            ax.plot(poly[:, 0], poly[:, 1],
                    color=SUPER_COLOR[s_id], alpha=0.45,
                    linewidth=1.0, zorder=1)
        # Scatter
        for c in range(n_classes):
            sc = sup[c]
            ax.scatter(coords[c, 0], coords[c, 1],
                       c=[SUPER_COLOR[sc]], s=150, alpha=0.95,
                       edgecolors="black", linewidth=0.7, zorder=3)
            name = CIFAR10_NAMES[c]
            xr, yr = coords[c, 0], coords[c, 1]
            if draw_disk:
                # Pixel-size offsets in disk units
                dx, dy, ha = LABEL_OFFSETS[name]
            else:
                # Use simple offset: text to the right, except for crowded
                # left-side points (we will rely on polygon outlines)
                dx, dy, ha = 0.0, 0.0, "left"
                # Build a simple rule: label to the right with small space
                dx = 0.0
                dy = 0.0
                ha = "left"
            if draw_disk:
                ax.text(xr + dx, yr + dy, name,
                        fontsize=9, ha=ha, va="center",
                        color="#222222", fontweight="bold", zorder=5)
            else:
                ax.annotate(name, (xr, yr), xytext=(6, 0),
                            textcoords="offset points",
                            fontsize=9, ha="left", va="center",
                            color="#222222", fontweight="bold", zorder=5)
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

    draw_panel(ax0, pca2, "(a) Euclidean: PCA in raw feature space")
    draw_panel(ax1, poin2_disk,
               r"(b) Poincar" + "é" + r" disk: hyperbolic-MDS",
               draw_disk=True)

    # ----- (c) Bar chart: within vs across superclass mean distance, each
    # normalised so within=1 in its own metric.
    bar_x = np.array([0, 1])
    width = 0.36
    ax2.bar(bar_x - width/2, e_norm, width=width,
            color="#9ec3df", edgecolor="black", linewidth=0.6,
            label="Euclidean")
    ax2.bar(bar_x + width/2, h_norm, width=width,
            color="#f4a3a3", edgecolor="black", linewidth=0.6,
            label="Hyperbolic")
    for i, v in enumerate(e_norm):
        ax2.text(bar_x[i] - width/2, v + 0.01, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=9)
    for i, v in enumerate(h_norm):
        ax2.text(bar_x[i] + width/2, v + 0.01, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=9)
    ax2.axhline(1.0, color="#888888", linewidth=0.8, linestyle="--", alpha=0.7)
    ax2.set_xticks(bar_x)
    ax2.set_xticklabels(["within-super", "across-super"])
    ax2.set_ylabel("mean d / mean within-d", fontsize=10.5)
    ax2.set_ylim(0, max(e_norm[1], h_norm[1]) * 1.18)
    ax2.set_title("(c) Mean distance ratio", fontsize=12)
    ax2.legend(frameon=False, fontsize=9, loc="upper left")
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    ax2.grid(axis="y", alpha=0.25)

    import matplotlib.lines as mlines
    handles = [mlines.Line2D([], [], marker="o", color="w",
                              markerfacecolor=SUPER_COLOR[s],
                              markeredgecolor="black",
                              markersize=10, label=SUPER_NAME[s])
               for s in [0, 1]]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.02),
               ncol=2, frameon=False, fontsize=11, columnspacing=2.5)

    fig.suptitle(f"{MODEL.upper()} on {DATASET.upper()} -- 10 classes, 2 superclasses; "
                 r"$\delta = 0.025$, ORC $= +0.89$",
                 fontsize=11, y=0.97)
    plt.subplots_adjust(top=0.86, bottom=0.10, wspace=0.18)
    plt.savefig(OUT / "fig_euclid_vs_poincare.pdf", dpi=200, bbox_inches="tight")
    plt.savefig(OUT / "fig_euclid_vs_poincare.png", dpi=200, bbox_inches="tight")
    print(f"\nSaved {OUT}/fig_euclid_vs_poincare.pdf")


if __name__ == "__main__":
    main()
