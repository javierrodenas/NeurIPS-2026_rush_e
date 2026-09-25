#!/usr/bin/env python3
"""
v8: pairwise-distance histograms, raw vs phi-projected.

Two panels (Euclidean / Poincare). In each, two histograms over the C(100,2) =
4950 pairwise distances among CIFAR-100 fine-class centroids:
  - within-super pairs (same CIFAR-100 superclass)
  - across-super pairs

Distances are normalised by the panel's own median pairwise distance, so the
two panels share a common reference of 1.0 and the SHAPE of the distributions
is comparable. The within/across separation is what NC and few-shot
classifiers exploit.
"""
import os, sys
sys.stdout.reconfigure(line_buffering=True)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

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
    cents = np.stack([X[y == c].mean(0) for c in range(100)])
    sup = fine_to_coarse[np.arange(100)]

    mu = cents.mean(0)
    p95 = np.percentile(np.linalg.norm(cents - mu, axis=1), 95)
    cents_h = project(cents, mu, p95)

    D_e = np.sqrt(((cents[:, None] - cents[None, :])**2).sum(-1))
    D_h = poincare_dist_matrix(cents_h)

    iu = np.triu_indices(100, k=1)
    pair_sup_eq = (sup[iu[0]] == sup[iu[1]])

    e_within = D_e[iu][pair_sup_eq]
    e_across = D_e[iu][~pair_sup_eq]
    h_within = D_h[iu][pair_sup_eq]
    h_across = D_h[iu][~pair_sup_eq]

    # Normalise by median pairwise distance under each metric
    e_med = np.median(D_e[iu])
    h_med = np.median(D_h[iu])
    e_within_n = e_within / e_med; e_across_n = e_across / e_med
    h_within_n = h_within / h_med; h_across_n = h_across / h_med

    print(f"Pairs: within={pair_sup_eq.sum()}  across={(~pair_sup_eq).sum()}")
    print(f"Euclidean (norm.):  within mean={e_within_n.mean():.3f}  "
          f"across mean={e_across_n.mean():.3f}  d'={(e_across_n.mean()-e_within_n.mean())/np.sqrt(0.5*(e_within_n.var()+e_across_n.var())):.3f}")
    print(f"Hyperbolic (norm.): within mean={h_within_n.mean():.3f}  "
          f"across mean={h_across_n.mean():.3f}  d'={(h_across_n.mean()-h_within_n.mean())/np.sqrt(0.5*(h_within_n.var()+h_across_n.var())):.3f}")

    plt.rcParams.update({"font.family":"serif","font.size":11,
                         "figure.facecolor":"white","savefig.facecolor":"white",
                         "axes.facecolor":"white"})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)

    C_WITHIN = "#2ca02c"
    C_ACROSS = "#d62728"

    def panel(ax, w, a, title, xmax):
        bins = np.linspace(0, xmax, 41)
        ax.hist(w, bins=bins, density=True, alpha=0.55,
                color=C_WITHIN, edgecolor="white", linewidth=0.4,
                label=f"within-super (n={len(w)})")
        ax.hist(a, bins=bins, density=True, alpha=0.55,
                color=C_ACROSS, edgecolor="white", linewidth=0.4,
                label=f"across-super (n={len(a)})")
        # KDE overlays
        xs = np.linspace(0, xmax, 400)
        try:
            ax.plot(xs, gaussian_kde(w)(xs), color=C_WITHIN, linewidth=1.6)
            ax.plot(xs, gaussian_kde(a)(xs), color=C_ACROSS, linewidth=1.6)
        except Exception:
            pass
        # Mean lines
        ax.axvline(w.mean(), color=C_WITHIN, linestyle="--",
                   linewidth=1.0, alpha=0.9)
        ax.axvline(a.mean(), color=C_ACROSS, linestyle="--",
                   linewidth=1.0, alpha=0.9)
        # Annotate gap
        gap = a.mean() - w.mean()
        # Cohen's d
        d_eff = gap / np.sqrt(0.5*(w.var() + a.var()))
        ax.text(0.97, 0.93,
                f"$\\bar{{d}}_{{\\rm across}} - \\bar{{d}}_{{\\rm within}} = {gap:+.3f}$\n"
                f"Cohen's $d = {d_eff:.2f}$",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                          edgecolor="#888888", linewidth=0.6, alpha=0.95))
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("pairwise distance / median", fontsize=10.5)
        ax.set_xlim(0, xmax)
        for sp in ("top","right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", alpha=0.18)
        ax.legend(frameon=False, fontsize=9.5, loc="upper left")

    xmax_e = max(e_within_n.max(), e_across_n.max()) * 1.02
    xmax_h = max(h_within_n.max(), h_across_n.max()) * 1.02
    xmax = max(xmax_e, xmax_h)
    panel(axes[0], e_within_n, e_across_n,
          "(a) Euclidean: pairwise centroid distances", xmax)
    panel(axes[1], h_within_n, h_across_n,
          r"(b) Hyperbolic: $d_\mathbb{B}(\phi(\cdot))$ pairwise distances",
          xmax)
    axes[0].set_ylabel("density", fontsize=10.5)

    fig.suptitle(f"{MODEL.upper()} on CIFAR-100 -- 100 fine centroids, 20 superclasses; "
                 r"$\delta = 0.07$, ORC $= +0.65$",
                 fontsize=11, y=1.02)
    plt.subplots_adjust(top=0.88, bottom=0.16, wspace=0.10)
    plt.savefig(OUT / "fig_euclid_vs_poincare.pdf", dpi=200, bbox_inches="tight")
    plt.savefig(OUT / "fig_euclid_vs_poincare.png", dpi=200, bbox_inches="tight")
    print(f"\nSaved {OUT}/fig_euclid_vs_poincare.pdf")


if __name__ == "__main__":
    main()
