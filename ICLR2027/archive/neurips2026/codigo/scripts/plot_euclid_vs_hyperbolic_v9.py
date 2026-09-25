#!/usr/bin/env python3
"""
v9: 4-point Gromov-defect histogram, Euclidean vs Hyperbolic.

For a sample of quadruples of fine-class centroids, compute the Gromov 4-point
defect (the gap between the two larger of the three pairwise sums of opposite
edges, divided by the diameter), under the raw Euclidean metric and under
the Poincare metric of the phi-projected centroids. A perfect tree has all
defects = 0; smaller defects = more tree-like = more hyperbolic.

This directly visualises what delta-hyperbolicity measures, and shows whether
phi makes the metric more tree-like.
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
N_QUADS = 200_000
SEED = 0


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


def four_point_defects(D, n_quads, rng):
    n = D.shape[0]
    idx = rng.integers(0, n, size=(n_quads, 4))
    keep = (idx[:, 0] != idx[:, 1]) & (idx[:, 0] != idx[:, 2]) & \
           (idx[:, 0] != idx[:, 3]) & (idx[:, 1] != idx[:, 2]) & \
           (idx[:, 1] != idx[:, 3]) & (idx[:, 2] != idx[:, 3])
    idx = idx[keep]
    a, b, c, d = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
    s1 = D[a, b] + D[c, d]
    s2 = D[a, c] + D[b, d]
    s3 = D[a, d] + D[b, c]
    sums = np.stack([s1, s2, s3], axis=1)
    sums.sort(axis=1)
    defect = sums[:, 2] - sums[:, 1]
    diam = D.max()
    return defect / max(diam, 1e-12)


def main():
    d = np.load(CACHE / f"{MODEL}_{DATASET}_train.npz")
    X = d["features"].astype(np.float32); y = d["labels"].astype(np.int64)
    cents = np.stack([X[y == c].mean(0) for c in range(100)])
    mu = cents.mean(0)
    p95 = np.percentile(np.linalg.norm(cents - mu, axis=1), 95)
    cents_h = project(cents, mu, p95)

    D_e = np.sqrt(((cents[:, None] - cents[None, :])**2).sum(-1))
    D_h = poincare_dist_matrix(cents_h)

    rng = np.random.default_rng(SEED)
    def_e = four_point_defects(D_e, N_QUADS, rng)
    rng = np.random.default_rng(SEED)
    def_h = four_point_defects(D_h, N_QUADS, rng)

    delta_e = def_e.max()
    delta_h = def_h.max()
    print(f"Euclidean: defect mean={def_e.mean():.4f}  median={np.median(def_e):.4f}  "
          f"delta(=max)={delta_e:.4f}")
    print(f"Hyperbolic: defect mean={def_h.mean():.4f}  median={np.median(def_h):.4f}  "
          f"delta(=max)={delta_h:.4f}")

    plt.rcParams.update({"font.family":"serif","font.size":11,
                         "figure.facecolor":"white","savefig.facecolor":"white",
                         "axes.facecolor":"white"})
    fig, ax = plt.subplots(1, 1, figsize=(7.6, 4.8))

    bins = np.linspace(0, max(def_e.max(), def_h.max()) * 1.02, 60)
    C_E = "#1f77b4"
    C_H = "#d62728"
    ax.hist(def_e, bins=bins, density=True, alpha=0.45,
            color=C_E, edgecolor="white", linewidth=0.4,
            label=f"Euclidean  ($\\bar{{\\rm defect}}={def_e.mean():.3f}$, "
                  f"$\\delta={delta_e:.3f}$)")
    ax.hist(def_h, bins=bins, density=True, alpha=0.45,
            color=C_H, edgecolor="white", linewidth=0.4,
            label=f"Hyperbolic  ($\\bar{{\\rm defect}}={def_h.mean():.3f}$, "
                  f"$\\delta={delta_h:.3f}$)")
    ax.axvline(def_e.mean(), color=C_E, linestyle="--",
               linewidth=1.0, alpha=0.9)
    ax.axvline(def_h.mean(), color=C_H, linestyle="--",
               linewidth=1.0, alpha=0.9)
    ax.set_xlabel(r"4-point Gromov defect / diameter", fontsize=11)
    ax.set_ylabel("density", fontsize=11)
    ax.set_title(f"{MODEL.upper()} on CIFAR-100 -- {N_QUADS//1000}K random quadruples of fine-class centroids",
                 fontsize=11.5)
    ax.legend(frameon=False, fontsize=10.5, loc="upper right")
    for sp in ("top","right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", alpha=0.18)

    plt.tight_layout()
    plt.savefig(OUT / "fig_euclid_vs_poincare.pdf", dpi=200, bbox_inches="tight")
    plt.savefig(OUT / "fig_euclid_vs_poincare.png", dpi=200, bbox_inches="tight")
    print(f"\nSaved {OUT}/fig_euclid_vs_poincare.pdf")


if __name__ == "__main__":
    main()
