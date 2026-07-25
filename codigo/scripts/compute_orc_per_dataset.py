#!/usr/bin/env python3
"""
Per-(model, dataset) Ollivier-Ricci curvature on the centroid kNN graph.

For each (model, dataset):
  - Load cached features (X_train, y_train).
  - Compute centroids per class.
  - Build k=10 kNN graph among centroids (Euclidean).
  - For each edge (u, v): kappa = 1 - W1(mu_u, mu_v) / d(u, v),
    where mu_u is uniform over u's k neighbors and W1 is the optimal-
    transport cost under Euclidean ground metric (solved as a linear
    assignment between the k neighbors of u and the k neighbors of v).
  - Aggregate ORC: mean over edges (also report fraction of negative edges).

Output: results/orc_per_dataset.csv
"""
import os, sys, time
sys.stdout.reconfigure(line_buffering=True)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

ROOT = Path("/home/javi/Platonic")
CACHE = ROOT / "results/practical_tasks_cache"
OUT = ROOT / "results"

PARADIGMS = {
    "i21k_t":"Supervised","i21k_s":"Supervised","i21k_b":"Supervised","i21k_l":"Supervised",
    "dinov1_b":"SSL","dinov2_s":"SSL","dinov2_b":"SSL","dinov2_l":"SSL","dinov2_g":"SSL",
    "clip_b":"Contrastive","clip_l":"Contrastive","siglip_b":"Contrastive",
}
PARAMS_M = {
    "i21k_t":5.5, "i21k_s":22, "i21k_b":86, "i21k_l":307,
    "dinov1_b":86, "dinov2_s":22, "dinov2_b":86, "dinov2_l":304, "dinov2_g":1100,
    "clip_b":86, "clip_l":307, "siglip_b":86,
}

MODELS = list(PARADIGMS.keys())
DATASETS = ["imagenet","cifar100","cifar10","dtd","fashionmnist","mnist"]


def adaptive_k(n_classes, k_max=10):
    """Cap k so the kNN graph is not near-complete: k <= n_classes / 3."""
    return max(2, min(k_max, n_classes // 3))


def load_centroids(model, dataset):
    """Load cached features and compute per-class centroids."""
    if dataset == "imagenet":
        # Prefer full-train if available, else fall back to standard
        full = CACHE / f"{model}_imagenet_fulltrain.npz"
        if full.exists():
            d = np.load(full)
            X = d["features"].astype(np.float32); y = d["labels"].astype(np.int64)
        else:
            d = np.load(CACHE / f"{model}_imagenet_train.npz")
            X = d["features"].astype(np.float32); y = d["labels"].astype(np.int64)
    else:
        d = np.load(CACHE / f"{model}_{dataset}_train.npz")
        X = d["features"].astype(np.float32); y = d["labels"].astype(np.int64)
    n_classes = int(y.max()) + 1
    cents = np.stack([X[y == c].mean(0) for c in range(n_classes)])
    return cents


def w1_uniform(neighbors_u, neighbors_v, points):
    """W1 between uniform distributions over given neighbor sets, Euclidean ground."""
    A = points[neighbors_u]
    B = points[neighbors_v]
    cost = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
    row_ind, col_ind = linear_sum_assignment(cost)
    return cost[row_ind, col_ind].mean()


def compute_orc(centroids, k=None):
    """Mean Ollivier-Ricci curvature on the k-NN graph of centroids."""
    D = cdist(centroids, centroids)
    n = len(centroids)
    if k is None:
        k = adaptive_k(n)
    # k+1 nearest including self
    knn = np.argsort(D, axis=1)[:, 1:k+1]
    edges = set()
    for u in range(n):
        for v in knn[u]:
            edges.add((min(u, int(v)), max(u, int(v))))
    edges = list(edges)
    kappas = []
    for u, v in edges:
        w1 = w1_uniform(knn[u], knn[v], centroids)
        kappas.append(1 - w1 / max(D[u, v], 1e-7))
    kappas = np.array(kappas)
    return float(kappas.mean()), float((kappas < 0).mean()), len(edges)


def main():
    rows = []
    for ds in DATASETS:
        for model in MODELS:
            try:
                t0 = time.time()
                cents = load_centroids(model, ds)
                k_used = adaptive_k(len(cents))
                orc_mean, frac_neg, n_edges = compute_orc(cents, k=k_used)
                dt = time.time() - t0
                print(f"  {ds:14s} {model:10s}: k={k_used}  ORC={orc_mean:+.3f}  neg={frac_neg*100:.1f}%  edges={n_edges}  ({dt:.1f}s)", flush=True)
                rows.append(dict(
                    model=model, dataset=ds, paradigm=PARADIGMS[model], params_M=PARAMS_M[model],
                    n_classes=len(cents), k=k_used, ORC_mean=orc_mean, frac_neg_edges=frac_neg, n_edges=n_edges,
                ))
                pd.DataFrame(rows).to_csv(OUT / "orc_per_dataset.csv", index=False)
            except FileNotFoundError as e:
                print(f"  SKIP {ds} {model}: {e}", flush=True)
            except Exception as e:
                print(f"  ERROR {ds} {model}: {e}", flush=True)
    print(f"\nDone: {len(rows)} (model, dataset) rows -> {OUT}/orc_per_dataset.csv")


if __name__ == "__main__":
    main()
