#!/usr/bin/env python3
"""
Recompute the canonical Gromov delta on centroid sets:
  delta = max_{x,y,z,w} (S_max - S_mid) / 2,   delta_norm = delta / diam(X).

We sample n_quads=500_000 quadruples per seed and average the MAX
across 10 seeds (stable estimator of the supremum over quadruples).

Targets all 12 vision panel models on the 6 datasets, at the 100
imgs/class regime (subsample of full-train for ImageNet).

Output: results/delta_max_per_dataset.csv
"""
import os, sys, time
sys.stdout.reconfigure(line_buffering=True)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist, squareform

ROOT = Path("/home/javi/Platonic")
CACHE = ROOT / "results/practical_tasks_cache"
OUT = ROOT / "results"

PARADIGMS = {
    "i21k_t":"Supervised","i21k_s":"Supervised","i21k_b":"Supervised","i21k_l":"Supervised",
    "dinov1_b":"SSL","dinov2_s":"SSL","dinov2_b":"SSL","dinov2_l":"SSL","dinov2_g":"SSL",
    "clip_b":"Contrastive","clip_l":"Contrastive","siglip_b":"Contrastive",
}
PER_CLASS = 100
DATASETS = ["imagenet","cifar100","cifar10","dtd","fashionmnist","mnist"]


def gromov_delta_max(X, n_quads=500_000, n_seeds=10):
    """Mean-of-max over `n_seeds` independent samples of `n_quads` quadruples."""
    D = squareform(pdist(X, 'euclidean'))
    diam = D.max()
    n = len(X)
    deltas = []
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        i = rng.randint(0, n, n_quads)
        j = rng.randint(0, n, n_quads)
        k = rng.randint(0, n, n_quads)
        l = rng.randint(0, n, n_quads)
        # Discard quadruples with any repeats (rare with n=1000)
        ok = (i != j) & (i != k) & (i != l) & (j != k) & (j != l) & (k != l)
        i, j, k, l = i[ok], j[ok], k[ok], l[ok]
        s = np.sort(np.stack([D[i,j]+D[k,l], D[i,k]+D[j,l], D[i,l]+D[j,k]], 1), 1)
        defects = (s[:,2] - s[:,1]) / 2
        deltas.append(defects.max() / diam)
    deltas = np.array(deltas)
    return float(deltas.mean()), float(deltas.std())


def load_centroids(model, dataset):
    if dataset == "imagenet":
        # Subsample 100/class from full-train
        d = np.load(CACHE / f"{model}_imagenet_fulltrain.npz")
        X_full = d["features"].astype(np.float32); y_full = d["labels"].astype(np.int64)
        rng = np.random.RandomState(0)
        sel = []
        for c in range(int(y_full.max()) + 1):
            ic = np.where(y_full == c)[0]
            sel.extend(rng.choice(ic, min(PER_CLASS, len(ic)), replace=False).tolist())
        X = X_full[np.array(sel)]; y = y_full[np.array(sel)]
    else:
        d = np.load(CACHE / f"{model}_{dataset}_train.npz")
        X = d["features"].astype(np.float32); y = d["labels"].astype(np.int64)
    n_classes = int(y.max()) + 1
    cents = np.stack([X[y == c].mean(0) for c in range(n_classes)])
    return cents


def main():
    rows = []
    for ds in DATASETS:
        for model in PARADIGMS.keys():
            try:
                t0 = time.time()
                cents = load_centroids(model, ds)
                d_mean, d_std = gromov_delta_max(cents)
                rows.append(dict(
                    model=model, dataset=ds, paradigm=PARADIGMS[model],
                    n_classes=len(cents), delta_max=d_mean, delta_max_std=d_std,
                ))
                print(f"  {ds:14s} {model:10s}: delta_max = {d_mean:.4f} +/- {d_std:.4f}  ({time.time()-t0:.1f}s)", flush=True)
                pd.DataFrame(rows).to_csv(OUT / "delta_max_per_dataset.csv", index=False)
            except FileNotFoundError as e:
                print(f"  SKIP {ds} {model}: {e}", flush=True)
            except Exception as e:
                print(f"  ERR  {ds} {model}: {e}", flush=True)
    print(f"\nDone: {len(rows)} rows -> {OUT}/delta_max_per_dataset.csv")


if __name__ == "__main__":
    main()
