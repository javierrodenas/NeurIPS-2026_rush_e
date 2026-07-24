#!/usr/bin/env python3
"""
REBUTTAL EXP 1 — delta controls (answers RJje: spherical/Euclidean/whitened/
dimension-matched baselines; random vs semantic class groupings).

For each of the 12 panel models, on the ImageNet 1000-class centroids
(100 imgs/class, same subsample protocol as delta_max_recompute.py):

  real          delta_max of the actual centroids (Euclidean metric)
  gauss         random Gaussian point cloud, same (n, d)
  sphere        uniform points on the unit sphere S^{d-1}, same (n, d)
  whitened      PCA-whitened centroids (same n, d)
  l2_chord      L2-normalized centroids, Euclidean (chord) distances
  l2_geodesic   L2-normalized centroids, arccos geodesic (spherical metric)
  shufcoord     coordinate-shuffled centroids (per-dim permutation: kills
                cross-dim correlation, keeps marginals)

Grouping controls (n=100 groups of 10 classes, matched n):
  grp_wordnet   centroids of 100 WordNet-coherent groups (agglomerative
                clustering of the 1000x1000 WordNet distance matrix)
  grp_random    centroids of 100 random groups of 10 classes
  cls_random100 delta of 100 randomly chosen REAL class centroids (n control)

delta_max estimator identical to the paper: 500k quadruples x 10 seeds,
max defect / diameter, mean over seeds.

Output: rebuttal/results/exp1_delta_controls.csv
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
OUT = ROOT / "rebuttal/results"
OUT.mkdir(parents=True, exist_ok=True)
WN_DIST = ROOT / "results/features/imagenet/imagenet1k_wordnet_dist.npy"

PARADIGMS = {
    "i21k_t":"Supervised","i21k_s":"Supervised","i21k_b":"Supervised","i21k_l":"Supervised",
    "dinov1_b":"SSL","dinov2_s":"SSL","dinov2_b":"SSL","dinov2_l":"SSL","dinov2_g":"SSL",
    "clip_b":"Contrastive","clip_l":"Contrastive","siglip_b":"Contrastive",
}
PER_CLASS = 100
N_QUADS = 500_000
N_SEEDS = 10


def delta_max_from_D(D, n_quads=N_QUADS, n_seeds=N_SEEDS):
    diam = D.max()
    n = D.shape[0]
    deltas = []
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        i = rng.randint(0, n, n_quads); j = rng.randint(0, n, n_quads)
        k = rng.randint(0, n, n_quads); l = rng.randint(0, n, n_quads)
        ok = (i!=j)&(i!=k)&(i!=l)&(j!=k)&(j!=l)&(k!=l)
        i,j,k,l = i[ok],j[ok],k[ok],l[ok]
        s = np.sort(np.stack([D[i,j]+D[k,l], D[i,k]+D[j,l], D[i,l]+D[j,k]],1),1)
        deltas.append(((s[:,2]-s[:,1])/2).max() / diam)
    d = np.array(deltas)
    return float(d.mean()), float(d.std())


def delta_max(X, **kw):
    return delta_max_from_D(squareform(pdist(X, 'euclidean')), **kw)


def load_centroids(model):
    d = np.load(CACHE / f"{model}_imagenet_fulltrain.npz")
    X_full = d["features"].astype(np.float32); y_full = d["labels"].astype(np.int64)
    rng = np.random.RandomState(0)
    sel = []
    for c in range(int(y_full.max()) + 1):
        ic = np.where(y_full == c)[0]
        sel.extend(rng.choice(ic, min(PER_CLASS, len(ic)), replace=False).tolist())
    X = X_full[np.array(sel)]; y = y_full[np.array(sel)]
    n_classes = int(y.max()) + 1
    cents = np.stack([X[y == c].mean(0) for c in range(n_classes)])
    return cents


def whiten(X):
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return (U * np.sqrt(len(X) - 1)).astype(np.float32)


def wordnet_groups(n_groups=100):
    from sklearn.cluster import AgglomerativeClustering
    Dw = np.load(WN_DIST)
    lab = AgglomerativeClustering(n_clusters=n_groups, metric="precomputed",
                                  linkage="average").fit_predict(Dw)
    return lab


def main():
    rows = []
    wn_lab = wordnet_groups(100)
    rng_g = np.random.RandomState(123)
    rand_lab = rng_g.permutation(np.repeat(np.arange(100), 10))

    for model in PARADIGMS:
        t0 = time.time()
        cents = load_centroids(model)
        n, d = cents.shape
        rng = np.random.RandomState(7)

        variants = {}
        variants["real"] = delta_max(cents)
        variants["gauss"] = delta_max(rng.randn(n, d).astype(np.float32))
        sph = rng.randn(n, d).astype(np.float32)
        sph /= np.linalg.norm(sph, axis=1, keepdims=True)
        variants["sphere"] = delta_max(sph)
        variants["whitened"] = delta_max(whiten(cents))
        cn = cents / np.linalg.norm(cents, axis=1, keepdims=True)
        variants["l2_chord"] = delta_max(cn)
        G = np.clip(cn @ cn.T, -1.0, 1.0)
        variants["l2_geodesic"] = delta_max_from_D(np.arccos(G))
        shuf = cents.copy()
        for col in range(d):
            shuf[:, col] = shuf[rng.permutation(n), col]
        variants["shufcoord"] = delta_max(shuf)

        g_wn = np.stack([cents[wn_lab == g].mean(0) for g in range(100)])
        variants["grp_wordnet"] = delta_max(g_wn)
        g_rd = np.stack([cents[rand_lab == g].mean(0) for g in range(100)])
        variants["grp_random"] = delta_max(g_rd)
        idx100 = rng.choice(n, 100, replace=False)
        variants["cls_random100"] = delta_max(cents[idx100])

        for variant, (dm, ds) in variants.items():
            rows.append(dict(model=model, paradigm=PARADIGMS[model], n=n, d=d,
                             variant=variant, delta_max=dm, delta_max_std=ds))
        pd.DataFrame(rows).to_csv(OUT / "exp1_delta_controls.csv", index=False)
        msg = "  ".join(f"{k}={v[0]:.3f}" for k, v in variants.items())
        print(f"{model:10s} ({time.time()-t0:.0f}s)  {msg}", flush=True)

    print("Done ->", OUT / "exp1_delta_controls.csv")


if __name__ == "__main__":
    main()
