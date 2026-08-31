#!/usr/bin/env python3
"""expR32 (mock-review Q4): stability of delta_norm and its excess under bootstrap
of the per-class images used to build the centroids.

Cells: {ViT-L, DINOv2-L, DINOv2-G, CLIP-B} x {ImageNet, CIFAR-100}. B=30 bootstrap
replicates: per class, resample its training images with replacement, rebuild the
centroid, then delta_norm (500K quads x 3 seeds) and excess over 2 spectrum-null
replicates (3 seeds each). Row b=-1 is the unresampled reference with the same
reduced seed counts. Output: expR32_centroid_bootstrap.csv."""
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
ROOT = Path("/media/HDD_4TB_2/javi/Platonic"); CACHE = ROOT/"results/practical_tasks_cache"
OUT = Path(__file__).resolve().parents[1]/"results"
def delta_norm(X, n_quads=500_000, n_seeds=3):
    D = squareform(pdist(X, "euclidean")); diam = D.max(); n = len(D)
    out = []
    for s in range(n_seeds):
        rng = np.random.RandomState(s)
        i, j, k, l = (rng.randint(0, n, n_quads) for _ in range(4))
        ok = (i!=j)&(i!=k)&(i!=l)&(j!=k)&(j!=l)&(k!=l); i,j,k,l = i[ok],j[ok],k[ok],l[ok]
        S = np.sort(np.stack([D[i,j]+D[k,l], D[i,k]+D[j,l], D[i,l]+D[j,k]], 1), 1)
        out.append(((S[:,2]-S[:,1])/2).max()/diam)
    return float(np.mean(out))
def specnull(C, rep):
    mu = C.mean(0); Cc = C - mu
    U, S, Vt = np.linalg.svd(Cc, full_matrices=False)
    rng = np.random.RandomState(300+rep)
    G = rng.randn(len(C), len(S)).astype(np.float32)
    G /= G.std(0, keepdims=True) * np.sqrt(len(C))
    return (G*S)@Vt + mu
def excess_of(C):
    dr = delta_norm(C)
    nm = np.mean([delta_norm(specnull(C, r)) for r in range(2)])
    return dr, float(dr - nm)
rows = []
for ds in ["cifar100","dtd"]:  # imagenet per-image features not cached locally
    for m in ["i21k_l","dinov2_l","dinov2_g","clip_b"]:
        d = np.load(CACHE/f"{m}_{ds}_train.npz")
        X = d["features"].astype(np.float32); y = d["labels"].astype(np.int64)
        idx = [np.where(y==c)[0] for c in range(int(y.max())+1)]
        C0 = np.stack([X[ii].mean(0) for ii in idx])
        dr, ex = excess_of(C0)
        rows.append(dict(model=m, dataset=ds, b=-1, n_per_class=int(np.median([len(ii) for ii in idx])), delta=dr, excess=ex))
        for b in range(30):
            t0=time.time(); rng = np.random.RandomState(1000+b)
            Cb = np.stack([X[rng.choice(ii, size=len(ii), replace=True)].mean(0) for ii in idx])
            dr, ex = excess_of(Cb)
            rows.append(dict(model=m, dataset=ds, b=b, n_per_class=int(np.median([len(ii) for ii in idx])), delta=dr, excess=ex))
            pd.DataFrame(rows).to_csv(OUT/"expR32_centroid_bootstrap.csv", index=False)
        sub=[r for r in rows if r["model"]==m and r["dataset"]==ds and r["b"]>=0]
        print(f"{m} {ds}: ref delta {rows[-31]['delta']:.4f} exc {rows[-31]['excess']:+.4f} | boot sd(delta)={np.std([r['delta'] for r in sub]):.4f} sd(exc)={np.std([r['excess'] for r in sub]):.4f}")
print("Done expR32")
