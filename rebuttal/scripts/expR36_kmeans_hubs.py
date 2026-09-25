#!/usr/bin/env python3
"""expR36 (fresh-review Q4): cluster-preserving null with DATA-DRIVEN hubs.
Same construction as expR29 but hubs from k-means on the centroids themselves
(k=30 ImageNet / 20 CIFAR-100, 10 restarts, seed 0), so the scope note (i) of
the flat-null table is addressed: hierarchy among model-defined clusters.
Output: expR36_kmeans_hubs.csv."""
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.vq import kmeans2
ROOT = Path("/media/HDD_4TB_2/javi/Platonic"); CACHE = ROOT/"results/practical_tasks_cache"
OUT = Path(__file__).resolve().parents[1]/"results"
MODELS = ["i21k_t","i21k_s","i21k_b","i21k_l","dinov1_b","dinov2_s","dinov2_b","dinov2_l","dinov2_g","clip_b","clip_l","siglip_b"]
def delta_norm(X, n_seeds):
    D = squareform(pdist(X, "euclidean")); diam = D.max(); n = len(D)
    out = []
    for s in range(n_seeds):
        rng = np.random.RandomState(s)
        i, j, k, l = (rng.randint(0, n, 500_000) for _ in range(4))
        ok = (i!=j)&(i!=k)&(i!=l)&(j!=k)&(j!=l)&(k!=l); i,j,k,l = i[ok],j[ok],k[ok],l[ok]
        S = np.sort(np.stack([D[i,j]+D[k,l], D[i,k]+D[j,l], D[i,l]+D[j,k]], 1), 1)
        out.append(((S[:,2]-S[:,1])/2).max()/diam)
    return float(np.mean(out)), float(np.std(out))
def spec_sample(M, rep, seed0=500):
    mu = M.mean(0); Mc = M - mu
    U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
    rng = np.random.RandomState(seed0+rep)
    G = rng.randn(len(M), len(S)).astype(np.float32)
    G /= G.std(0, keepdims=True) * np.sqrt(len(M))
    return (G*S)@Vt + mu
def cents(m, ds):
    if ds == "imagenet":
        return np.load(ROOT/f"results/centroids/imagenet_train/{m}.npy").astype(np.float32)
    d = np.load(CACHE/f"{m}_{ds}_train.npz")
    X = d["features"].astype(np.float32); y = d["labels"].astype(np.int64)
    return np.stack([X[y==c].mean(0) for c in range(int(y.max())+1)])
rows=[]
for ds, k in [("imagenet",30),("cifar100",20)]:
    for m in MODELS:
        t0=time.time()
        C = cents(m, ds)
        np.random.seed(0)
        _, sup = kmeans2(C.astype(np.float64), k, minit='++', seed=0)
        keep = np.unique(sup)                      # drop empty clusters if any
        remap = {c: i for i, c in enumerate(keep)}
        sup = np.array([remap[c] for c in sup]); k_eff = len(keep)
        hubs = np.stack([C[sup==s].mean(0) for s in range(k_eff)]).astype(np.float32)
        dr, dr_sd = delta_norm(C, 10)
        nB = [delta_norm(C - hubs[sup] + spec_sample(hubs, r)[sup], 5)[0] for r in range(5)]
        bmu, bsd = float(np.mean(nB)), float(np.std(nB, ddof=1))
        zB = (dr-bmu)/max(np.sqrt(bsd**2+dr_sd**2),1e-9)
        rows.append(dict(model=m, dataset=ds, k=int(k_eff), delta=dr, nullB_mean=bmu,
                         excessB=dr-bmu, zB=zB, time_s=time.time()-t0))
        print(f"{m:9s} {ds:9s} k={k} excB {dr-bmu:+.4f} zB {zB:+.1f} ({rows[-1]['time_s']:.0f}s)")
        pd.DataFrame(rows).to_csv(OUT/"expR36_kmeans_hubs.csv", index=False)
print("Done expR36")
