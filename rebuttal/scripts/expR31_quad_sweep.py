#!/usr/bin/env python3
"""expR31 (mock-review W6): stability of the excess under the quadruple budget,
plus a 99.9th-percentile variant of the four-point statistic.

Cells: {ViT-L, DINOv2-L, CLIP-B} x {ImageNet, CIFAR-100, DTD}. For each cell and
n_quads in {1e4,5e4,1e5,5e5,1e6,2e6}: delta_norm real (10 seeds) and 3 spectrum-null
replicates (5 seeds each), with the SAME null configurations reused across budgets.
Rows stat='p999' use the 99.9th percentile of the defect instead of the supremum
(at every budget). Output: expR31_quad_sweep.csv (protocol otherwise = exp20)."""
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
ROOT = Path("/media/HDD_4TB_2/javi/Platonic"); CACHE = ROOT/"results/practical_tasks_cache"
OUT = Path(__file__).resolve().parents[1]/"results"
def delta_stats(X, n_quads, n_seeds):
    D = squareform(pdist(X, "euclidean")); diam = D.max(); n = len(D)
    mx, p9 = [], []
    for s in range(n_seeds):
        rng = np.random.RandomState(s)
        i, j, k, l = (rng.randint(0, n, n_quads) for _ in range(4))
        ok = (i!=j)&(i!=k)&(i!=l)&(j!=k)&(j!=l)&(k!=l); i,j,k,l = i[ok],j[ok],k[ok],l[ok]
        S = np.sort(np.stack([D[i,j]+D[k,l], D[i,k]+D[j,l], D[i,l]+D[j,k]], 1), 1)
        d = (S[:,2]-S[:,1])/2
        mx.append(d.max()/diam); p9.append(np.percentile(d, 99.9)/diam)
    return (float(np.mean(mx)), float(np.std(mx))), (float(np.mean(p9)), float(np.std(p9)))
def specnull(C, rep):
    mu = C.mean(0); Cc = C - mu
    U, S, Vt = np.linalg.svd(Cc, full_matrices=False)
    rng = np.random.RandomState(300+rep)
    G = rng.randn(len(C), len(S)).astype(np.float32)
    G /= G.std(0, keepdims=True) * np.sqrt(len(C))
    return (G*S)@Vt + mu
def cents(m, ds):
    if ds == "imagenet":   # per-image ImageNet features are not cached on this machine
        return np.load(ROOT/f"results/centroids/imagenet_train/{m}.npy").astype(np.float32)
    d = np.load(CACHE/f"{m}_{ds}_train.npz")
    X = d["features"].astype(np.float32); y = d["labels"].astype(np.int64)
    return np.stack([X[y==c].mean(0) for c in range(int(y.max())+1)])
rows = []
for ds in ["imagenet","cifar100","dtd"]:
    for m in ["i21k_l","dinov2_l","clip_b"]:
        C = cents(m, ds); nulls = [specnull(C, r) for r in range(3)]
        for nq in [10_000, 50_000, 100_000, 500_000, 1_000_000, 2_000_000]:
            t0=time.time()
            (dm, dm_sd), (dp, dp_sd) = delta_stats(C, nq, 10)
            nm  = [delta_stats(N, nq, 5) for N in nulls]
            for stat, real, real_sd, nvals in [("max", dm, dm_sd, [x[0][0] for x in nm]),
                                               ("p999", dp, dp_sd, [x[1][0] for x in nm])]:
                mu_n, sd_n = float(np.mean(nvals)), float(np.std(nvals, ddof=1))
                rows.append(dict(model=m, dataset=ds, stat=stat, n_quads=nq, delta=real,
                                 delta_sd=real_sd, null_mean=mu_n, null_sd=sd_n,
                                 excess=real-mu_n, time_s=time.time()-t0))
            print(f"{m} {ds} nq={nq:>8} max exc {rows[-2]['excess']:+.4f}  p999 exc {rows[-1]['excess']:+.4f} ({time.time()-t0:.0f}s)")
            pd.DataFrame(rows).to_csv(OUT/"expR31_quad_sweep.csv", index=False)
print("Done expR31")
