#!/usr/bin/env python3
"""expR37 (fresh-review W5): does the calibration lesson hold on prior work's
object, sample-level features? 1000 stratified training images (10/class,
seed 0) per model on CIFAR-100 and DTD (full per-image caches): raw delta_norm
and excess over 5 spectrum-null replicates. Output: expR37_sample_level.csv."""
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
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
def specnull(C, rep):
    mu = C.mean(0); Cc = C - mu
    U, S, Vt = np.linalg.svd(Cc, full_matrices=False)
    rng = np.random.RandomState(300+rep)
    G = rng.randn(len(C), len(S)).astype(np.float32)
    G /= G.std(0, keepdims=True) * np.sqrt(len(C))
    return (G*S)@Vt + mu
rows=[]
for ds, per in [("cifar100",10),("dtd",22)]:
    for m in MODELS:
        t0=time.time()
        d = np.load(CACHE/f"{m}_{ds}_train.npz")
        X = d["features"].astype(np.float32); y = d["labels"].astype(np.int64)
        rng = np.random.RandomState(0); idx=[]
        for c in range(int(y.max())+1):
            ii = np.where(y==c)[0]; idx += list(rng.choice(ii, size=min(per,len(ii)), replace=False))
        Xs = X[np.array(idx)]
        dr, dr_sd = delta_norm(Xs, 10)
        nulls = [delta_norm(specnull(Xs, r), 5)[0] for r in range(5)]
        nm, nsd = float(np.mean(nulls)), float(np.std(nulls, ddof=1))
        z = (dr-nm)/max(np.sqrt(nsd**2+dr_sd**2),1e-9)
        rows.append(dict(model=m, dataset=ds, n=len(Xs), delta=dr, null_mean=nm,
                         excess=dr-nm, z=z, time_s=time.time()-t0))
        print(f"{m:9s} {ds:9s} n={len(Xs)} raw {dr:.4f} null {nm:.4f} exc {dr-nm:+.4f} z {z:+.1f} ({rows[-1]['time_s']:.0f}s)")
        pd.DataFrame(rows).to_csv(OUT/"expR37_sample_level.csv", index=False)
print("Done expR37")
