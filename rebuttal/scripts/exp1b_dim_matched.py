#!/usr/bin/env python3
"""REBUTTAL EXP 1b — dimension-matched delta: PCA all models' ImageNet
centroids to a common d=192, recompute delta_max. Removes the ambient-
dimension confound from the cross-model ordering."""
import os, sys
sys.stdout.reconfigure(line_buffering=True)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist, squareform

ROOT = Path("/home/javi/Platonic"); CACHE = ROOT/"results/practical_tasks_cache"
OUT = ROOT/"rebuttal/results"
PARADIGMS = {"i21k_t":"Supervised","i21k_s":"Supervised","i21k_b":"Supervised","i21k_l":"Supervised",
 "dinov1_b":"SSL","dinov2_s":"SSL","dinov2_b":"SSL","dinov2_l":"SSL","dinov2_g":"SSL",
 "clip_b":"Contrastive","clip_l":"Contrastive","siglip_b":"Contrastive"}
D_COMMON = 192

def delta_max(X, n_quads=500_000, n_seeds=10):
    D = squareform(pdist(X,'euclidean')); diam = D.max(); n = len(X); out=[]
    for s in range(n_seeds):
        rng = np.random.RandomState(s)
        i,j,k,l = (rng.randint(0,n,n_quads) for _ in range(4))
        ok=(i!=j)&(i!=k)&(i!=l)&(j!=k)&(j!=l)&(k!=l); i,j,k,l=i[ok],j[ok],k[ok],l[ok]
        S = np.sort(np.stack([D[i,j]+D[k,l],D[i,k]+D[j,l],D[i,l]+D[j,k]],1),1)
        out.append(((S[:,2]-S[:,1])/2).max()/diam)
    return float(np.mean(out)), float(np.std(out))

def load_centroids(model):
    d = np.load(CACHE/f"{model}_imagenet_fulltrain.npz")
    X_full = d["features"].astype(np.float32); y_full = d["labels"].astype(np.int64)
    rng = np.random.RandomState(0); sel=[]
    for c in range(int(y_full.max())+1):
        ic = np.where(y_full==c)[0]
        sel.extend(rng.choice(ic, min(100,len(ic)), replace=False).tolist())
    X = X_full[np.array(sel)]; y = y_full[np.array(sel)]
    return np.stack([X[y==c].mean(0) for c in range(int(y.max())+1)])

rows=[]
for m, par in PARADIGMS.items():
    C = load_centroids(m)
    Cc = C - C.mean(0)
    U,S,Vt = np.linalg.svd(Cc, full_matrices=False)
    Cp = (U[:, :D_COMMON] * S[:D_COMMON])          # PCA to common dim, keeps scale
    dm, ds = delta_max(Cp)
    dg, _ = delta_max(np.random.RandomState(7).randn(*Cp.shape).astype(np.float32))
    rows.append(dict(model=m, paradigm=par, d_orig=C.shape[1], d_common=D_COMMON,
                     delta_pca=dm, delta_pca_std=ds, delta_gauss_matched=dg))
    print(f"{m:10s} d={C.shape[1]:>4} delta_pca192={dm:.4f}±{ds:.4f}  gauss192={dg:.4f}", flush=True)
pd.DataFrame(rows).to_csv(OUT/"exp1b_dim_matched.csv", index=False)
print("Done")
