#!/usr/bin/env python3
"""REBUTTAL EXP 1c — spectrum-matched Gaussian null: for each model's ImageNet
centroids, sample a Gaussian cloud with IDENTICAL covariance spectrum (same
singular values in the PCA basis). delta(real) vs delta(null) then isolates
higher-order (hierarchical) structure from all second-order structure."""
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
    C = load_centroids(m); n, d = C.shape
    mu = C.mean(0); Cc = C - mu
    U,S,Vt = np.linalg.svd(Cc, full_matrices=False)
    d_real, s_real = delta_max(C)
    nulls = []
    for rep in range(3):
        rng = np.random.RandomState(100+rep)
        G = rng.randn(n, len(S)).astype(np.float32)
        G /= G.std(0, keepdims=True) * np.sqrt(n)   # unit-norm columns
        null = (G * S) @ Vt                          # same spectrum, same basis
        nulls.append(delta_max(null)[0])
    d_null = float(np.mean(nulls)); s_null = float(np.std(nulls))
    rows.append(dict(model=m, paradigm=par, n=n, d=d, delta_real=d_real,
                     delta_specnull=d_null, delta_specnull_std=s_null,
                     excess=d_real-d_null))
    pd.DataFrame(rows).to_csv(OUT/"exp1c_spectrum_null.csv", index=False)
    print(f"{m:10s} real={d_real:.4f}  specnull={d_null:.4f}±{s_null:.4f}  excess={d_real-d_null:+.4f}", flush=True)
print("Done")
