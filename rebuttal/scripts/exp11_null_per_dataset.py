#!/usr/bin/env python3
"""REBUTTAL EXP 11 — is "DINOv2 at its spectrum null" ImageNet-specific, and
does its tree appear in the ANGULAR metric?
For each model x dataset: delta_max real vs spectrum-matched Gaussian null,
in (a) raw Euclidean metric and (b) angular metric (L2-normalized, chord).
Small-n datasets (<=10 classes) use exact enumeration of all quadruples."""
import os, sys, itertools
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
DATASETS = ["imagenet","cifar100","dtd","cifar10","fashionmnist","mnist"]

def delta_max(X, n_quads=500_000, n_seeds=10):
    D = squareform(pdist(X,'euclidean')); diam = D.max(); n = len(D)
    if n <= 16:   # exact over all quadruples
        q = np.array(list(itertools.combinations(range(n), 4)))
        i,j,k,l = q.T
        S = np.sort(np.stack([D[i,j]+D[k,l],D[i,k]+D[j,l],D[i,l]+D[j,k]],1),1)
        return float(((S[:,2]-S[:,1])/2).max()/diam), 0.0
    out=[]
    for s in range(n_seeds):
        rng = np.random.RandomState(s)
        i,j,k,l = (rng.randint(0,n,n_quads) for _ in range(4))
        ok=(i!=j)&(i!=k)&(i!=l)&(j!=k)&(j!=l)&(k!=l); i,j,k,l=i[ok],j[ok],k[ok],l[ok]
        S = np.sort(np.stack([D[i,j]+D[k,l],D[i,k]+D[j,l],D[i,l]+D[j,k]],1),1)
        out.append(((S[:,2]-S[:,1])/2).max()/diam)
    return float(np.mean(out)), float(np.std(out))

def specnull(C, rep):
    """Gaussian cloud with identical covariance spectrum (and mean) as C."""
    mu = C.mean(0); Cc = C - mu
    U,S,Vt = np.linalg.svd(Cc, full_matrices=False)
    rng = np.random.RandomState(300+rep)
    G = rng.randn(len(C), len(S)).astype(np.float32)
    G /= G.std(0, keepdims=True) * np.sqrt(len(C))
    return (G*S)@Vt + mu

def cents(model, ds):
    d = np.load(CACHE/f"{model}_{ds}_train.npz")
    X = d["features"].astype(np.float32); y = d["labels"].astype(np.int64)
    return np.stack([X[y==c].mean(0) for c in range(int(y.max())+1)])

def norm_rows(X):
    return X/np.linalg.norm(X,axis=1,keepdims=True).clip(min=1e-12)

rows=[]
for ds in DATASETS:
    for m, par in PARADIGMS.items():
        C = cents(m, ds)
        n_seeds = 10 if len(C) > 16 else 1
        de_real,_ = delta_max(C, n_seeds=n_seeds)
        dc_real,_ = delta_max(norm_rows(C), n_seeds=n_seeds)
        ne, nc = [], []
        for rep in range(3):
            N = specnull(C, rep)
            ne.append(delta_max(N, n_seeds=max(3,n_seeds//3))[0])
            Cn = norm_rows(C)
            Nn = specnull(Cn, rep)
            nc.append(delta_max(norm_rows(Nn), n_seeds=max(3,n_seeds//3))[0])
        rows.append(dict(model=m, paradigm=par, dataset=ds, n_classes=len(C),
            d_eucl=de_real, null_eucl=float(np.mean(ne)),
            exc_eucl=de_real-float(np.mean(ne)),
            d_cos=dc_real, null_cos=float(np.mean(nc)),
            exc_cos=dc_real-float(np.mean(nc))))
        pd.DataFrame(rows).to_csv(OUT/"exp11_null_per_dataset.csv", index=False)
        print(f"{ds:13s} {m:10s} eucl {de_real:.3f} vs null {np.mean(ne):.3f} "
              f"(exc {de_real-np.mean(ne):+.3f}) | cos {dc_real:.3f} vs {np.mean(nc):.3f} "
              f"(exc {dc_real-np.mean(nc):+.3f})", flush=True)
print("Done")
