#!/usr/bin/env python3
"""
REBUTTAL EXP 12 — direct curvature-SIGN estimation (RJje Q1/Q5).

Parallelogram-law estimator on the centroid metric (Gu et al. 2019 style):
for a triangle (b, c) with approximate midpoint m (the point minimizing
d(x,b)+d(x,c)+|d(x,b)-d(x,c)|) and a reference point a:

  xi(a; b,c) = [ d(a,m)^2 + d(b,c)^2/4 - (d(a,b)^2 + d(a,c)^2)/2 ] / (2 d(a,m))

xi = 0 in Euclidean space, > 0 in spherical, < 0 in hyperbolic/tree metrics.
We CALIBRATE the exact same estimator on reference geometries first; if the
calibration separates the signs, model values are interpretable.

Output: rebuttal/results/exp12_curvature_sign.csv
"""
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

def xi_stats(D, n_tri=20000, seed=0):
    n = len(D); rng = np.random.RandomState(seed)
    xs = []
    for _ in range(n_tri):
        b, c = rng.choice(n, 2, replace=False)
        score = D[:, b] + D[:, c] + np.abs(D[:, b] - D[:, c])
        score[b] = score[c] = np.inf
        m = int(np.argmin(score))
        a = rng.randint(n)
        if a in (b, c, m): continue
        dam = D[a, m]
        if dam < 1e-9: continue
        xi = (dam**2 + D[b, c]**2/4 - (D[a, b]**2 + D[a, c]**2)/2) / (2*dam)
        xs.append(xi / D.max())        # scale-normalize by diameter
    xs = np.array(xs)
    return float(np.mean(xs)), float(np.median(xs)), float((xs < 0).mean())

def h2_D(R=6, n=1000, seed=0):
    rng = np.random.RandomState(seed)
    u = rng.rand(n); r = np.arccosh(1 + u*(np.cosh(R)-1))
    th = rng.rand(n)*2*np.pi
    er = np.tanh(r/2); P = np.stack([er*np.cos(th), er*np.sin(th)], 1)
    sq = (P**2).sum(1)
    d2 = np.maximum(sq[:,None]+sq[None,:]-2*P@P.T, 0)
    den = np.maximum((1-sq[:,None])*(1-sq[None,:]), 1e-15)
    return np.arccosh(np.maximum(1+2*d2/den, 1.0))

def tree_D(depth=9):
    import itertools
    n = 2**depth - 1
    def td(a, b):
        da, db, d = int(np.log2(a+1)), int(np.log2(b+1)), 0
        while da > db: a=(a-1)//2; da-=1; d+=1
        while db > da: b=(b-1)//2; db-=1; d+=1
        while a != b: a=(a-1)//2; b=(b-1)//2; d+=2
        return d
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n): D[i,j]=D[j,i]=td(i,j)
    return D

rows = []
rng = np.random.RandomState(1)
# calibration references
refs = {}
G = rng.randn(1000, 768).astype(np.float32)
refs["ref_gauss768"] = squareform(pdist(G))
S = G/np.linalg.norm(G, axis=1, keepdims=True)
refs["ref_sphere_geo"] = np.arccos(np.clip(S@S.T, -1, 1))
refs["ref_sphere_chord"] = squareform(pdist(S))
refs["ref_H2_R6"] = h2_D()
refs["ref_tree_d9"] = tree_D()
for name, D in refs.items():
    m, md, fn = xi_stats(D)
    rows.append(dict(model=name, paradigm="reference", xi_mean=m, xi_median=md, frac_neg=fn))
    pd.DataFrame(rows).to_csv(OUT/"exp12_curvature_sign.csv", index=False)
    print(f"{name:18s} xi_mean={m:+.4f} median={md:+.4f} frac_neg={fn:.3f}", flush=True)

def cents(model):
    d = np.load(CACHE/f"{model}_imagenet_train.npz")
    X = d["features"].astype(np.float32); y = d["labels"].astype(np.int64)
    return np.stack([X[y==c].mean(0) for c in range(int(y.max())+1)])

for m, par in PARADIGMS.items():
    C = cents(m)
    D = squareform(pdist(C))
    mu, md, fn = xi_stats(D)
    Cn = C/np.linalg.norm(C, axis=1, keepdims=True)
    Dc = squareform(pdist(Cn))
    muc, mdc, fnc = xi_stats(Dc)
    rows.append(dict(model=m, paradigm=par, xi_mean=mu, xi_median=md, frac_neg=fn,
                     xi_mean_cos=muc, xi_median_cos=mdc, frac_neg_cos=fnc))
    pd.DataFrame(rows).to_csv(OUT/"exp12_curvature_sign.csv", index=False)
    print(f"{m:18s} xi_mean={mu:+.4f} median={md:+.4f} frac_neg={fn:.3f} | "
          f"cos: {muc:+.4f}/{mdc:+.4f}/{fnc:.3f}", flush=True)
print("Done")
