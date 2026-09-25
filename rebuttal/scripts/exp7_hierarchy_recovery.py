#!/usr/bin/env python3
"""REBUTTAL EXP 7 — (a) hierarchy recovery as an additional downstream task:
average-linkage clustering of CIFAR-100 class centroids using Euclidean vs
Poincare distances, cut at 20 -> ARI/NMI vs the TRUE 20 superclasses.
(b) single-image (no centroid pooling) WordNet alignment on ImageNet:
Spearman(dist of 1 random image per class, WordNet dist), 5 draws.
Answers pux6 "centroid circularity" and Xbn5 "one more task"."""
import os, sys
sys.stdout.reconfigure(line_buffering=True)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path("/home/javi/Platonic"); CACHE = ROOT/"results/practical_tasks_cache"
OUT = ROOT/"rebuttal/results"
WN = np.load(ROOT/"results/features/imagenet/imagenet1k_wordnet_dist.npy")
IU = np.triu_indices(1000, 1)
PARADIGMS = {"i21k_t":"Supervised","i21k_s":"Supervised","i21k_b":"Supervised","i21k_l":"Supervised",
 "dinov1_b":"SSL","dinov2_s":"SSL","dinov2_b":"SSL","dinov2_l":"SSL","dinov2_g":"SSL",
 "clip_b":"Contrastive","clip_l":"Contrastive","siglip_b":"Contrastive"}
COARSE = {0:[4,30,55,72,95],1:[1,32,67,73,91],2:[54,62,70,82,92],3:[9,10,16,28,61],4:[0,51,53,57,83],
 5:[22,39,40,86,87],6:[5,20,25,84,94],7:[6,7,14,18,24],8:[3,42,43,88,97],9:[12,17,37,68,76],
 10:[23,33,49,60,71],11:[15,19,21,31,38],12:[34,63,64,66,75],13:[26,45,77,79,99],
 14:[2,11,35,46,98],15:[27,29,44,78,93],16:[36,50,65,74,80],17:[47,52,56,59,96],
 18:[8,13,48,58,90],19:[41,69,81,85,89]}
SUP = np.zeros(100, dtype=int)
for c, fs in COARSE.items():
    for f in fs: SUP[f] = c
TARGET = 0.70710678

def s4(X):
    mu = X.mean(0); Xc = X-mu
    p95 = np.percentile(np.linalg.norm(Xc,axis=1),95)
    s = 2*np.arctanh(TARGET)/max(p95,1e-7); Xs = Xc*s
    nrm = np.linalg.norm(Xs,axis=1,keepdims=True).clip(min=1e-7)
    Y = np.tanh(nrm/2)/nrm*Xs
    cur = np.linalg.norm(Y,axis=1,keepdims=True).clip(min=1e-7)
    return Y*np.clip((1-1e-3)/cur, None, 1.0)

def poinc_D(Y):
    sq = (Y**2).sum(1)
    d2 = np.maximum(sq[:,None]+sq[None,:]-2*Y@Y.T,0)
    den = np.maximum((1-sq[:,None])*(1-sq[None,:]),1e-12)
    return np.arccosh(np.maximum(1+2*d2/den,1+1e-12))

rows = []
for m, par in PARADIGMS.items():
    # (a) CIFAR-100 hierarchy recovery, R vs H linkage
    d = np.load(CACHE/f"{m}_cifar100_train.npz")
    X = d["features"].astype(np.float32); y = d["labels"].astype(np.int64)
    C = np.stack([X[y==c].mean(0) for c in range(100)])
    from scipy.spatial.distance import squareform as sqf
    D_r = pdist(C, 'euclidean')
    D_h = sqf(poinc_D(s4(C)), checks=False)
    res = {}
    for tag, D in [("R", D_r), ("H", D_h)]:
        cut = fcluster(linkage(D, method="average"), t=20, criterion="maxclust")
        res[f"ari_{tag}"] = adjusted_rand_score(SUP, cut)
        res[f"nmi_{tag}"] = normalized_mutual_info_score(SUP, cut)
    # (b) single-image WordNet alignment (ImageNet, no pooling), 5 draws
    dd = np.load(CACHE/f"{m}_imagenet_train.npz")
    Xi = dd["features"].astype(np.float32); yi = dd["labels"].astype(np.int64)
    rhos = []
    for s in range(5):
        rng = np.random.RandomState(s)
        idx = np.array([rng.choice(np.where(yi==c)[0]) for c in range(1000)])
        Ds = squareform(pdist(Xi[idx],'euclidean'))
        rhos.append(spearmanr(Ds[IU], WN[IU]).statistic)
    # centroid alignment for reference
    Cc = np.stack([Xi[yi==c].mean(0) for c in range(1000)])
    rho_cent = spearmanr(squareform(pdist(Cc))[IU], WN[IU]).statistic
    rows.append(dict(model=m, paradigm=par, **res,
                     rho_single_mean=float(np.mean(rhos)), rho_single_std=float(np.std(rhos)),
                     rho_centroid=float(rho_cent)))
    pd.DataFrame(rows).to_csv(OUT/"exp7_hierarchy_recovery.csv", index=False)
    print(f"{m:10s} ARI R={res['ari_R']:.3f} H={res['ari_H']:.3f} | "
          f"rho single={np.mean(rhos):+.3f}±{np.std(rhos):.3f} centroid={rho_cent:+.3f}", flush=True)
print("Done")
