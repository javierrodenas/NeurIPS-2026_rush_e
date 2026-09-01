#!/usr/bin/env python3
"""expR42 (round-7 Q1, v2): what excess does a pure star (mixture of Gaussians, no
hierarchy) get under the spectrum null A, and under the cluster-preserving null B?
Synthetic clouds n=1000, d=768: K=30 Gaussian centers + isotropic within-cluster
noise at three tightness levels, and a 2-level hierarchical variant (6 super-centers
x 5 sub-centers) for contrast. Nulls in two constructions: Gaussian coefficients (paper)
and Haar-rotated coefficients (exact sample spectrum). 10 delta seeds; 10 replicates per null."""
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
OUT = Path(__file__).resolve().parents[1]/"results"
def delta_norm(X, n_seeds):
    D = squareform(pdist(X, "euclidean")); diam = D.max(); n = len(D); out = []
    for s in range(n_seeds):
        rng = np.random.RandomState(s)
        i, j, k, l = (rng.randint(0, n, 500_000) for _ in range(4))
        ok = (i!=j)&(i!=k)&(i!=l)&(j!=k)&(j!=l)&(k!=l); i,j,k,l = i[ok],j[ok],k[ok],l[ok]
        S = np.sort(np.stack([D[i,j]+D[k,l], D[i,k]+D[j,l], D[i,l]+D[j,k]], 1), 1)
        out.append(((S[:,2]-S[:,1])/2).max()/diam)
    return float(np.mean(out)), float(np.std(out))
def spec_sample(M, rep, seed0):
    mu = M.mean(0); Mc = M - mu
    U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
    rng = np.random.RandomState(seed0+rep)
    G = rng.randn(len(M), len(S)).astype(np.float32); G /= G.std(0, keepdims=True)*np.sqrt(len(M))
    return (G*S)@Vt + mu
def haar_sample(M, rep, seed0):
    mu = M.mean(0); Mc = M - mu
    U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
    rng = np.random.RandomState(seed0+rep); Z = rng.randn(len(M), len(M)); Q, R = np.linalg.qr(Z); Q = Q*np.sign(np.diag(R))
    return ((Q[:, :len(S)]*S)@Vt + mu).astype(np.float32)
def flatnull(C, sup, rep, sampler):
    hubs = np.stack([C[sup==s].mean(0) for s in range(sup.max()+1)])
    return C - hubs[sup] + sampler(hubs, rep, 500)[sup]
def star(K, ratio, seed, n=1000, d=768):
    rng = np.random.RandomState(seed)
    centers = rng.randn(K, d); lab = rng.randint(0, K, n)
    return (centers[lab] + ratio*rng.randn(n, d)).astype(np.float32), lab
def hier(K1, K2, ratio, seed, n=1000, d=768):
    rng = np.random.RandomState(seed)
    sup = rng.randn(K1, d); sub = sup[:,None,:] + 0.5*rng.randn(K1, K2, d)
    lab = rng.randint(0, K1*K2, n); s1, s2 = lab//K2, lab%K2
    return (sub[s1, s2] + ratio*rng.randn(n, d)).astype(np.float32), s1
rows=[]
for name, mk, sups in [("star30_tight", lambda s: star(30, 0.1, s), None), ("star30_mid", lambda s: star(30, 0.3, s), None),
                       ("star30_loose", lambda s: star(30, 0.6, s), None), ("hier6x5_mid", lambda s: hier(6, 5, 0.3, s), None)]:
    for seed in range(2):
        t0=time.time(); C, sup = mk(seed)
        dr, dr_sd = delta_norm(C, 10)
        res = dict(config=name, seed=seed, delta=dr)
        for tag, sampler in [("gauss", spec_sample), ("haar", haar_sample)]:
            nA = [delta_norm(sampler(C, r, 300), 3)[0] for r in range(10)]
            nB = [delta_norm(flatnull(C, sup, r, sampler), 3)[0] for r in range(10)]
            amu, asd = float(np.mean(nA)), float(np.std(nA, ddof=1)); bmu, bsd = float(np.mean(nB)), float(np.std(nB, ddof=1))
            res[f"excessA_{tag}"] = dr-amu; res[f"zA_{tag}"] = (dr-amu)/max(np.sqrt(asd**2+dr_sd**2),1e-9)
            res[f"excessB_{tag}"] = dr-bmu; res[f"zB_{tag}"] = (dr-bmu)/max(np.sqrt(bsd**2+dr_sd**2),1e-9)
        res["time_s"] = time.time()-t0; rows.append(res)
        print(f"{name:13s} s{seed} delta {dr:.4f} | gauss A {res['excessA_gauss']:+.4f} B {res['excessB_gauss']:+.4f} | haar A {res['excessA_haar']:+.4f} B {res['excessB_haar']:+.4f} ({time.time()-t0:.0f}s)")
        pd.DataFrame(rows).to_csv(OUT/"expR42_star_calibration.csv", index=False)
print("Done expR42")
