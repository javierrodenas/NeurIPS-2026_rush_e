#!/usr/bin/env python3
"""expR45 (round-7 Q2): ConvNet rows for the ImageNet census from the centroid store
(ResNet-50 Barlow Twins and BYOL; plus MAE-B and I-JEPA-H as extra ViT-SSL rows):
delta, spectrum null A (20 reps) and WordNet-hub null B (20 reps), percentile ranks."""
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
ROOT = Path("/media/HDD_4TB_2/javi/Platonic"); OUT = Path(__file__).resolve().parents[1]/"results"
WN = np.load(ROOT/"results/features/imagenet/imagenet1k_wordnet_dist.npy")
WN30 = AgglomerativeClustering(n_clusters=30, metric="precomputed", linkage="average").fit_predict(WN)
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
    mu = M.mean(0); Mc = M - mu; U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
    rng = np.random.RandomState(seed0+rep); G = rng.randn(len(M), len(S)).astype(np.float32); G /= G.std(0, keepdims=True)*np.sqrt(len(M))
    return (G*S)@Vt + mu
def flatnull(C, sup, rep):
    hubs = np.stack([C[sup==s].mean(0) for s in range(sup.max()+1)]); return C - hubs[sup] + spec_sample(hubs, rep, 500)[sup]
rows=[]
for m, kind in [("barlow_r50","ConvNet SSL"),("byol_r50","ConvNet SSL"),("mae_b","ViT SSL (MAE)"),("ijepa_h","ViT SSL (I-JEPA)")]:
    f = ROOT/f"results/centroids/imagenet_train/{m}.npy"
    if not f.exists(): print("missing", m); continue
    t0=time.time(); C = np.load(f).astype(np.float32)
    if C.shape[0] != 1000: print("skip", m, C.shape); continue
    dr, dr_sd = delta_norm(C, 10)
    nA = [delta_norm(spec_sample(C, r, 300), 3)[0] for r in range(20)]; nB = nA
    amu, asd = float(np.mean(nA)), float(np.std(nA, ddof=1)); bmu, bsd = float(np.mean(nB)), float(np.std(nB, ddof=1))
    rows.append(dict(model=m, kind=kind, d=C.shape[1], delta=dr, excessA=dr-amu, zA=(dr-amu)/max(np.sqrt(asd**2+dr_sd**2),1e-9),
                     rankA=float(np.mean([x>dr for x in nA])), excessB=dr-bmu, zB=(dr-bmu)/max(np.sqrt(bsd**2+dr_sd**2),1e-9), time_s=time.time()-t0))
    print(f"{m:11s} {kind:16s} d={C.shape[1]} delta {dr:.4f} excA {dr-amu:+.4f} zA {rows[-1]['zA']:+.1f} | excB {dr-bmu:+.4f} zB {rows[-1]['zB']:+.1f} ({time.time()-t0:.0f}s)")
    pd.DataFrame(rows).to_csv(OUT/"expR45_convnet_rows.csv", index=False)
print("Done expR45")
