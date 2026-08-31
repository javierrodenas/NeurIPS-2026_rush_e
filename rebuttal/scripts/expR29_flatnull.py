#!/usr/bin/env python3
"""expR29 (mock-review W1): does the excess measure hierarchy, or just clusters?

For each of the 12 vision models on ImageNet (wn30 superclasses: average-linkage
clustering of the WordNet distance matrix into 30 groups, as in exp3/exp10) and
CIFAR-100 (the 20 standard superclasses): delta_norm real (500K x 10 seeds), the
spectrum null A (5 reps, exp20 protocol), and a NEW cluster-preserving,
hierarchy-flattening null B: each centroid keeps its offset to its superclass hub,
but the hub configuration is replaced by a spectrum-matched Gaussian sample of the
hubs (5 reps). excessB < 0 means tree organization AMONG the hubs beyond their
second moments; excessB ~ 0 means the spectrum-null excess only reflected
clustering. Output: expR29_flatnull.csv."""
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
ROOT = Path("/media/HDD_4TB_2/javi/Platonic"); CACHE = ROOT/"results/practical_tasks_cache"
OUT = Path(__file__).resolve().parents[1]/"results"
MODELS = ["i21k_t","i21k_s","i21k_b","i21k_l","dinov1_b","dinov2_s","dinov2_b",
          "dinov2_l","dinov2_g","clip_b","clip_l","siglip_b"]
COARSE = {0:[4,30,55,72,95],1:[1,32,67,73,91],2:[54,62,70,82,92],3:[9,10,16,28,61],4:[0,51,53,57,83],
 5:[22,39,40,86,87],6:[5,20,25,84,94],7:[6,7,14,18,24],8:[3,42,43,88,97],9:[12,17,37,68,76],
 10:[23,33,49,60,71],11:[15,19,21,31,38],12:[34,63,64,66,75],13:[26,45,77,79,99],
 14:[2,11,35,46,98],15:[27,29,44,78,93],16:[36,50,65,74,80],17:[47,52,56,59,96],
 18:[8,13,48,58,90],19:[41,69,81,85,89]}
SUP100 = np.zeros(100, dtype=int)
for s, cls in COARSE.items():
    for c in cls: SUP100[c] = s
WN = np.load(ROOT/"results/features/imagenet/imagenet1k_wordnet_dist.npy")
WN30 = AgglomerativeClustering(n_clusters=30, metric="precomputed",
                               linkage="average").fit_predict(WN)
def delta_norm(X, n_quads=500_000, n_seeds=10):
    D = squareform(pdist(X, "euclidean")); diam = D.max(); n = len(D)
    out = []
    for s in range(n_seeds):
        rng = np.random.RandomState(s)
        i, j, k, l = (rng.randint(0, n, n_quads) for _ in range(4))
        ok = (i!=j)&(i!=k)&(i!=l)&(j!=k)&(j!=l)&(k!=l); i,j,k,l = i[ok],j[ok],k[ok],l[ok]
        S = np.sort(np.stack([D[i,j]+D[k,l], D[i,k]+D[j,l], D[i,l]+D[j,k]], 1), 1)
        out.append(((S[:,2]-S[:,1])/2).max()/diam)
    return float(np.mean(out)), float(np.std(out))
def spec_sample(M, rep, seed0):
    mu = M.mean(0); Mc = M - mu
    U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
    rng = np.random.RandomState(seed0+rep)
    G = rng.randn(len(M), len(S)).astype(np.float32)
    G /= G.std(0, keepdims=True) * np.sqrt(len(M))
    return (G*S)@Vt + mu
def flatnull(C, sup, rep):
    hubs = np.stack([C[sup==s].mean(0) for s in range(sup.max()+1)])
    hubs_null = spec_sample(hubs, rep, 500)
    return C - hubs[sup] + hubs_null[sup]
def cents(m, ds):
    if ds == "imagenet":
        return np.load(ROOT/f"results/centroids/imagenet_train/{m}.npy").astype(np.float32)
    d = np.load(CACHE/f"{m}_{ds}_train.npz")
    X = d["features"].astype(np.float32); y = d["labels"].astype(np.int64)
    return np.stack([X[y==c].mean(0) for c in range(int(y.max())+1)])
def main():
    csv_path = OUT/"expR29_flatnull.csv"
    rows = pd.read_csv(csv_path).to_dict("records") if csv_path.exists() else []
    done = {(r["model"], r["dataset"]) for r in rows}
    for ds, sup in [("imagenet", WN30), ("cifar100", SUP100)]:
        for m in MODELS:
            if (m, ds) in done: continue
            t0 = time.time()
            C = cents(m, ds)
            dr, dr_sd = delta_norm(C)
            nA = [delta_norm(spec_sample(C, r, 300), n_seeds=5)[0] for r in range(5)]
            nB = [delta_norm(flatnull(C, sup, r), n_seeds=5)[0] for r in range(5)]
            amu, asd = float(np.mean(nA)), float(np.std(nA, ddof=1))
            bmu, bsd = float(np.mean(nB)), float(np.std(nB, ddof=1))
            zB = (dr-bmu)/max(np.sqrt(bsd**2+dr_sd**2), 1e-9)
            rows.append(dict(model=m, dataset=ds, n_sup=int(sup.max()+1), delta=dr,
                             delta_sd=dr_sd, nullA_mean=amu, nullA_sd=asd, excessA=dr-amu,
                             nullB_mean=bmu, nullB_sd=bsd, excessB=dr-bmu, zB=zB,
                             time_s=time.time()-t0))
            print(f"{m:9s} {ds:9s} delta {dr:.4f} | A {amu:.4f} excA {dr-amu:+.4f} | "
                  f"B {bmu:.4f} excB {dr-bmu:+.4f} zB={zB:+.1f} ({rows[-1]['time_s']:.0f}s)")
            pd.DataFrame(rows).to_csv(csv_path, index=False)
    print("Done expR29")
if __name__ == "__main__":
    main()
