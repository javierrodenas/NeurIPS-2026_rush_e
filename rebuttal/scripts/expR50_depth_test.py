#!/usr/bin/env python3
"""expR50 (round-10 central ask): a star-calibrated depth test.
For each real cell (12 models x {ImageNet wn30, CIFAR-100 20 superclasses}) we build a
MATCHED STAR: same number of hubs K, same cluster sizes, same per-cluster within-spread
(RMS offset norm) and same hub spread (RMS hub norm about the grand mean), Gaussian
everywhere, no hierarchy. Both the real cloud and its matched star are scored with the
cluster-preserving null B in the Haar (exact-spectrum) construction; the DEPTH EXCESS is
excessB(real) - excessB(matched star). Negative depth excess beyond the star's seed noise
means the hub arrangement is more tree-like than a star of the same clustering.
Output: expR50_depth_test.csv."""
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
ROOT = Path("/media/HDD_4TB_2/javi/Platonic"); CACHE = ROOT/"results/practical_tasks_cache"; OUT = Path(__file__).resolve().parents[1]/"results"
MODELS = ["i21k_t","i21k_s","i21k_b","i21k_l","dinov1_b","dinov2_s","dinov2_b","dinov2_l","dinov2_g","clip_b","clip_l","siglip_b"]
COARSE = {0:[4,30,55,72,95],1:[1,32,67,73,91],2:[54,62,70,82,92],3:[9,10,16,28,61],4:[0,51,53,57,83],5:[22,39,40,86,87],6:[5,20,25,84,94],7:[6,7,14,18,24],8:[3,42,43,88,97],9:[12,17,37,68,76],10:[23,33,49,60,71],11:[15,19,21,31,38],12:[34,63,64,66,75],13:[26,45,77,79,99],14:[2,11,35,46,98],15:[27,29,44,78,93],16:[36,50,65,74,80],17:[47,52,56,59,96],18:[8,13,48,58,90],19:[41,69,81,85,89]}
SUP100 = np.zeros(100, dtype=int)
for s, cls in COARSE.items():
    for c in cls: SUP100[c] = s
WN = np.load(ROOT/"results/features/imagenet/imagenet1k_wordnet_dist.npy")
WN30 = AgglomerativeClustering(n_clusters=30, metric="precomputed", linkage="average").fit_predict(WN)
def delta_norm(X, n_seeds):
    D = squareform(pdist(X, "euclidean")); diam = D.max(); n = len(D); out = []
    for s in range(n_seeds):
        rng = np.random.RandomState(s); i, j, k, l = (rng.randint(0, n, 500_000) for _ in range(4))
        ok = (i!=j)&(i!=k)&(i!=l)&(j!=k)&(j!=l)&(k!=l); i,j,k,l = i[ok],j[ok],k[ok],l[ok]
        S = np.sort(np.stack([D[i,j]+D[k,l], D[i,k]+D[j,l], D[i,l]+D[j,k]], 1), 1); out.append(((S[:,2]-S[:,1])/2).max()/diam)
    return float(np.mean(out)), float(np.std(out))
def haar_sample(M, rep, seed0=700):
    mu = M.mean(0); Mc = M - mu; U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
    rng = np.random.RandomState(seed0+rep); Z = rng.randn(len(M), len(M)); Q, R = np.linalg.qr(Z); Q = Q*np.sign(np.diag(R))
    return ((Q[:, :len(S)]*S)@Vt + mu).astype(np.float32)
def flatnull(C, sup, rep):
    hubs = np.stack([C[sup==s].mean(0) for s in range(sup.max()+1)]); return C - hubs[sup] + haar_sample(hubs, rep)[sup]
def excessB(C, sup, n_real=10, n_rep=10):
    dr, dr_sd = delta_norm(C, n_real); nB = [delta_norm(flatnull(C, sup, r), 3)[0] for r in range(n_rep)]
    return dr, dr_sd, float(dr-np.mean(nB)), float(np.std(nB, ddof=1))
def matched_star(C, sup, seed):
    rng = np.random.RandomState(seed); K = sup.max()+1; n, d = C.shape
    hubs = np.stack([C[sup==s].mean(0) for s in range(K)]); hub_rms = np.sqrt(((hubs-hubs.mean(0))**2).sum(1).mean())
    off = C - hubs[sup]; wr = np.array([np.sqrt((off[sup==s]**2).sum(1).mean()) for s in range(K)])   # per-cluster within RMS
    H = rng.randn(K, d); H *= hub_rms/np.sqrt((H**2).sum(1).mean())                                   # Gaussian hubs, matched RMS
    Z = rng.randn(n, d); Z *= (wr[sup]/np.sqrt(d))[:, None]                                            # per-cluster matched within-spread
    return (H[sup] + Z).astype(np.float32)
def cents(m, ds):
    if ds == "imagenet": return np.load(ROOT/f"results/centroids/imagenet_train/{m}.npy").astype(np.float32)
    d = np.load(CACHE/f"{m}_{ds}_train.npz"); X = d["features"].astype(np.float32); y = d["labels"].astype(np.int64)
    return np.stack([X[y==c].mean(0) for c in range(int(y.max())+1)])
rows=[]
for ds, sup in [("cifar100", SUP100), ("imagenet", WN30)]:
    for m in MODELS:
        t0=time.time(); C = cents(m, ds)
        dr, dr_sd, exB, nBsd = excessB(C, sup)
        stars = [excessB(matched_star(C, sup, s), sup, n_real=5, n_rep=5) for s in range(3)]
        exS = float(np.mean([x[2] for x in stars])); exS_sd = float(np.std([x[2] for x in stars], ddof=1))
        depth = exB - exS; z = depth/max(np.sqrt(exS_sd**2 + nBsd**2 + dr_sd**2), 1e-9)
        rows.append(dict(model=m, dataset=ds, K=int(sup.max()+1), delta=dr, excessB_real=exB, excessB_star=exS, excessB_star_sd=exS_sd,
                         depth_excess=depth, z_depth=z, star_delta=float(np.mean([x[0] for x in stars])), time_s=time.time()-t0))
        print(f"{m:9s} {ds:9s} excB real {exB:+.4f} | matched star {exS:+.4f}±{exS_sd:.4f} | depth {depth:+.4f} z {z:+.1f} ({time.time()-t0:.0f}s)")
        pd.DataFrame(rows).to_csv(OUT/"expR50_depth_test.csv", index=False)
print("Done expR50")
