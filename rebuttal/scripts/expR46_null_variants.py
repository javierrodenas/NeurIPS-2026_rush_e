#!/usr/bin/env python3
"""expR46: sensitivity of the spectrum null to its construction, on real cells.
Gaussian coefficients (paper), Haar-rotated coefficients (exact sample spectrum),
PC-permutation (exact per-PC marginals). 12 models x {ImageNet, CIFAR-100, CIFAR-10}, 10 reps x 3 seeds."""
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
ROOT = Path("/media/HDD_4TB_2/javi/Platonic"); CACHE = ROOT/"results/practical_tasks_cache"; OUT = Path(__file__).resolve().parents[1]/"results"
MODELS = ["i21k_t","i21k_s","i21k_b","i21k_l","dinov1_b","dinov2_s","dinov2_b","dinov2_l","dinov2_g","clip_b","clip_l","siglip_b"]
def delta_norm(X, n_seeds):
    D = squareform(pdist(X, "euclidean")); diam = D.max(); n = len(D); out = []
    for s in range(n_seeds):
        rng = np.random.RandomState(s); i, j, k, l = (rng.randint(0, n, 500_000) for _ in range(4))
        ok = (i!=j)&(i!=k)&(i!=l)&(j!=k)&(j!=l)&(k!=l); i,j,k,l = i[ok],j[ok],k[ok],l[ok]
        S = np.sort(np.stack([D[i,j]+D[k,l], D[i,k]+D[j,l], D[i,l]+D[j,k]], 1), 1); out.append(((S[:,2]-S[:,1])/2).max()/diam)
    return float(np.mean(out)), float(np.std(out))
def svd(M): mu = M.mean(0); U,S,Vt = np.linalg.svd(M-mu, full_matrices=False); return mu,U,S,Vt
def gauss(M, rep):
    mu,U,S,Vt = svd(M); rng = np.random.RandomState(300+rep); G = rng.randn(len(M), len(S)).astype(np.float32); G /= G.std(0, keepdims=True)*np.sqrt(len(M)); return (G*S)@Vt+mu
def haar(M, rep):
    mu,U,S,Vt = svd(M); rng = np.random.RandomState(700+rep); Z = rng.randn(len(M), len(M)); Q,R = np.linalg.qr(Z); Q = Q*np.sign(np.diag(R)); return ((Q[:, :len(S)]*S)@Vt+mu).astype(np.float32)
def pcperm(M, rep):
    mu,U,S,Vt = svd(M); rng = np.random.RandomState(900+rep); Up = np.stack([rng.permutation(U[:,k]) for k in range(U.shape[1])],1); return ((Up*S)@Vt+mu).astype(np.float32)
def cents(m, ds):
    if ds == "imagenet": return np.load(ROOT/f"results/centroids/imagenet_train/{m}.npy").astype(np.float32)
    d = np.load(CACHE/f"{m}_{ds}_train.npz"); X = d["features"].astype(np.float32); y = d["labels"].astype(np.int64)
    return np.stack([X[y==c].mean(0) for c in range(int(y.max())+1)])
rows=[]
for ds in ["cifar100","imagenet","cifar10"]:
    for m in MODELS:
        t0=time.time(); C = cents(m, ds); dr, dr_sd = delta_norm(C, 5)
        res = dict(model=m, dataset=ds, delta=dr)
        for tag, fn in [("gauss",gauss),("haar",haar),("pcperm",pcperm)]:
            nulls = [delta_norm(fn(C, r), 3)[0] for r in range(10)]
            nm, nsd = float(np.mean(nulls)), float(np.std(nulls, ddof=1))
            res[f"excess_{tag}"] = dr-nm; res[f"z_{tag}"] = (dr-nm)/max(np.sqrt(nsd**2+dr_sd**2),1e-9)
        res["time_s"]=time.time()-t0; rows.append(res)
        print(f"{m:9s} {ds:9s} gauss {res['excess_gauss']:+.4f} haar {res['excess_haar']:+.4f} pcperm {res['excess_pcperm']:+.4f} ({res['time_s']:.0f}s)")
        pd.DataFrame(rows).to_csv(OUT/"expR46_null_variants.csv", index=False)
print("Done expR46")
