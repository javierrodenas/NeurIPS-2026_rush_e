#!/usr/bin/env python3
"""expR34 (fresh-review W1/Q3): p99.9 four-point statistic as primary on ImageNet.
12 models, centroid store, excess over 20 spectrum-null replicates (3 seeds each),
z AND percentile rank (fraction of null replicates above the real value).
Output: expR34_p999_imagenet.csv."""
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
ROOT = Path("/media/HDD_4TB_2/javi/Platonic"); OUT = Path(__file__).resolve().parents[1]/"results"
MODELS = ["i21k_t","i21k_s","i21k_b","i21k_l","dinov1_b","dinov2_s","dinov2_b","dinov2_l","dinov2_g","clip_b","clip_l","siglip_b"]
def p999(X, n_seeds):
    D = squareform(pdist(X, "euclidean")); diam = D.max(); n = len(D)
    out = []
    for s in range(n_seeds):
        rng = np.random.RandomState(s)
        i, j, k, l = (rng.randint(0, n, 500_000) for _ in range(4))
        ok = (i!=j)&(i!=k)&(i!=l)&(j!=k)&(j!=l)&(k!=l); i,j,k,l = i[ok],j[ok],k[ok],l[ok]
        S = np.sort(np.stack([D[i,j]+D[k,l], D[i,k]+D[j,l], D[i,l]+D[j,k]], 1), 1)
        out.append(np.percentile((S[:,2]-S[:,1])/2, 99.9)/diam)
    return float(np.mean(out)), float(np.std(out))
def specnull(C, rep):
    mu = C.mean(0); Cc = C - mu
    U, S, Vt = np.linalg.svd(Cc, full_matrices=False)
    rng = np.random.RandomState(300+rep)
    G = rng.randn(len(C), len(S)).astype(np.float32)
    G /= G.std(0, keepdims=True) * np.sqrt(len(C))
    return (G*S)@Vt + mu
rows=[]
for m in MODELS:
    t0=time.time()
    C = np.load(ROOT/f"results/centroids/imagenet_train/{m}.npy").astype(np.float32)
    dr, dr_sd = p999(C, 10)
    nulls = [p999(specnull(C, r), 3)[0] for r in range(20)]
    nm, nsd = float(np.mean(nulls)), float(np.std(nulls, ddof=1))
    pr = float(np.mean([n > dr for n in nulls]))   # fraction of null reps above real (1.0 = below all)
    z = (dr-nm)/max(np.sqrt(nsd**2+dr_sd**2),1e-9)
    rows.append(dict(model=m, p999=dr, p999_sd=dr_sd, null_mean=nm, null_sd=nsd,
                     excess=dr-nm, z=z, frac_null_above=pr, time_s=time.time()-t0))
    print(f"{m:9s} p999 {dr:.4f} null {nm:.4f} exc {dr-nm:+.4f} z {z:+.1f} null-above {pr:.2f} ({rows[-1]['time_s']:.0f}s)")
    pd.DataFrame(rows).to_csv(OUT/"expR34_p999_imagenet.csv", index=False)
print("Done expR34")
