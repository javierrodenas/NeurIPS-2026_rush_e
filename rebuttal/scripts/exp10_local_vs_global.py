#!/usr/bin/env python3
"""REBUTTAL EXP 10 — reconcile Figure 3 (DINOv2-G looks semantically organized
on CIFAR-100) with the audit (DINOv2 lowest WordNet rho / ARI).
Hypothesis: DINOv2 has strong LOCAL semantic structure (siblings adjacent,
compact spokes) but weak GLOBAL tree alignment (near-equidistant superclasses).

CIFAR-100 centroid measures (per model, Euclid + cosine):
  sib_triplet   P[d(a, sibling) < d(a, non-sibling)]  (scale-free, local)
  sib_recall4   mean fraction of the 4 same-superclass siblings among the
                4 nearest centroids
  spoke_ratio   mean ||fine - hub|| / mean inter-hub dist  (Figure 3's claim;
                lower = more compact spokes)
ImageNet (wn30 groups): same sib_triplet at the coarse level.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering

ROOT = Path("/home/javi/Platonic"); CACHE = ROOT/"results/practical_tasks_cache"
OUT = ROOT/"rebuttal/results"
WN = np.load(ROOT/"results/features/imagenet/imagenet1k_wordnet_dist.npy")
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

def cents(model, ds):
    d = np.load(CACHE/f"{model}_{ds}_train.npz")
    X = d["features"].astype(np.float32); y = d["labels"].astype(np.int64)
    return np.stack([X[y==c].mean(0) for c in range(int(y.max())+1)])

def cosD(C):
    Cn = C/np.linalg.norm(C,axis=1,keepdims=True).clip(min=1e-12)
    return 1.0 - Cn@Cn.T

def sib_triplet(D, sup):
    n = len(D); agree = tot = 0
    for a in range(n):
        sib = np.where((sup==sup[a]) & (np.arange(n)!=a))[0]
        non = np.where(sup!=sup[a])[0]
        ds = D[a, sib][:,None]; dn = D[a, non][None,:]
        agree += (ds < dn).sum(); tot += ds.size*dn.shape[1]
    return agree/tot

def sib_recall(D, sup, k=4):
    n = len(D); rec = []
    for a in range(n):
        nn = np.argsort(D[a])[1:k+1]
        rec.append((sup[nn]==sup[a]).mean())
    return float(np.mean(rec))

def spoke_ratio(C, sup):
    hubs = np.stack([C[sup==g].mean(0) for g in range(sup.max()+1)])
    spokes = np.mean([np.linalg.norm(C[i]-hubs[sup[i]]) for i in range(len(C))])
    inter = pdist(hubs).mean()
    return spokes/inter

wn30_cache = {}
rows = []
for m, par in PARADIGMS.items():
    C = cents(m, "cifar100")
    D_e = squareform(pdist(C)); D_c = cosD(C)
    row = dict(model=m, paradigm=par,
        c100_sibtrip_e=sib_triplet(D_e, SUP), c100_sibtrip_c=sib_triplet(D_c, SUP),
        c100_rec4_e=sib_recall(D_e, SUP), c100_rec4_c=sib_recall(D_c, SUP),
        c100_spoke=spoke_ratio(C, SUP))
    Ci = cents(m, "imagenet")
    wn30 = AgglomerativeClustering(n_clusters=30, metric="precomputed",
                                   linkage="average").fit_predict(WN)
    Di = squareform(pdist(Ci))
    row["in_sibtrip_e"] = sib_triplet(Di, wn30)
    row["in_sibtrip_c"] = sib_triplet(cosD(Ci), wn30)
    rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/"exp10_local_vs_global.csv", index=False)
    print(f"{m:10s} C100 sibtrip e/c={row['c100_sibtrip_e']:.3f}/{row['c100_sibtrip_c']:.3f} "
          f"rec4 e/c={row['c100_rec4_e']:.3f}/{row['c100_rec4_c']:.3f} "
          f"spoke={row['c100_spoke']:.3f} | IN sibtrip e/c={row['in_sibtrip_e']:.3f}/{row['in_sibtrip_c']:.3f}",
          flush=True)
print("Done")
