#!/usr/bin/env python3
"""
ICLR EXP 23 — controls for the tree map (W2/W3 of the v2 review). THE critical
experiment: does the DINOv2 island survive (a) cosine distance, the metric the
paper itself identifies as carrying DINOv2's semantics, (b) Ward and complete
linkage, and is it (c) a degenerate-cut artifact (chaining)?

Configs per dataset: {euclid, cosine} x {average, complete} + ward on raw
features and on L2-normalized features (proper cosine-Ward proxy).
Per config: pairwise ARI at the reference cut, island stats split by scale
(DINO-B and DINOv2-S vs DINOv2-B/L/G, per W3), cluster-size diagnostics
(max-cluster fraction, singleton count), cophenetic correlation matrices
(reported this time), and ARI as a function of cut level k.

Output: exp23_treemap_controls.npz + log.
"""
import sys, itertools
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
from pathlib import Path
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from sklearn.metrics import adjusted_rand_score

ROOT = Path("/home/javi/Platonic"); CACHE = ROOT/"results/practical_tasks_cache"
OUT = ROOT/"rebuttal/results"
MODELS = ["i21k_t","i21k_s","i21k_b","i21k_l","dinov1_b","dinov2_s","dinov2_b",
          "dinov2_l","dinov2_g","clip_b","clip_l","siglip_b"]
SUP = ["i21k_t","i21k_s","i21k_b","i21k_l","clip_b","clip_l","siglip_b"]
D2_BIG = ["dinov2_b","dinov2_l","dinov2_g"]
D2_SMALL = ["dinov2_s","dinov1_b"]

def cents(m, ds, n_cls):
    d = np.load(CACHE/f"{m}_{ds}_train.npz")
    X = d["features"].astype(np.float32); y = d["labels"].astype(np.int64)
    return np.stack([X[y==c].mean(0) for c in range(n_cls)])

def build_tree(X, metric, link):
    if link == "ward":
        Xf = X/np.linalg.norm(X, axis=1, keepdims=True) if metric == "cosine" else X
        return linkage(Xf, method="ward")
    D = pdist(X, "cosine" if metric == "cosine" else "euclidean")
    return linkage(D, method=link)

def cut_stats(labels):
    _, counts = np.unique(labels, return_counts=True)
    return float(counts.max()/len(labels)), int((counts == 1).sum())

def run(ds, n_cls, k_ref, k_sweep):
    Xs = {m: cents(m, ds, n_cls) for m in MODELS}
    results = {}
    for metric, link in [("euclid","average"),("euclid","complete"),("euclid","ward"),
                         ("cosine","average"),("cosine","complete"),("cosine","ward")]:
        trees = {m: build_tree(Xs[m], metric, link) for m in MODELS}
        cuts = {m: fcluster(trees[m], k_ref, criterion="maxclust") for m in MODELS}
        ari = np.eye(len(MODELS))
        for i, j in itertools.combinations(range(len(MODELS)), 2):
            ari[i,j] = ari[j,i] = adjusted_rand_score(cuts[MODELS[i]], cuts[MODELS[j]])
        cop = np.eye(len(MODELS))
        cvs = {m: cophenet(trees[m]) for m in MODELS}
        for i, j in itertools.combinations(range(len(MODELS)), 2):
            cop[i,j] = cop[j,i] = np.corrcoef(cvs[MODELS[i]], cvs[MODELS[j]])[0,1]
        idx = {m: i for i, m in enumerate(MODELS)}
        deg = {m: cut_stats(cuts[m]) for m in MODELS}
        big_vs_sup = np.mean([ari[idx[a], idx[b]] for a in D2_BIG for b in SUP])
        small_vs_sup = np.mean([ari[idx[a], idx[b]] for a in D2_SMALL for b in SUP])
        sup_vs_sup = np.mean([ari[idx[a], idx[b]] for a, b in itertools.combinations(SUP, 2)])
        cop_big_sup = np.mean([cop[idx[a], idx[b]] for a in D2_BIG for b in SUP])
        cop_sup_sup = np.mean([cop[idx[a], idx[b]] for a, b in itertools.combinations(SUP, 2)])
        maxfrac = max(deg[m][0] for m in D2_BIG)
        results[(metric, link)] = dict(ari=ari, cop=cop,
            big_vs_sup=big_vs_sup, small_vs_sup=small_vs_sup, sup_vs_sup=sup_vs_sup,
            cop_big_sup=cop_big_sup, cop_sup_sup=cop_sup_sup,
            deg=deg)
        print(f"{ds:9s} {metric:6s}-{link:8s} | Dv2-B/L/G vs SUP: ARI {big_vs_sup:.3f} "
              f"coph {cop_big_sup:.3f} | Dv2-S+DINO-B vs SUP: {small_vs_sup:.3f} | "
              f"SUP vs SUP: ARI {sup_vs_sup:.3f} coph {cop_sup_sup:.3f} | "
              f"max Dv2big maxclust-frac {maxfrac:.2f}")
    # k-sweep for the two headline configs
    sweep = {}
    for metric in ["euclid","cosine"]:
        trees = {m: build_tree(Xs[m], metric, "average") for m in MODELS}
        for k in k_sweep:
            cuts = {m: fcluster(trees[m], k, criterion="maxclust") for m in MODELS}
            idx = {m: i for i, m in enumerate(MODELS)}
            bs = np.mean([adjusted_rand_score(cuts[a], cuts[b]) for a in D2_BIG for b in SUP])
            ss = np.mean([adjusted_rand_score(cuts[a], cuts[b]) for a, b in itertools.combinations(SUP, 2)])
            sweep[(metric, k)] = (bs, ss)
            print(f"  k-sweep {ds} {metric} k={k:4d}: island {bs:.3f} vs block {ss:.3f}")
    return results, sweep

r_in, s_in = run("imagenet", 1000, 30, [10, 30, 100, 300])
r_c1, s_c1 = run("cifar100", 100, 20, [5, 10, 20, 40])
np.savez(OUT/"exp23_treemap_controls.npz",
         summary_in={str(k): {kk: vv for kk, vv in v.items() if kk != "deg"} for k, v in r_in.items()},
         summary_c1={str(k): {kk: vv for kk, vv in v.items() if kk != "deg"} for k, v in r_c1.items()},
         allow_pickle=True)
print("Done")
