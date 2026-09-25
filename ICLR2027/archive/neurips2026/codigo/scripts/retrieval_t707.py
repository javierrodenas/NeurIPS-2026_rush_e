#!/usr/bin/env python3
"""
Sample-to-sample retrieval (P@10) at t=1/sqrt(2) for the 12 vision panel
models on all 6 datasets. Uses X_tr-based mu/p95 (NOT centroids) for the
projection, since we project samples not prototypes here.

Output: results/retrieval_t707.csv
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(8)
from pathlib import Path
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering

ROOT = Path("/home/javi/Platonic")
CACHE = ROOT / "results/practical_tasks_cache"
WN_DIST = ROOT / "results/features/imagenet/imagenet1k_wordnet_dist.npy"
OUT = ROOT / "results"

PARADIGMS = {
    "i21k_t":"Supervised","i21k_s":"Supervised","i21k_b":"Supervised","i21k_l":"Supervised",
    "dinov1_b":"SSL","dinov2_s":"SSL","dinov2_b":"SSL","dinov2_l":"SSL","dinov2_g":"SSL",
    "clip_b":"Contrastive","clip_l":"Contrastive","siglip_b":"Contrastive",
}
TARGET = 1/np.sqrt(2)

# CIFAR-100 coarse mapping (same as in practitioner_tasks_CDE.py)
COARSE = {0:[4,30,55,72,95],1:[1,32,67,73,91],2:[54,62,70,82,92],3:[9,10,16,28,61],4:[0,51,53,57,83],
 5:[22,39,40,86,87],6:[5,20,25,84,94],7:[6,7,14,18,24],8:[3,42,43,88,97],9:[12,17,37,68,76],
 10:[23,33,49,60,71],11:[15,19,21,31,38],12:[34,63,64,66,75],13:[26,45,77,79,99],
 14:[2,11,35,46,98],15:[27,29,44,78,93],16:[36,50,65,74,80],17:[47,52,56,59,96],
 18:[8,13,48,58,90],19:[41,69,81,85,89]}
FINE_TO_COARSE_C100 = np.array([0]*100)
for c, fs in COARSE.items():
    for f in fs: FINE_TO_COARSE_C100[f] = c

# CIFAR-10 vehicle/animal: 0=airplane,1=auto,8=ship,9=truck (vehicles); rest=animals
CIFAR10_SUPER = np.array([0,0,1,1,1,1,1,1,0,0])

# FashionMNIST 4 super: 0=top, 1=bottom, 2=outerwear, 3=footwear/bags
# 0:T-shirt, 1:Trouser, 2:Pullover, 3:Dress, 4:Coat, 5:Sandal, 6:Shirt, 7:Sneaker, 8:Bag, 9:Ankle boot
FMNIST_SUPER = np.array([0,1,0,1,2,3,0,3,3,3])


def project(X, mu, p95, target=TARGET):
    s = 2 * np.arctanh(target) / max(p95, 1e-7)
    Xs = (X - mu) * s
    nrm = np.linalg.norm(Xs, axis=-1, keepdims=True).clip(min=1e-7)
    Y = np.tanh(nrm/2) / nrm * Xs
    cur = np.linalg.norm(Y, axis=-1, keepdims=True).clip(min=1e-7)
    return Y * np.minimum((1.0-1e-3)/cur, 1.0)


def poincare_dist(X, Y, chunk=200):
    n=len(X); m=len(Y); out=np.empty((n,m),dtype=np.float32)
    y_sq=(Y*Y).sum(-1)[None,:]
    for i in range(0,n,chunk):
        Xc=X[i:i+chunk]; x_sq=(Xc*Xc).sum(-1,keepdims=True)
        d_sq=(x_sq+y_sq-2.0*Xc@Y.T).clip(min=0)
        denom=((1-x_sq)*(1-y_sq)).clip(min=1e-7)
        out[i:i+chunk]=np.arccosh((1+2*d_sq/denom).clip(min=1+1e-7))
    return out


def p_at_k(D, ql, dbl, k=10):
    nn = np.argpartition(D, k, axis=1)[:, :k]
    nn_sorted = np.empty_like(nn)
    for i in range(len(D)):
        order = np.argsort(D[i, nn[i]])
        nn_sorted[i] = nn[i][order]
    return float((dbl[nn_sorted] == ql[:, None]).mean())


def load_features(model, dataset):
    if dataset == "imagenet":
        # Subsample 100/class from full-train for consistency
        full = CACHE / f"{model}_imagenet_fulltrain.npz"
        d_tr = np.load(full); X_full = d_tr["features"].astype(np.float32); y_full = d_tr["labels"].astype(np.int64)
        rng = np.random.RandomState(0)
        sel = []
        for c in range(int(y_full.max())+1):
            ic = np.where(y_full == c)[0]
            sel.extend(rng.choice(ic, min(100, len(ic)), replace=False).tolist())
        X_tr = X_full[np.array(sel)]; y_tr = y_full[np.array(sel)]
        d_te = np.load(CACHE / f"{model}_imagenet_test.npz")
        X_te = d_te["features"].astype(np.float32); y_te = d_te["labels"].astype(np.int64)
    else:
        d_tr = np.load(CACHE / f"{model}_{dataset}_train.npz")
        X_tr = d_tr["features"].astype(np.float32); y_tr = d_tr["labels"].astype(np.int64)
        d_te = np.load(CACHE / f"{model}_{dataset}_test.npz")
        X_te = d_te["features"].astype(np.float32); y_te = d_te["labels"].astype(np.int64)
    return X_tr, y_tr, X_te, y_te


def superclass_map(dataset, n_classes, train_only_imagenet_super=None):
    if dataset == "imagenet":
        if train_only_imagenet_super is None:
            wn = np.load(WN_DIST)
            train_only_imagenet_super = AgglomerativeClustering(n_clusters=30, metric='precomputed', linkage='average').fit_predict(wn)
        return train_only_imagenet_super
    if dataset == "cifar100":
        return FINE_TO_COARSE_C100
    if dataset == "cifar10":
        return CIFAR10_SUPER
    if dataset == "fashionmnist":
        return FMNIST_SUPER
    return None  # dtd, mnist: no superclass


def evaluate(model, dataset, super_imagenet=None, n_queries=1000, train_per_class=50, seed=42):
    try:
        X_tr, y_tr, X_te, y_te = load_features(model, dataset)
    except FileNotFoundError as e:
        print(f"  SKIP {model}/{dataset}: {e}", flush=True); return None
    n_classes = int(max(y_tr.max(), y_te.max())) + 1
    rng = np.random.RandomState(seed)
    # Subsample DB to 50/class
    sub = []
    for c in range(n_classes):
        ic = np.where(y_tr == c)[0]
        if len(ic) > train_per_class:
            sub.extend(rng.choice(ic, train_per_class, replace=False).tolist())
        else:
            sub.extend(ic.tolist())
    sub = np.array(sub)
    X_db = X_tr[sub]; y_db = y_tr[sub]
    if len(X_te) > n_queries:
        q = rng.choice(len(X_te), n_queries, replace=False)
    else:
        q = np.arange(len(X_te))
    X_q = X_te[q]; y_q = y_te[q]

    # X_tr-based mu/p95 (not centroid-based)
    mu = X_tr.mean(0)
    p95 = np.percentile(np.linalg.norm(X_tr - mu, axis=1), 95)

    Xq_h = project(X_q, mu, p95)
    Xdb_h = project(X_db, mu, p95)

    DR = cdist(X_q, X_db, metric="euclidean")
    DH = poincare_dist(Xq_h, Xdb_h, chunk=200)

    fine_R = p_at_k(DR, y_q, y_db)
    fine_H = p_at_k(DH, y_q, y_db)
    fine_adv = (fine_H - fine_R) * 100

    super_map = superclass_map(dataset, n_classes, super_imagenet)
    if super_map is not None:
        sup_R = p_at_k(DR, super_map[y_q], super_map[y_db])
        sup_H = p_at_k(DH, super_map[y_q], super_map[y_db])
        sup_adv = (sup_H - sup_R) * 100
    else:
        sup_R = sup_H = sup_adv = float("nan")

    return dict(
        model=model, dataset=dataset, paradigm=PARADIGMS[model],
        R_fine=fine_R, H_fine=fine_H, adv_fine_pp=fine_adv,
        R_sup=sup_R, H_sup=sup_H, adv_sup_pp=sup_adv,
    )


def main():
    print("Loading WordNet superclasses for ImageNet...", flush=True)
    if WN_DIST.exists():
        wn = np.load(WN_DIST)
        super_imagenet = AgglomerativeClustering(n_clusters=30, metric='precomputed', linkage='average').fit_predict(wn)
    else:
        super_imagenet = None
        print("  WordNet dist not found; ImageNet super skipped.", flush=True)

    rows = []
    order = ["i21k_t","i21k_s","i21k_b","i21k_l","dinov1_b","dinov2_s","dinov2_b","dinov2_l","dinov2_g","clip_b","clip_l","siglip_b"]
    datasets = ["imagenet","cifar100","cifar10","dtd","fashionmnist","mnist"]
    for ds in datasets:
        for m in order:
            t0 = time.time()
            r = evaluate(m, ds, super_imagenet=super_imagenet)
            if r is not None:
                r["time_s"] = time.time() - t0
                rows.append(r)
                print(f"  {ds:14s} {m:10s}: R-fine adv={r['adv_fine_pp']:+.2f}  R-sup adv={r['adv_sup_pp']:+.2f}  ({r['time_s']:.0f}s)", flush=True)
                pd.DataFrame(rows).to_csv(OUT/"retrieval_t707.csv", index=False)
    print(f"\nDone: {OUT}/retrieval_t707.csv  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
