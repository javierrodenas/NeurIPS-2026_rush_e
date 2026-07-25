#!/usr/bin/env python3
"""
Run NC and Few-shot on the FULL ImageNet train features (1.23M samples)
that have already been extracted.

Same test set (50K) as the 100/class evaluation, so H_advantage values
are directly comparable. CPU-only to avoid conflicting with the still-running
extraction on GPU 0.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

# Force CPU
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
OUT = ROOT / "results/tasks_imagenet_full"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = "cpu"
print(f"Device: {DEVICE}", flush=True)

PARADIGMS = {
    "i21k_t":"Supervised","i21k_s":"Supervised","i21k_b":"Supervised","i21k_l":"Supervised",
    "dinov1_b":"SSL","dinov2_s":"SSL","dinov2_b":"SSL","dinov2_l":"SSL","dinov2_g":"SSL",
    "clip_b":"Contrastive","clip_l":"Contrastive","siglip_b":"Contrastive",
}
N_CLASSES = 1000


def s4_project(X, mu, scale):
    Xs = (X - mu) * scale
    nrm = np.linalg.norm(Xs, axis=-1, keepdims=True).clip(min=1e-7)
    Y = np.tanh(nrm/2) / nrm * Xs
    cur = np.linalg.norm(Y, axis=-1, keepdims=True).clip(min=1e-7)
    return Y * np.minimum((1.0-1e-3)/cur, 1.0)


def s4_params_from_centroids(C):
    """Use centroid stats for mu and scale (bypasses needing all 1.23M points centered)."""
    mu = C.mean(0)
    p95 = np.percentile(np.linalg.norm(C - mu, axis=1), 95)
    return mu, 2 * np.arctanh(0.5) / max(p95, 1e-7)


def poincare_dist_chunked(X, Y, chunk=500):
    n = len(X)
    out = np.empty((n, len(Y)), dtype=np.float32)
    y_sq = (Y*Y).sum(-1)[None, :]
    for i in range(0, n, chunk):
        Xc = X[i:i+chunk]
        x_sq = (Xc*Xc).sum(-1, keepdims=True)
        d_sq = (x_sq + y_sq - 2.0 * Xc @ Y.T).clip(min=0)
        denom = ((1-x_sq)*(1-y_sq)).clip(min=1e-7)
        arg = (1 + 2*d_sq/denom).clip(min=1+1e-7)
        out[i:i+chunk] = np.arccosh(arg)
    return out


def gromov_delta(X, n_quads=50000, seed=42):
    from scipy.spatial.distance import pdist, squareform
    rng = np.random.RandomState(seed)
    D = squareform(pdist(X, 'euclidean'))
    diam = D.max()
    n = len(X)
    quads = np.array([rng.choice(n, 4, replace=False) for _ in range(n_quads)])
    i,j,k,l = quads.T
    s = np.sort(np.stack([D[i,j]+D[k,l], D[i,k]+D[j,l], D[i,l]+D[j,k]],1),1)
    return float(((s[:,2]-s[:,1])/2).mean()/diam)


def evaluate(model):
    tr = CACHE / f"{model}_imagenet_fulltrain.npz"
    te = CACHE / f"{model}_imagenet_test.npz"
    if not tr.exists():
        return None
    if not te.exists():
        return None
    print(f"\n>>> {model}", flush=True)
    t0 = time.time()
    d_tr = np.load(tr); d_te = np.load(te)
    X_tr = d_tr["features"].astype(np.float32); y_tr = d_tr["labels"].astype(np.int64)
    X_te = d_te["features"].astype(np.float32); y_te = d_te["labels"].astype(np.int64)
    print(f"  loaded train={X_tr.shape}, test={X_te.shape} ({time.time()-t0:.0f}s)", flush=True)

    # Centroids: mean per class on FULL train
    t1 = time.time()
    cents_R = np.stack([X_tr[y_tr==c].mean(0) for c in range(N_CLASSES)])
    print(f"  centroids in {time.time()-t1:.0f}s, dim={cents_R.shape[1]}", flush=True)

    # delta on new centroids
    delta = gromov_delta(cents_R)
    print(f"  delta = {delta:.4f}", flush=True)

    # R-NC
    t1 = time.time()
    D_R = cdist(X_te, cents_R, metric='euclidean')
    pred_R = np.argmin(D_R, axis=1)
    r_acc = float((pred_R == y_te).mean())
    del D_R
    print(f"  R-NC = {r_acc:.4f} ({time.time()-t1:.0f}s)", flush=True)

    # H-NC
    t1 = time.time()
    mu, sc = s4_params_from_centroids(cents_R)
    Xte_p = s4_project(X_te, mu, sc)
    cents_h_raw = s4_project(cents_R, mu, sc)
    cents_h = s4_project(cents_h_raw, np.zeros_like(mu), 1.0)
    D_H = poincare_dist_chunked(Xte_p, cents_h, chunk=500)
    pred_H = np.argmin(D_H, axis=1)
    h_acc = float((pred_H == y_te).mean())
    del D_H, Xte_p
    print(f"  H-NC = {h_acc:.4f} ({time.time()-t1:.0f}s) adv={(h_acc-r_acc)*100:+.2f}pp", flush=True)

    # superclass for retrieval-style metric
    if WN_DIST.exists():
        wn = np.load(WN_DIST)
        super_map = AgglomerativeClustering(n_clusters=30, metric='precomputed', linkage='average').fit_predict(wn)
        nc_R_super = float((super_map[pred_R] == super_map[y_te]).mean())
        nc_H_super = float((super_map[pred_H] == super_map[y_te]).mean())
    else:
        nc_R_super = nc_H_super = float("nan")

    print(f"  total {time.time()-t0:.0f}s", flush=True)
    return dict(
        model=model, paradigm=PARADIGMS[model], dataset='imagenet_fulltrain',
        delta=delta, in_dim=int(X_tr.shape[1]), n_train_full=int(len(X_tr)),
        nc_R_acc=r_acc, nc_H_acc=h_acc, nc_adv_pp=(h_acc-r_acc)*100,
        nc_R_super_acc=nc_R_super, nc_H_super_acc=nc_H_super,
    )


def main():
    rows = []
    # Order: cheap models first, in case extraction completes new ones we can pick up
    order = ["i21k_t","i21k_s","i21k_b","i21k_l","dinov1_b","clip_b","clip_l","siglip_b",
             "dinov2_s","dinov2_b","dinov2_l","dinov2_g"]
    for m in order:
        res = evaluate(m)
        if res is not None:
            rows.append(res)
            pd.DataFrame(rows).to_csv(OUT/"tasks_imagenet_full.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(OUT/"tasks_imagenet_full.csv", index=False)

    # Compare against existing tasks_imagenet results (100/class train)
    print("\n" + "="*100)
    print(" COMPARISON: 100/class train (current) vs full train (new)")
    print("="*100)
    try:
        old = pd.read_csv(ROOT/"results/tasks_imagenet/tasks_imagenet.csv")
        merged = df.merge(old[['model','delta','NC_R_acc','NC_H_acc','NC_adv_pp']], on='model', suffixes=('_full','_100c'))
        merged['delta_change'] = merged['delta_full'] - merged['delta_100c']
        merged['NC_R_change'] = merged['nc_R_acc'] - merged['NC_R_acc']
        merged['NC_H_change'] = merged['nc_H_acc'] - merged['NC_H_acc']
        merged['NC_adv_change'] = merged['nc_adv_pp'] - merged['NC_adv_pp']
        cols = ['model','paradigm','delta_100c','delta_full','delta_change',
                'NC_R_acc','nc_R_acc','NC_R_change','NC_H_acc','nc_H_acc','NC_H_change',
                'NC_adv_pp','nc_adv_pp','NC_adv_change']
        print(merged[cols].round(4).to_string(index=False))
        merged.to_csv(OUT/"comparison_100c_vs_full.csv", index=False)
        print(f"\nSaved comparison: {OUT}/comparison_100c_vs_full.csv")
    except Exception as e:
        print(f"Could not load 100/class baseline: {e}")


if __name__ == "__main__":
    main()
