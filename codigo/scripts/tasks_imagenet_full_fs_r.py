#!/usr/bin/env python3
"""
Add Few-shot and Retrieval to the full-train models that already have NC.
CPU-only, runs in parallel with the ongoing GPU extraction.
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
from sklearn.cluster import AgglomerativeClustering

ROOT = Path("/home/javi/Platonic")
CACHE = ROOT / "results/practical_tasks_cache"
WN_DIST = ROOT / "results/features/imagenet/imagenet1k_wordnet_dist.npy"
OUT = ROOT / "results/tasks_imagenet_full"

PARADIGMS = {
    "i21k_t":"Supervised","i21k_s":"Supervised","i21k_b":"Supervised","i21k_l":"Supervised",
    "dinov1_b":"SSL","dinov2_s":"SSL","dinov2_b":"SSL","dinov2_l":"SSL","dinov2_g":"SSL",
    "clip_b":"Contrastive","clip_l":"Contrastive","siglip_b":"Contrastive",
}


def s4_project(X, mu, scale):
    Xs = (X - mu) * scale
    nrm = np.linalg.norm(Xs, axis=-1, keepdims=True).clip(min=1e-7)
    Y = np.tanh(nrm/2) / nrm * Xs
    cur = np.linalg.norm(Y, axis=-1, keepdims=True).clip(min=1e-7)
    return Y * np.minimum((1.0-1e-3)/cur, 1.0)


def s4_params_from_centroids(C):
    mu = C.mean(0)
    p95 = np.percentile(np.linalg.norm(C - mu, axis=1), 95)
    return mu, 2 * np.arctanh(0.5) / max(p95, 1e-7)


def euclid_dist(A, B):
    a_sq = (A*A).sum(-1, keepdims=True)
    b_sq = (B*B).sum(-1, keepdims=True).T
    return np.sqrt((a_sq + b_sq - 2.0 * A @ B.T).clip(min=0))


def poincare_dist_chunked(X, Y, chunk=500):
    n = len(X); m = len(Y)
    out = np.empty((n, m), dtype=np.float32)
    y_sq = (Y*Y).sum(-1)[None, :]
    for i in range(0, n, chunk):
        Xc = X[i:i+chunk]
        x_sq = (Xc*Xc).sum(-1, keepdims=True)
        d_sq = (x_sq + y_sq - 2.0 * Xc @ Y.T).clip(min=0)
        denom = ((1-x_sq)*(1-y_sq)).clip(min=1e-7)
        arg = (1 + 2*d_sq/denom).clip(min=1+1e-7)
        out[i:i+chunk] = np.arccosh(arg)
    return out


def task_FS(X_tr, y_tr, X_te, y_te, n_episodes=1000, n_way=5, k_shot=5, q_query=15, seed=42):
    """Few-shot 5-way 5-shot, 1000 episodes."""
    X = np.concatenate([X_tr, X_te], 0); y = np.concatenate([y_tr, y_te], 0)
    classes = np.unique(y)
    rng = np.random.RandomState(seed)
    idx_per_c = {c: np.where(y==c)[0] for c in classes}

    # S4 params on full pool
    mu = X.mean(0)
    p95 = np.percentile(np.linalg.norm(X - mu, axis=1), 95)
    sc = 2*np.arctanh(0.5)/max(p95, 1e-7)
    X_h = s4_project(X, mu, sc)

    R_acc = []; H_acc = []
    for ep in range(n_episodes):
        sel = rng.choice(classes, n_way, replace=False)
        sup = []; qry = []; qlab = []
        for li, c in enumerate(sel):
            ic = idx_per_c[c]
            if len(ic) < k_shot+q_query:
                continue
            pick = rng.choice(ic, k_shot+q_query, replace=False)
            sup.extend(pick[:k_shot])
            qry.extend(pick[k_shot:k_shot+q_query])
            qlab.extend([li]*q_query)
        if not qry:
            continue
        sup = np.array(sup); qry = np.array(qry); qlab = np.array(qlab)
        # R
        prots_R = X[sup].reshape(n_way, k_shot, -1).mean(1)
        D_R = euclid_dist(X[qry], prots_R)
        R_acc.append((np.argmin(D_R, 1) == qlab).mean())
        # H
        sup_h = X_h[sup]; qry_h = X_h[qry]
        prots_h = sup_h.reshape(n_way, k_shot, -1).mean(1)
        prots_h = s4_project(prots_h, np.zeros_like(mu), 1.0)
        D_H = poincare_dist_chunked(qry_h, prots_h)
        H_acc.append((np.argmin(D_H, 1) == qlab).mean())
    return float(np.mean(R_acc)), float(np.mean(H_acc))


def task_R(X_tr, y_tr, X_te, y_te, n_classes, super_map, n_queries=1000, train_per_class=50, seed=42):
    """Sample-to-sample retrieval P@10 on subsampled DB."""
    rng = np.random.RandomState(seed)
    sub = []
    for c in range(n_classes):
        ic = np.where(y_tr==c)[0]
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

    mu = X_tr.mean(0)
    p95 = np.percentile(np.linalg.norm(X_tr - mu, axis=1), 95)
    sc = 2*np.arctanh(0.5)/max(p95, 1e-7)
    Xq_h = s4_project(X_q, mu, sc)
    Xdb_h = s4_project(X_db, mu, sc)

    DR = euclid_dist(X_q, X_db)
    DH = poincare_dist_chunked(Xq_h, Xdb_h, chunk=200)

    def p_at_k(D, ql, dbl, k=10):
        nn = np.argpartition(D, k, axis=1)[:, :k]
        # sort within for top-k
        nn_sorted = np.empty_like(nn)
        for i in range(len(D)):
            order = np.argsort(D[i, nn[i]])
            nn_sorted[i] = nn[i][order]
        return float((dbl[nn_sorted] == ql[:, None]).mean())

    return dict(
        R_fine_p10=p_at_k(DR, y_q, y_db),
        H_fine_p10=p_at_k(DH, y_q, y_db),
        R_sup_p10=p_at_k(DR, super_map[y_q], super_map[y_db]),
        H_sup_p10=p_at_k(DH, super_map[y_q], super_map[y_db]),
    )


def main():
    print("Loading WordNet superclasses...", flush=True)
    wn = np.load(WN_DIST)
    super_map = AgglomerativeClustering(n_clusters=30, metric='precomputed', linkage='average').fit_predict(wn)

    rows = []
    order = ["i21k_t","i21k_s","i21k_b","i21k_l","dinov1_b","clip_b","clip_l","siglip_b","dinov2_s","dinov2_b","dinov2_l","dinov2_g"]
    for model in order:
        tr = CACHE / f"{model}_imagenet_fulltrain.npz"
        te = CACHE / f"{model}_imagenet_test.npz"
        if not tr.exists() or not te.exists():
            print(f"  SKIP {model}: missing npz", flush=True); continue
        t0 = time.time()
        d_tr = np.load(tr); d_te = np.load(te)
        X_tr = d_tr["features"].astype(np.float32); y_tr = d_tr["labels"].astype(np.int64)
        X_te = d_te["features"].astype(np.float32); y_te = d_te["labels"].astype(np.int64)
        n_classes = int(max(y_tr.max(), y_te.max())) + 1
        print(f"\n>>> {model}  loaded in {time.time()-t0:.0f}s", flush=True)

        # FS
        t1 = time.time()
        fs_R, fs_H = task_FS(X_tr, y_tr, X_te, y_te, n_episodes=1000)
        print(f"  FS: R={fs_R:.4f} H={fs_H:.4f} adv={(fs_H-fs_R)*100:+.2f}pp ({time.time()-t1:.0f}s)", flush=True)

        # R retrieval
        t1 = time.time()
        r = task_R(X_tr, y_tr, X_te, y_te, n_classes, super_map)
        print(f"  R-fine: R={r['R_fine_p10']:.4f} H={r['H_fine_p10']:.4f} adv={(r['H_fine_p10']-r['R_fine_p10'])*100:+.2f}pp", flush=True)
        print(f"  R-sup:  R={r['R_sup_p10']:.4f} H={r['H_sup_p10']:.4f} adv={(r['H_sup_p10']-r['R_sup_p10'])*100:+.2f}pp ({time.time()-t1:.0f}s)", flush=True)

        rows.append(dict(
            model=model, paradigm=PARADIGMS[model],
            FS_R_acc=fs_R, FS_H_acc=fs_H, FS_adv_pp=(fs_H-fs_R)*100,
            **{f"R_{k}": v for k, v in r.items()},
            R_fine_adv_pp=(r['H_fine_p10']-r['R_fine_p10'])*100,
            R_sup_adv_pp=(r['H_sup_p10']-r['R_sup_p10'])*100,
            time_s=time.time()-t0,
        ))
        pd.DataFrame(rows).to_csv(OUT/"tasks_imagenet_full_fs_r.csv", index=False)


if __name__ == "__main__":
    main()
