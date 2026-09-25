#!/usr/bin/env python3
"""
Curvature sweep for NC, FS and R on the 12 vision full-train models on ImageNet.
For each (model, c) we compute the H-version of each task and the R baseline
(curvature-independent, computed once per model per task).

Pipeline matches production tasks_imagenet_full*.py: double-projection of
prototypes/database (s4_project applied twice with target=0.5/sqrt(c) and then
mu=0,scale=1).
"""
import os, sys, time
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
OUT = ROOT / "results/tasks_imagenet_full"

PARADIGMS = {
    "i21k_t":"Supervised","i21k_s":"Supervised","i21k_b":"Supervised","i21k_l":"Supervised",
    "dinov1_b":"SSL","dinov2_s":"SSL","dinov2_b":"SSL","dinov2_l":"SSL","dinov2_g":"SSL",
    "clip_b":"Contrastive","clip_l":"Contrastive","siglip_b":"Contrastive",
}
N_CLASSES = 1000
CURVATURES = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0]


def project_to_ball(X, mu, scale):
    """exp-map at origin in unit Poincare ball, with scale s applied to (X-mu)."""
    Xs = (X - mu) * scale
    nrm = np.linalg.norm(Xs, axis=-1, keepdims=True).clip(min=1e-7)
    Y = np.tanh(nrm / 2) / nrm * Xs
    cur = np.linalg.norm(Y, axis=-1, keepdims=True).clip(min=1e-7)
    return Y * np.minimum((1.0 - 1e-3) / cur, 1.0)


def project_for_curvature(X, mu, p95, c):
    target = min(0.5 / np.sqrt(c), 0.95)
    return project_to_ball(X, mu, 2 * np.arctanh(target) / max(p95, 1e-7))


def double_project(X, mu, p95, c):
    Y = project_for_curvature(X, mu, p95, c)
    return project_to_ball(Y, np.zeros_like(mu), 1.0)


def poincare_dist_chunked(X, Y, chunk=500):
    n = len(X); m = len(Y)
    out = np.empty((n, m), dtype=np.float32)
    y_sq = (Y * Y).sum(-1)[None, :]
    for i in range(0, n, chunk):
        Xc = X[i:i + chunk]
        x_sq = (Xc * Xc).sum(-1, keepdims=True)
        d_sq = (x_sq + y_sq - 2.0 * Xc @ Y.T).clip(min=0)
        denom = ((1 - x_sq) * (1 - y_sq)).clip(min=1e-7)
        arg = (1 + 2 * d_sq / denom).clip(min=1 + 1e-7)
        out[i:i + chunk] = np.arccosh(arg)
    return out


def euclid_dist(A, B):
    a_sq = (A * A).sum(-1, keepdims=True)
    b_sq = (B * B).sum(-1, keepdims=True).T
    return np.sqrt((a_sq + b_sq - 2.0 * A @ B.T).clip(min=0))


def task_NC(X_tr, y_tr, X_te, y_te, mu, p95, c):
    """Returns (R_acc, H_acc)."""
    cents_R = np.stack([X_tr[y_tr == k].mean(0) for k in range(N_CLASSES)])
    # R
    D_R = cdist(X_te, cents_R, metric='euclidean')
    pred_R = np.argmin(D_R, axis=1); del D_R
    r_acc = float((pred_R == y_te).mean())
    # H
    Xte_p = project_for_curvature(X_te, mu, p95, c)
    cents_h = double_project(cents_R, mu, p95, c)
    D_H = poincare_dist_chunked(Xte_p, cents_h, chunk=500)
    pred_H = np.argmin(D_H, axis=1); del D_H, Xte_p
    h_acc = float((pred_H == y_te).mean())
    return r_acc, h_acc


def task_FS(X, y, mu, p95, c, n_episodes=1000, n_way=5, k_shot=5, q_query=15, seed=42):
    classes = np.unique(y)
    rng = np.random.RandomState(seed)
    idx_per_c = {k: np.where(y == k)[0] for k in classes}
    X_h = project_for_curvature(X, mu, p95, c)
    R_acc = []; H_acc = []
    for _ in range(n_episodes):
        sel = rng.choice(classes, n_way, replace=False)
        sup = []; qry = []; qlab = []
        for li, k in enumerate(sel):
            ic = idx_per_c[k]
            if len(ic) < k_shot + q_query:
                continue
            pick = rng.choice(ic, k_shot + q_query, replace=False)
            sup.extend(pick[:k_shot])
            qry.extend(pick[k_shot:k_shot + q_query])
            qlab.extend([li] * q_query)
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
        prots_h = project_to_ball(prots_h, np.zeros_like(mu), 1.0)
        D_H = poincare_dist_chunked(qry_h, prots_h)
        H_acc.append((np.argmin(D_H, 1) == qlab).mean())
    return float(np.mean(R_acc)), float(np.mean(H_acc))


def task_R(X_tr, y_tr, X_te, y_te, mu, p95, c, super_map, n_queries=1000, train_per_class=50, seed=42):
    rng = np.random.RandomState(seed)
    sub = []
    for k in range(N_CLASSES):
        ic = np.where(y_tr == k)[0]
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

    Xq_h = project_for_curvature(X_q, mu, p95, c)
    Xdb_h = project_for_curvature(X_db, mu, p95, c)

    DR = euclid_dist(X_q, X_db)
    DH = poincare_dist_chunked(Xq_h, Xdb_h, chunk=200)

    def p_at_k(D, ql, dbl, k=10):
        nn = np.argpartition(D, k, axis=1)[:, :k]
        nn_sorted = np.empty_like(nn)
        for i in range(len(D)):
            order = np.argsort(D[i, nn[i]])
            nn_sorted[i] = nn[i][order]
        return float((dbl[nn_sorted] == ql[:, None]).mean())

    return dict(
        R_fine=p_at_k(DR, y_q, y_db),
        H_fine=p_at_k(DH, y_q, y_db),
        R_sup=p_at_k(DR, super_map[y_q], super_map[y_db]),
        H_sup=p_at_k(DH, super_map[y_q], super_map[y_db]),
    )


def main():
    print("Loading WordNet superclasses...", flush=True)
    wn = np.load(WN_DIST)
    super_map = AgglomerativeClustering(n_clusters=30, metric='precomputed', linkage='average').fit_predict(wn)

    rows_nc = []; rows_fs = []; rows_r = []
    order = ["i21k_t","i21k_s","i21k_b","i21k_l","dinov1_b","clip_b","clip_l","siglip_b",
             "dinov2_s","dinov2_b","dinov2_l","dinov2_g"]

    for model in order:
        tr = CACHE / f"{model}_imagenet_fulltrain.npz"
        te = CACHE / f"{model}_imagenet_test.npz"
        if not tr.exists() or not te.exists():
            print(f"  SKIP {model}", flush=True); continue
        print(f"\n>>> {model}", flush=True)
        t0 = time.time()
        d_tr = np.load(tr); d_te = np.load(te)
        X_tr = d_tr["features"].astype(np.float32); y_tr = d_tr["labels"].astype(np.int64)
        X_te = d_te["features"].astype(np.float32); y_te = d_te["labels"].astype(np.int64)
        print(f"  loaded ({time.time()-t0:.0f}s)", flush=True)

        # mu, p95 from centroids on full train (to match NC pipeline)
        cents = np.stack([X_tr[y_tr == k].mean(0) for k in range(N_CLASSES)])
        mu = cents.mean(0)
        p95 = np.percentile(np.linalg.norm(cents - mu, axis=1), 95)

        # FS uses concatenated train+test pool
        X_all = np.concatenate([X_tr, X_te], 0); y_all = np.concatenate([y_tr, y_te], 0)
        # mu/p95 for FS computed on full pool
        mu_fs = X_all.mean(0)
        p95_fs = np.percentile(np.linalg.norm(X_all - mu_fs, axis=1), 95)

        # Track c-independent baselines (use first c just to get them)
        for c in CURVATURES:
            t1 = time.time()
            r_nc, h_nc = task_NC(X_tr, y_tr, X_te, y_te, mu, p95, c)
            r_fs, h_fs = task_FS(X_all, y_all, mu_fs, p95_fs, c)
            r_dict = task_R(X_tr, y_tr, X_te, y_te, mu, p95, c, super_map)
            adv_nc = (h_nc - r_nc) * 100
            adv_fs = (h_fs - r_fs) * 100
            adv_r_fine = (r_dict['H_fine'] - r_dict['R_fine']) * 100
            adv_r_sup = (r_dict['H_sup'] - r_dict['R_sup']) * 100
            print(f"  c={c:>5}: NC adv={adv_nc:+.2f}  FS adv={adv_fs:+.2f}  R-fine adv={adv_r_fine:+.2f}  R-sup adv={adv_r_sup:+.2f}  ({time.time()-t1:.0f}s)", flush=True)
            rows_nc.append(dict(model=model, paradigm=PARADIGMS[model], c=c, R_acc=r_nc, H_acc=h_nc, H_adv_pp=adv_nc))
            rows_fs.append(dict(model=model, paradigm=PARADIGMS[model], c=c, R_acc=r_fs, H_acc=h_fs, H_adv_pp=adv_fs))
            rows_r.append(dict(model=model, paradigm=PARADIGMS[model], c=c,
                               R_fine=r_dict['R_fine'], H_fine=r_dict['H_fine'], adv_fine_pp=adv_r_fine,
                               R_sup=r_dict['R_sup'], H_sup=r_dict['H_sup'], adv_sup_pp=adv_r_sup))
            pd.DataFrame(rows_nc).to_csv(OUT/"sweep_curvature_nc.csv", index=False)
            pd.DataFrame(rows_fs).to_csv(OUT/"sweep_curvature_fs.csv", index=False)
            pd.DataFrame(rows_r).to_csv(OUT/"sweep_curvature_r.csv", index=False)
        print(f"  total {time.time()-t0:.0f}s", flush=True)

    print(f"\nDone. {len(rows_nc)} NC rows / {len(rows_fs)} FS rows / {len(rows_r)} R rows.")


if __name__ == "__main__":
    main()
