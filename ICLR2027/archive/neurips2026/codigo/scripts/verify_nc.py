#!/usr/bin/env python3
"""
Independent re-implementation of NC (R and H) to verify the original numbers.
Different code path, different libraries (numpy/scipy where possible),
to catch any bugs in the original tasks_CDE.py / tasks_imagenet.py.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist

ROOT = Path("/home/javi/Platonic")
CACHE = ROOT / "results/practical_tasks_cache"

def s4_project_np(X, mu, scale):
    """Pure numpy S4 projection."""
    Xs = (X - mu) * scale
    nrm = np.linalg.norm(Xs, axis=-1, keepdims=True).clip(min=1e-7)
    Y = np.tanh(nrm/2) / nrm * Xs
    cur = np.linalg.norm(Y, axis=-1, keepdims=True).clip(min=1e-7)
    return Y * np.minimum((1.0-1e-3)/cur, 1.0)

def s4_params_np(X):
    mu = X.mean(0)
    p95 = np.percentile(np.linalg.norm(X - mu, axis=1), 95)
    scale = 2 * np.arctanh(0.5) / max(p95, 1e-7)
    return mu, scale

def poincare_dist_np(X, Y):
    """Pure numpy Poincare distance, n x m matrix."""
    x_sq = (X*X).sum(-1, keepdims=True)
    y_sq = (Y*Y).sum(-1, keepdims=True).T
    d_sq = (x_sq + y_sq - 2.0 * X @ Y.T).clip(min=0)
    denom = ((1-x_sq)*(1-y_sq)).clip(min=1e-7)
    arg = (1 + 2*d_sq/denom).clip(min=1+1e-7)
    return np.arccosh(arg)

def nc_R(X_tr, y_tr, X_te, y_te, n_classes):
    cents = np.stack([X_tr[y_tr==c].mean(0) for c in range(n_classes)])
    D = cdist(X_te, cents, metric='euclidean')
    pred = np.argmin(D, axis=1)
    return float((pred == y_te).mean()), pred

def nc_H(X_tr, y_tr, X_te, y_te, n_classes):
    mu, sc = s4_params_np(X_tr)
    Xtr_p = s4_project_np(X_tr, mu, sc)
    Xte_p = s4_project_np(X_te, mu, sc)
    # centroids in ball
    cents_raw = np.stack([Xtr_p[y_tr==c].mean(0) for c in range(n_classes)])
    cents_h = s4_project_np(cents_raw, np.zeros_like(mu), 1.0)  # re-clip with exp_map
    D = poincare_dist_np(Xte_p, cents_h)
    pred = np.argmin(D, axis=1)
    return float((pred == y_te).mean()), pred

def verify(model, dataset):
    tr_p = CACHE / f"{model}_{dataset}_train.npz"
    te_p = CACHE / f"{model}_{dataset}_test.npz"
    if not tr_p.exists() or not te_p.exists():
        return None
    tr = np.load(tr_p); te = np.load(te_p)
    X_tr, y_tr = tr["features"].astype(np.float32), tr["labels"].astype(np.int64)
    X_te, y_te = te["features"].astype(np.float32), te["labels"].astype(np.int64)
    n_classes = int(max(y_tr.max(), y_te.max())) + 1

    r_acc, _ = nc_R(X_tr, y_tr, X_te, y_te, n_classes)
    h_acc, _ = nc_H(X_tr, y_tr, X_te, y_te, n_classes)
    return r_acc, h_acc

# Compare with stored CSVs
mp = pd.read_csv(ROOT/"results/master_per_dataset.csv")
samples = [
    # (model, dataset, expected_R, expected_H)
    ("dinov2_l", "cifar100"),
    ("dinov2_l", "imagenet"),
    ("dinov2_g", "imagenet"),
    ("dinov2_b", "imagenet"),
    ("i21k_t", "cifar100"),
    ("i21k_l", "imagenet"),
    ("clip_b", "cifar100"),
    ("siglip_b", "mnist"),
]

print(f"{'Model':10} {'Dataset':12} | {'CSV R':>7} {'verif R':>8} {'Δ R':>6} | {'CSV H':>7} {'verif H':>8} {'Δ H':>6} | {'CSV adv':>8} {'verif adv':>9}")
print("-" * 115)
for m, ds in samples:
    row = mp[(mp.model == m) & (mp.dataset == ds)]
    if len(row) == 0:
        print(f"{m}/{ds}: not in CSV"); continue
    csv_R = float(row.iloc[0]['NC_R_acc'])
    csv_H = float(row.iloc[0]['NC_H_acc'])
    csv_adv = float(row.iloc[0]['NC_adv_pp'])

    res = verify(m, ds)
    if res is None:
        print(f"{m}/{ds}: missing npz"); continue
    v_R, v_H = res
    v_adv = (v_H - v_R) * 100
    flag_R = "✓" if abs(v_R - csv_R) < 0.005 else "✗"
    flag_H = "✓" if abs(v_H - csv_H) < 0.005 else "✗"
    print(f"{m:10} {ds:12} | {csv_R:.4f}  {v_R:.4f} {flag_R} {(v_R-csv_R)*100:+.2f} | "
          f"{csv_H:.4f}  {v_H:.4f} {flag_H} {(v_H-csv_H)*100:+.2f} | {csv_adv:+.2f}    {v_adv:+.2f}")
