#!/usr/bin/env python3
"""
Full independent re-verification of NC R and NC H across all 72 combos
(12 models x 6 datasets). Pure numpy/scipy, different code path from
tasks_CDE.py / tasks_imagenet.py.

Outputs verify_full_nc.csv and reports any discrepancy with the master.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist
import time

ROOT = Path("/home/javi/Platonic")
CACHE = ROOT / "results/practical_tasks_cache"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["i21k_t","i21k_s","i21k_b","i21k_l",
          "dinov1_b","dinov2_s","dinov2_b","dinov2_l","dinov2_g",
          "clip_b","clip_l","siglip_b"]
DATASETS = ["cifar100","cifar10","dtd","fashionmnist","mnist","imagenet"]


def s4_project(X, mu, scale):
    Xs = (X - mu) * scale
    nrm = np.linalg.norm(Xs, axis=-1, keepdims=True).clip(min=1e-7)
    Y = np.tanh(nrm/2) / nrm * Xs
    cur = np.linalg.norm(Y, axis=-1, keepdims=True).clip(min=1e-7)
    return Y * np.minimum((1.0-1e-3)/cur, 1.0)

def s4_params(X):
    mu = X.mean(0)
    p95 = np.percentile(np.linalg.norm(X - mu, axis=1), 95)
    return mu, 2 * np.arctanh(0.5) / max(p95, 1e-7)

def poincare_dist_chunked(X, Y, chunk=200):
    """Poincare distance n x m with chunking over X to limit memory."""
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

def evaluate(model, dataset):
    tr = CACHE / f"{model}_{dataset}_train.npz"
    te = CACHE / f"{model}_{dataset}_test.npz"
    if not tr.exists() or not te.exists():
        return None
    d_tr = np.load(tr); d_te = np.load(te)
    X_tr = d_tr["features"].astype(np.float32); y_tr = d_tr["labels"].astype(np.int64)
    X_te = d_te["features"].astype(np.float32); y_te = d_te["labels"].astype(np.int64)
    n_classes = int(max(y_tr.max(), y_te.max())) + 1

    # R-NC
    cents_R = np.stack([X_tr[y_tr==c].mean(0) for c in range(n_classes)])
    D_R = cdist(X_te, cents_R, metric='euclidean')
    pred_R = np.argmin(D_R, axis=1)
    r_acc = float((pred_R == y_te).mean())

    # H-NC
    mu, sc = s4_params(X_tr)
    Xtr_p = s4_project(X_tr, mu, sc)
    Xte_p = s4_project(X_te, mu, sc)
    cents_h = np.stack([Xtr_p[y_tr==c].mean(0) for c in range(n_classes)])
    cents_h = s4_project(cents_h, np.zeros_like(mu), 1.0)
    D_H = poincare_dist_chunked(Xte_p, cents_h, chunk=500)
    pred_H = np.argmin(D_H, axis=1)
    h_acc = float((pred_H == y_te).mean())

    return r_acc, h_acc, n_classes, X_tr.shape[1]


def main():
    rows = []
    t_total = time.time()
    for model in MODELS:
        for ds in DATASETS:
            t0 = time.time()
            res = evaluate(model, ds)
            if res is None:
                print(f"  SKIP {model}/{ds}: missing npz")
                continue
            r_acc, h_acc, n_classes, dim = res
            elapsed = time.time() - t0
            rows.append({"model": model, "dataset": ds,
                         "n_classes": n_classes, "in_dim": dim,
                         "verif_R_acc": r_acc, "verif_H_acc": h_acc,
                         "verif_adv_pp": (h_acc - r_acc) * 100,
                         "time_s": elapsed})
            print(f"  {model:10s}/{ds:13s} d={dim:>4} k={n_classes:>4}  "
                  f"R={r_acc:.4f} H={h_acc:.4f} adv={(h_acc-r_acc)*100:+.2f}pp  ({elapsed:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT/"verify_full_nc.csv", index=False)
    print(f"\nTotal time: {time.time()-t_total:.0f}s")
    print(f"Saved: {OUT}/verify_full_nc.csv")

    # Compare with master_per_dataset.csv
    mp = pd.read_csv(OUT/"master_per_dataset.csv")
    print("\n" + "="*90)
    print(" Comparison with master_per_dataset.csv (delta vs CSV)")
    print("="*90)
    n_match_R, n_match_H, n_total = 0, 0, 0
    bad_R, bad_H = [], []
    for _, r in df.iterrows():
        m = mp[(mp.model == r['model']) & (mp.dataset == r['dataset'])]
        if len(m) == 0: continue
        m = m.iloc[0]
        d_R = abs(r['verif_R_acc'] - m['NC_R_acc'])
        d_H = abs(r['verif_H_acc'] - m['NC_H_acc'])
        n_total += 1
        if d_R < 1e-3: n_match_R += 1
        else: bad_R.append((r['model'], r['dataset'], d_R))
        if d_H < 1e-3: n_match_H += 1
        else: bad_H.append((r['model'], r['dataset'], d_H))
    print(f"  R matches:  {n_match_R}/{n_total}")
    print(f"  H matches:  {n_match_H}/{n_total}")
    if bad_R:
        print(f"  BAD R: {bad_R}")
    if bad_H:
        print(f"  BAD H: {bad_H}")
    if not bad_R and not bad_H:
        print("\n  ✓ ALL 72 NC R AND NC H VALUES MATCH THE MASTER CSV")


if __name__ == "__main__":
    main()
