#!/usr/bin/env python3
"""
Curvature sweep for the post-hoc Poincare projection on the 12 vision
full-train models on ImageNet. For each (model, c) we compute the H-NC
accuracy and the hyperbolic advantage H_adv = (acc_H - acc_R) * 100.

Curvature interpretation: in a Poincare ball of curvature -c, the boundary
is at norm 1/sqrt(c). We map the 95th-percentile feature norm to half the
boundary, which is equivalent to setting target radius t = 0.5/sqrt(c) in
the unit ball with the canonical exp-map.

Output: results/tasks_imagenet_full/sweep_curvature_nc.csv
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

ROOT = Path("/home/javi/Platonic")
CACHE = ROOT / "results/practical_tasks_cache"
OUT = ROOT / "results/tasks_imagenet_full"

PARADIGMS = {
    "i21k_t":"Supervised","i21k_s":"Supervised","i21k_b":"Supervised","i21k_l":"Supervised",
    "dinov1_b":"SSL","dinov2_s":"SSL","dinov2_b":"SSL","dinov2_l":"SSL","dinov2_g":"SSL",
    "clip_b":"Contrastive","clip_l":"Contrastive","siglip_b":"Contrastive",
}
N_CLASSES = 1000
CURVATURES = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0]


def project_unit_ball(X, mu, p95, target):
    """Map features to unit ball so 95th-pct centred norm lands at `target`."""
    s = 2 * np.arctanh(target) / max(p95, 1e-7)
    Xs = (X - mu) * s
    nrm = np.linalg.norm(Xs, axis=-1, keepdims=True).clip(min=1e-7)
    Y = np.tanh(nrm / 2) / nrm * Xs
    cur = np.linalg.norm(Y, axis=-1, keepdims=True).clip(min=1e-7)
    return Y * np.minimum((1.0 - 1e-3) / cur, 1.0)


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


def evaluate(model):
    tr = CACHE / f"{model}_imagenet_fulltrain.npz"
    te = CACHE / f"{model}_imagenet_test.npz"
    if not tr.exists() or not te.exists():
        print(f"  SKIP {model}: missing npz", flush=True); return []
    print(f"\n>>> {model}", flush=True)
    t0 = time.time()
    d_tr = np.load(tr); d_te = np.load(te)
    X_tr = d_tr["features"].astype(np.float32); y_tr = d_tr["labels"].astype(np.int64)
    X_te = d_te["features"].astype(np.float32); y_te = d_te["labels"].astype(np.int64)
    print(f"  loaded train={X_tr.shape}, test={X_te.shape} ({time.time()-t0:.0f}s)", flush=True)

    cents_R = np.stack([X_tr[y_tr == c].mean(0) for c in range(N_CLASSES)])

    # R baseline (curvature-independent)
    D_R = cdist(X_te, cents_R, metric='euclidean')
    pred_R = np.argmin(D_R, axis=1)
    r_acc = float((pred_R == y_te).mean())
    del D_R
    print(f"  R-NC = {r_acc:.4f}", flush=True)

    mu = cents_R.mean(0)
    p95 = np.percentile(np.linalg.norm(cents_R - mu, axis=1), 95)

    rows = []
    for c in CURVATURES:
        t1 = time.time()
        target = 0.5 / np.sqrt(c)
        target = min(target, 0.95)  # safety cap so we don't sit on the boundary
        Xte_p = project_unit_ball(X_te, mu, p95, target)
        # Match production tasks_imagenet_full.py: double-project centroids
        cents_h_raw = project_unit_ball(cents_R, mu, p95, target)
        # Second projection: no shift, scale=1 (target = tanh(0.5) ~ 0.4621 at scale 1)
        # Actually production uses s4_project(., 0, 1.0) which = tanh(|X|/2) at scale 1
        nrm = np.linalg.norm(cents_h_raw, axis=-1, keepdims=True).clip(min=1e-7)
        cents_h = np.tanh(nrm / 2) / nrm * cents_h_raw
        cur = np.linalg.norm(cents_h, axis=-1, keepdims=True).clip(min=1e-7)
        cents_h = cents_h * np.minimum((1.0 - 1e-3) / cur, 1.0)
        D_H = poincare_dist_chunked(Xte_p, cents_h, chunk=500)
        pred_H = np.argmin(D_H, axis=1)
        h_acc = float((pred_H == y_te).mean())
        del D_H, Xte_p
        adv = (h_acc - r_acc) * 100
        print(f"  c={c:>5} (target={target:.3f}): H-NC={h_acc:.4f}  adv={adv:+.2f}pp  ({time.time()-t1:.0f}s)", flush=True)
        rows.append(dict(
            model=model, paradigm=PARADIGMS[model],
            c=c, target_radius=target,
            R_acc=r_acc, H_acc=h_acc, H_adv_pp=adv,
        ))
    print(f"  total {time.time()-t0:.0f}s", flush=True)
    return rows


def main():
    all_rows = []
    order = ["i21k_t","i21k_s","i21k_b","i21k_l","dinov1_b","clip_b","clip_l","siglip_b",
             "dinov2_s","dinov2_b","dinov2_l","dinov2_g"]
    for m in order:
        rows = evaluate(m)
        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(OUT / "sweep_curvature_nc.csv", index=False)
    print(f"\nDone: {OUT}/sweep_curvature_nc.csv  ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
