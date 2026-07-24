#!/usr/bin/env python3
"""
REBUTTAL EXP 2b — does the hyperbolic metric stack ON TOP of L2
normalization?  (follow-up to exp2's finding that plain cosine is a strong
baseline on SSL backbones.)

All metrics operate on L2-NORMALIZED features (norm-noise removed):

  COS   cosine distance on normalized features (reference)
  HN    s4 pipeline applied to normalized features (center, p95-scale,
        exp-map) + Poincare distance
  RTN   same transform as HN, Euclidean distance (control)
  RN    Euclidean distance on normalized features (chord; equivalent
        ranking to COS for NC by monotonicity, kept as sanity check)

Tasks NC + FS (1000 episodes), 10-model panel x 6 datasets.
Per-episode paired CI for HN - COS.

Output: rebuttal/results/exp2b_normalized_stack.csv
"""
import os, sys, time
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import pandas as pd
import torch
from pathlib import Path

ROOT = Path("/home/javi/Platonic")
CACHE = ROOT / "results/practical_tasks_cache"
OUT = ROOT / "rebuttal/results"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = ["i21k_t","i21k_s","i21k_b","i21k_l",
          "dinov2_s","dinov2_b","dinov2_l","dinov2_g",
          "clip_b","clip_l"]
PARADIGMS = {"i21k_t":"Supervised","i21k_s":"Supervised","i21k_b":"Supervised",
             "i21k_l":"Supervised","dinov2_s":"SSL","dinov2_b":"SSL",
             "dinov2_l":"SSL","dinov2_g":"SSL","clip_b":"Contrastive",
             "clip_l":"Contrastive"}
DATASETS = ["imagenet","cifar100","cifar10","dtd","fashionmnist","mnist"]
TARGET = 0.70710678
METRICS = ["COS","HN","RTN","RN"]


def s4_project_torch(X, mu, scale):
    Xs = (X - mu) * scale
    nrm = Xs.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    Y = torch.tanh(nrm/2) / nrm * Xs
    cur = Y.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    return Y * torch.clamp((1.0-1e-3) / cur, max=1.0)


def poincare_dist(X, Y):
    x_sq = (X*X).sum(-1, keepdim=True)
    y_sq = (Y*Y).sum(-1, keepdim=True).T
    d_sq = (x_sq + y_sq - 2.0 * X @ Y.T).clamp(min=0)
    denom = ((1-x_sq)*(1-y_sq)).clamp(min=1e-7)
    arg = (1 + 2*d_sq/denom).clamp(min=1+1e-7)
    return 2.0 * torch.acosh(arg)


def cos_dist(X, Y):
    Xn = X / X.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    Yn = Y / Y.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    return 1.0 - Xn @ Yn.T


def build_spaces(Xn_tr, Xn_te):
    """All inputs already L2-normalized. Returns metric -> (tr, te, dist)."""
    mu = Xn_tr.mean(0)
    p95 = torch.quantile((Xn_tr - mu).norm(dim=1), 0.95).item()
    scale = 2 * float(np.arctanh(TARGET)) / max(p95, 1e-7)
    Htr = s4_project_torch(Xn_tr, mu, scale)
    Hte = s4_project_torch(Xn_te, mu, scale)
    ed = lambda A, B: torch.cdist(A, B, p=2)
    return {
        "COS": (Xn_tr, Xn_te, cos_dist),
        "HN":  (Htr, Hte, poincare_dist),
        "RTN": (Htr, Hte, ed),
        "RN":  (Xn_tr, Xn_te, ed),
    }


def protos(train_pts, y_tr, n_classes, metric):
    P = torch.stack([train_pts[torch.tensor(y_tr==c)].mean(0) for c in range(n_classes)])
    if metric == "HN":
        P = s4_project_torch(P, torch.zeros(P.shape[-1], device=DEVICE), 1.0)
    return P


def task_NC(Xn_tr, y_tr, Xn_te, y_te):
    n_classes = int(max(y_tr.max(), y_te.max())) + 1
    spaces = build_spaces(Xn_tr, Xn_te)
    out = {}
    for m, (tr, te, dist) in spaces.items():
        P = protos(tr, y_tr, n_classes, m)
        pred = dist(te, P).argmin(1).cpu().numpy()
        out[m] = float((pred == y_te).mean())
    return out


def task_FS(Xn, y, n_episodes=1000, n_way=5, k_shot=5, q_query=15, seed=42):
    classes = np.unique(y)
    if len(classes) < n_way: return None
    idx_per_c = {c: np.where(y == c)[0] for c in classes}
    rng = np.random.RandomState(seed)
    spaces = build_spaces(Xn, Xn)
    accs = {m: [] for m in METRICS}
    for ep in range(n_episodes):
        sel = rng.choice(classes, n_way, replace=False)
        sup_idx, qry_idx, qry_lab = [], [], []
        for li, c in enumerate(sel):
            pick = rng.choice(idx_per_c[c], k_shot+q_query, replace=False)
            sup_idx.extend(pick[:k_shot]); qry_idx.extend(pick[k_shot:])
            qry_lab.extend([li]*q_query)
        sup_idx = np.array(sup_idx); qry_idx = np.array(qry_idx)
        qry_lab = np.array(qry_lab)
        for m, (tr_pts, _, dist) in spaces.items():
            sup = tr_pts[sup_idx]; qry = tr_pts[qry_idx]
            P = sup.view(n_way, k_shot, -1).mean(1)
            if m == "HN":
                P = s4_project_torch(P, torch.zeros(P.shape[-1], device=DEVICE), 1.0)
            pred = dist(qry, P).argmin(1).cpu().numpy()
            accs[m].append((pred == qry_lab).mean())
    res = {m: float(np.mean(v)) for m, v in accs.items()}
    diff = np.array(accs["HN"]) - np.array(accs["COS"])
    res["HN_COS_diff"] = float(diff.mean())
    res["HN_COS_ci95"] = float(1.96 * diff.std() / np.sqrt(n_episodes))
    return res


def main():
    rows = []
    for model in MODELS:
        for ds in DATASETS:
            t0 = time.time()
            try:
                tr = np.load(CACHE / f"{model}_{ds}_train.npz")
                te = np.load(CACHE / f"{model}_{ds}_test.npz")
            except FileNotFoundError:
                print(f"SKIP {model}/{ds}", flush=True); continue
            X_tr = tr["features"].astype(np.float32); y_tr = tr["labels"].astype(np.int64)
            X_te = te["features"].astype(np.float32); y_te = te["labels"].astype(np.int64)
            Xn_tr = torch.tensor(X_tr, device=DEVICE)
            Xn_tr = Xn_tr / Xn_tr.norm(dim=1, keepdim=True).clamp(min=1e-7)
            Xn_te = torch.tensor(X_te, device=DEVICE)
            Xn_te = Xn_te / Xn_te.norm(dim=1, keepdim=True).clamp(min=1e-7)
            nc = task_NC(Xn_tr, y_tr, Xn_te, y_te)
            Xall = torch.cat([Xn_tr, Xn_te], 0)
            yall = np.concatenate([y_tr, y_te], 0)
            fs = task_FS(Xall, yall)
            row = dict(model=model, dataset=ds, paradigm=PARADIGMS[model])
            for m in METRICS: row[f"NC_{m}"] = nc[m]
            if fs:
                for m in METRICS: row[f"FS_{m}"] = fs[m]
                row["FS_HN_COS_diff"] = fs["HN_COS_diff"]
                row["FS_HN_COS_ci95"] = fs["HN_COS_ci95"]
            row["time_s"] = time.time() - t0
            rows.append(row)
            pd.DataFrame(rows).to_csv(OUT / "exp2b_normalized_stack.csv", index=False)
            print(f"{model:10s}/{ds:13s} "
                  f"NC COS={nc['COS']:.4f} HN={nc['HN']:.4f} RTN={nc['RTN']:.4f} | "
                  f"FS COS={fs['COS']:.4f} HN={fs['HN']:.4f} RTN={fs['RTN']:.4f} "
                  f"(HN-COS {fs['HN_COS_diff']*100:+.2f}±{fs['HN_COS_ci95']*100:.2f}pp) "
                  f"({time.time()-t0:.0f}s)", flush=True)
    print("Done ->", OUT / "exp2b_normalized_stack.csv")


if __name__ == "__main__":
    main()
