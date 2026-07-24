#!/usr/bin/env python3
"""
REBUTTAL EXP 2 — metric controls on the downstream tasks (answers RJje Q3/Q4
and Xbn5 Q3).

Same features, same prototypes-by-argmin protocol as the paper, five metrics:

  R        raw Euclidean (paper baseline)
  H        post-hoc Poincare projection + Poincare distance (paper method)
  RT       SAME radial tanh transform phi(x) as H, but EUCLIDEAN distances
           on the transformed points  -> isolates "radial rescaling" from
           "hyperbolic metric"
  COS      cosine distance on raw features (equivalent ranking to spherical
           geodesic arccos, since arccos is monotone)
  COSC     cosine distance on centered features (same centering as H)

Tasks: NC (nearest-centroid) and FS (5-way 5-shot, 1000 episodes) on the
10-model paper panel x 6 datasets (ImageNet at 100/class train, 50/class test).
FS reports per-episode paired std of (H - R) -> 95% CI of the advantage.

Output: rebuttal/results/exp2_metric_controls.csv
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
METRICS = ["R","H","RT","COS","COSC"]


def s4_project_torch(X, mu, scale):
    Xs = (X - mu) * scale
    nrm = Xs.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    Y = torch.tanh(nrm/2) / nrm * Xs
    cur = Y.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    return Y * torch.clamp((1.0-1e-3) / cur, max=1.0)


def s4_params(X_train, target_p95=TARGET):
    mu = X_train.mean(0)
    p95 = np.percentile(np.linalg.norm(X_train - mu, axis=1), 95)
    scale = 2 * np.arctanh(target_p95) / max(p95, 1e-7)
    return mu, scale


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


def build_spaces(Xtr, Xte, mu_t, scale):
    """Returns dict metric -> (train_pts, test_pts, distfn)."""
    zeros = torch.zeros_like(mu_t)
    Xtr_p = s4_project_torch(Xtr, mu_t, scale)
    Xte_p = s4_project_torch(Xte, mu_t, scale)
    ed = lambda A, B: torch.cdist(A, B, p=2)
    return {
        "R":    (Xtr, Xte, ed),
        "H":    (Xtr_p, Xte_p, poincare_dist),
        "RT":   (Xtr_p, Xte_p, ed),
        "COS":  (Xtr, Xte, cos_dist),
        "COSC": (Xtr - mu_t, Xte - mu_t, cos_dist),
    }, zeros


def protos(train_pts, y_tr, n_classes, metric, zeros):
    P = torch.stack([train_pts[torch.tensor(y_tr==c)].mean(0) for c in range(n_classes)])
    if metric == "H":
        P = s4_project_torch(P, zeros, 1.0)   # re-exp-map, as in the paper
    return P


def task_NC(X_tr, y_tr, X_te, y_te):
    n_classes = int(max(y_tr.max(), y_te.max())) + 1
    Xtr = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    Xte = torch.tensor(X_te, dtype=torch.float32, device=DEVICE)
    mu, scale = s4_params(X_tr)
    mu_t = torch.tensor(mu, dtype=torch.float32, device=DEVICE)
    spaces, zeros = build_spaces(Xtr, Xte, mu_t, scale)
    out = {}
    for m, (tr, te, dist) in spaces.items():
        P = protos(tr, y_tr, n_classes, m, zeros)
        pred = dist(te, P).argmin(1).cpu().numpy()
        out[m] = float((pred == y_te).mean())
    return out


def task_FS(X_tr, y_tr, X_te, y_te, n_episodes=1000, n_way=5, k_shot=5,
            q_query=15, seed=42):
    X = np.concatenate([X_tr, X_te], 0); y = np.concatenate([y_tr, y_te], 0)
    classes = np.unique(y)
    if len(classes) < n_way: return None
    idx_per_c = {c: np.where(y == c)[0] for c in classes}
    rng = np.random.RandomState(seed)
    mu, scale = s4_params(X)
    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    mu_t = torch.tensor(mu, dtype=torch.float32, device=DEVICE)
    spaces, zeros = build_spaces(Xt, Xt, mu_t, scale)  # same pts both sides
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
            if m == "H":
                P = s4_project_torch(P, torch.zeros(P.shape[-1], device=DEVICE), 1.0)
            pred = dist(qry, P).argmin(1).cpu().numpy()
            accs[m].append((pred == qry_lab).mean())
    res = {m: float(np.mean(v)) for m, v in accs.items()}
    diff_hr = np.array(accs["H"]) - np.array(accs["R"])
    res["HR_diff_mean"] = float(diff_hr.mean())
    res["HR_diff_ci95"] = float(1.96 * diff_hr.std() / np.sqrt(n_episodes))
    diff_hrt = np.array(accs["H"]) - np.array(accs["RT"])
    res["HRT_diff_mean"] = float(diff_hrt.mean())
    res["HRT_diff_ci95"] = float(1.96 * diff_hrt.std() / np.sqrt(n_episodes))
    return res


def load(model, ds):
    tr = np.load(CACHE / f"{model}_{ds}_train.npz")
    te = np.load(CACHE / f"{model}_{ds}_test.npz")
    return (tr["features"].astype(np.float32), tr["labels"].astype(np.int64),
            te["features"].astype(np.float32), te["labels"].astype(np.int64))


def main():
    rows = []
    for model in MODELS:
        for ds in DATASETS:
            t0 = time.time()
            try:
                X_tr, y_tr, X_te, y_te = load(model, ds)
            except FileNotFoundError:
                print(f"SKIP {model}/{ds}", flush=True); continue
            nc = task_NC(X_tr, y_tr, X_te, y_te)
            fs = task_FS(X_tr, y_tr, X_te, y_te)
            row = dict(model=model, dataset=ds, paradigm=PARADIGMS[model])
            for m in METRICS: row[f"NC_{m}"] = nc[m]
            if fs:
                for m in METRICS: row[f"FS_{m}"] = fs[m]
                for k in ["HR_diff_mean","HR_diff_ci95","HRT_diff_mean","HRT_diff_ci95"]:
                    row[f"FS_{k}"] = fs[k]
            row["time_s"] = time.time() - t0
            rows.append(row)
            pd.DataFrame(rows).to_csv(OUT / "exp2_metric_controls.csv", index=False)
            print(f"{model:10s}/{ds:13s} "
                  f"NC R={nc['R']:.4f} H={nc['H']:.4f} RT={nc['RT']:.4f} "
                  f"COS={nc['COS']:.4f} COSC={nc['COSC']:.4f} | "
                  f"FS R={fs['R']:.4f} H={fs['H']:.4f} RT={fs['RT']:.4f} "
                  f"COS={fs['COS']:.4f}  (H-R CI95 ±{fs['HR_diff_ci95']*100:.2f}pp) "
                  f"({time.time()-t0:.0f}s)", flush=True)
    print("Done ->", OUT / "exp2_metric_controls.csv")


if __name__ == "__main__":
    main()
