#!/usr/bin/env python3
"""
REBUTTAL EXP 9 — independent re-verification of the "cosine > hyperbolic for
DINOv2" finding, fresh code path (pure numpy/torch, written from scratch).

Checks:
  V1  Feature sanity: are cached features pre-normalized? (norm stats)
  V2  NC on dinov2_l/dinov2_g x cifar100/dtd/imagenet with:
        R    Euclidean on raw
        H    paper pipeline (s4 project, re-exp-mapped prototype, Poincare)
        H2   variant prototype: phi(Euclidean mean of RAW train features)
        COS  cosine on raw
      Cross-checked against exp2 and the paper's Table 1 numbers.
  V3  FS 5w5s on cifar100/dtd for dinov2_g: seeds 123 and 7, 2000 episodes
      each, R/H/COS with paired CI.
  V4  NC on the FULL ImageNet train (1.23M, independent npz) for dinov2_g:
      R/H/COS.

Output: rebuttal/results/exp9_dino_verify.log (stdout only)
"""
import sys, time
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import torch
from pathlib import Path

CACHE = Path("/home/javi/Platonic/results/practical_tasks_cache")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ATANH_T = float(np.arctanh(0.70710678))

def load(model, ds, split):
    d = np.load(CACHE / f"{model}_{ds}_{split}.npz")
    return d["features"].astype(np.float32), d["labels"].astype(np.int64)

def project(X, mu, scale):
    """Fresh implementation of the paper's exp-map projection."""
    Z = (X - mu) * scale
    n = Z.norm(dim=-1, keepdim=True).clamp_min(1e-7)
    Y = torch.tanh(n / 2) * Z / n
    return Y * ((1 - 1e-3) / Y.norm(dim=-1, keepdim=True).clamp_min(1e-7)).clamp(max=1.0)

def pdist_poincare(A, B):
    a2 = (A * A).sum(-1, keepdim=True)
    b2 = (B * B).sum(-1, keepdim=True).T
    sq = (a2 + b2 - 2 * A @ B.T).clamp_min(0)
    x = 1 + 2 * sq / ((1 - a2) * (1 - b2)).clamp_min(1e-7)
    return torch.acosh(x.clamp_min(1 + 1e-7))

def nc_all(model, ds):
    Xtr, ytr = load(model, ds, "train"); Xte, yte = load(model, ds, "test")
    K = int(max(ytr.max(), yte.max())) + 1
    tr = torch.tensor(Xtr, device=DEV); te = torch.tensor(Xte, device=DEV)
    yt = torch.tensor(ytr, device=DEV)
    P_raw = torch.stack([tr[yt == c].mean(0) for c in range(K)])
    # R
    acc = {}
    pred = torch.cdist(te, P_raw).argmin(1).cpu().numpy()
    acc["R"] = (pred == yte).mean()
    # H (paper pipeline)
    mu = tr.mean(0)
    p95 = torch.quantile((tr - mu).norm(dim=1), 0.95)
    sc = 2 * ATANH_T / p95.clamp_min(1e-7)
    trp, tep = project(tr, mu, sc), project(te, mu, sc)
    Ph_raw = torch.stack([trp[yt == c].mean(0) for c in range(K)])
    Ph = project(Ph_raw, torch.zeros_like(mu), 1.0)
    pred = pdist_poincare(tep, Ph).argmin(1).cpu().numpy()
    acc["H"] = (pred == yte).mean()
    # H2 (alternative prototype: project the raw-space class mean)
    Ph2 = project(P_raw, mu, sc)
    pred = pdist_poincare(tep, Ph2).argmin(1).cpu().numpy()
    acc["H2"] = (pred == yte).mean()
    # COS
    tn = te / te.norm(dim=1, keepdim=True).clamp_min(1e-7)
    pn = P_raw / P_raw.norm(dim=1, keepdim=True).clamp_min(1e-7)
    pred = (tn @ pn.T).argmax(1).cpu().numpy()
    acc["COS"] = (pred == yte).mean()
    return acc, float(np.linalg.norm(Xtr, axis=1).std() / np.linalg.norm(Xtr, axis=1).mean())

def fs(model, ds, seed, n_ep=2000):
    Xtr, ytr = load(model, ds, "train"); Xte, yte = load(model, ds, "test")
    X = np.vstack([Xtr, Xte]); y = np.hstack([ytr, yte])
    Xg = torch.tensor(X, device=DEV)
    mu = Xg.mean(0)
    p95 = torch.quantile((Xg - mu).norm(dim=1), 0.95)
    sc = 2 * ATANH_T / p95.clamp_min(1e-7)
    Xp = project(Xg, mu, sc)
    Xn = Xg / Xg.norm(dim=1, keepdim=True).clamp_min(1e-7)
    idx = {c: np.where(y == c)[0] for c in np.unique(y)}
    rng = np.random.RandomState(seed)
    accs = {m: [] for m in ["R", "H", "COS"]}
    cls = np.array(sorted(idx))
    for _ in range(n_ep):
        sel = rng.choice(cls, 5, replace=False)
        sup, qry, ql = [], [], []
        for li, c in enumerate(sel):
            p = rng.choice(idx[c], 20, replace=False)
            sup += p[:5].tolist(); qry += p[5:].tolist(); ql += [li] * 15
        sup, qry, ql = np.array(sup), np.array(qry), np.array(ql)
        # R
        P = Xg[sup].view(5, 5, -1).mean(1)
        accs["R"].append((torch.cdist(Xg[qry], P).argmin(1).cpu().numpy() == ql).mean())
        # H
        Ph = project(Xp[sup].view(5, 5, -1).mean(1), torch.zeros_like(mu), 1.0)
        accs["H"].append((pdist_poincare(Xp[qry], Ph).argmin(1).cpu().numpy() == ql).mean())
        # COS
        Pn = Xn[sup].view(5, 5, -1).mean(1)
        Pn = Pn / Pn.norm(dim=1, keepdim=True).clamp_min(1e-7)
        accs["COS"].append(((Xn[qry] @ Pn.T).argmax(1).cpu().numpy() == ql).mean())
    out = {m: float(np.mean(v)) for m, v in accs.items()}
    d = np.array(accs["COS"]) - np.array(accs["H"])
    out["COS_H_diff"] = float(d.mean()); out["COS_H_ci"] = float(1.96 * d.std() / np.sqrt(n_ep))
    return out

print("V1/V2 — NC (independent reimplementation), norm-variation coefficient:")
for m in ["dinov2_l", "dinov2_g"]:
    for ds in ["cifar100", "dtd", "imagenet"]:
        a, cv = nc_all(m, ds)
        print(f"  {m}/{ds:9s} R={a['R']:.4f} H={a['H']:.4f} H2={a['H2']:.4f} "
              f"COS={a['COS']:.4f}  (feat-norm CV={cv:.3f})", flush=True)

print("V3 — FS dinov2_g, 2000 episodes, two fresh seeds:")
for ds in ["cifar100", "dtd"]:
    for seed in [123, 7]:
        r = fs("dinov2_g", ds, seed)
        print(f"  {ds:9s} seed={seed:3d} R={r['R']:.4f} H={r['H']:.4f} COS={r['COS']:.4f} "
              f"COS-H={r['COS_H_diff']*100:+.2f}±{r['COS_H_ci']*100:.2f}pp", flush=True)

print("V4 — NC on FULL ImageNet train (independent npz), dinov2_g:")
d = np.load(CACHE / "dinov2_g_imagenet_fulltrain.npz")
Xtr, ytr = d["features"].astype(np.float32), d["labels"].astype(np.int64)
Xte, yte = load("dinov2_g", "imagenet", "test")
tr = torch.tensor(Xtr, device=DEV); te = torch.tensor(Xte, device=DEV)
yt = torch.tensor(ytr, device=DEV)
P = torch.stack([tr[yt == c].mean(0) for c in range(1000)])
accR = (torch.cdist(te, P).argmin(1).cpu().numpy() == yte).mean()
mu = tr.mean(0); p95 = torch.quantile((tr - mu).norm(dim=1), 0.95)
sc = 2 * ATANH_T / p95.clamp_min(1e-7)
trp, tep = project(tr, mu, sc), project(te, mu, sc)
Ph = project(torch.stack([trp[yt == c].mean(0) for c in range(1000)]),
             torch.zeros_like(mu), 1.0)
accH = (pdist_poincare(tep, Ph).argmin(1).cpu().numpy() == yte).mean()
tn = te / te.norm(dim=1, keepdim=True); pn = P / P.norm(dim=1, keepdim=True)
accC = ((tn @ pn.T).argmax(1).cpu().numpy() == yte).mean()
print(f"  fulltrain NC: R={accR:.4f} H={accH:.4f} COS={accC:.4f}", flush=True)
print("Done")
