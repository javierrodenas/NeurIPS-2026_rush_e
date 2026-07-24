#!/usr/bin/env python3
"""REBUTTAL EXP 13 — McNemar tests for the NC cells (promised to Xbn5).
For each model x dataset: paired NC predictions under R, H, COS; two-sided
exact binomial McNemar p for H vs R and H vs COS.
Output: rebuttal/results/exp13_mcnemar.csv"""
import os, sys, time
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd, torch
from pathlib import Path
from scipy.stats import binomtest

ROOT = Path("/home/javi/Platonic"); CACHE = ROOT/"results/practical_tasks_cache"
OUT = ROOT/"rebuttal/results"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = ["i21k_t","i21k_s","i21k_b","i21k_l","dinov2_s","dinov2_b","dinov2_l","dinov2_g","clip_b","clip_l"]
DATASETS = ["imagenet","cifar100","cifar10","dtd","fashionmnist","mnist"]
ATANH_T = float(np.arctanh(0.70710678))

def project(X, mu, scale):
    Z = (X - mu) * scale
    n = Z.norm(dim=-1, keepdim=True).clamp_min(1e-7)
    Y = torch.tanh(n/2) * Z / n
    return Y * ((1-1e-3)/Y.norm(dim=-1, keepdim=True).clamp_min(1e-7)).clamp(max=1.0)

def poinc(A, B):
    a2=(A*A).sum(-1,keepdim=True); b2=(B*B).sum(-1,keepdim=True).T
    sq=(a2+b2-2*A@B.T).clamp_min(0)
    return torch.acosh((1+2*sq/((1-a2)*(1-b2)).clamp_min(1e-7)).clamp_min(1+1e-7))

def mcnemar(right1, right2):
    b = int((~right1 & right2).sum()); c = int((right1 & ~right2).sum())
    if b + c == 0: return 1.0, b, c
    return float(binomtest(min(b, c), b+c, 0.5).pvalue), b, c

rows=[]
for m in MODELS:
    for ds in DATASETS:
        try:
            tr = np.load(CACHE/f"{m}_{ds}_train.npz"); te = np.load(CACHE/f"{m}_{ds}_test.npz")
        except FileNotFoundError: continue
        Xtr, ytr = tr["features"].astype(np.float32), tr["labels"].astype(np.int64)
        Xte, yte = te["features"].astype(np.float32), te["labels"].astype(np.int64)
        K = int(max(ytr.max(), yte.max()))+1
        a = torch.tensor(Xtr, device=DEV); b_ = torch.tensor(Xte, device=DEV)
        yt = torch.tensor(ytr, device=DEV)
        P = torch.stack([a[yt==c].mean(0) for c in range(K)])
        pr_R = torch.cdist(b_, P).argmin(1).cpu().numpy()
        mu = a.mean(0); p95 = torch.quantile((a-mu).norm(dim=1), 0.95)
        sc = 2*ATANH_T/p95.clamp_min(1e-7)
        ap, bp = project(a, mu, sc), project(b_, mu, sc)
        Ph = project(torch.stack([ap[yt==c].mean(0) for c in range(K)]),
                     torch.zeros_like(mu), 1.0)
        pr_H = poinc(bp, Ph).argmin(1).cpu().numpy()
        bn = b_/b_.norm(dim=1,keepdim=True).clamp_min(1e-7)
        Pn = P/P.norm(dim=1,keepdim=True).clamp_min(1e-7)
        pr_C = (bn@Pn.T).argmax(1).cpu().numpy()
        rR, rH, rC = pr_R==yte, pr_H==yte, pr_C==yte
        pHR, b1, c1 = mcnemar(rR, rH)
        pHC, b2, c2 = mcnemar(rC, rH)
        rows.append(dict(model=m, dataset=ds, n_test=len(yte),
            acc_R=rR.mean(), acc_H=rH.mean(), acc_COS=rC.mean(),
            p_H_vs_R=pHR, flips_HR=f"{b1}/{c1}", p_H_vs_COS=pHC, flips_HC=f"{b2}/{c2}"))
        pd.DataFrame(rows).to_csv(OUT/"exp13_mcnemar.csv", index=False)
        print(f"{m:10s}/{ds:13s} H-R p={pHR:.2e} ({b1}/{c1})  H-COS p={pHC:.2e} ({b2}/{c2})", flush=True)
print("Done")
