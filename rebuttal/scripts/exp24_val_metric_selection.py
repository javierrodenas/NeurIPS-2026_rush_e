#!/usr/bin/env python3
"""
ICLR EXP 24 — metric selection on a VALIDATION split (W5 of the v2 review).

Per cell (10 models x 6 datasets), 1000 FS episodes (paper protocol, seed 42):
per-episode accuracy under R / COS / H. Episodes 0-499 = validation,
500-999 = test. Policies compared on TEST only:
  always-R, always-COS, val-picked (best metric on validation, per cell),
  test-oracle (what a max-over-metrics plot implicitly does; for reference),
  rule (the paper's two-step: flat dataset -> R; else SSL/supervised -> COS,
        contrastive -> H).
Merit: mean test advantage over R; agreement of rule vs val-picked.

Output: exp24_val_metric_selection.csv + log.
"""
import os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd, torch
from pathlib import Path

ROOT = Path("/home/javi/Platonic"); CACHE = ROOT/"results/practical_tasks_cache"
OUT = ROOT/"rebuttal/results"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ATANH = 2*np.arctanh(0.70710678)
MODELS = ["i21k_t","i21k_s","i21k_b","i21k_l","dinov2_s","dinov2_b","dinov2_l",
          "dinov2_g","clip_b","clip_l"]
DATASETS = ["imagenet","cifar100","cifar10","dtd","fashionmnist","mnist"]
FLAT = {"fashionmnist","mnist"}
def fam(m): return "ssl" if m.startswith("dinov") else "sup" if m.startswith("i21k") else "con"

def s4(X, mu, scale):
    Xs = (X-mu)*scale
    n = Xs.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    return torch.tanh(n/2)/n*Xs

def poinc(X, Y):
    x2=(X*X).sum(-1,keepdim=True); y2=(Y*Y).sum(-1,keepdim=True).T
    num=(x2-2*X@Y.T+y2).clamp(min=0); den=((1-x2)*(1-y2.T).T).clamp(min=1e-9)
    return torch.acosh(1+2*num/den)

def episode_accs(Xtr, ytr, mu, scale, n_ep=1000, seed=42):
    rng = np.random.RandomState(seed)
    classes = np.unique(ytr)
    by = {c: np.where(ytr==c)[0] for c in classes}
    A = torch.tensor(Xtr, device=DEV)
    An = A/A.norm(dim=1, keepdim=True).clamp(min=1e-9)
    Ap = s4(A, mu, scale)
    accs = np.zeros((n_ep, 3), dtype=np.float32)
    for e in range(n_ep):
        cls = rng.choice(classes, 5, replace=False)
        sup_idx, qry_idx, qlab = [], [], []
        for ci, c in enumerate(cls):
            pick = rng.choice(by[c], 20, replace=False)
            sup_idx.append(pick[:5]); qry_idx.append(pick[5:]); qlab += [ci]*15
        sup_idx = np.concatenate(sup_idx); qry_idx = np.concatenate(qry_idx)
        qlab = np.array(qlab)
        for k, (Xa, dist) in enumerate([(A, "euc"), (An, "euc"), (Ap, "poi")]):
            S = Xa[sup_idx].view(5, 5, -1).mean(1)
            if k == 2:
                S = s4(S, torch.zeros_like(S[0]), 1.0)
                Dq = poinc(Xa[qry_idx], S)
            else:
                Dq = torch.cdist(Xa[qry_idx], S)
            accs[e, k] = (Dq.argmin(1).cpu().numpy() == qlab).mean()
    return accs  # columns: R, COS, H

def main():
    csv_path = OUT/"exp24_val_metric_selection.csv"
    rows = pd.read_csv(csv_path).to_dict("records") if csv_path.exists() else []
    done = {(r["model"], r["dataset"]) for r in rows}
    for ds in DATASETS:
        for m in MODELS:
            if (m, ds) in done: continue
            t0 = time.time()
            d = np.load(CACHE/f"{m}_{ds}_train.npz")
            X = d["features"].astype(np.float32); y = d["labels"].astype(np.int64)
            Xt = torch.tensor(X, device=DEV)
            mu = Xt.mean(0)
            p95 = torch.quantile((Xt-mu).norm(dim=1), 0.95).item()
            scale = ATANH/max(p95, 1e-7)
            accs = episode_accs(X, y, mu, scale)
            val, tst = accs[:500], accs[500:]
            names = ["R", "COS", "H"]
            val_pick = int(val.mean(0).argmax())
            test_oracle = int(tst.mean(0).argmax())
            rule_pick = 0 if ds in FLAT else (1 if fam(m) in ("ssl","sup") else 2)
            r = dict(model=m, dataset=ds,
                     test_R=tst[:,0].mean(), test_COS=tst[:,1].mean(), test_H=tst[:,2].mean(),
                     val_pick=names[val_pick], rule_pick=names[rule_pick],
                     oracle_pick=names[test_oracle],
                     adv_val=(tst[:,val_pick]-tst[:,0]).mean()*100,
                     adv_rule=(tst[:,rule_pick]-tst[:,0]).mean()*100,
                     adv_cos=(tst[:,1]-tst[:,0]).mean()*100,
                     adv_oracle=(tst[:,test_oracle]-tst[:,0]).mean()*100,
                     time_s=time.time()-t0)
            rows.append(r)
            print(f"{m:9s} {ds:12s} val={r['val_pick']:3s} rule={r['rule_pick']:3s} | "
                  f"adv val {r['adv_val']:+.2f} rule {r['adv_rule']:+.2f} cos {r['adv_cos']:+.2f} "
                  f"oracle {r['adv_oracle']:+.2f} ({r['time_s']:.0f}s)")
            pd.DataFrame(rows).to_csv(csv_path, index=False)
    df = pd.DataFrame(rows)
    print("\n== MEANS (60 cells) ==")
    for c in ["adv_val","adv_rule","adv_cos","adv_oracle"]:
        print(f"  {c:10s} {df[c].mean():+.3f}pp")
    print("rule == val-pick:", (df.val_pick == df.rule_pick).sum(), "/", len(df))
    print("Done")

if __name__ == "__main__":
    main()
