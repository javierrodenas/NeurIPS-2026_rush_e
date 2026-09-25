#!/usr/bin/env python3
"""
Practitioner tasks C, D, E: H vs R across multiple downstream tasks.

  C. Nearest-Centroid Classification (NC)         — 12 models x 5 datasets
  D. kNN classification (k=5)                     — 12 models x 5 datasets
  E. Few-shot 5-way 5-shot (1000 episodes/combo)  — 12 models x 2 datasets

Each task is run with R (Euclidean) and H (S4 Poincare projection).
Outputs per-task CSVs and a combined paradigm x dataset summary.
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import time, warnings, sys
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

ROOT = Path("/home/javi/Platonic")
CACHE = ROOT / "results/practical_tasks_cache"
OUT = ROOT / "results/tasks_CDE"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}", flush=True)

PARADIGMS = {
    "i21k_t":"Supervised","i21k_s":"Supervised","i21k_b":"Supervised","i21k_l":"Supervised",
    "dinov1_b":"SSL","dinov2_s":"SSL","dinov2_b":"SSL","dinov2_l":"SSL","dinov2_g":"SSL",
    "clip_b":"Contrastive","clip_l":"Contrastive","siglip_b":"Contrastive",
}
import os
_subset = os.environ.get("PRACT_MODELS", "").strip()
if _subset:
    ALL_MODELS = [m.strip() for m in _subset.split(",") if m.strip()]
else:
    ALL_MODELS = list(PARADIGMS.keys())
ALL_DATASETS = ["cifar100","cifar10","dtd","fashionmnist","mnist"]
FEWSHOT_DATASETS = ALL_DATASETS  # uniform across all tasks

COARSE = {0:[4,30,55,72,95],1:[1,32,67,73,91],2:[54,62,70,82,92],3:[9,10,16,28,61],4:[0,51,53,57,83],
 5:[22,39,40,86,87],6:[5,20,25,84,94],7:[6,7,14,18,24],8:[3,42,43,88,97],9:[12,17,37,68,76],
 10:[23,33,49,60,71],11:[15,19,21,31,38],12:[34,63,64,66,75],13:[26,45,77,79,99],
 14:[2,11,35,46,98],15:[27,29,44,78,93],16:[36,50,65,74,80],17:[47,52,56,59,96],
 18:[8,13,48,58,90],19:[41,69,81,85,89]}
FINE_TO_COARSE_C100 = np.array([0]*100)
for c, fs in COARSE.items():
    for f in fs: FINE_TO_COARSE_C100[f] = c
CIFAR10_SUPER = np.array([1,1,0,0,0,0,0,0,1,1])
FMNIST_SUPER  = np.array([0,1,0,3,0,2,0,2,3,2])
MNIST_SUPER   = np.array([0,1,2,2,1,2,0,1,0,0])

def super_map(ds):
    return {"cifar100":FINE_TO_COARSE_C100,"cifar10":CIFAR10_SUPER,
            "fashionmnist":FMNIST_SUPER,"mnist":MNIST_SUPER}.get(ds)


def super_err(preds, labels, ds):
    sm = super_map(ds)
    if sm is None: return float("nan")
    s = sm[labels]; sp = sm[preds]
    wrong = preds != labels
    if wrong.sum()==0: return float("nan")
    return float((sp[wrong] == s[wrong]).mean())


# ─── S4 projection (Poincare ball, c=1) ───────────────────────────────────
def s4_project_torch(X, mu, scale):
    """X torch tensor, returns Poincare-projected tensor."""
    Xs = (X - mu) * scale
    nrm = Xs.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    Y = torch.tanh(nrm/2) / nrm * Xs
    cur = Y.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    return Y * torch.clamp((1.0-1e-3) / cur, max=1.0)


def s4_params(X_train, target_p95=0.5):
    mu = X_train.mean(0)
    p95 = np.percentile(np.linalg.norm(X_train - mu, axis=1), 95)
    scale = 2 * np.arctanh(target_p95) / max(p95, 1e-7)
    return mu, scale


def poincare_dist(X, Y):
    """X (n,d), Y (m,d) on Poincare ball -> (n,m). Torch. Memory-efficient: avoids (n,m,d) tensor."""
    x_sq = (X*X).sum(-1, keepdim=True)               # (n,1)
    y_sq = (Y*Y).sum(-1, keepdim=True).T             # (1,m)
    d_sq = (x_sq + y_sq - 2.0 * X @ Y.T).clamp(min=0)  # (n,m)
    denom = ((1-x_sq)*(1-y_sq)).clamp(min=1e-7)
    arg = (1 + 2*d_sq/denom).clamp(min=1+1e-7)
    return 2.0 * torch.acosh(arg)


def euclid_dist(X, Y):
    """Pairwise L2 distance (n,m). Torch."""
    return torch.cdist(X, Y, p=2)


# ─── Task C: Nearest-Centroid ─────────────────────────────────────────────
def task_C(X_tr, y_tr, X_te, y_te, ds):
    n_classes = int(max(y_tr.max(), y_te.max())) + 1
    Xtr = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    Xte = torch.tensor(X_te, dtype=torch.float32, device=DEVICE)
    # Centroids in R
    centroids_r = torch.stack([Xtr[torch.tensor(y_tr==c)].mean(0) for c in range(n_classes)])
    D_r = euclid_dist(Xte, centroids_r)
    pred_r = D_r.argmin(1).cpu().numpy()
    # H side: project with S4
    mu, scale = s4_params(X_tr)
    mu_t = torch.tensor(mu, dtype=torch.float32, device=DEVICE)
    zeros_t = torch.zeros_like(mu_t)
    Xtr_p = s4_project_torch(Xtr, mu_t, scale)
    Xte_p = s4_project_torch(Xte, mu_t, scale)
    centroids_h_raw = torch.stack([Xtr_p[torch.tensor(y_tr==c)].mean(0) for c in range(n_classes)])
    # Re-apply exp_map at origin to centroids (matches shell_collapse_diagnose pipeline)
    centroids_h = s4_project_torch(centroids_h_raw, zeros_t, 1.0)
    D_h = poincare_dist(Xte_p, centroids_h)
    pred_h = D_h.argmin(1).cpu().numpy()
    return dict(
        r_acc=float((pred_r==y_te).mean()), h_acc=float((pred_h==y_te).mean()),
        r_super=super_err(pred_r,y_te,ds), h_super=super_err(pred_h,y_te,ds),
    )


# ─── Task D: kNN (k=5) ────────────────────────────────────────────────────
def task_D(X_tr, y_tr, X_te, y_te, ds, k=5, train_per_class=50):
    """kNN with subsampled train (for speed). Returns R-kNN and H-kNN accuracy."""
    n_classes = int(max(y_tr.max(), y_te.max())) + 1
    rng = np.random.RandomState(42)
    sub_idx = []
    for c in range(n_classes):
        idx_c = np.where(y_tr == c)[0]
        if len(idx_c) > train_per_class:
            sub_idx.extend(rng.choice(idx_c, train_per_class, replace=False).tolist())
        else:
            sub_idx.extend(idx_c.tolist())
    sub_idx = np.array(sub_idx)
    X_sub = X_tr[sub_idx]; y_sub = y_tr[sub_idx]

    Xtr = torch.tensor(X_sub, dtype=torch.float32, device=DEVICE)
    Xte = torch.tensor(X_te, dtype=torch.float32, device=DEVICE)
    ysub_t = torch.tensor(y_sub, dtype=torch.long, device=DEVICE)

    # R-kNN
    D_r = euclid_dist(Xte, Xtr)
    nn_idx = D_r.topk(k, largest=False, dim=1).indices
    pred_r = []
    for i in range(len(Xte)):
        neigh_y = ysub_t[nn_idx[i]]
        # majority vote
        vals, counts = torch.unique(neigh_y, return_counts=True)
        pred_r.append(vals[counts.argmax()].item())
    pred_r = np.array(pred_r)

    # H-kNN with S4
    mu, scale = s4_params(X_tr)
    mu_t = torch.tensor(mu, dtype=torch.float32, device=DEVICE)
    Xtr_p = s4_project_torch(Xtr, mu_t, scale)
    Xte_p = s4_project_torch(Xte, mu_t, scale)
    # Chunk over test for memory
    pred_h = np.empty(len(Xte), dtype=np.int64)
    chunk = 1000
    for i in range(0, len(Xte), chunk):
        D_h = poincare_dist(Xte_p[i:i+chunk], Xtr_p)
        nn = D_h.topk(k, largest=False, dim=1).indices
        for j in range(D_h.shape[0]):
            neigh_y = ysub_t[nn[j]]
            vals, counts = torch.unique(neigh_y, return_counts=True)
            pred_h[i+j] = vals[counts.argmax()].item()

    return dict(
        r_acc=float((pred_r==y_te).mean()), h_acc=float((pred_h==y_te).mean()),
        r_super=super_err(pred_r,y_te,ds), h_super=super_err(pred_h,y_te,ds),
        n_train=len(X_sub),
    )


# ─── Task E: Few-shot 5-way 5-shot ────────────────────────────────────────
def task_E(X_tr, y_tr, X_te, y_te, ds, n_episodes=1000, n_way=5, k_shot=5, q_query=15, seed=42):
    """
    Pool train+test together (we have features, labels) and build episodes:
      - Sample n_way classes
      - From each class, sample k_shot support + q_query query (no overlap)
      - Predict each query by nearest support-centroid distance (R or H)
      - Episode accuracy = mean correct across all queries
    Average over n_episodes.
    """
    X = np.concatenate([X_tr, X_te], axis=0)
    y = np.concatenate([y_tr, y_te], axis=0)
    classes = np.unique(y)
    n_classes = len(classes)
    if n_classes < n_way:
        return None
    # Pre-index per class
    idx_per_c = {c: np.where(y == c)[0] for c in classes}
    rng = np.random.RandomState(seed)

    # S4 params on full data
    mu = X.mean(0)
    p95 = np.percentile(np.linalg.norm(X - mu, axis=1), 95)
    scale = 2 * np.arctanh(0.5) / max(p95, 1e-7)

    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    mu_t = torch.tensor(mu, dtype=torch.float32, device=DEVICE)
    X_h = s4_project_torch(Xt, mu_t, scale)

    r_acc_list = []
    h_acc_list = []
    for ep in range(n_episodes):
        sel = rng.choice(classes, n_way, replace=False)
        sup_idx = []
        qry_idx = []
        qry_labels_local = []  # 0..n_way-1
        for li, c in enumerate(sel):
            ic = idx_per_c[c]
            if len(ic) < k_shot + q_query:
                # not enough samples for this class: take what we can
                rng.shuffle(ic)
                k = min(k_shot, max(1, len(ic)-1))
                q = max(1, len(ic) - k)
                pick = ic
            else:
                pick = rng.choice(ic, k_shot+q_query, replace=False)
                k = k_shot; q = q_query
            sup_idx.extend(pick[:k].tolist())
            qry_idx.extend(pick[k:k+q].tolist())
            qry_labels_local.extend([li]*q)
        sup_idx = np.array(sup_idx); qry_idx = np.array(qry_idx)
        qry_labels = np.array(qry_labels_local)

        # R: support means in R
        sup_X = Xt[sup_idx]
        qry_X = Xt[qry_idx]
        prots_r = []
        offset = 0
        for li in range(n_way):
            n = (np.array(qry_labels_local) == li).sum() // q_query if False else None
            # Recompute support count per class:
            pass
        # Easier: rebuild prototypes from shape we know (k_shot per class in order)
        # Support order: [class0 k_shot, class1 k_shot, ...]
        sup_per_class = len(sup_idx) // n_way  # may be <k_shot for tiny classes
        prots_r = sup_X.view(n_way, sup_per_class, -1).mean(1)
        D_r = euclid_dist(qry_X, prots_r)
        pred_r = D_r.argmin(1).cpu().numpy()
        r_acc_list.append((pred_r == qry_labels).mean())

        # H: support means in H ball, then re-apply exp_map at origin
        sup_h = X_h[sup_idx]
        qry_h = X_h[qry_idx]
        prots_h_raw = sup_h.view(n_way, sup_per_class, -1).mean(1)
        zeros_t2 = torch.zeros(prots_h_raw.shape[-1], device=DEVICE)
        prots_h = s4_project_torch(prots_h_raw, zeros_t2, 1.0)
        D_h = poincare_dist(qry_h, prots_h)
        pred_h = D_h.argmin(1).cpu().numpy()
        h_acc_list.append((pred_h == qry_labels).mean())

    return dict(
        r_acc=float(np.mean(r_acc_list)), r_acc_std=float(np.std(r_acc_list)),
        h_acc=float(np.mean(h_acc_list)), h_acc_std=float(np.std(h_acc_list)),
        n_episodes=n_episodes, n_way=n_way, k_shot=k_shot, q_query=q_query,
    )


def gromov_delta(X, n_quads=50000, seed=42):
    from scipy.spatial.distance import pdist, squareform
    rng = np.random.RandomState(seed)
    D = squareform(pdist(X, 'euclidean'))
    diam = D.max()
    n = len(X)
    if n < 4: return float("nan")
    quads = np.array([rng.choice(n, 4, replace=False) for _ in range(n_quads)])
    i,j,k,l = quads.T
    s1=D[i,j]+D[k,l]; s2=D[i,k]+D[j,l]; s3=D[i,l]+D[j,k]
    sums = np.sort(np.stack([s1,s2,s3],1),1)
    return float(((sums[:,2]-sums[:,1])/2).mean()/diam)


def main():
    rows = []
    for model in ALL_MODELS:
        for ds in ALL_DATASETS:
            t0 = time.time()
            try:
                Xtr_d = np.load(CACHE/f"{model}_{ds}_train.npz")
                Xte_d = np.load(CACHE/f"{model}_{ds}_test.npz")
                X_tr = Xtr_d["features"].astype(np.float32); y_tr = Xtr_d["labels"].astype(np.int64)
                X_te = Xte_d["features"].astype(np.float32); y_te = Xte_d["labels"].astype(np.int64)
            except FileNotFoundError:
                print(f"SKIP {model}/{ds}", flush=True); continue
            n_classes = int(max(y_tr.max(), y_te.max())) + 1

            # delta on backbone centroids
            cents_e = np.stack([X_tr[y_tr==c].mean(0) for c in range(n_classes)])
            delta = gromov_delta(cents_e)

            base = dict(model=model, dataset=ds, paradigm=PARADIGMS[model],
                        in_dim=int(X_tr.shape[1]), n_classes=n_classes, delta=delta)

            print(f"{model:10s}/{ds:13s} d={X_tr.shape[1]:>4} k={n_classes:>3}  δ={delta:.4f}", flush=True)

            # Task C
            t1 = time.time()
            c = task_C(X_tr, y_tr, X_te, y_te, ds)
            adv_c = (c["h_acc"] - c["r_acc"])*100
            print(f"  C(NC)   R={c['r_acc']:.4f} H={c['h_acc']:.4f} adv={adv_c:+.2f}pp  ({time.time()-t1:.1f}s)", flush=True)

            # Task D
            t1 = time.time()
            d = task_D(X_tr, y_tr, X_te, y_te, ds)
            adv_d = (d["h_acc"] - d["r_acc"])*100
            print(f"  D(kNN)  R={d['r_acc']:.4f} H={d['h_acc']:.4f} adv={adv_d:+.2f}pp  ({time.time()-t1:.1f}s)", flush=True)

            # Task E (only for FEWSHOT_DATASETS)
            e_r = float("nan"); e_h = float("nan"); adv_e = float("nan")
            if ds in FEWSHOT_DATASETS:
                t1 = time.time()
                e = task_E(X_tr, y_tr, X_te, y_te, ds, n_episodes=1000)
                if e is not None:
                    e_r = e["r_acc"]; e_h = e["h_acc"]
                    adv_e = (e_h - e_r)*100
                    print(f"  E(FS)   R={e_r:.4f} H={e_h:.4f} adv={adv_e:+.2f}pp ({time.time()-t1:.1f}s, {e['n_episodes']} ep)", flush=True)

            row = {**base,
                   "C_r_acc":c["r_acc"], "C_h_acc":c["h_acc"], "C_adv":adv_c,
                   "C_r_super":c["r_super"], "C_h_super":c["h_super"],
                   "D_r_acc":d["r_acc"], "D_h_acc":d["h_acc"], "D_adv":adv_d,
                   "D_r_super":d["r_super"], "D_h_super":d["h_super"],
                   "E_r_acc":e_r, "E_h_acc":e_h, "E_adv":adv_e,
                   "time_s":time.time()-t0}
            rows.append(row)
            pd.DataFrame(rows).to_csv(OUT/"tasks_CDE.csv", index=False)

    df = pd.DataFrame(rows)

    # ─── Summary by paradigm ──────────────────────────────────────────────
    print("\n" + "="*80)
    print("SUMMARY: H_advantage (pp) by paradigm × task")
    print("="*80)
    for task, col in [("C (NC)","C_adv"), ("D (kNN)","D_adv"), ("E (5-way 5-shot)","E_adv")]:
        v = df.dropna(subset=[col])
        print(f"\n  {task}  (n={len(v)})")
        agg = v.groupby("paradigm")[col].agg(["mean","std","count"]).round(2)
        print(agg.to_string())

    # Correlations with delta
    print("\n" + "="*80)
    print("CORRELATIONS: δ vs H_advantage per task")
    print("="*80)
    for task, col in [("C", "C_adv"), ("D", "D_adv"), ("E", "E_adv")]:
        v = df.dropna(subset=["delta", col])
        if len(v) >= 4:
            r_p, p_p = pearsonr(v["delta"], v[col])
            r_s, p_s = spearmanr(v["delta"], v[col])
            print(f"  Task {task}:  pearson r={r_p:+.3f} (p={p_p:.4f})  spearman r={r_s:+.3f} (p={p_s:.4f})  n={len(v)}")

    df.to_csv(OUT/"tasks_CDE.csv", index=False)
    print(f"\nSaved: {OUT/'tasks_CDE.csv'}")


if __name__ == "__main__":
    main()
