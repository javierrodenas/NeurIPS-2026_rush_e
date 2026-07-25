#!/usr/bin/env python3
"""
Tasks NC, Few-shot 5w5s, and Sample retrieval on ImageNet using S4 pipeline.
Same recipe as tasks_CDE.py + sample_retrieval.py, but for the 1000-class ImageNet
features extracted on the remote (~/Platonic/results/practical_tasks_cache/<model>_imagenet_*.npz).

Superclass labels = AgglomerativeClustering(30) on WordNet distance matrix,
matching the original paper's setup.
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import AgglomerativeClustering
import time, warnings, sys
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

ROOT = Path("/home/javi/Platonic")
CACHE = ROOT / "results/practical_tasks_cache"
WN_DIST = ROOT / "results/features/imagenet/imagenet1k_wordnet_dist.npy"
OUT = ROOT / "results/tasks_imagenet"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PARADIGMS = {
    "i21k_t":"Supervised","i21k_s":"Supervised","i21k_b":"Supervised","i21k_l":"Supervised",
    "dinov1_b":"SSL","dinov2_s":"SSL","dinov2_b":"SSL","dinov2_l":"SSL","dinov2_g":"SSL",
    "clip_b":"Contrastive","clip_l":"Contrastive","siglip_b":"Contrastive",
}
ALL_MODELS = list(PARADIGMS.keys())

N_CLASSES = 1000
N_SUPER = 30
N_QUERIES_RETR = 1000        # subsample test queries for retrieval (speed)
TRAIN_PER_CLASS_RETR = 50    # subsample train DB for retrieval
N_EPISODES_FS = 1000

# ─── helpers ─────────────────────────────────────────────────────────────
def s4_project_torch(X, mu, scale):
    Xs = (X - mu) * scale
    nrm = Xs.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    Y = torch.tanh(nrm/2) / nrm * Xs
    cur = Y.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    return Y * torch.clamp((1.0-1e-3) / cur, max=1.0)

def s4_params(X_np):
    mu = X_np.mean(0)
    p95 = np.percentile(np.linalg.norm(X_np - mu, axis=1), 95)
    scale = 2 * np.arctanh(0.5) / max(p95, 1e-7)
    return mu, scale

def euclid_dist(X, Y): return torch.cdist(X, Y, p=2)

def poincare_dist(X, Y):
    x_sq = (X*X).sum(-1, keepdim=True)
    y_sq = (Y*Y).sum(-1, keepdim=True).T
    d_sq = (x_sq + y_sq - 2.0 * X @ Y.T).clamp(min=0)
    denom = ((1-x_sq)*(1-y_sq)).clamp(min=1e-7)
    arg = (1 + 2*d_sq/denom).clamp(min=1+1e-7)
    return torch.acosh(arg)  # factor 2 dropped (rank-equivalent), prevents huge magnitudes

def gromov_delta(X, n_quads=50000, seed=42):
    rng = np.random.RandomState(seed)
    D = squareform(pdist(X, 'euclidean'))
    diam = D.max()
    n = len(X)
    quads = np.array([rng.choice(n, 4, replace=False) for _ in range(n_quads)])
    i,j,k,l = quads.T
    s = np.sort(np.stack([D[i,j]+D[k,l], D[i,k]+D[j,l], D[i,l]+D[j,k]],1),1)
    return float(((s[:,2]-s[:,1])/2).mean()/diam)

# ─── tasks ───────────────────────────────────────────────────────────────
def task_NC(X_tr, y_tr, X_te, y_te, n_classes, super_map=None):
    Xtr = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    Xte = torch.tensor(X_te, dtype=torch.float32, device=DEVICE)
    cents_R = torch.stack([Xtr[torch.tensor(y_tr==c)].mean(0) for c in range(n_classes)])
    pred_R = euclid_dist(Xte, cents_R).argmin(1).cpu().numpy()
    mu, sc = s4_params(X_tr)
    mu_t = torch.tensor(mu, dtype=torch.float32, device=DEVICE)
    Xtr_h = s4_project_torch(Xtr, mu_t, sc)
    Xte_h = s4_project_torch(Xte, mu_t, sc)
    cents_h = torch.stack([Xtr_h[torch.tensor(y_tr==c)].mean(0) for c in range(n_classes)])
    cents_h = s4_project_torch(cents_h, torch.zeros_like(mu_t), 1.0)
    pred_H = poincare_dist(Xte_h, cents_h).argmin(1).cpu().numpy()
    return dict(
        nc_R_acc=float((pred_R==y_te).mean()), nc_H_acc=float((pred_H==y_te).mean()),
        nc_R_super=float((super_map[pred_R]==super_map[y_te]).mean()) if super_map is not None else float("nan"),
        nc_H_super=float((super_map[pred_H]==super_map[y_te]).mean()) if super_map is not None else float("nan"),
    )

def task_FS(X_tr, y_tr, X_te, y_te, n_episodes=1000, n_way=5, k_shot=5, q_query=15, seed=42):
    X = np.concatenate([X_tr, X_te], 0); y = np.concatenate([y_tr, y_te], 0)
    classes = np.unique(y)
    rng = np.random.RandomState(seed)
    idx_per_c = {c: np.where(y==c)[0] for c in classes}
    mu = X.mean(0); p95 = np.percentile(np.linalg.norm(X - mu, axis=1), 95)
    sc = 2*np.arctanh(0.5)/max(p95, 1e-7)
    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    mu_t = torch.tensor(mu, dtype=torch.float32, device=DEVICE)
    X_h = s4_project_torch(Xt, mu_t, sc)
    R_acc, H_acc = [], []
    for ep in range(n_episodes):
        sel = rng.choice(classes, n_way, replace=False)
        sup, qry, qlab = [], [], []
        for li, c in enumerate(sel):
            ic = idx_per_c[c]
            if len(ic) < k_shot+q_query: continue
            pick = rng.choice(ic, k_shot+q_query, replace=False)
            sup.extend(pick[:k_shot]); qry.extend(pick[k_shot:k_shot+q_query]); qlab.extend([li]*q_query)
        if not qry: continue
        sup = np.array(sup); qry = np.array(qry); qlab = np.array(qlab)
        prots_R = Xt[sup].view(n_way, k_shot, -1).mean(1)
        D_R = euclid_dist(Xt[qry], prots_R)
        R_acc.append((D_R.argmin(1).cpu().numpy() == qlab).mean())
        sup_h = X_h[sup]; qry_h = X_h[qry]
        prots_h = sup_h.view(n_way, k_shot, -1).mean(1)
        prots_h = s4_project_torch(prots_h, torch.zeros(prots_h.shape[-1], device=DEVICE), 1.0)
        D_H = poincare_dist(qry_h, prots_h)
        H_acc.append((D_H.argmin(1).cpu().numpy() == qlab).mean())
    return dict(fs_R_acc=float(np.mean(R_acc)), fs_H_acc=float(np.mean(H_acc)))

def task_R_sample(X_tr, y_tr, X_te, y_te, super_map, n_classes,
                   n_queries=N_QUERIES_RETR, train_per_class=TRAIN_PER_CLASS_RETR, seed=42):
    rng = np.random.RandomState(seed)
    sub = []
    for c in range(n_classes):
        ic = np.where(y_tr==c)[0]
        if len(ic) > train_per_class: sub.extend(rng.choice(ic, train_per_class, replace=False).tolist())
        else: sub.extend(ic.tolist())
    sub = np.array(sub)
    X_db = X_tr[sub]; y_db = y_tr[sub]
    if len(X_te) > n_queries:
        q = rng.choice(len(X_te), n_queries, replace=False)
    else:
        q = np.arange(len(X_te))
    X_q = X_te[q]; y_q = y_te[q]
    mu, sc = s4_params(X_tr)
    mu_t = torch.tensor(mu, dtype=torch.float32, device=DEVICE)
    Xq_t = torch.tensor(X_q, dtype=torch.float32, device=DEVICE)
    Xdb_t = torch.tensor(X_db, dtype=torch.float32, device=DEVICE)
    Xq_h = s4_project_torch(Xq_t, mu_t, sc)
    Xdb_h = s4_project_torch(Xdb_t, mu_t, sc)
    chunk = 200
    DR = torch.empty((len(Xq_t), len(Xdb_t)), dtype=torch.float32, device=DEVICE)
    DH = torch.empty_like(DR)
    for i in range(0, len(Xq_t), chunk):
        DR[i:i+chunk] = euclid_dist(Xq_t[i:i+chunk], Xdb_t)
        DH[i:i+chunk] = poincare_dist(Xq_h[i:i+chunk], Xdb_h)
    def p_at_k(D, ql, dbl, k=10):
        nn = D.topk(k, largest=False, dim=1).indices.cpu().numpy()
        m = (dbl[nn] == ql[:, None]).mean(axis=1)
        return float(m.mean())
    out = dict(
        r_R_fine_p10=p_at_k(DR, y_q, y_db),
        r_H_fine_p10=p_at_k(DH, y_q, y_db),
        r_R_sup_p10=p_at_k(DR, super_map[y_q], super_map[y_db]),
        r_H_sup_p10=p_at_k(DH, super_map[y_q], super_map[y_db]),
    )
    del DR, DH, Xq_t, Xdb_t, Xq_h, Xdb_h
    torch.cuda.empty_cache()
    return out

# ─── main ────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}", flush=True)
    print("Loading WordNet distance matrix and computing 30 superclasses...", flush=True)
    wn = np.load(WN_DIST)
    super_map = AgglomerativeClustering(n_clusters=N_SUPER, metric='precomputed', linkage='average').fit_predict(wn)
    print(f"  WN superclasses: {N_SUPER}, sizes mean={pd.Series(super_map).value_counts().mean():.1f}", flush=True)

    rows = []
    for model in ALL_MODELS:
        tr_p = CACHE / f"{model}_imagenet_train.npz"
        te_p = CACHE / f"{model}_imagenet_test.npz"
        if not tr_p.exists() or not te_p.exists():
            print(f"SKIP {model}: missing npz", flush=True); continue
        t0 = time.time()
        Xtr = np.load(tr_p)["features"].astype(np.float32)
        ytr = np.load(tr_p)["labels"].astype(np.int64)
        Xte = np.load(te_p)["features"].astype(np.float32)
        yte = np.load(te_p)["labels"].astype(np.int64)
        print(f"\n>>> {model}  train={Xtr.shape}  test={Xte.shape}", flush=True)

        # delta on backbone centroids
        cents = np.stack([Xtr[ytr==c].mean(0) for c in range(N_CLASSES)])
        delta = gromov_delta(cents)

        nc = task_NC(Xtr, ytr, Xte, yte, N_CLASSES, super_map=super_map)
        print(f"  NC: R={nc['nc_R_acc']:.4f}  H={nc['nc_H_acc']:.4f}  adv={(nc['nc_H_acc']-nc['nc_R_acc'])*100:+.2f}pp", flush=True)
        fs = task_FS(Xtr, ytr, Xte, yte, n_episodes=N_EPISODES_FS)
        print(f"  FS: R={fs['fs_R_acc']:.4f}  H={fs['fs_H_acc']:.4f}  adv={(fs['fs_H_acc']-fs['fs_R_acc'])*100:+.2f}pp", flush=True)
        rs = task_R_sample(Xtr, ytr, Xte, yte, super_map, N_CLASSES)
        print(f"  R-fine: R={rs['r_R_fine_p10']:.4f}  H={rs['r_H_fine_p10']:.4f}  adv={(rs['r_H_fine_p10']-rs['r_R_fine_p10'])*100:+.2f}pp", flush=True)
        print(f"  R-sup:  R={rs['r_R_sup_p10']:.4f}  H={rs['r_H_sup_p10']:.4f}  adv={(rs['r_H_sup_p10']-rs['r_R_sup_p10'])*100:+.2f}pp", flush=True)

        row = dict(model=model, paradigm=PARADIGMS[model], dataset='imagenet',
                   delta=delta, in_dim=int(Xtr.shape[1]),
                   NC_R_acc=nc['nc_R_acc'], NC_H_acc=nc['nc_H_acc'], NC_adv_pp=(nc['nc_H_acc']-nc['nc_R_acc'])*100,
                   FewShot_R_acc=fs['fs_R_acc'], FewShot_H_acc=fs['fs_H_acc'], FewShot_adv_pp=(fs['fs_H_acc']-fs['fs_R_acc'])*100,
                   RetrFine_R=rs['r_R_fine_p10'], RetrFine_H=rs['r_H_fine_p10'], RetrFine_adv_pp=(rs['r_H_fine_p10']-rs['r_R_fine_p10'])*100,
                   RetrSup_R=rs['r_R_sup_p10'], RetrSup_H=rs['r_H_sup_p10'], RetrSup_adv_pp=(rs['r_H_sup_p10']-rs['r_R_sup_p10'])*100,
                   time_s=time.time()-t0)
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT/"tasks_imagenet.csv", index=False)
        print(f"  done ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT/"tasks_imagenet.csv", index=False)

    # Summary
    print("\n" + "="*70 + "\n SUMMARY (mean H-advantage by paradigm)\n" + "="*70)
    for col in ['NC_adv_pp','FewShot_adv_pp','RetrFine_adv_pp','RetrSup_adv_pp']:
        print(f"\n{col}:")
        print(df.groupby('paradigm')[col].agg(['mean','std','count']).round(2).to_string())

    print("\n" + "="*70 + "\n CORRELATIONS (delta vs adv)\n" + "="*70)
    for col in ['NC_adv_pp','FewShot_adv_pp','RetrFine_adv_pp','RetrSup_adv_pp']:
        v = df.dropna(subset=['delta', col])
        if len(v) >= 4:
            r,p = pearsonr(v['delta'], v[col])
            print(f"  delta vs {col:18s}: r={r:+.3f}, p={p:.4f}, n={len(v)}")

if __name__ == "__main__":
    main()
