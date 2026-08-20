#!/usr/bin/env python3
"""
ICLR EXP 19 — number-of-classes (C) vs hierarchy deconfound (W3 of the mock
review). ImageNet subsampled to C in {10,20,50,100,200,500,1000} classes,
with RANDOM subsets vs WORDNET-COHERENT subsets (greedy min mean WordNet
distance), for 4 representative backbones.

Per (model, C, mode, seed): delta_norm (exact enumeration for C<=30),
spectrum-null excess, and NC H_adv on the class subset using the exact
task pipeline of the paper (prototype = mean of projected points, re-mapped;
practitioner_tasks_t707.task_C convention).

Output: rebuttal/results/exp19_c_sweep.csv + exp19.log (incremental).
"""
import os, sys, time, itertools
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd, torch
from pathlib import Path
from scipy.spatial.distance import pdist, squareform

ROOT = Path("/home/javi/Platonic")
CACHE = ROOT/"results/practical_tasks_cache"
WND = np.load(ROOT/"results/features/imagenet/imagenet1k_wordnet_dist.npy")
OUT = ROOT/"rebuttal/results"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ATANH = 2*np.arctanh(0.70710678)

MODELS = ["i21k_l", "dinov2_l", "dinov2_g", "clip_l"]
CS = [10, 20, 50, 100, 200, 500, 1000]

def delta_norm(X, n_quads=200_000, n_seeds=5):
    D = squareform(pdist(X, "euclidean")); diam = D.max(); n = len(D)
    if n <= 30:
        q = np.array(list(itertools.combinations(range(n), 4)))
        i, j, k, l = q.T
        S = np.sort(np.stack([D[i,j]+D[k,l], D[i,k]+D[j,l], D[i,l]+D[j,k]], 1), 1)
        return float(((S[:,2]-S[:,1])/2).max()/diam), 0.0
    out = []
    for s in range(n_seeds):
        rng = np.random.RandomState(s)
        i, j, k, l = (rng.randint(0, n, n_quads) for _ in range(4))
        ok = (i!=j)&(i!=k)&(i!=l)&(j!=k)&(j!=l)&(k!=l); i,j,k,l = i[ok],j[ok],k[ok],l[ok]
        S = np.sort(np.stack([D[i,j]+D[k,l], D[i,k]+D[j,l], D[i,l]+D[j,k]], 1), 1)
        out.append(((S[:,2]-S[:,1])/2).max()/diam)
    return float(np.mean(out)), float(np.std(out))

def specnull(C, rep):
    mu = C.mean(0); Cc = C - mu
    U, S, Vt = np.linalg.svd(Cc, full_matrices=False)
    rng = np.random.RandomState(300+rep)
    G = rng.randn(len(C), len(S)).astype(np.float32)
    G /= G.std(0, keepdims=True) * np.sqrt(len(C))
    return (G*S)@Vt + mu

# --- exact task_C convention from scripts/practitioner_tasks_t707.py ---
def s4_project(X, mu, scale):
    Xs = (X - mu) * scale
    nrm = Xs.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    return torch.tanh(nrm/2) / nrm * Xs

def poincare_dist(X, Y):
    x2 = (X*X).sum(-1, keepdim=True); y2 = (Y*Y).sum(-1, keepdim=True).T
    xy = X @ Y.T
    num = (x2 - 2*xy + y2.clamp(min=0)).clamp(min=0)
    den = ((1-x2)*(1-y2.T).T).clamp(min=1e-9)
    return torch.acosh(1 + 2*num/den)

def nc_adv(Xtr, ytr, Xte, yte, classes):
    m = np.isin(ytr, classes); mt = np.isin(yte, classes)
    remap = {c: i for i, c in enumerate(classes)}
    ytr2 = np.array([remap[c] for c in ytr[m]]); yte2 = np.array([remap[c] for c in yte[mt]])
    A = torch.tensor(Xtr[m], dtype=torch.float32, device=DEV)
    B = torch.tensor(Xte[mt], dtype=torch.float32, device=DEV)
    n_cls = len(classes)
    cent_r = torch.stack([A[torch.tensor(ytr2==c, device=DEV)].mean(0) for c in range(n_cls)])
    pr = torch.cdist(B, cent_r).argmin(1).cpu().numpy()
    acc_r = float((pr == yte2).mean())
    mu = A.mean(0)
    p95 = torch.quantile((A-mu).norm(dim=1), 0.95).item()
    scale = ATANH / max(p95, 1e-7)
    Ap = s4_project(A, mu, scale); Bp = s4_project(B, mu, scale)
    cent_raw = torch.stack([Ap[torch.tensor(ytr2==c, device=DEV)].mean(0) for c in range(n_cls)])
    cent_h = s4_project(cent_raw, torch.zeros_like(mu), 1.0)
    ph = poincare_dist(Bp, cent_h).argmin(1).cpu().numpy()
    acc_h = float((ph == yte2).mean())
    return acc_r, acc_h, len(yte2)

def coherent_subset(C, seed):
    rng = np.random.RandomState(seed)
    cur = [int(rng.randint(1000))]
    cand = set(range(1000)) - set(cur)
    while len(cur) < C:
        cl = np.array(sorted(cand))
        best = cl[np.argmin(WND[np.ix_(cl, cur)].mean(1))]
        cur.append(int(best)); cand.discard(int(best))
    return np.array(cur)

def main():
    csv_path = OUT/"exp19_c_sweep.csv"
    rows = pd.read_csv(csv_path).to_dict("records") if csv_path.exists() else []
    done = {(r["model"], r["C"], r["mode"], r["seed"]) for r in rows}
    for m in MODELS:
        tr = np.load(CACHE/f"{m}_imagenet_train.npz"); te = np.load(CACHE/f"{m}_imagenet_test.npz")
        Xtr, ytr = tr["features"].astype(np.float32), tr["labels"].astype(np.int64)
        Xte, yte = te["features"].astype(np.float32), te["labels"].astype(np.int64)
        cents_all = np.stack([Xtr[ytr==c].mean(0) for c in range(1000)])
        for C in CS:
            seeds = range(5) if C <= 200 else range(2) if C < 1000 else range(1)
            modes = ["random"] if C == 1000 else ["random", "coherent"]
            for mode in modes:
                for seed in seeds:
                    key = (m, C, mode, seed)
                    if key in done: continue
                    t0 = time.time()
                    if C == 1000:
                        cls = np.arange(1000)
                    elif mode == "random":
                        cls = np.random.RandomState(seed).choice(1000, C, replace=False)
                    else:
                        cls = coherent_subset(C, seed)
                    sub = cents_all[cls]
                    dm, dsd = delta_norm(sub)
                    nulls = [delta_norm(specnull(sub, rep), n_seeds=3)[0] for rep in range(2)]
                    exc = dm - float(np.mean(nulls))
                    accr, acch, nte = nc_adv(Xtr, ytr, Xte, yte, cls)
                    rows.append(dict(model=m, C=C, mode=mode, seed=seed, delta=dm,
                                     delta_sd=dsd, null_mean=float(np.mean(nulls)),
                                     excess=exc, nc_R=accr, nc_H=acch,
                                     nc_adv_pp=(acch-accr)*100, n_test=nte,
                                     time_s=time.time()-t0))
                    r = rows[-1]
                    print(f"{m:9s} C={C:4d} {mode:8s} s{seed} | delta {dm:.4f} exc {exc:+.4f} "
                          f"| NC {100*accr:.2f}->{100*acch:.2f} ({r['nc_adv_pp']:+.2f}pp, n={nte}) "
                          f"| {r['time_s']:.0f}s")
                    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print("Done")

if __name__ == "__main__":
    main()
