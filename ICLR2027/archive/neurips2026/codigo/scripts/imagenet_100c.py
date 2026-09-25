#!/usr/bin/env python3
"""
Recompute ImageNet results at 100 imgs/class (subsample from full-train),
matching the 5 non-ImageNet datasets which already use 100/class.
Outputs delta, ORC, NC R/H/adv, FS R/H/adv at t=1/sqrt(2) for the 12
vision panel models.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(8)
from pathlib import Path
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.optimize import linear_sum_assignment

ROOT = Path("/home/javi/Platonic")
CACHE = ROOT / "results/practical_tasks_cache"
OUT = ROOT / "results"

PARADIGMS = {
    "i21k_t":"Supervised","i21k_s":"Supervised","i21k_b":"Supervised","i21k_l":"Supervised",
    "dinov1_b":"SSL","dinov2_s":"SSL","dinov2_b":"SSL","dinov2_l":"SSL","dinov2_g":"SSL",
    "clip_b":"Contrastive","clip_l":"Contrastive","siglip_b":"Contrastive",
}
PARAMS_M = {"i21k_t":5,"i21k_s":22,"i21k_b":86,"i21k_l":307,"dinov1_b":86,
            "dinov2_s":22,"dinov2_b":86,"dinov2_l":307,"dinov2_g":1100,
            "clip_b":86,"clip_l":307,"siglip_b":86}

N_CLASSES = 1000
PER_CLASS = 100
T_TARGET = 1/np.sqrt(2)


def project_unit_ball(X, mu, p95, target):
    s = 2 * np.arctanh(target) / max(p95, 1e-7)
    Xs = (X - mu) * s
    nrm = np.linalg.norm(Xs, axis=-1, keepdims=True).clip(min=1e-7)
    Y = np.tanh(nrm/2) / nrm * Xs
    cur = np.linalg.norm(Y, axis=-1, keepdims=True).clip(min=1e-7)
    return Y * np.minimum((1.0-1e-3)/cur, 1.0)


def double_project(X, mu, p95, target):
    Y = project_unit_ball(X, mu, p95, target)
    nrm = np.linalg.norm(Y, axis=-1, keepdims=True).clip(min=1e-7)
    Z = np.tanh(nrm/2) / nrm * Y
    cur = np.linalg.norm(Z, axis=-1, keepdims=True).clip(min=1e-7)
    return Z * np.minimum((1.0-1e-3)/cur, 1.0)


def poincare_dist(X, Y, chunk=500):
    n=len(X); m=len(Y); out=np.empty((n,m),dtype=np.float32)
    y_sq=(Y*Y).sum(-1)[None,:]
    for i in range(0,n,chunk):
        Xc=X[i:i+chunk]; x_sq=(Xc*Xc).sum(-1,keepdims=True)
        d_sq=(x_sq+y_sq-2.0*Xc@Y.T).clip(min=0)
        denom=((1-x_sq)*(1-y_sq)).clip(min=1e-7)
        out[i:i+chunk]=np.arccosh((1+2*d_sq/denom).clip(min=1+1e-7))
    return out


def euclid_dist(A, B):
    a=(A*A).sum(-1,keepdims=True); b=(B*B).sum(-1,keepdims=True).T
    return np.sqrt((a+b-2.0*A@B.T).clip(min=0))


def gromov_delta(X, n_quads=50000, seed=42):
    rng=np.random.RandomState(seed)
    D=squareform(pdist(X,'euclidean')); diam=D.max(); n=len(X)
    quads=np.array([rng.choice(n,4,replace=False) for _ in range(n_quads)])
    i,j,k,l=quads.T
    s=np.sort(np.stack([D[i,j]+D[k,l], D[i,k]+D[j,l], D[i,l]+D[j,k]],1),1)
    return float(((s[:,2]-s[:,1])/2).mean()/diam)


def w1_uniform(neighbors_u, neighbors_v, points):
    A=points[neighbors_u]; B=points[neighbors_v]
    cost=np.sqrt(((A[:,None,:]-B[None,:,:])**2).sum(-1))
    r,c=linear_sum_assignment(cost)
    return cost[r,c].mean()


def compute_orc(centroids, k=10):
    D=cdist(centroids,centroids); n=len(centroids)
    knn=np.argsort(D,axis=1)[:,1:k+1]
    edges=set()
    for u in range(n):
        for v in knn[u]:
            edges.add((min(u,int(v)),max(u,int(v))))
    edges=list(edges); kappas=[]
    for u,v in edges:
        w1=w1_uniform(knn[u],knn[v],centroids)
        kappas.append(1-w1/max(D[u,v],1e-7))
    return float(np.mean(kappas))


def task_FS(X_all, y_all, mu, p95, target, n_episodes=1000, n_way=5, k_shot=5, q_query=15, seed=42):
    classes=np.unique(y_all); rng=np.random.RandomState(seed)
    idx_per={c:np.where(y_all==c)[0] for c in classes}
    X_h=project_unit_ball(X_all, mu, p95, target)
    R=[]; H=[]
    for _ in range(n_episodes):
        sel=rng.choice(classes,n_way,replace=False)
        sup,qry,qlab=[],[],[]
        for li,c in enumerate(sel):
            ic=idx_per[c]
            if len(ic)<k_shot+q_query: continue
            pick=rng.choice(ic,k_shot+q_query,replace=False)
            sup.extend(pick[:k_shot]); qry.extend(pick[k_shot:k_shot+q_query]); qlab.extend([li]*q_query)
        if not qry: continue
        sup=np.array(sup); qry=np.array(qry); qlab=np.array(qlab)
        prots=X_all[sup].reshape(n_way,k_shot,-1).mean(1)
        D_R=euclid_dist(X_all[qry], prots)
        R.append((np.argmin(D_R,1)==qlab).mean())
        sup_h=X_h[sup]; qry_h=X_h[qry]
        prots_h=sup_h.reshape(n_way,k_shot,-1).mean(1)
        nrm=np.linalg.norm(prots_h,axis=-1,keepdims=True).clip(min=1e-7)
        prots_h=np.tanh(nrm/2)/nrm*prots_h
        cur=np.linalg.norm(prots_h,axis=-1,keepdims=True).clip(min=1e-7)
        prots_h=prots_h*np.minimum((1.0-1e-3)/cur,1.0)
        D_H=poincare_dist(qry_h, prots_h)
        H.append((np.argmin(D_H,1)==qlab).mean())
    return float(np.mean(R)), float(np.mean(H))


def evaluate(model):
    tr=CACHE/f"{model}_imagenet_fulltrain.npz"; te=CACHE/f"{model}_imagenet_test.npz"
    if not tr.exists() or not te.exists():
        print(f"  SKIP {model}", flush=True); return None
    print(f"\n>>> {model}", flush=True); t0=time.time()
    d_tr=np.load(tr); d_te=np.load(te)
    X_tr_full=d_tr["features"].astype(np.float32); y_tr_full=d_tr["labels"].astype(np.int64)
    X_te=d_te["features"].astype(np.float32); y_te=d_te["labels"].astype(np.int64)

    # Subsample 100 per class from full-train (deterministic)
    rng=np.random.RandomState(0)
    sel=[]
    for c in range(N_CLASSES):
        ic=np.where(y_tr_full==c)[0]
        if len(ic)>PER_CLASS:
            sel.extend(rng.choice(ic,PER_CLASS,replace=False).tolist())
        else:
            sel.extend(ic.tolist())
    sel=np.array(sel)
    X_tr=X_tr_full[sel]; y_tr=y_tr_full[sel]
    print(f"  loaded+subsampled n_train={len(X_tr)} ({time.time()-t0:.0f}s)", flush=True)

    # Centroids
    cents=np.stack([X_tr[y_tr==c].mean(0) for c in range(N_CLASSES)])

    # delta on centroids
    delta=gromov_delta(cents)
    # ORC on centroids (k=10 for 1000 classes is fine)
    orc=compute_orc(cents, k=10)

    # NC at t=1/sqrt(2)
    mu=cents.mean(0); p95=np.percentile(np.linalg.norm(cents-mu,axis=1),95)
    D_R=cdist(X_te, cents, metric='euclidean'); pred_R=np.argmin(D_R,1); r_acc=float((pred_R==y_te).mean()); del D_R
    Xte_p=project_unit_ball(X_te, mu, p95, T_TARGET)
    cents_h=double_project(cents, mu, p95, T_TARGET)
    D_H=poincare_dist(Xte_p, cents_h, chunk=500); pred_H=np.argmin(D_H,1); h_acc=float((pred_H==y_te).mean()); del D_H
    nc_adv=(h_acc-r_acc)*100

    # FS at t=1/sqrt(2)
    X_all=np.concatenate([X_tr,X_te],0); y_all=np.concatenate([y_tr,y_te],0)
    mu_fs=X_all.mean(0); p95_fs=np.percentile(np.linalg.norm(X_all-mu_fs,axis=1),95)
    fs_R, fs_H = task_FS(X_all, y_all, mu_fs, p95_fs, T_TARGET)
    fs_adv=(fs_H-fs_R)*100

    print(f"  delta={delta:.4f}  ORC={orc:+.3f}  NC R={r_acc:.4f} H={h_acc:.4f} adv={nc_adv:+.2f}  FS R={fs_R:.4f} H={fs_H:.4f} adv={fs_adv:+.2f}  ({time.time()-t0:.0f}s)", flush=True)
    return dict(model=model, paradigm=PARADIGMS[model], params_M=PARAMS_M[model],
                delta=delta, ORC=orc,
                NC_R=r_acc, NC_H=h_acc, NC_adv=nc_adv,
                FS_R=fs_R, FS_H=fs_H, FS_adv=fs_adv)


def main():
    rows=[]
    order=["i21k_t","i21k_s","i21k_b","i21k_l","dinov1_b","dinov2_s","dinov2_b","dinov2_l","dinov2_g","clip_b","clip_l","siglip_b"]
    for m in order:
        r=evaluate(m)
        if r is not None:
            rows.append(r)
            pd.DataFrame(rows).to_csv(OUT/"imagenet_100c.csv", index=False)
    print(f"\nDone: {OUT}/imagenet_100c.csv  ({len(rows)} rows)")


if __name__=="__main__":
    main()
