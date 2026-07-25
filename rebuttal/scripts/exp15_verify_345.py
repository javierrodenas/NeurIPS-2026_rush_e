#!/usr/bin/env python3
"""
REBUTTAL EXP 15 — independent re-run of tasks 3 (kNN), 4 (retrieval P@10)
and 5 (clustering ARI), fresh code path, to verify the repo's May results.

Sampling replicates the original scripts' RNG call order (so cells are
comparable one-to-one); all math (projection, distances, voting, P@10,
linkage) is written from scratch here.

Outputs: rebuttal/results/exp15_verify_345.csv  (mine vs repo, per cell)
"""
import os, sys, time
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd, torch
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

ROOT = Path("/home/javi/Platonic"); CACHE = ROOT/"results/practical_tasks_cache"
OUT = ROOT/"rebuttal/results"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = ["i21k_t","i21k_s","i21k_b","i21k_l","dinov1_b","dinov2_s","dinov2_b",
          "dinov2_l","dinov2_g","clip_b","clip_l","siglip_b"]
AT = float(np.arctanh(1/np.sqrt(2)))

COARSE = {0:[4,30,55,72,95],1:[1,32,67,73,91],2:[54,62,70,82,92],3:[9,10,16,28,61],4:[0,51,53,57,83],
 5:[22,39,40,86,87],6:[5,20,25,84,94],7:[6,7,14,18,24],8:[3,42,43,88,97],9:[12,17,37,68,76],
 10:[23,33,49,60,71],11:[15,19,21,31,38],12:[34,63,64,66,75],13:[26,45,77,79,99],
 14:[2,11,35,46,98],15:[27,29,44,78,93],16:[36,50,65,74,80],17:[47,52,56,59,96],
 18:[8,13,48,58,90],19:[41,69,81,85,89]}
SUP100 = np.zeros(100, dtype=int)
for c, fs in COARSE.items():
    for f in fs: SUP100[f] = c

def proj(X, mu, p95):
    s = 2*AT/max(p95, 1e-7)
    Z = (X-mu)*s
    n = Z.norm(dim=-1, keepdim=True).clamp_min(1e-7)
    Y = torch.tanh(n/2)*Z/n
    return Y*((1-1e-3)/Y.norm(dim=-1, keepdim=True).clamp_min(1e-7)).clamp(max=1.0)

def pdist_h(A, B, chunk=1000):
    outs=[]
    b2=(B*B).sum(-1)[None,:]
    for i in range(0, len(A), chunk):
        Ac=A[i:i+chunk]; a2=(Ac*Ac).sum(-1,keepdim=True)
        sq=(a2+b2-2*Ac@B.T).clamp_min(0)
        outs.append(torch.acosh((1+2*sq/((1-a2)*(1-b2)).clamp_min(1e-7)).clamp_min(1+1e-7)))
    return torch.cat(outs,0)

def load(model, ds, split):
    d = np.load(CACHE/f"{model}_{ds}_{split}.npz")
    return d["features"].astype(np.float32), d["labels"].astype(np.int64)

def load_in_train(model):
    d = np.load(CACHE/f"{model}_imagenet_fulltrain.npz")
    Xf, yf = d["features"].astype(np.float32), d["labels"].astype(np.int64)
    rng = np.random.RandomState(0); sel=[]
    for c in range(int(yf.max())+1):
        ic = np.where(yf==c)[0]
        sel.extend(rng.choice(ic, min(100,len(ic)), replace=False).tolist())
    sel=np.array(sel)
    return Xf[sel], yf[sel]

def knn5(Xq, Xdb, ydb, dist):
    D = dist(Xq, Xdb)
    nn = D.topk(5, largest=False, dim=1).indices
    lab = torch.tensor(ydb, device=DEV)[nn]
    return lab.mode(dim=1).values.cpu().numpy()

def p10(D, yq, ydb):
    nn = D.topk(10, largest=False, dim=1).indices
    lab = torch.tensor(ydb, device=DEV)[nn]
    return float((lab == torch.tensor(yq, device=DEV)[:,None]).float().mean())

rows=[]
# ── TAREA 3: kNN, 12 modelos × 5 datasets (protocolo t707) ──
t707 = pd.read_csv(ROOT/"results/tasks_t707/tasks_t707.csv").set_index(['model','dataset'])
for m in MODELS:
    for ds in ["cifar100","cifar10","dtd","fashionmnist","mnist"]:
        Xtr,ytr = load(m,ds,"train"); Xte,yte = load(m,ds,"test")
        K = int(max(ytr.max(),yte.max()))+1
        rng = np.random.RandomState(42); sub=[]
        for c in range(K):
            ic = np.where(ytr==c)[0]
            sub.extend(rng.choice(ic,50,replace=False).tolist() if len(ic)>50 else ic.tolist())
        sub=np.array(sub)
        a=torch.tensor(Xtr[sub],device=DEV); b=torch.tensor(Xte,device=DEV)
        pr=knn5(b,a,ytr[sub],lambda A,B: torch.cdist(A,B))
        mu=Xtr.mean(0); p95=np.percentile(np.linalg.norm(Xtr-mu,axis=1),95)
        mt=torch.tensor(mu,device=DEV)
        ah,bh=proj(a,mt,p95),proj(b,mt,p95)
        ph=knn5(bh,ah,ytr[sub],pdist_h)
        accR,accH=float((pr==yte).mean()),float((ph==yte).mean())
        ref=t707.loc[(m,ds)]
        rows.append(dict(task="kNN",model=m,dataset=ds,mine_R=accR,mine_H=accH,
                         repo_R=ref.D_r_acc,repo_H=ref.D_h_acc,
                         dR=(accR-ref.D_r_acc)*100,dH=(accH-ref.D_h_acc)*100))
        pd.DataFrame(rows).to_csv(OUT/"exp15_verify_345.csv",index=False)
    print(f"kNN {m} listo",flush=True)

# ── TAREA 4: retrieval P@10, 12 modelos × 6 datasets (protocolo retrieval_t707) ──
ret = pd.read_csv(ROOT/"results/retrieval_t707.csv").set_index(['model','dataset'])
for m in MODELS:
    for ds in ["cifar100","cifar10","dtd","fashionmnist","mnist","imagenet"]:
        if ds=="imagenet": Xtr,ytr = load_in_train(m)
        else: Xtr,ytr = load(m,ds,"train")
        Xte,yte = load(m,ds,"test")
        K=int(max(ytr.max(),yte.max()))+1
        rng=np.random.RandomState(42); sub=[]
        for c in range(K):
            ic=np.where(ytr==c)[0]
            sub.extend(rng.choice(ic,50,replace=False).tolist() if len(ic)>50 else ic.tolist())
        sub=np.array(sub)
        q = rng.choice(len(Xte),1000,replace=False) if len(Xte)>1000 else np.arange(len(Xte))
        Xdb,ydb=Xtr[sub],ytr[sub]; Xq,yq=Xte[q],yte[q]
        mu=Xtr.mean(0); p95=np.percentile(np.linalg.norm(Xtr-mu,axis=1),95)
        a=torch.tensor(Xdb,device=DEV); b=torch.tensor(Xq,device=DEV); mt=torch.tensor(mu,device=DEV)
        fR=p10(torch.cdist(b,a),yq,ydb)
        fH=p10(pdist_h(proj(b,mt,p95),proj(a,mt,p95)),yq,ydb)
        ref=ret.loc[(m,ds)]
        rows.append(dict(task="retr",model=m,dataset=ds,mine_R=fR,mine_H=fH,
                         repo_R=ref.R_fine,repo_H=ref.H_fine,
                         dR=(fR-ref.R_fine)*100,dH=(fH-ref.H_fine)*100))
        pd.DataFrame(rows).to_csv(OUT/"exp15_verify_345.csv",index=False)
    print(f"retr {m} listo",flush=True)

# ── TAREA 5: clustering ARI (average linkage), CIFAR-100 ──
prev = pd.read_csv(OUT/"exp8_p1_recovery.csv")
for m in MODELS:
    Xtr,ytr = load(m,"cifar100","train")
    C = np.stack([Xtr[ytr==c].mean(0) for c in range(100)])
    D_r = squareform(np.sqrt(((C[:,None]-C[None,:])**2).sum(-1)), checks=False)
    cutR = fcluster(linkage(D_r,method="average"),t=20,criterion="maxclust")
    Ct = torch.tensor(C)
    mu=Ct.mean(0); p95=float(np.percentile((Ct-mu).norm(dim=1).numpy(),95))
    Ch = proj(Ct,mu,p95)
    D_h = squareform(pdist_h(Ch,Ch,chunk=100).numpy(), checks=False)
    cutH = fcluster(linkage(D_h,method="average"),t=20,criterion="maxclust")
    ariR, ariH = adjusted_rand_score(SUP100,cutR), adjusted_rand_score(SUP100,cutH)
    pR = prev[(prev.model==m)&(prev.method=="average")&(prev.space=="R")].ari.iloc[0]
    pH = prev[(prev.model==m)&(prev.method=="average")&(prev.space=="H")].ari.iloc[0]
    rows.append(dict(task="clust",model=m,dataset="cifar100",mine_R=ariR,mine_H=ariH,
                     repo_R=pR,repo_H=pH,dR=(ariR-pR)*100,dH=(ariH-pH)*100))
    pd.DataFrame(rows).to_csv(OUT/"exp15_verify_345.csv",index=False)
    print(f"clust {m} listo",flush=True)

df=pd.DataFrame(rows)
print("\nRESUMEN |dif| mío vs repo (pp):")
print(df.groupby("task")[["dR","dH"]].apply(lambda g: g.abs().max()).round(2).to_string())
print("Done")
