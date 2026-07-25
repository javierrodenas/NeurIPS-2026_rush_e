#!/usr/bin/env python3
"""
REBUTTAL EXP 14 — text prototype task with a real 3-level hierarchy
(DBpedia Classes: 9 L1 -> 70 L2 -> 219 L3), answering Xbn5 Q1's literal ask
("a text-based task") with the paper's exact protocol.

Per embedder (BGE-base CLS-pool, E5-base mean-pool+"query: ", GTE-base
mean-pool), raw (UNnormalized) pooled embeddings as the analog of vision CLS:

  Geometry : delta_max (paper estimator) on the 219 L3 centroids,
             spectrum-matched null excess, sibling-triplet vs L2 (eucl+cos).
  Tool     : NC and FS(5w5s, 1000 ep) with R / H / COS + paired CIs.

Success criterion (fixed BEFORE running): FS H-R > 0 with CI95 excluding 0
for >=2 of 3 embedders.

Output: rebuttal/results/exp14_dbpedia.csv + log
"""
import os, sys, time
os.environ.setdefault("HF_HOME", "/home/javi/Platonic/.hf_cache")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd, torch
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from huggingface_hub import hf_hub_download

OUT = Path("/home/javi/Platonic/rebuttal/results"); OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ATANH_T = float(np.arctanh(0.70710678))
N_TRAIN, N_TEST = 100, 30

MODELS = [
    ("bge_base", "BAAI/bge-base-en-v1.5", "cls",  ""),
    ("e5_base",  "intfloat/e5-base-v2",   "mean", "query: "),
    ("gte_base", "thenlper/gte-base",     "mean", ""),
]

def load_dbpedia():
    tr = pd.read_csv(hf_hub_download("DeveloperOats/DBPedia_Classes", "DBPEDIA_train.csv", repo_type="dataset"))
    te = pd.read_csv(hf_hub_download("DeveloperOats/DBPedia_Classes", "DBPEDIA_test.csv", repo_type="dataset"))
    rng = np.random.RandomState(0)
    def sub(df, n):
        return df.groupby("l3", group_keys=False).apply(
            lambda g: g.sample(min(n, len(g)), random_state=rng))
    tr, te = sub(tr, N_TRAIN), sub(te, N_TEST)
    l3 = sorted(tr.l3.unique())
    l3i = {c: i for i, c in enumerate(l3)}
    l2_of = tr.drop_duplicates("l3").set_index("l3").l2.to_dict()
    y_tr = tr.l3.map(l3i).values; y_te = te.l3.map(l3i).values
    sup = np.array([hash(l2_of[c]) for c in l3])   # L2 group id per L3 class
    _, sup = np.unique(sup, return_inverse=True)
    return tr.text.tolist(), y_tr, te.text.tolist(), y_te, sup, len(l3)

@torch.no_grad()
def encode(hf, pool, prefix, texts, bs=128):
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(hf)
    mod = AutoModel.from_pretrained(hf).to(DEV).eval()
    vs = []
    for i in range(0, len(texts), bs):
        enc = tok([prefix + t for t in texts[i:i+bs]], return_tensors="pt",
                  padding=True, truncation=True, max_length=256).to(DEV)
        h = mod(**enc).last_hidden_state
        if pool == "cls": v = h[:, 0]
        else:
            m = enc["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        vs.append(v.float().cpu().numpy())          # RAW, unnormalized
    del mod; torch.cuda.empty_cache()
    return np.concatenate(vs, 0)

def delta_max(X, n_quads=500_000, n_seeds=10):
    D = squareform(pdist(X, 'euclidean')); diam = D.max(); n = len(D); out = []
    for s in range(n_seeds):
        rng = np.random.RandomState(s)
        i,j,k,l = (rng.randint(0,n,n_quads) for _ in range(4))
        ok=(i!=j)&(i!=k)&(i!=l)&(j!=k)&(j!=l)&(k!=l); i,j,k,l=i[ok],j[ok],k[ok],l[ok]
        S = np.sort(np.stack([D[i,j]+D[k,l],D[i,k]+D[j,l],D[i,l]+D[j,k]],1),1)
        out.append(((S[:,2]-S[:,1])/2).max()/diam)
    return float(np.mean(out))

def spec_null_excess(C, reps=3):
    d_real = delta_max(C)
    mu = C.mean(0); U,S,Vt = np.linalg.svd(C-mu, full_matrices=False)
    nulls = []
    for r in range(reps):
        rng = np.random.RandomState(300+r)
        G = rng.randn(len(C), len(S)).astype(np.float32)
        G /= G.std(0, keepdims=True)*np.sqrt(len(C))
        nulls.append(delta_max((G*S)@Vt + mu, n_seeds=4))
    return d_real, float(np.mean(nulls)), d_real - float(np.mean(nulls))

def sib_triplet(C, sup):
    D = squareform(pdist(C)); n=len(D); agree=tot=0
    for a in range(n):
        sib = np.where((sup==sup[a]) & (np.arange(n)!=a))[0]
        non = np.where(sup!=sup[a])[0]
        if not len(sib): continue
        ds = D[a,sib][:,None]; dn = D[a,non][None,:]
        agree += (ds<dn).sum(); tot += ds.size*dn.shape[1]
    return agree/tot

def s4t(X, mu, sc):
    Z=(X-mu)*sc; n=Z.norm(dim=-1,keepdim=True).clamp_min(1e-7)
    Y=torch.tanh(n/2)*Z/n
    return Y*((1-1e-3)/Y.norm(dim=-1,keepdim=True).clamp_min(1e-7)).clamp(max=1.0)

def poinc(A,B):
    a2=(A*A).sum(-1,keepdim=True); b2=(B*B).sum(-1,keepdim=True).T
    sq=(a2+b2-2*A@B.T).clamp_min(0)
    return torch.acosh((1+2*sq/((1-a2)*(1-b2)).clamp_min(1e-7)).clamp_min(1+1e-7))

def run_tool(Xtr, ytr, Xte, yte, K):
    a=torch.tensor(Xtr,device=DEV); b=torch.tensor(Xte,device=DEV)
    yt=torch.tensor(ytr,device=DEV)
    P=torch.stack([a[yt==c].mean(0) for c in range(K)])
    accR=float((torch.cdist(b,P).argmin(1).cpu().numpy()==yte).mean())
    mu=a.mean(0); p95=torch.quantile((a-mu).norm(dim=1),0.95)
    sc=2*ATANH_T/p95.clamp_min(1e-7)
    ap,bp=s4t(a,mu,sc),s4t(b,mu,sc)
    Ph=s4t(torch.stack([ap[yt==c].mean(0) for c in range(K)]),torch.zeros_like(mu),1.0)
    accH=float((poinc(bp,Ph).argmin(1).cpu().numpy()==yte).mean())
    bn=b/b.norm(dim=1,keepdim=True).clamp_min(1e-7); pn=P/P.norm(dim=1,keepdim=True).clamp_min(1e-7)
    accC=float(((bn@pn.T).argmax(1).cpu().numpy()==yte).mean())
    return accR, accH, accC

def run_fs(X, y, n_ep=1000, seed=42):
    cls=np.unique(y); idx={c:np.where(y==c)[0] for c in cls}
    cls=np.array([c for c in cls if len(idx[c])>=20])
    Xg=torch.tensor(X,device=DEV)
    mu=Xg.mean(0); p95=torch.quantile((Xg-mu).norm(dim=1),0.95)
    sc=2*ATANH_T/p95.clamp_min(1e-7)
    Xp=s4t(Xg,mu,sc); Xn=Xg/Xg.norm(dim=1,keepdim=True).clamp_min(1e-7)
    rng=np.random.RandomState(seed); acc={m:[] for m in "RHC"}
    for _ in range(n_ep):
        sel=rng.choice(cls,5,replace=False); sup,qry,ql=[],[],[]
        for li,c in enumerate(sel):
            p=rng.choice(idx[c],20,replace=False)
            sup+=p[:5].tolist(); qry+=p[5:].tolist(); ql+=[li]*15
        sup,qry,ql=np.array(sup),np.array(qry),np.array(ql)
        P=Xg[sup].view(5,5,-1).mean(1)
        acc["R"].append((torch.cdist(Xg[qry],P).argmin(1).cpu().numpy()==ql).mean())
        Ph=s4t(Xp[sup].view(5,5,-1).mean(1),torch.zeros(P.shape[-1],device=DEV),1.0)
        acc["H"].append((poinc(Xp[qry],Ph).argmin(1).cpu().numpy()==ql).mean())
        Pn=Xn[sup].view(5,5,-1).mean(1); Pn=Pn/Pn.norm(dim=1,keepdim=True).clamp_min(1e-7)
        acc["C"].append(((Xn[qry]@Pn.T).argmax(1).cpu().numpy()==ql).mean())
    out={m:float(np.mean(v)) for m,v in acc.items()}
    dHR=np.array(acc["H"])-np.array(acc["R"]); dHC=np.array(acc["H"])-np.array(acc["C"])
    out["HR"]=float(dHR.mean()); out["HR_ci"]=float(1.96*dHR.std()/np.sqrt(n_ep))
    out["HC"]=float(dHC.mean()); out["HC_ci"]=float(1.96*dHC.std()/np.sqrt(n_ep))
    return out

def main():
    texts_tr, y_tr, texts_te, y_te, sup, K = load_dbpedia()
    print(f"DBpedia: {K} clases L3, {len(set(sup))} grupos L2, "
          f"{len(texts_tr)} train / {len(texts_te)} test", flush=True)
    rows=[]
    for short, hf, pool, pre in MODELS:
        t0=time.time()
        Xtr=encode(hf,pool,pre,texts_tr); Xte=encode(hf,pool,pre,texts_te)
        C=np.stack([Xtr[y_tr==c].mean(0) for c in range(K)])
        d_real,d_null,exc = spec_null_excess(C)
        Cn=C/np.linalg.norm(C,axis=1,keepdims=True)
        trip_e=sib_triplet(C,sup); trip_c=sib_triplet(Cn,sup)
        ncR,ncH,ncC = run_tool(Xtr,y_tr,Xte,y_te,K)
        fs = run_fs(np.vstack([Xtr,Xte]), np.concatenate([y_tr,y_te]))
        rows.append(dict(model=short, delta=d_real, null=d_null, excess=exc,
            trip_eucl=trip_e, trip_cos=trip_c,
            NC_R=ncR, NC_H=ncH, NC_COS=ncC,
            FS_R=fs["R"], FS_H=fs["H"], FS_COS=fs["C"],
            FS_HR_pp=fs["HR"]*100, FS_HR_ci=fs["HR_ci"]*100,
            FS_HC_pp=fs["HC"]*100, FS_HC_ci=fs["HC_ci"]*100))
        pd.DataFrame(rows).to_csv(OUT/"exp14_dbpedia.csv", index=False)
        print(f"{short:9s} δ={d_real:.3f} null={d_null:.3f} exc={exc:+.3f} "
              f"trip e/c={trip_e:.3f}/{trip_c:.3f} | "
              f"NC R={ncR:.4f} H={ncH:.4f} C={ncC:.4f} | "
              f"FS R={fs['R']:.4f} H={fs['H']:.4f} C={fs['C']:.4f} "
              f"H-R={fs['HR']*100:+.2f}±{fs['HR_ci']*100:.2f}pp "
              f"H-C={fs['HC']*100:+.2f}±{fs['HC_ci']*100:.2f}pp ({time.time()-t0:.0f}s)", flush=True)
    print("Done")

if __name__ == "__main__":
    main()
