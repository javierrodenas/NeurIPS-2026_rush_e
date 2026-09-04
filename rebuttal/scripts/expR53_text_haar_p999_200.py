#!/usr/bin/env python3
"""expR53 (review-response, A1): the text census under {null construction} x {statistic}, 200 replicates,
padding-free extraction exactly as expR48b (batch size 1, last token for causal LMs, model-specific
pooling for embedders, fp16 where expR48b used it). The embeddings of each model are extracted once
and cached (results/text_cache/{model}_photo_bs1.npz) so the four constructions score the same vectors.

  --null gauss|haar, --stat sup|p999 as in expR52. Real value over quadruple seeds 0..9; each null
  replicate over seeds 0..2 (as expR48/expR48b); null seeds 300+rep. Per model: raw statistic +- s.d.,
  null mean/s.d., excess, r_above, p_left; at merge: Benjamini-Hochberg across the 15 models.

Outputs: expR53_text_haar_p999_200.csv (record), expR53_text_haar_sup_200.csv, expR53_text_gauss_p999_200.csv
(the Gaussian x supremum cell is expR48b_text_census200_bs1.csv). Parallel by model with --part.
"""
import os, sys, time, argparse, glob
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
ROOT = Path(os.environ.get("PLATONIC_ROOT", "/media/HDD_4TB_2/javi/Platonic"))
OUT = Path(os.environ.get("PLATONIC_RESULTS", str(Path(__file__).resolve().parents[1]/"results")))
os.environ.setdefault("HF_HOME", "/media/HDD_4TB_2/javi/hf_cache")
TCACHE = ROOT/"results/text_cache"; TCACHE.mkdir(parents=True, exist_ok=True)
N_REP = 200
NAMES = {("haar","p999"): "expR53_text_haar_p999_200", ("haar","sup"): "expR53_text_haar_sup_200",
         ("gauss","p999"): "expR53_text_gauss_p999_200", ("gauss","sup"): "expR53_text_gauss_sup_200"}
MODELS = [("gpt2","causal","gpt2",None,"",False), ("gpt2_m","causal","gpt2-medium",None,"",False),
          ("gpt2_l","causal","gpt2-large",None,"",False), ("gpt2_xl","causal","gpt2-xl",None,"",True),
          ("pythia_410m","causal","EleutherAI/pythia-410m",None,"",False), ("pythia_1b","causal","EleutherAI/pythia-1b",None,"",True),
          ("pythia_2b8","causal","EleutherAI/pythia-2.8b",None,"",True), ("olmo_1b","causal","allenai/OLMo-1B-hf",None,"",True),
          ("bge_base","embed","BAAI/bge-base-en-v1.5","cls","",False), ("bge_large","embed","BAAI/bge-large-en-v1.5","cls","",False),
          ("gte_base","embed","thenlper/gte-base","mean","",False), ("gte_large","embed","thenlper/gte-large","mean","",False),
          ("gte_qwen2","embed","Alibaba-NLP/gte-Qwen2-1.5B-instruct","last","",True),
          ("e5_base","embed","intfloat/e5-base-v2","mean","query: ",False), ("e5_large","embed","intfloat/e5-large-v2","mean","query: ",False)]
ORDER = [m[0] for m in MODELS]

def imagenet_class_names():
    from timm.data import ImageNetInfo
    info = ImageNetInfo(); return [info.index_to_description(i).split(",")[0].strip() for i in range(1000)]

def encode(kind, hf, pool, prefix, texts, fp16, bs=1):
    import torch
    from transformers import AutoTokenizer, AutoModel
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    kw = dict(torch_dtype=torch.float16) if fp16 else {}
    tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
    mod = AutoModel.from_pretrained(hf, trust_remote_code=True, **kw).to(DEV).eval()
    if kind == "causal":
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        tok.padding_side = "left"
    vs = []
    with torch.no_grad():
        for a in range(0, len(texts), bs):
            b = tok([prefix+t for t in texts[a:a+bs]], return_tensors="pt", padding=True, truncation=True, max_length=64).to(DEV)
            h = mod(**b).last_hidden_state
            if kind == "causal": v = h[:, -1, :]
            elif pool == "last":
                idx = b.attention_mask.sum(1) - 1; v = h[torch.arange(len(h)), idx]
            elif pool == "cls": v = h[:, 0]
            else:
                m = b.attention_mask.unsqueeze(-1).to(h.dtype); v = (h*m).sum(1)/m.sum(1).clamp(min=1e-9)
            vs.append(v.float().cpu().numpy())
    del mod; torch.cuda.empty_cache()
    return np.concatenate(vs).astype(np.float32)

def embeddings(short, kind, hf, pool, prefix, fp16):
    f = TCACHE/f"{short}_photo_bs1.npz"
    if f.exists(): return np.load(f)["X"]
    names = imagenet_class_names(); X = encode(kind, hf, pool, prefix, [f"a photo of a {n}" for n in names], fp16)
    np.savez_compressed(f, X=X); return X

def make_delta(stat):
    def delta_norm(X, n_seeds):
        D = squareform(pdist(X, "euclidean")); diam = D.max(); n = len(D); out = []
        for s in range(n_seeds):
            rng = np.random.RandomState(s)
            i, j, k, l = (rng.randint(0, n, 500_000) for _ in range(4))
            ok = (i!=j)&(i!=k)&(i!=l)&(j!=k)&(j!=l)&(k!=l); i,j,k,l = i[ok],j[ok],k[ok],l[ok]
            S = np.sort(np.stack([D[i,j]+D[k,l], D[i,k]+D[j,l], D[i,l]+D[j,k]], 1), 1)
            dfc = (S[:,2]-S[:,1])/2
            out.append((dfc.max() if stat == "sup" else np.percentile(dfc, 99.9))/diam)
        return float(np.mean(out)), float(np.std(out))
    return delta_norm
def svd(M): mu = M.mean(0); U, S, Vt = np.linalg.svd(M-mu, full_matrices=False); return mu, U, S, Vt
def null_gauss(M, rep):
    mu, U, S, Vt = svd(M); rng = np.random.RandomState(300+rep)
    G = rng.randn(len(M), len(S)).astype(np.float32); G /= G.std(0, keepdims=True)*np.sqrt(len(M)); return (G*S)@Vt+mu
def null_haar(M, rep):
    mu, U, S, Vt = svd(M); rng = np.random.RandomState(300+rep)
    Z = rng.randn(len(M), len(M)); Q, R = np.linalg.qr(Z); Q = Q*np.sign(np.diag(R))
    return ((Q[:, :len(S)]*S)@Vt+mu).astype(np.float32)
NULLS = {"gauss": null_gauss, "haar": null_haar}
def bh(p):
    p = np.asarray(p, dtype=float); n = len(p); order = np.argsort(p); ranked = p[order]*n/np.arange(1, n+1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]; out = np.empty(n); out[order] = np.minimum(adj, 1.0); return out

def run(null, stat, part, models):
    delta_norm = make_delta(stat); nf = NULLS[null]; base = NAMES[(null, stat)]
    csv_path = OUT/f"{base}.part_{part}.csv"
    rows = pd.read_csv(csv_path).to_dict("records") if csv_path.exists() else []
    done = {r["model"] for r in rows}
    for short, kind, hf, pool, prefix, fp16 in MODELS:
        if short not in models or short in done: continue
        t0 = time.time()
        try: X = embeddings(short, kind, hf, pool, prefix, fp16)
        except Exception as e: print(f"{short}: FAILED {type(e).__name__}: {str(e)[:120]}"); continue
        dr, dr_sd = delta_norm(X, 10)
        nulls = np.array([delta_norm(nf(X, r), 3)[0] for r in range(N_REP)])
        nm, nsd = float(nulls.mean()), float(nulls.std(ddof=1))
        r_above = int((nulls > dr).sum()); p_left = (1 + int((nulls <= dr).sum()))/(N_REP+1)
        rows.append(dict(model=short, kind=kind, null=null, stat=stat, d=X.shape[1], delta=dr, delta_sd=dr_sd, null_mean=nm, null_sd=nsd,
                         excess=dr-nm, z=(dr-nm)/max(np.sqrt(nsd**2+dr_sd**2),1e-9), frac_null_above=r_above/N_REP,
                         r_above=r_above, p_left=p_left, time_s=time.time()-t0))
        print(f"{short:12s} {null}/{stat} delta {dr:.4f} null {nm:.4f} exc {dr-nm:+.4f} r {r_above}/{N_REP} p {p_left:.4f} ({rows[-1]['time_s']:.0f}s)")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Done part {part} ({null}/{stat})")

def merge(null, stat):
    base = NAMES[(null, stat)]
    parts = sorted(glob.glob(str(OUT/f"{base}.part_*.csv")))
    df = pd.concat([pd.read_csv(p) for p in parts]).drop_duplicates(subset=["model"], keep="first")
    df["_o"] = df.model.map({m: i for i, m in enumerate(ORDER)}); df = df.sort_values("_o").drop(columns=["_o"])
    df["genuine"] = df.p_left <= 0.05; df["p_bh"] = bh(df.p_left.values); df["genuine_bh"] = df.p_bh <= 0.05
    df.to_csv(OUT/f"{base}.csv", index=False)
    print(f"SUMMARY {base}: {len(df)} models; genuine {int(df.genuine.sum())} (BH {int(df.genuine_bh.sum())}); genuine_bh: {list(df[df.genuine_bh].model)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--null", default="haar", choices=["gauss","haar"]); ap.add_argument("--stat", default="p999", choices=["sup","p999"])
    ap.add_argument("--part", default=None); ap.add_argument("--models", nargs="+", default=ORDER); ap.add_argument("--merge", action="store_true")
    A = ap.parse_args()
    if A.merge: merge(A.null, A.stat)
    else:
        assert A.part, "--part TAG required"; run(A.null, A.stat, A.part, A.models)
