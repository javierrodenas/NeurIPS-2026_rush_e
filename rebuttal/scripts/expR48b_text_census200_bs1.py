#!/usr/bin/env python3
"""expR48b (R6): the padding-free text census at 200 spectrum-null replicates (Groger resolution).

Copy of expR48_text_census_bs1.py with ONE change: range(200) replicates instead of 20. Same
extraction (padding-free, batch size 1, last token for causal LMs, model-specific pooling for
embedders, fp16 where expR48 used it), same estimator (real delta with 10 seeds), same null
construction and seeds (300+rep), same 3 quadruple seeds per replicate as expR48 (the vision
census uses 5). Stored per model: frac_null_above (= r/200), r_above = #{null > real}, and the
left-tail add-one p-value p_left = (1 + #{null <= real}) / 201; "genuine" = p_left <= 0.05.

Parallel by model: `--part TAG --models ...` writes expR48b_text_census200_bs1.part_TAG.csv;
`--merge` concatenates the parts into expR48b_text_census200_bs1.csv in MODELS order.
"""
import os, sys, time, argparse, glob
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd, torch
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
ROOT = Path(os.environ.get("PLATONIC_ROOT", "/media/HDD_4TB_2/javi/Platonic"))
OUT = Path(os.environ.get("PLATONIC_RESULTS", str(Path(__file__).resolve().parents[1]/"results")))
os.environ.setdefault("HF_HOME", "/media/HDD_4TB_2/javi/hf_cache")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_REP = 200
def imagenet_class_names():
    from timm.data import ImageNetInfo
    info = ImageNetInfo()
    return [info.index_to_description(i).split(",")[0].strip() for i in range(1000)]
def delta_norm(X, n_quads=500_000, n_seeds=10):
    D = squareform(pdist(X, "euclidean")); diam = D.max(); n = len(D)
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
@torch.no_grad()
def encode(kind, hf, pool, prefix, texts, fp16, bs=1):
    from transformers import AutoTokenizer, AutoModel
    kw = dict(torch_dtype=torch.float16) if fp16 else {}
    tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
    mod = AutoModel.from_pretrained(hf, trust_remote_code=True, **kw).to(DEV).eval()
    if kind == "causal":
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        tok.padding_side = "left"
    vs = []
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
MODELS = [("gpt2","causal","gpt2",None,"",False), ("gpt2_m","causal","gpt2-medium",None,"",False),
          ("gpt2_l","causal","gpt2-large",None,"",False), ("gpt2_xl","causal","gpt2-xl",None,"",True),
          ("pythia_410m","causal","EleutherAI/pythia-410m",None,"",False), ("pythia_1b","causal","EleutherAI/pythia-1b",None,"",True),
          ("pythia_2b8","causal","EleutherAI/pythia-2.8b",None,"",True), ("olmo_1b","causal","allenai/OLMo-1B-hf",None,"",True),
          ("bge_base","embed","BAAI/bge-base-en-v1.5","cls","",False), ("bge_large","embed","BAAI/bge-large-en-v1.5","cls","",False),
          ("gte_base","embed","thenlper/gte-base","mean","",False), ("gte_large","embed","thenlper/gte-large","mean","",False),
          ("gte_qwen2","embed","Alibaba-NLP/gte-Qwen2-1.5B-instruct","last","",True),
          ("e5_base","embed","intfloat/e5-base-v2","mean","query: ",False), ("e5_large","embed","intfloat/e5-large-v2","mean","query: ",False)]
ORDER = [m[0] for m in MODELS]
def run(part, models):
    names = imagenet_class_names()
    prompts = [f"a photo of a {n}" for n in names]
    csv_path = OUT/f"expR48b_text_census200_bs1.part_{part}.csv"
    rows = pd.read_csv(csv_path).to_dict("records") if csv_path.exists() else []
    done = {r["model"] for r in rows}
    for short, kind, hf, pool, prefix, fp16 in MODELS:
        if short not in models or short in done: continue
        t0 = time.time()
        try:
            X = encode(kind, hf, pool, prefix, prompts, fp16)
        except Exception as e:
            print(f"{short}: FAILED {type(e).__name__}: {str(e)[:120]}"); continue
        dr, dr_sd = delta_norm(X)
        nulls = np.array([delta_norm(specnull(X, r), n_seeds=3)[0] for r in range(N_REP)])
        nm, nsd = float(nulls.mean()), float(nulls.std(ddof=1))
        r_above = int((nulls > dr).sum()); p_left = (1 + int((nulls <= dr).sum())) / (N_REP + 1)
        rows.append(dict(model=short, kind=kind, d=X.shape[1], delta=dr, delta_sd=dr_sd, null_mean=nm, null_sd=nsd,
                         excess=dr-nm, z=(dr-nm)/max(np.sqrt(nsd**2+dr_sd**2),1e-9), frac_null_above=r_above/N_REP,
                         r_above=r_above, p_left=p_left, time_s=time.time()-t0))
        print(f"{short:12s} delta {dr:.4f} null {nm:.4f} exc {dr-nm:+.4f} r {r_above}/{N_REP} p {p_left:.4f} ({rows[-1]['time_s']:.0f}s)")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Done part {part}")
def merge():
    parts = sorted(glob.glob(str(OUT/"expR48b_text_census200_bs1.part_*.csv")))
    df = pd.concat([pd.read_csv(p) for p in parts]).drop_duplicates(subset=["model"], keep="first")
    df["_o"] = df.model.map({m: i for i, m in enumerate(ORDER)}); df = df.sort_values("_o").drop(columns=["_o"])
    df.to_csv(OUT/"expR48b_text_census200_bs1.csv", index=False)
    print(f"merged {len(df)} models; genuine (p_left<=0.05): {int((df.p_left<=0.05).sum())}/{len(df)}")
    for _, r in df.iterrows(): print(f"  {r.model:12s} exc {r.excess:+.4f} r {int(r.r_above)}/200 p {r.p_left:.4f}")
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default=None); ap.add_argument("--models", nargs="+", default=ORDER)
    ap.add_argument("--merge", action="store_true"); A = ap.parse_args()
    if A.merge: merge()
    else:
        assert A.part, "--part TAG required"; run(A.part, A.models)
