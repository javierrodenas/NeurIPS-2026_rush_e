#!/usr/bin/env python3
"""expR48 (round-7 P7): the text census (paper template) re-run PADDING-FREE (batch size 1) with 20 spectrum-null replicates.

All four GPT-2 sizes x the 10 distinct templates of exp4+exp17 (photo shared),
encoded with the exp18 protocol (left-padded last token, fp16 for XL), then
delta_norm (500K x 10 seeds) and excess over 3 spectrum-null replicates (5 seeds
each) per (model, template). Output: expR48_text_census_bs1.csv."""
import os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd, torch
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
ROOT = Path("/media/HDD_4TB_2/javi/Platonic")
OUT = Path(__file__).resolve().parents[1]/"results"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
def imagenet_class_names():
    from timm.data import ImageNetInfo
    info = ImageNetInfo()
    return [info.index_to_description(i).split(",")[0].strip() for i in range(1000)]
def glosses():
    from timm.data import ImageNetInfo
    from nltk.corpus import wordnet as wn
    wnids = ImageNetInfo().label_names()
    return [wn.synset_from_pos_and_offset('n', int(w[1:])).definition() for w in wnids]
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
def main():
    names = imagenet_class_names()
    prompts = [f"a photo of a {n}" for n in names]
    csv_path = OUT/"expR48_text_census_bs1.csv"
    rows = pd.read_csv(csv_path).to_dict("records") if csv_path.exists() else []
    done = {r["model"] for r in rows}
    for short, kind, hf, pool, prefix, fp16 in MODELS:
        if short in done: continue
        t0 = time.time()
        try:
            X = encode(kind, hf, pool, prefix, prompts, fp16)
        except Exception as e:
            print(f"{short}: FAILED {type(e).__name__}: {str(e)[:120]}"); continue
        dr, dr_sd = delta_norm(X)
        nulls = [delta_norm(specnull(X, r), n_seeds=3)[0] for r in range(20)]
        nm, nsd = float(np.mean(nulls)), float(np.std(nulls, ddof=1))
        pr = float(np.mean([n > dr for n in nulls]))
        rows.append(dict(model=short, kind=kind, d=X.shape[1], delta=dr, delta_sd=dr_sd, null_mean=nm, null_sd=nsd,
                         excess=dr-nm, z=(dr-nm)/max(np.sqrt(nsd**2+dr_sd**2),1e-9), frac_null_above=pr, time_s=time.time()-t0))
        print(f"{short:12s} delta {dr:.4f} null {nm:.4f} exc {dr-nm:+.4f} z {rows[-1]['z']:+.1f} above {pr:.2f} ({rows[-1]['time_s']:.0f}s)")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
    print("Done expR48")
if __name__ == "__main__":
    main()
