#!/usr/bin/env python3
"""expR49 (mock-review W3): does the GPT-2 scale emergence survive the template? PADDING-FREE (batch 1) re-run.

All four GPT-2 sizes x the 10 distinct templates of exp4+exp17 (photo shared),
encoded with the exp18 protocol (left-padded last token, fp16 for XL), then
delta_norm (500K x 10 seeds) and excess over 3 spectrum-null replicates (5 seeds
each) per (model, template). Output: expR49_template_nulls_bs1.csv."""
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
def encode_causal(hf, texts, fp16, bs=1):
    from transformers import AutoTokenizer, AutoModel
    kw = dict(torch_dtype=torch.float16) if fp16 else {}
    tok = AutoTokenizer.from_pretrained(hf)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    mod = AutoModel.from_pretrained(hf, **kw).to(DEV).eval()
    vs = []
    for a in range(0, len(texts), bs):
        b = tok(texts[a:a+bs], return_tensors="pt", padding=True, truncation=True,
                max_length=64).to(DEV)
        h = mod(**b).last_hidden_state
        vs.append(h[:, -1, :].float().cpu().numpy())
    del mod; torch.cuda.empty_cache()
    return np.concatenate(vs).astype(np.float32)
MODELS = [("gpt2","gpt2",False), ("gpt2_m","gpt2-medium",False),
          ("gpt2_l","gpt2-large",False), ("gpt2_xl","gpt2-xl",True)]
def main():
    names = imagenet_class_names(); gl = glosses()
    TEMPLATES = {
        "photo":      [f"a photo of a {n}" for n in names],
        "name_only":  names,
        "image":      [f"an image of a {n}" for n in names],
        "closeup":    [f"a close-up photo of a {n}" for n in names],
        "this_is":    [f"this is a photo of a {n}" for n in names],
        "wild":       [f"a {n} in the wild" for n in names],
        "def_pair":   [f"{n}: {g}" for n, g in zip(names, gl)],
        "gloss_only": gl,
        "concept":    [f"the concept of {n}" for n in names],
        "discussion": [f"a discussion about {n}" for n in names],
    }
    csv_path = OUT/"expR49_template_nulls_bs1.csv"
    rows = pd.read_csv(csv_path).to_dict("records") if csv_path.exists() else []
    done = {(r["model"], r["template"]) for r in rows}
    for short, hf, fp16 in MODELS:
        todo = [t for t in TEMPLATES if (short, t) not in done]
        if not todo: continue
        for tkey in todo:
            t0 = time.time()
            X = encode_causal(hf, TEMPLATES[tkey], fp16)
            dr, dr_sd = delta_norm(X)
            nulls = [delta_norm(specnull(X, r), n_seeds=5)[0] for r in range(3)]
            nm, nsd = float(np.mean(nulls)), float(np.std(nulls, ddof=1))
            rows.append(dict(model=short, template=tkey, d=X.shape[1], delta=dr,
                             delta_sd=dr_sd, null_mean=nm, null_sd=nsd,
                             excess=dr-nm, z=(dr-nm)/max(np.sqrt(nsd**2+dr_sd**2),1e-9),
                             time_s=time.time()-t0))
            print(f"{short:8s} {tkey:11s} delta {dr:.4f} null {nm:.4f} exc {dr-nm:+.4f} z={rows[-1]['z']:+.1f} ({rows[-1]['time_s']:.0f}s)")
            pd.DataFrame(rows).to_csv(csv_path, index=False)
    print("Done expR49")
if __name__ == "__main__":
    main()
