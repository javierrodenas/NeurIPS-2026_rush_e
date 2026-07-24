#!/usr/bin/env python3
"""
REBUTTAL EXP 4 — prompt robustness of the text-model geometry
(answers Xbn5 Q2 and RJje Q6 / prompt-confound concern).

Models: GPT-2 S, GPT-2 M (causal LMs; last-token hidden state) and
BGE-base, E5-base (text embedders; native pooling + L2 norm).

For each of 6 prompt templates over the 1000 ImageNet class names,
compute delta_max (paper estimator) and mean ORC (k=10) of the 1000
prompt embeddings.

Claim to test: the ranking causal-LM-more-tree-like-than-embedder and the
absolute delta levels are stable across templates, including the
class-name-only template (no shared lexical scaffold).

Output: rebuttal/results/exp4_prompt_variation.csv
"""
import os, sys, time
os.environ.setdefault("HF_HOME", "/home/javi/Platonic/.hf_cache")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment

ROOT = Path("/home/javi/Platonic")
OUT = ROOT / "rebuttal/results"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEMPLATES = {
    "photo":    "a photo of a {}",
    "name_only":"{}",
    "image":    "an image of a {}",
    "closeup":  "a close-up photo of a {}",
    "this_is":  "this is a photo of a {}",
    "wild":     "a {} in the wild",
}

MODELS = [
    ("gpt2",        "causal",  "gpt2"),
    ("gpt2_m",      "causal",  "gpt2-medium"),
    ("bge_base",    "embed",   "BAAI/bge-base-en-v1.5"),
    ("e5_base",     "embed",   "intfloat/e5-base-v2"),
]


def imagenet_class_names():
    try:
        from timm.data import ImageNetInfo
        info = ImageNetInfo()
        names = [info.index_to_description(i).split(",")[0].strip()
                 for i in range(1000)]
        assert len(names) == 1000
        return names
    except Exception as e:
        print("timm names failed:", e)
        import urllib.request
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        txt = urllib.request.urlopen(url, timeout=30).read().decode()
        names = [l.strip() for l in txt.strip().split("\n")]
        assert len(names) == 1000
        return names


def delta_max_from_D(D, n_quads=500_000, n_seeds=10):
    diam = D.max(); n = D.shape[0]; deltas = []
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        i = rng.randint(0, n, n_quads); j = rng.randint(0, n, n_quads)
        k = rng.randint(0, n, n_quads); l = rng.randint(0, n, n_quads)
        ok = (i!=j)&(i!=k)&(i!=l)&(j!=k)&(j!=l)&(k!=l)
        i,j,k,l = i[ok],j[ok],k[ok],l[ok]
        s = np.sort(np.stack([D[i,j]+D[k,l], D[i,k]+D[j,l], D[i,l]+D[j,k]],1),1)
        deltas.append(((s[:,2]-s[:,1])/2).max() / diam)
    d = np.array(deltas)
    return float(d.mean()), float(d.std())


def orc_mean(X, k=10):
    D = squareform(pdist(X, 'euclidean'))
    n = len(X)
    nn = np.argsort(D, axis=1)[:, 1:k+1]
    edges = set()
    for u in range(n):
        for v in nn[u]:
            edges.add((min(u, int(v)), max(u, int(v))))
    kap = []
    for (u, v) in edges:
        M = D[np.ix_(nn[u], nn[v])]
        r, c = linear_sum_assignment(M)
        kap.append(1.0 - M[r, c].mean() / max(D[u, v], 1e-12))
    kap = np.array(kap)
    return float(kap.mean()), float((kap < 0).mean())


@torch.no_grad()
def encode_causal(hf_name, prompts, batch=32):
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(hf_name)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModel.from_pretrained(hf_name).to(DEVICE).eval()
    vecs = []
    for i in range(0, len(prompts), batch):
        enc = tok(prompts[i:i+batch], return_tensors="pt", padding=True).to(DEVICE)
        h = model(**enc).last_hidden_state       # (b, T, d), left-padded
        vecs.append(h[:, -1, :].float().cpu().numpy())  # last token
    del model
    torch.cuda.empty_cache()
    return np.concatenate(vecs, 0)


@torch.no_grad()
def encode_embedder(hf_name, prompts, pool, batch=64):
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModel.from_pretrained(hf_name).to(DEVICE).eval()
    vecs = []
    for i in range(0, len(prompts), batch):
        enc = tok(prompts[i:i+batch], return_tensors="pt", padding=True,
                  truncation=True).to(DEVICE)
        out = model(**enc).last_hidden_state
        if pool == "cls":
            v = out[:, 0]
        else:  # mean pooling with attention mask
            m = enc["attention_mask"].unsqueeze(-1).float()
            v = (out * m).sum(1) / m.sum(1).clamp(min=1e-9)
        v = torch.nn.functional.normalize(v, dim=-1)
        vecs.append(v.float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(vecs, 0)


def main():
    names = imagenet_class_names()
    print(f"{len(names)} class names; e.g. {names[:3]}", flush=True)
    rows = []
    for short, kind, hf_name in MODELS:
        for tkey, tpl in TEMPLATES.items():
            t0 = time.time()
            prompts = [tpl.format(n) for n in names]
            if kind == "causal":
                X = encode_causal(hf_name, prompts)
            else:
                pool = "cls" if "bge" in hf_name.lower() else "mean"
                if "e5" in hf_name.lower():
                    prompts = [f"query: {p}" for p in prompts]
                X = encode_embedder(hf_name, prompts, pool)
            D = squareform(pdist(X, 'euclidean'))
            dm, dstd = delta_max_from_D(D)
            om, fneg = orc_mean(X)
            rows.append(dict(model=short, kind=kind, template=tkey,
                             delta_max=dm, delta_max_std=dstd,
                             ORC_mean=om, frac_neg=fneg))
            pd.DataFrame(rows).to_csv(OUT / "exp4_prompt_variation.csv", index=False)
            print(f"{short:9s} {tkey:9s} delta={dm:.4f}±{dstd:.4f} "
                  f"ORC={om:+.3f} fneg={fneg:.3f} ({time.time()-t0:.0f}s)", flush=True)
    print("Done ->", OUT / "exp4_prompt_variation.csv")


if __name__ == "__main__":
    main()
