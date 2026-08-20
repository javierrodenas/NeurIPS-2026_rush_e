#!/usr/bin/env python3
"""
ICLR EXP 18 — spectrum-matched nulls for ALL text models (W2 corollary of the
mock review; tests the cross-modality half of H1).

For each of the 16 text models of the breadth panel (9 causal LMs + 7 text
embedders), encode "a photo of a {class}" for the 1000 ImageNet classes
(paper protocol, exp4-validated encoder), then compute delta_norm and the
excess over a spectrum-matched null (3 reps). If causal LMs do not beat their
nulls, the "causal LMs are among the most hierarchical" claim is scoped out.

Output: rebuttal/results/exp18_text_nulls.csv + exp18.log (incremental).
"""
import os, sys, time
os.environ.setdefault("HF_HOME", "/home/javi/Platonic/.hf_cache")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd, torch
from pathlib import Path
from scipy.spatial.distance import pdist, squareform

ROOT = Path("/home/javi/Platonic"); OUT = ROOT/"rebuttal/results"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
sys.path.insert(0, str(ROOT/"rebuttal/scripts"))
from exp4_prompt_variation import imagenet_class_names, delta_max_from_D

# (short, kind, hf_id, pool, prefix, fp16)
MODELS = [
    ("gpt2",       "causal", "gpt2",                                None,   "", False),
    ("gpt2_m",     "causal", "gpt2-medium",                         None,   "", False),
    ("gpt2_l",     "causal", "gpt2-large",                          None,   "", False),
    ("gpt2_xl",    "causal", "gpt2-xl",                             None,   "", True),
    ("pythia_410m","causal", "EleutherAI/pythia-410m",              None,   "", False),
    ("pythia_1b",  "causal", "EleutherAI/pythia-1b",                None,   "", True),
    ("pythia_2b8", "causal", "EleutherAI/pythia-2.8b",              None,   "", True),
    ("olmo_1b",    "causal", "allenai/OLMo-1B-hf",                  None,   "", True),
    ("olmo_7b",    "causal", "allenai/OLMo-7B-hf",                  None,   "", True),
    ("bge_base",   "embed",  "BAAI/bge-base-en-v1.5",               "cls",  "", False),
    ("bge_large",  "embed",  "BAAI/bge-large-en-v1.5",              "cls",  "", False),
    ("gte_base",   "embed",  "thenlper/gte-base",                   "mean", "", False),
    ("gte_large",  "embed",  "thenlper/gte-large",                  "mean", "", False),
    ("gte_qwen2",  "embed",  "Alibaba-NLP/gte-Qwen2-1.5B-instruct", "last", "", True),
    ("e5_base",    "embed",  "intfloat/e5-base-v2",                 "mean", "query: ", False),
    ("e5_large",   "embed",  "intfloat/e5-large-v2",                "mean", "query: ", False),
]

def specnull(C, rep):
    mu = C.mean(0); Cc = C - mu
    U, S, Vt = np.linalg.svd(Cc, full_matrices=False)
    rng = np.random.RandomState(300+rep)
    G = rng.randn(len(C), len(S)).astype(np.float32)
    G /= G.std(0, keepdims=True) * np.sqrt(len(C))
    return (G*S)@Vt + mu

@torch.no_grad()
def encode(kind, hf, pool, prefix, texts, fp16, bs=32):
    from transformers import AutoTokenizer, AutoModel
    kw = dict(torch_dtype=torch.float16) if fp16 else {}
    tok = AutoTokenizer.from_pretrained(hf)
    mod = AutoModel.from_pretrained(hf, **kw).to(DEV).eval()
    if kind == "causal":
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
    vs = []
    for a in range(0, len(texts), bs):
        b = tok([prefix+t for t in texts[a:a+bs]], return_tensors="pt", padding=True,
                truncation=True, max_length=64).to(DEV)
        h = mod(**b).last_hidden_state
        if kind == "causal" or pool == "last":
            if kind == "causal":
                v = h[:, -1, :]                       # left-padded: last token
            else:                                     # right-padded last non-pad token
                idx = b.attention_mask.sum(1) - 1
                v = h[torch.arange(len(h)), idx]
        elif pool == "cls":
            v = h[:, 0]
        else:
            m = b.attention_mask.unsqueeze(-1).to(h.dtype)
            v = (h*m).sum(1)/m.sum(1).clamp(min=1e-9)
        vs.append(v.float().cpu().numpy())
    del mod; torch.cuda.empty_cache()
    return np.concatenate(vs).astype(np.float32)

def main():
    names = imagenet_class_names()
    prompts = [f"a photo of a {n}" for n in names]
    csv_path = OUT/"exp18_text_nulls.csv"
    rows = pd.read_csv(csv_path).to_dict("records") if csv_path.exists() else []
    done = {r["model"] for r in rows}
    for short, kind, hf, pool, prefix, fp16 in MODELS:
        if short in done:
            print(f"{short}: skip"); continue
        t0 = time.time()
        try:
            X = encode(kind, hf, pool, prefix, prompts, fp16)
        except Exception as e:
            print(f"{short}: ENCODE FAILED: {type(e).__name__}: {str(e)[:150]}")
            continue
        D = squareform(pdist(X, "euclidean"))
        dm, ds = delta_max_from_D(D)
        nulls = []
        for rep in range(3):
            Dn = squareform(pdist(specnull(X, rep), "euclidean"))
            nulls.append(delta_max_from_D(Dn, n_seeds=3)[0])
        nm = float(np.mean(nulls)); nsd = float(np.std(nulls))
        rows.append(dict(model=short, kind=kind, d=X.shape[1], delta=dm, delta_sd=ds,
                         null_mean=nm, null_sd=nsd, excess=dm-nm,
                         time_s=time.time()-t0))
        r = rows[-1]
        print(f"{short:12s} {kind:6s} d={r['d']:5d} delta {dm:.4f}±{ds:.4f} | "
              f"null {nm:.4f}±{nsd:.4f} | excess {dm-nm:+.4f} | {r['time_s']:.0f}s")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
    print("Done")

if __name__ == "__main__":
    main()
