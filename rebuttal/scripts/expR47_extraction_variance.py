#!/usr/bin/env python3
"""expR47 (round-9 question): extraction-to-extraction variability of the text census.
GPT-2 M and OLMo-1B (the two models whose delta-hat moved most between exp18 and expR44),
same 1000 prompts, encoded under batch size {1, 16, 32} x precision {fp32, fp16}
(bs=1 has no padding). Reports delta_norm (10 seeds) per setting, so padding/batch
composition vs precision can be told apart. Output: expR47_extraction_variance.csv."""
import os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd, torch
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
OUT = Path(__file__).resolve().parents[1]/"results"; DEV = "cuda"
def imagenet_class_names():
    from timm.data import ImageNetInfo
    info = ImageNetInfo(); return [info.index_to_description(i).split(",")[0].strip() for i in range(1000)]
def delta_norm(X, n_seeds=10):
    D = squareform(pdist(X, "euclidean")); diam = D.max(); n = len(D); out = []
    for s in range(n_seeds):
        rng = np.random.RandomState(s); i, j, k, l = (rng.randint(0, n, 500_000) for _ in range(4))
        ok = (i!=j)&(i!=k)&(i!=l)&(j!=k)&(j!=l)&(k!=l); i,j,k,l = i[ok],j[ok],k[ok],l[ok]
        S = np.sort(np.stack([D[i,j]+D[k,l], D[i,k]+D[j,l], D[i,l]+D[j,k]], 1), 1); out.append(((S[:,2]-S[:,1])/2).max()/diam)
    return float(np.mean(out)), float(np.std(out))
@torch.no_grad()
def encode(hf, texts, fp16, bs):
    from transformers import AutoTokenizer, AutoModel
    kw = dict(torch_dtype=torch.float16) if fp16 else {}
    tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True); mod = AutoModel.from_pretrained(hf, trust_remote_code=True, **kw).to(DEV).eval()
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"; vs = []
    for a in range(0, len(texts), bs):
        b = tok(texts[a:a+bs], return_tensors="pt", padding=True, truncation=True, max_length=64).to(DEV)
        h = mod(**b).last_hidden_state; vs.append(h[:, -1, :].float().cpu().numpy())
    del mod; torch.cuda.empty_cache(); return np.concatenate(vs).astype(np.float32)
prompts = [f"a photo of a {n}" for n in imagenet_class_names()]
rows = []
for short, hf in [("gpt2_m","gpt2-medium"), ("olmo_1b","allenai/OLMo-1B-hf")]:
    X1 = None
    for fp16 in (False, True):
        for bs in (1, 16, 32):
            t0 = time.time(); X = encode(hf, prompts, fp16, bs); dr, sd = delta_norm(X)
            if X1 is None: X1 = X
            cos = float(np.mean(np.sum(X*X1,1)/(np.linalg.norm(X,axis=1)*np.linalg.norm(X1,axis=1)+1e-9)))
            rows.append(dict(model=short, fp16=int(fp16), batch=bs, delta=dr, delta_sd=sd, mean_cos_to_bs1_fp32=cos, time_s=time.time()-t0))
            print(f"{short:8s} fp16={int(fp16)} bs={bs:2d} delta {dr:.4f}±{sd:.4f} cos-to-ref {cos:.4f} ({time.time()-t0:.0f}s)")
            pd.DataFrame(rows).to_csv(OUT/"expR47_extraction_variance.csv", index=False)
print("Done expR47")
