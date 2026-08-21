#!/usr/bin/env python3
"""
ICLR EXP 18b — why does the matched null rise with scale? (W4 of the v2 review)

For the GPT-2 family (the scale-emergence story) plus the borderline-verdict
cells (OLMo-1B, GTE-base): re-encode the 1000 class prompts (saving embeddings
this time), and compute
  - participation ratio of the centroid covariance (effective rank),
  - spectrum-matched null excess (sanity vs exp18),
  - PC-permutation null excess (the second null, shown not just mentioned),
  - z-scores for both nulls.
If "genuine" merely tracked anisotropy, excess and effective rank would move
together; the GPT-2 ladder tests this directly.

Output: exp18b_text_anisotropy.csv + embeddings in results/text_embeddings/.
"""
import os, sys, time
os.environ.setdefault("HF_HOME", "/home/javi/Platonic/.hf_cache")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd, torch
from pathlib import Path
from scipy.spatial.distance import pdist, squareform

ROOT = Path("/home/javi/Platonic"); OUT = ROOT/"rebuttal/results"
EMB = ROOT/"results/text_embeddings"; EMB.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT/"rebuttal/scripts"))
from exp4_prompt_variation import imagenet_class_names, delta_max_from_D
from exp18_text_nulls import encode, specnull

MODELS = [
    ("gpt2",     "causal", "gpt2",                  None,   "", False),
    ("gpt2_m",   "causal", "gpt2-medium",           None,   "", False),
    ("gpt2_l",   "causal", "gpt2-large",            None,   "", False),
    ("gpt2_xl",  "causal", "gpt2-xl",               None,   "", True),
    ("olmo_1b",  "causal", "allenai/OLMo-1B-hf",    None,   "", True),
    ("gte_base", "embed",  "thenlper/gte-base",     "mean", "", False),
]

def pcperm(C, rep):
    """PC-permutation null: shuffle each principal component across classes."""
    mu = C.mean(0); Cc = C - mu
    U, S, Vt = np.linalg.svd(Cc, full_matrices=False)
    rng = np.random.RandomState(500+rep)
    Up = U.copy()
    for j in range(Up.shape[1]):
        rng.shuffle(Up[:, j])
    return (Up*S)@Vt + mu

def erank(C):
    Cc = C - C.mean(0)
    lam = np.linalg.svd(Cc, compute_uv=False)**2
    return float(lam.sum()**2/ (lam**2).sum())

def main():
    prompts = [f"a photo of a {n}" for n in imagenet_class_names()]
    csv_path = OUT/"exp18b_text_anisotropy.csv"
    rows = pd.read_csv(csv_path).to_dict("records") if csv_path.exists() else []
    done = {r["model"] for r in rows}
    for short, kind, hf, pool, prefix, fp16 in MODELS:
        if short in done:
            print(f"{short}: skip"); continue
        t0 = time.time()
        try:
            X = encode(kind, hf, pool, prefix, prompts, fp16)
        except Exception as e:
            print(f"{short}: ENCODE FAILED: {type(e).__name__}: {str(e)[:140]}"); continue
        np.save(EMB/f"{short}_imagenet_prompts.npy", X)
        D = squareform(pdist(X, "euclidean"))
        dm, dsd = delta_max_from_D(D)
        spec = [delta_max_from_D(squareform(pdist(specnull(X, r), "euclidean")), n_seeds=3)[0] for r in range(3)]
        pcp  = [delta_max_from_D(squareform(pdist(pcperm(X, r), "euclidean")), n_seeds=3)[0] for r in range(3)]
        sm, ssd = float(np.mean(spec)), float(np.std(spec))
        pm, psd = float(np.mean(pcp)), float(np.std(pcp))
        er = erank(X)
        z_s = (dm-sm)/max(np.sqrt(ssd**2+dsd**2), 1e-9)
        z_p = (dm-pm)/max(np.sqrt(psd**2+dsd**2), 1e-9)
        rows.append(dict(model=short, d=X.shape[1], erank=er, delta=dm, delta_sd=dsd,
                         null_spec=sm, null_spec_sd=ssd, exc_spec=dm-sm, z_spec=z_s,
                         null_pcperm=pm, null_pcperm_sd=psd, exc_pcperm=dm-pm, z_pcperm=z_p,
                         time_s=time.time()-t0))
        print(f"{short:9s} d={X.shape[1]:5d} erank={er:6.1f} | delta {dm:.4f} | "
              f"spec-null {sm:.4f} exc {dm-sm:+.4f} (z={z_s:+.1f}) | "
              f"pcperm-null {pm:.4f} exc {dm-pm:+.4f} (z={z_p:+.1f}) | {rows[-1]['time_s']:.0f}s")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
    print("Done")

if __name__ == "__main__":
    main()
