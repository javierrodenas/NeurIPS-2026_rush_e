#!/usr/bin/env python3
"""
ICLR EXP 16 — HierarCaps multi-level evaluation (public commitment to pux6).

Test set: 1000 four-level caption chains (general => ... => specific), COCO val
(Alper & Averbuch-Elor, ECCV 2024). Scope as approved: level-by-level alignment
+ radial ordering, nothing more.

Per model (paper text embedders BGE/E5/GTE + the text towers of the SAME
checkpoints as the vision panel: CLIP-B/32 openai, CLIP-L/14 openai,
SigLIP-B/16 webli), raw (UNnormalized) pooled embeddings:

  (a) RADIAL ORDERING under the paper's exact Poincare map (center mu over all
      4000 captions, s = atanh(1/sqrt2)/p95): per chain Spearman rho(level,
      radius), mean +- sd, % strictly increasing chains (chance 1/24 ~ 4.2%),
      and a within-chain shuffle control.
  (b) SIBLING TRIPLETS at L1 and L2 (leaf embeddings; chains sharing the L1/L2
      caption are siblings): agreement vs 0.5 chance, cosine and Euclidean.
  (c) WITHIN-CHAIN ENTAILMENT per level k in {1,2,3}: cos-dist(leaf, own
      level-k caption) < cos-dist(leaf, random other chain's level-k caption)?
  (d) delta-hat on the 1000 leaf embeddings + spectrum-matched null excess
      (secondary line, 3 reps).

Output: rebuttal/results/exp16_hierarcaps.csv + exp16.log
"""
import os, sys, itertools, time
os.environ.setdefault("HF_HOME", "/home/javi/Platonic/.hf_cache")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd, torch
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

ROOT = Path("/home/javi/Platonic"); OUT = ROOT/"rebuttal/results"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ATANH_T = float(np.arctanh(0.70710678))

MODELS = [
    ("bge_base",  "hf",     "BAAI/bge-base-en-v1.5",             "cls",  ""),
    ("e5_base",   "hf",     "intfloat/e5-base-v2",               "mean", "query: "),
    ("gte_base",  "hf",     "thenlper/gte-base",                 "mean", ""),
    ("clip_b",    "clip",   "openai/clip-vit-base-patch32",      "",     ""),
    ("clip_l",    "clip",   "openai/clip-vit-large-patch14",     "",     ""),
    ("siglip_b",  "siglip", "google/siglip-base-patch16-224",    "",     ""),
]

def delta_max(X, n_quads=500_000, n_seeds=10):
    D = squareform(pdist(X, 'euclidean')); diam = D.max(); n = len(D)
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
def encode(kind, hf, pool, prefix, texts, bs=64):
    if kind == "hf":
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained(hf)
        mod = AutoModel.from_pretrained(hf).to(DEV).eval()
        vs = []
        for a in range(0, len(texts), bs):
            b = tok([prefix+t for t in texts[a:a+bs]], padding=True, truncation=True,
                    max_length=128, return_tensors="pt").to(DEV)
            h = mod(**b).last_hidden_state
            if pool == "cls": v = h[:, 0]
            else:
                m = b.attention_mask.unsqueeze(-1).float()
                v = (h*m).sum(1)/m.sum(1)
            vs.append(v.float().cpu().numpy())
    elif kind == "clip":
        from transformers import CLIPTokenizer, CLIPModel
        tok = CLIPTokenizer.from_pretrained(hf)
        mod = CLIPModel.from_pretrained(hf).to(DEV).eval()
        vs = []
        for a in range(0, len(texts), bs):
            b = tok(texts[a:a+bs], padding=True, truncation=True, max_length=77,
                    return_tensors="pt").to(DEV)
            tm = mod.text_model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
            v = mod.text_projection(tm.pooler_output)
            vs.append(v.float().cpu().numpy())
    else:  # siglip
        from transformers import SiglipTokenizer, SiglipModel
        pr = SiglipTokenizer.from_pretrained(hf)
        mod = SiglipModel.from_pretrained(hf).to(DEV).eval()
        vs = []
        for a in range(0, len(texts), bs):
            b = pr(texts[a:a+bs], padding="max_length", truncation=True,
                   max_length=64, return_tensors="pt").to(DEV)
            tm = mod.text_model(input_ids=b["input_ids"])
            vs.append(tm.pooler_output.float().cpu().numpy())
    del mod; torch.cuda.empty_cache()
    return np.concatenate(vs).astype(np.float32)

def cosd(A, B):
    An = A/np.linalg.norm(A, axis=-1, keepdims=True)
    Bn = B/np.linalg.norm(B, axis=-1, keepdims=True)
    return 1 - (An*Bn).sum(-1)

def main():
    df = pd.read_csv(ROOT/"data/hierarcaps_test.csv")
    chains = [ [c.strip() for c in s.split("=>")] for s in df.captions ]
    assert all(len(c) == 4 for c in chains)
    n = len(chains)
    texts = [c[k] for c in chains for k in range(4)]           # 4 per chain
    idx = lambda i, k: 4*i + k                                  # chain i, level k

    # sibling groups at L1/L2 (only groups with >= 2 chains)
    import collections
    g1 = collections.defaultdict(list); g2 = collections.defaultdict(list)
    for i, c in enumerate(chains):
        g1[c[0]].append(i); g2[c[1]].append(i)
    g1 = {k: v for k, v in g1.items() if len(v) >= 2}
    g2 = {k: v for k, v in g2.items() if len(v) >= 2}
    print(f"chains {n} | L1 sibling groups {len(g1)} | L2 sibling groups {len(g2)}")

    csv_path = OUT/"exp16_hierarcaps.csv"
    rows = pd.read_csv(csv_path).to_dict("records") if csv_path.exists() else []
    done = {r["model"] for r in rows}
    for name, kind, hf, pool, prefix in MODELS:
        if name in done:
            print(f"{name}: already in CSV, skipping"); continue
        t0 = time.time()
        X = encode(kind, hf, pool, prefix, texts)
        d = X.shape[1]

        # (a) radial ordering under the paper map
        mu = X.mean(0); Z = X - mu
        norms = np.linalg.norm(Z, axis=1)
        s = ATANH_T/np.percentile(norms, 95)
        r = np.tanh(0.5*s*norms)                    # Poincare radius, monotone in norm
        R = r.reshape(n, 4)
        rhos = np.array([spearmanr([1,2,3,4], R[i]).statistic for i in range(n)])
        mono = float(np.mean((np.diff(R, axis=1) > 0).all(1)))
        rng = np.random.RandomState(0)
        Rs = R.copy(); [rng.shuffle(row) for row in Rs]
        rho_shuf = float(np.mean([spearmanr([1,2,3,4], Rs[i]).statistic for i in range(n)]))

        # (b) sibling triplets on leaf embeddings
        L = X[[idx(i,3) for i in range(n)]]
        def trip(groups, metric, n_trip=20000, seed=1):
            rr = np.random.RandomState(seed); keys = list(groups); wins = 0; tot = 0
            for _ in range(n_trip):
                k = keys[rr.randint(len(keys))]; g = groups[k]
                a, b = rr.choice(g, 2, replace=False)
                c = rr.randint(n)
                while c in g: c = rr.randint(n)
                if metric == "cos":
                    dab = cosd(L[a], L[b]); dac = cosd(L[a], L[c])
                else:
                    dab = np.linalg.norm(L[a]-L[b]); dac = np.linalg.norm(L[a]-L[c])
                wins += dab < dac; tot += 1
            return wins/tot
        t1c, t1e = trip(g1, "cos"), trip(g1, "euc")
        t2c, t2e = trip(g2, "cos"), trip(g2, "euc")

        # (c) within-chain entailment per level (cosine)
        ent = []
        rr = np.random.RandomState(2)
        for k in range(3):
            A = X[[idx(i,3) for i in range(n)]]
            P = X[[idx(i,k) for i in range(n)]]
            wins = 0; tot = 0
            for i in range(n):
                for _ in range(10):
                    j = rr.randint(n)
                    while j == i: j = rr.randint(n)
                    wins += cosd(A[i], P[i]) < cosd(A[i], P[j]); tot += 1
            ent.append(wins/tot)

        # (d) delta on leaves + spectrum-null excess
        dm, dsd = delta_max(L)
        nulls = [delta_max(specnull(L, rep), n_seeds=3)[0] for rep in range(3)]
        excess = dm - float(np.mean(nulls))

        rows.append(dict(model=name, d=d,
            rho_mean=float(np.nanmean(rhos)), rho_sd=float(np.nanstd(rhos)),
            pct_monotone=mono, rho_shuffle=rho_shuf,
            trip_L1_cos=t1c, trip_L1_euc=t1e, trip_L2_cos=t2c, trip_L2_euc=t2e,
            ent_L1=ent[0], ent_L2=ent[1], ent_L3=ent[2],
            delta_leaves=dm, delta_sd=dsd, null_mean=float(np.mean(nulls)), excess=excess,
            time_s=time.time()-t0))
        r0 = rows[-1]
        print(f"{name:9s} d={d:4d} | rho {r0['rho_mean']:+.3f}±{r0['rho_sd']:.2f} mono {100*mono:.1f}% (shuf {rho_shuf:+.3f}) | "
              f"trip L1 {t1c:.3f}/{t1e:.3f} L2 {t2c:.3f}/{t2e:.3f} | ent {ent[0]:.3f}/{ent[1]:.3f}/{ent[2]:.3f} | "
              f"delta {dm:.3f} null {np.mean(nulls):.3f} exc {excess:+.3f} | {r0['time_s']:.0f}s")
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    print("Done")

if __name__ == "__main__":
    main()
