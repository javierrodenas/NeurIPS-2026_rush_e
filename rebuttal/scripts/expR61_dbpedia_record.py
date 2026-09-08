#!/usr/bin/env python3
"""expR61 -- DBpedia Classes (219 leaf classes, three levels) for the three sentence embedders of exp14, re-read under the
census of record: Haar spectrum-matched null x 99.9th-percentile statistic x 200 replicates (calibrated_delta.py), plus
the two other constructions for the 2x2 (Haar x sup, Gauss x p99.9). Embeddings follow exp14 exactly (raw pooled
embeddings: BGE-base CLS, E5-base mean + "query: ", GTE-base mean; 100 train texts per leaf class, RandomState(0)
subsample); the 219 centroids and the L2 group labels are cached in $PLATONIC_ROOT/results/text_cache/dbpedia_{model}.npz
so the census never needs the GPU again. BH across the three embedders.
Output: rebuttal/results/expR61_dbpedia_record.csv
"""
import os, sys, time
os.environ.setdefault("HF_HOME", "/media/HDD_4TB_2/javi/hf_cache")
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "ICLR2027" / "tool"))
import calibrated_delta as cd
ROOT = Path(os.environ.get("PLATONIC_ROOT", "/media/HDD_4TB_2/javi/Platonic"))
OUT = Path(os.environ.get("PLATONIC_RESULTS", str(HERE.parents[1] / "rebuttal/results")))
TC = ROOT / "results/text_cache"; TC.mkdir(parents=True, exist_ok=True)
N_TRAIN = 100
MODELS = [("bge_base", "BAAI/bge-base-en-v1.5", "cls", ""), ("e5_base", "intfloat/e5-base-v2", "mean", "query: "), ("gte_base", "thenlper/gte-base", "mean", "")]

def load_dbpedia():
    from huggingface_hub import hf_hub_download
    tr = pd.read_csv(hf_hub_download("DeveloperOats/DBPedia_Classes", "DBPEDIA_train.csv", repo_type="dataset"))
    rng = np.random.RandomState(0)
    tr = tr.groupby("l3", group_keys=False).apply(lambda g: g.sample(min(N_TRAIN, len(g)), random_state=rng))
    l3 = sorted(tr.l3.unique()); l3i = {c: i for i, c in enumerate(l3)}
    l2_of = tr.drop_duplicates("l3").set_index("l3").l2.to_dict()
    _, sup = np.unique(np.array([l2_of[c] for c in l3]), return_inverse=True)
    return tr.text.tolist(), tr.l3.map(l3i).values, sup, len(l3)

def encode(hf, pool, prefix, texts, bs=128):
    import torch
    from transformers import AutoTokenizer, AutoModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(hf); mod = AutoModel.from_pretrained(hf).to(dev).eval(); vs = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = tok([prefix + t for t in texts[i:i+bs]], return_tensors="pt", padding=True, truncation=True, max_length=256).to(dev)
            h = mod(**enc).last_hidden_state
            if pool == "cls": v = h[:, 0]
            else:
                m = enc["attention_mask"].unsqueeze(-1).float(); v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            vs.append(v.float().cpu().numpy())
    del mod; torch.cuda.empty_cache()
    return np.concatenate(vs, 0)

def centroids(short, hf, pool, prefix):
    f = TC / f"dbpedia_{short}.npz"
    if f.exists(): d = np.load(f); return d["centroids"], d["sup"]
    texts, y, sup, K = load_dbpedia(); t0 = time.time()
    X = encode(hf, pool, prefix, texts); C = np.stack([X[y == c].mean(0) for c in range(K)])
    np.savez(f, centroids=C.astype(np.float32), sup=sup); print(f"{short}: encoded {len(texts)} texts -> {C.shape} ({time.time()-t0:.0f}s)")
    return C, sup

def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = p[o] * n / np.arange(1, n+1)
    adj = np.minimum.accumulate(r[::-1])[::-1]; out = np.empty(n); out[o] = np.minimum(adj, 1); return out

def main():
    rows = []
    for short, hf, pool, prefix in MODELS:
        C, sup = centroids(short, hf, pool, prefix); t0 = time.time()
        rec = cd.census(C, 200, "haar", "p999"); hs = cd.census(C, 200, "haar", "sup"); gp = cd.census(C, 200, "gauss", "p999")
        rows.append(dict(model=short, n=rec["n"], d=rec["d"], delta_999=rec["delta"], delta_sd=rec["delta_sd"], null_mean=rec["null_mean"], null_sd=rec["null_sd"],
                         excess=rec["excess"], r_above=rec["r_above"], p_left=rec["p_left"],
                         excess_haar_sup=hs["excess"], r_haar_sup=hs["r_above"], p_haar_sup=hs["p_left"],
                         excess_gauss_p999=gp["excess"], r_gauss_p999=gp["r_above"], p_gauss_p999=gp["p_left"], time_s=time.time()-t0))
        print(f"{short:9s} record: d999 {rec['delta']:.4f} null {rec['null_mean']:.4f} exc {rec['excess']:+.4f} r={rec['r_above']} p={rec['p_left']:.3f} | Hs exc {hs['excess']:+.4f} r={hs['r_above']} | Gp exc {gp['excess']:+.4f} r={gp['r_above']} ({time.time()-t0:.0f}s)")
    df = pd.DataFrame(rows); df["p_bh"] = bh(df.p_left); df["genuine_bh"] = df.p_bh <= 0.05
    df.to_csv(OUT / "expR61_dbpedia_record.csv", index=False); print(df[["model", "excess", "r_above", "p_left", "p_bh", "genuine_bh"]].to_string()); print("DONE expR61")

if __name__ == "__main__":
    main()
