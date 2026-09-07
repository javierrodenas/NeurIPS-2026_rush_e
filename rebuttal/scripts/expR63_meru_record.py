#!/usr/bin/env python3
"""expR63 (R8 of the restructuring brief) -- the hyperbolic-backbone control of expR51 (Table B30: MERU vs its CLIP twin,
S/B/L, CIFAR-100 and ImageNet, image and text modalities; cached embeddings in Platonic/results/meru_cache) under the census
of record and the validated depth test:
  * census excess on the class centroids of the projected embeddings (MERU: space-like hyperboloid coordinates; CLIP: unit
    vectors), Euclidean distances: Haar spectrum-matched null x 99.9th-percentile statistic x 200 replicates (real 10 seeds,
    replicate 5), BH across the 24 cells at merge -- the code of Table 1 (calibrated_delta.py);
  * the same statistic in each model's native metric (MERU: Lorentz distance between tangent-mean centroids; CLIP: angular
    distance), raw only, to show the native and Euclidean readings agree;
  * depth: the anisotropic matched-star test of Table B34 (10 star seeds) on ImageNet (n = 1000, WordNet K = 30: the validated
    regime) and on CIFAR-100 (n = 100, coarse K = 20: reported as unvalidated regime).

    python expR63_meru_record.py --part <model>   (meru_s clip_s meru_b clip_b meru_l clip_l)
    python expR63_meru_record.py --merge          -> expR63_meru_record.csv
"""
import os, sys, time, argparse
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd, torch
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "iclr2027" / "tool")); sys.path.insert(0, str(HERE))
import calibrated_delta as cd
import expR51_meru_control as R51          # frames (coarse-20, WordNet-30), meru path, cache location
from meru import lorentz as L
OUT = Path(os.environ.get("PLATONIC_RESULTS", str(HERE.parents[1] / "rebuttal/results")))
MODELS = ["meru_s", "clip_s", "meru_b", "clip_b", "meru_l", "clip_l"]

def stat_from_D(D, n_seeds, stat="p999", n_quads=500_000):
    diam = D.max(); n = len(D); out = []
    for s in range(n_seeds):
        rng = np.random.RandomState(s); i, j, k, l = (rng.randint(0, n, n_quads) for _ in range(4))
        ok = (i != j) & (i != k) & (i != l) & (j != k) & (j != l) & (k != l); i, j, k, l = i[ok], j[ok], k[ok], l[ok]
        S = np.sort(np.stack([D[i, j] + D[k, l], D[i, k] + D[j, l], D[i, l] + D[j, k]], 1), 1); dfc = (S[:, 2] - S[:, 1]) / 2
        out.append((dfc.max() if stat == "sup" else np.percentile(dfc, 99.9)) / diam)
    return float(np.mean(out)), float(np.std(out))

def native_D(members, labels, n_cls, curv):
    X = torch.tensor(members)
    if curv > 0:
        V = L.log_map0(X, curv); Cm = torch.stack([V[labels == c].mean(0) for c in range(n_cls)]); Cc = L.exp_map0(Cm, curv)
        D = L.pairwise_dist(Cc, Cc, curv).numpy(); D = 0.5 * (D + D.T)
    else:
        Cm = torch.stack([X[labels == c].mean(0) for c in range(n_cls)]); Cm = torch.nn.functional.normalize(Cm, dim=-1)
        D = np.arccos(np.clip((Cm @ Cm.T).numpy(), -1, 1))
    np.fill_diagonal(D, 0); return D

def run_part(key):
    f = OUT / f"expR63_meru_record.part_{key}.csv"; rows = pd.read_csv(f).to_dict("records") if f.exists() else []
    done = {(r["dataset"], r["modality"]) for r in rows}
    for ds in ("cifar100", "imagenet"):
        for mod in ("image", "text"):
            if (ds, mod) in done: continue
            src = R51.CACHE / f"{key}_{ds}_{'img' if mod == 'image' else 'txt'}.npz"
            if not src.exists(): print("missing", src); continue
            t0 = time.time(); d = np.load(src); proj, lab, curv = d["proj"].astype(np.float32), d["labels"], float(d["curv"])
            n_cls = int(lab.max()) + 1; sup = R51.FRAMES[ds](); C = np.stack([proj[lab == c].mean(0) for c in range(n_cls)])
            c = cd.census(C, 200, "haar", "p999"); nat, nat_sd = stat_from_D(native_D(proj, lab, n_cls, curv), 10, "p999")
            dt = cd.depth_test(C, sup, "aniso", 10)
            rows.append(dict(model=key, family=key.split("_")[0], size=key.split("_")[1], dataset=ds, modality=mod, K=int(sup.max() + 1), n=n_cls, d=int(C.shape[1]), curv=curv,
                             delta_999=c["delta"], delta_sd=c["delta_sd"], null_mean=c["null_mean"], null_sd=c["null_sd"], excess=c["excess"], z=c["z"], r_above=c["r_above"], p_left=c["p_left"],
                             delta_999_native=nat, delta_999_native_sd=nat_sd, excessB_real=dt["excessB_real"], excessB_star=dt["excessB_star"], depth=dt["depth_excess"], z_depth=dt["z_depth"],
                             depth_regime="validated" if ds == "imagenet" else "unvalidated", time_s=time.time() - t0))
            pd.DataFrame(rows).to_csv(f, index=False)
            print(f"{key:7s} {ds:9s} {mod:5s} K={rows[-1]['K']} n={n_cls} d999 {c['delta']:.4f} null {c['null_mean']:.4f} exc {c['excess']:+.4f} r={c['r_above']} p={c['p_left']:.3f} | native {nat:.4f} | depth {dt['depth_excess']:+.4f} z {dt['z_depth']:+.2f} ({time.time()-t0:.0f}s)")
    print(f"PART DONE {key}")

def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = p[o] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(r[::-1])[::-1]; out = np.empty(n); out[o] = np.minimum(adj, 1); return out

def merge():
    df = pd.concat([pd.read_csv(p) for p in sorted(OUT.glob("expR63_meru_record.part_*.csv"))], ignore_index=True)
    df["order"] = df.model.map({m: i for i, m in enumerate(MODELS)}); df = df.sort_values(["order", "dataset", "modality"]).drop(columns="order")
    df["p_bh"] = bh(df.p_left); df["genuine_bh"] = df.p_bh <= 0.05; df.to_csv(OUT / "expR63_meru_record.csv", index=False)
    print(f"merged {len(df)} cells; genuine BH {int(df.genuine_bh.sum())}/{len(df)}")
    for (ds, mod), g in df.groupby(["dataset", "modality"]):
        for sz in ("s", "b", "l"):
            m = g[g.model == f"meru_{sz}"]; cl = g[g.model == f"clip_{sz}"]
            if len(m) and len(cl):
                m, cl = m.iloc[0], cl.iloc[0]
                print(f"  {ds:9s} {mod:5s} {sz}: MERU exc {m.excess:+.4f} (r={m.r_above}) native {m.delta_999_native:.4f}/eucl {m.delta_999:.4f} depth z {m.z_depth:+.2f} | CLIP exc {cl.excess:+.4f} (r={cl.r_above}) native {cl.delta_999_native:.4f}/eucl {cl.delta_999:.4f} depth z {cl.z_depth:+.2f} | exc diff {m.excess-cl.excess:+.4f} depth z diff {m.z_depth-cl.z_depth:+.2f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--part"); ap.add_argument("--merge", action="store_true"); A = ap.parse_args()
    merge() if A.merge else run_part(A.part)
