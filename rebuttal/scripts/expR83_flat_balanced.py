#!/usr/bin/env python3
"""expR83 (ninth review, CPU): the decoupled false-alarm rate of the balanced frame.

The flat synthetic control of expR79 (30 iid hubs assigned by the WordNet-30 frame, offsets at ViT-L's real within/between ratio,
the whole cloud re-spectred to ViT-L's centered singular values; 5 seeds, the same clouds as expR79's flat rows) read on the balanced
frame (expR67: the complete-linkage 30-cut of the WordNet distance matrix, as in expR77): the depth test with the matched anisotropic
star (10 star seeds) on the intact cloud, and the decoupling control (10 seeds) on each, i.e. 50 decoupled runs, so the balanced
frame gets the same decoupled false-alarm rate as the 8-of-50 of the frame of record.
Output: expR83_flat_balanced.csv (rows) and expR83_flat_balanced_summary.csv (rates).
Two constructions: --built wn30 (expR79 clouds, hubs by the WordNet-30 frame, read on the balanced frame: the brief's literal construction, frame mismatch included)
and --built balanced (hubs assigned by the balanced frame itself, read on it: the matched construction, as the 8-of-50 is matched to WordNet-30).
Usage: python expR83_flat_balanced.py [--built wn30|balanced] [--seeds 0 1 2 3 4]"""
import os, sys, time, json, argparse
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "ICLR2027" / "tool"))
import expR56_depth_variants as R56, expR74_decoupling as R74, expR79_synthetic_deep_poincare as R79
from sklearn.cluster import AgglomerativeClustering
ROOT = R56.ROOT; OUT = R56.OUT

def balanced_frame():
    WN = np.load(ROOT / "results/features/imagenet/imagenet1k_wordnet_dist.npy")
    link = json.load(open(OUT / "expR67_frame_choice.json"))["chosen"]   # the balanced frame of record (expR67), as in expR77
    return AgglomerativeClustering(n_clusters=30, metric="precomputed", linkage=link).fit_predict(WN).astype(np.int64)

def main(A):
    cuts = {int(k): np.asarray(v) for k, v in json.load(open(OUT / "expR76_frames.json")).items()}
    ratio = float(pd.read_csv(OUT / "expR64b_wn30_summary.csv").set_index("model").loc["i21k_l", "ratio_real"]); C_real = R79.real_vitl(); sup_bal = balanced_frame()
    if A.built == "balanced": cuts = {30: sup_bal, 6: cuts[6], 2: cuts[2]}   # matched construction: the flat hubs assigned by the balanced frame itself (frame-mismatch excluded)
    f = OUT / f"expR83_flat_balanced.part_{A.built}_{'_'.join(map(str, A.seeds))}.csv"; rows = pd.read_csv(f).to_dict("records") if f.exists() else []
    done = {(int(r["seed"]), int(r["dec_seed"])) for r in rows}
    for s in A.seeds:
        X = R79.synthetic(C_real, cuts, ratio, s, deep=False)
        if (s, -1) not in done:
            t0 = time.time(); r = R56.depth_test(X, sup_bal, "aniso"); rows.append(dict(cloud="synthetic_flat_vitl_spectrum", built=A.built, frame="balanced", seed=s, dec_seed=-1, depth=r["depth_excess"], z=r["z_depth"], star_sd=r["excessB_star_sd"], time_s=time.time() - t0))
            print(f"flat seed {s} balanced intact z {r['z_depth']:+.2f} ({time.time() - t0:.0f}s)"); pd.DataFrame(rows).to_csv(f, index=False)
        for d in range(10):
            if (s, d) in done: continue
            t0 = time.time(); r = R56.depth_test(R74.decouple(X, sup_bal, d), sup_bal, "aniso"); rows.append(dict(cloud="synthetic_flat_vitl_spectrum", built=A.built, frame="balanced", seed=s, dec_seed=d, depth=r["depth_excess"], z=r["z_depth"], star_sd=r["excessB_star_sd"], time_s=time.time() - t0))
            pd.DataFrame(rows).to_csv(f, index=False)
        zs = [r["z"] for r in rows if r["seed"] == s and r["dec_seed"] >= 0]; print(f"flat seed {s} balanced decoupled z mean {np.mean(zs):+.2f} fired {int(np.sum(np.array(zs) <= -2))}/10")
    print("Done part", A.seeds)

def merge():
    import glob
    df = pd.concat([pd.read_csv(p) for p in glob.glob(str(OUT / "expR83_flat_balanced.part_*.csv"))]); df["built"] = df.get("built", "wn30").fillna("wn30")
    df = df.drop_duplicates(subset=["built", "seed", "dec_seed"]).sort_values(["built", "seed", "dec_seed"]); df.to_csv(OUT / "expR83_flat_balanced.csv", index=False)
    S = []
    for built, g in df.groupby("built"):
        dec = g[g.dec_seed >= 0]; it = g[g.dec_seed < 0]
        S.append(dict(built=built, frame="balanced", n_intact=len(it), intact_fired=int((it.z <= -2).sum()), intact_z_mean=float(it.z.mean()), n_decoupled=len(dec), decoupled_fired=int((dec.z <= -2).sum()), decoupled_rate=float((dec.z <= -2).mean()), dec_z_mean=float(dec.z.mean()), dec_z_min=float(dec.z.min()), dec_z_max=float(dec.z.max())))
    S = pd.DataFrame(S)
    S.to_csv(OUT / "expR83_flat_balanced_summary.csv", index=False); print(S.to_string())

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4]); ap.add_argument("--built", choices=["wn30", "balanced"], default="wn30", help="frame that assigns the flat hubs: wn30 = the brief's literal construction (expR79 clouds), balanced = matched to the frame read"); ap.add_argument("--merge", action="store_true"); A = ap.parse_args()
    merge() if A.merge else main(A)
