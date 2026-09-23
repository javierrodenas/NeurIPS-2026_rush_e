#!/usr/bin/env python3
"""expR77 (parallel track, positive control): the tests of the brief on the frozen checkpoint and on the two fine-tuned models of
expR76 (leaf CE vs leaf CE + hierarchical CE at the 30/6/2 cuts), all under the record:
  * census excess on the ImageNet centroids (100 images per class): centered Haar null (Q orthogonal to 1, as expR75), 99.9th-percentile
    statistic, 200 replicates, left-tail p;
  * depth test with the matched anisotropic star (expR56) at the WordNet 30-cut of record and on the balanced frame (expR67: the
    complete-linkage 30-cut), z against the combined spread, 10 star seeds;
  * decoupling control of expR74 (real hubs kept, each cluster's offsets Haar-rotated, 10 seeds) on both frames.
Success criterion (fixed in the brief): the hierarchical model is certified (z <= -2) under both frames and still fires with the
offsets decoupled, while the leaf-CE model and the frozen checkpoint do not. Output: expR77_positive_control.csv and a plain verdict.
Usage: python expR77_positive_control_tests.py [--models frozen ce_seed0 hier_seed0] [--n_star 10] [--n_dec 10]"""
import os, sys, time, argparse, json
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "ICLR2027" / "tool"))
import calibrated_delta as cd, expR56_depth_variants as R56, expR74_decoupling as R74, expR75_census_centered_haar as R75
from sklearn.cluster import AgglomerativeClustering
ROOT = R56.ROOT; OUT = R56.OUT; CACHE = R56.CACHE
FILES = {"frozen": "i21k_b_imagenet_train.npz", "ce_seed0": "vitb_ft_ce_seed0_imagenet_train.npz", "hier_seed0": "vitb_ft_hier_seed0_imagenet_train.npz", "ce_seed1": "vitb_ft_ce_seed1_imagenet_train.npz", "hier_seed1": "vitb_ft_hier_seed1_imagenet_train.npz"}

def centroids(f):
    d = np.load(CACHE / f); X = d["features"].astype(np.float32); y = d["labels"].astype(np.int64)
    return np.stack([X[y == c].mean(0) for c in range(int(y.max()) + 1)])

def frames_():
    WN = np.load(ROOT / "results/features/imagenet/imagenet1k_wordnet_dist.npy")
    rec = AgglomerativeClustering(n_clusters=30, metric="precomputed", linkage="average").fit_predict(WN).astype(int)
    bal_link = json.load(open(OUT / "expR67_frame_choice.json"))["chosen"]
    bal = AgglomerativeClustering(n_clusters=30, metric="precomputed", linkage=bal_link).fit_predict(WN).astype(int)
    return {"wn30": rec, "wn30bal": bal}

def main(A):
    cd.haarnull = R75.null_haar_centered   # the record: centered Haar null
    FR = frames_(); rows = []
    for m in A.models:
        f = FILES[m]
        if not (CACHE / f).exists(): print(f"{m}: cache {f} not found, skipped"); continue
        C = centroids(f); t0 = time.time()
        cen = cd.census(C, 200, "haar", "p999"); row = dict(model=m, n=len(C), delta=cen["delta"], excess=cen["excess"], null_sd=cen["null_sd"], r_above=cen["r_above"], p_left=cen["p_left"])
        for fname, sup in FR.items():
            res = R56.depth_test(C, sup, "aniso", n_star=A.n_star); row[f"depth_{fname}"] = res["depth_excess"]; row[f"z_{fname}"] = res["z_depth"]
            zs = [R56.depth_test(R74.decouple(C, sup, s), sup, "aniso", n_star=A.n_star)["z_depth"] for s in range(A.n_dec)]
            row[f"zdec_mean_{fname}"] = float(np.mean(zs)); row[f"zdec_sd_{fname}"] = float(np.std(zs, ddof=1)); row[f"dec_frac_cert_{fname}"] = float(np.mean(np.array(zs) <= -2))
        row["time_s"] = time.time() - t0; rows.append(row)
        print(f"{m:10s} excess {row['excess']:+.4f} (r {row['r_above']}/200, p {row['p_left']:.3f}) | z wn30 {row['z_wn30']:+.2f} bal {row['z_wn30bal']:+.2f} | decoupled z wn30 {row['zdec_mean_wn30']:+.2f} bal {row['zdec_mean_wn30bal']:+.2f} ({row['time_s']:.0f}s)")
        pd.DataFrame(rows).to_csv(OUT / (f"expR77_positive_control.part_{A.part}.csv" if A.part else "expR77_positive_control.csv"), index=False)
    if A.part: print("PART DONE", A.part); return   # a shard (seed 1, 2026-09-23): merged into the main file and the verdict by --merge, so the seed-0 rows of record are never overwritten
    S = {r["model"]: r for r in rows}
    def cert(r, fr): return r[f"z_{fr}"] <= -2
    def fires_dec(r, fr): return r[f"zdec_mean_{fr}"] <= -2
    verdict = []
    for seed in ("seed0", "seed1"):
        h, c = S.get(f"hier_{seed}"), S.get(f"ce_{seed}")
        if h is None: continue
        ok_h = all(cert(h, fr) for fr in FR) and all(fires_dec(h, fr) for fr in FR)
        ok_c = c is not None and not any(cert(c, fr) for fr in FR)
        ok_f = "frozen" in S and not any(cert(S["frozen"], fr) for fr in FR)
        verdict.append(dict(seed=seed, hier_certified_both_frames=all(cert(h, fr) for fr in FR), hier_fires_decoupled_both=all(fires_dec(h, fr) for fr in FR), ce_not_certified=ok_c, frozen_not_certified=ok_f, criterion_met=bool(ok_h and ok_c and ok_f)))
    json.dump(dict(rows=rows, verdict=verdict, criterion="hier certified (z<=-2) under both frames and still fires with offsets decoupled; ce and frozen not certified"), open(OUT / "expR77_positive_control_verdict.json", "w"), indent=1)
    for v in verdict: print("VERDICT", v)

def merge():
    import glob
    main_f = OUT / "expR77_positive_control.csv"; parts = sorted(glob.glob(str(OUT / "expR77_positive_control.part_*.csv")))
    df = pd.concat([pd.read_csv(main_f)] + [pd.read_csv(f) for f in parts]).drop_duplicates(subset=["model"], keep="last")
    order = ["frozen", "ce_seed0", "hier_seed0", "ce_seed1", "hier_seed1"]; df["_o"] = df.model.map({m: i for i, m in enumerate(order)}); df = df.sort_values("_o").drop(columns="_o"); df.to_csv(main_f, index=False)
    rows = df.to_dict("records"); S = {r["model"]: r for r in rows}; FR = ("wn30", "wn30bal")
    def cert(r, fr): return r[f"z_{fr}"] <= -2
    def fires_dec(r, fr): return r[f"zdec_mean_{fr}"] <= -2
    verdict = []
    for seed in ("seed0", "seed1"):
        h, c = S.get(f"hier_{seed}"), S.get(f"ce_{seed}")
        if h is None: continue
        ok_h = all(cert(h, fr) for fr in FR) and all(fires_dec(h, fr) for fr in FR)
        ok_c = c is not None and not any(cert(c, fr) for fr in FR)
        ok_f = "frozen" in S and not any(cert(S["frozen"], fr) for fr in FR)
        verdict.append(dict(seed=seed, hier_certified_both_frames=all(cert(h, fr) for fr in FR), hier_fires_decoupled_both=all(fires_dec(h, fr) for fr in FR), ce_not_certified=ok_c, frozen_not_certified=ok_f, criterion_met=bool(ok_h and ok_c and ok_f)))
    json.dump(dict(rows=rows, verdict=verdict, criterion="hier certified (z<=-2) under both frames and still fires with offsets decoupled; ce and frozen not certified"), open(OUT / "expR77_positive_control_verdict.json", "w"), indent=1)
    print(df.round(3).to_string())
    for v in verdict: print("VERDICT", v)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", default=["frozen", "ce_seed0", "hier_seed0"]); ap.add_argument("--part", default=None, help="write a shard file instead of the main csv (merge with --merge)"); ap.add_argument("--merge", action="store_true"); ap.add_argument("--n_star", type=int, default=10); ap.add_argument("--n_dec", type=int, default=10); A = ap.parse_args()
    merge() if A.merge else main(A)
