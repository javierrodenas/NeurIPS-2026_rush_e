#!/usr/bin/env python3
"""expR81 (seventh review, priority 1d, CPU): power of the depth test for a deep hierarchy at every backbone's own noise level.

The deep synthetic hierarchy of expR79 (three nested levels 2 / 6 / 30 over the hubs of the WordNet frame of record) is built for each
of the 12 ImageNet backbones with that backbone's own centered spectrum (expR79.synthetic re-spectres the synthetic cloud to the real
one) and its own within/between ratio (expR64b_wn30_summary.csv, ratio_real), 5 seeds each; the depth test with the matched anisotropic
star at K = 30 (10 star seeds) on the intact cloud, and the decoupling control of expR74 (real hubs kept, offsets Haar-rotated, 10
seeds) on each. Every row keeps the excess B of the cloud and of its star, the star spread and z, so the change of the reading under
decoupling can be decomposed (star spread against excess).
Output: expR81_deep_per_backbone.part_<name>.csv per shard; --merge writes expR81_deep_per_backbone.csv (all rows),
expR81_deep_per_backbone_summary.csv (per backbone: power over the 5 seeds, decoupled power over the 50 runs, mean excess B and star
spread before and after decoupling) and expR81_deep_per_backbone_families.csv (power pooled per family).
Usage: python expR81_deep_per_backbone.py --part <name> --models m1 m2 [--seeds 0 1 2 3 4] ; python expR81_deep_per_backbone.py --merge"""
import os, sys, time, json, argparse, glob
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "ICLR2027" / "tool"))
import expR56_depth_variants as R56, expR74_decoupling as R74, expR79_synthetic_deep_poincare as R79
OUT = R56.OUT; CACHE = R56.CACHE
MODELS = ["i21k_t", "i21k_s", "i21k_b", "i21k_l", "dinov1_b", "dinov2_s", "dinov2_b", "dinov2_l", "dinov2_g", "clip_b", "clip_l", "siglip_b"]
FAMILY = {m: ("supervised ViT" if m.startswith("i21k") else ("DINO" if m.startswith("dino") else "contrastive")) for m in MODELS}
N_DEC = 10

def centroids(m):
    d = np.load(CACHE / f"{m}_imagenet_train.npz"); X = d["features"].astype(np.float32); y = d["labels"]
    return np.stack([X[y == c].mean(0) for c in range(1000)])

def run(part, models, seeds):
    cuts = {int(k): np.asarray(v) for k, v in json.load(open(OUT / "expR76_frames.json")).items()}; sup = cuts[30]
    ratio = pd.read_csv(OUT / "expR64b_wn30_summary.csv").set_index("model").ratio_real
    csv_path = OUT / f"expR81_deep_per_backbone.part_{part}.csv"; rows = list(pd.read_csv(csv_path).to_dict("records")) if csv_path.exists() else []
    done = {(r["model"], int(r["seed"])) for r in rows}
    for m in models:
        C_real = centroids(m); rr = float(ratio[m])
        for seed in seeds:
            if (m, seed) in done: continue
            t0 = time.time(); X = R79.synthetic(C_real, cuts, rr, seed, deep=True); res = R56.depth_test(X, sup, "aniso")
            base = dict(model=m, family=FAMILY[m], seed=seed, ratio=rr, dim=int(C_real.shape[1]), K=30)
            rows.append(dict(**base, kind="intact", dec_seed=-1, excessB_real=res["excessB_real"], excessB_star=res["excessB_star"], star_sd=res["excessB_star_sd"], depth_excess=res["depth_excess"], z_depth=res["z_depth"], time_s=time.time() - t0))
            for s in range(N_DEC):
                t1 = time.time(); rd = R56.depth_test(R74.decouple(X, sup, s), sup, "aniso")
                rows.append(dict(**base, kind="decoupled", dec_seed=s, excessB_real=rd["excessB_real"], excessB_star=rd["excessB_star"], star_sd=rd["excessB_star_sd"], depth_excess=rd["depth_excess"], z_depth=rd["z_depth"], time_s=time.time() - t1))
            zs = [r["z_depth"] for r in rows if r["model"] == m and r["seed"] == seed and r["kind"] == "decoupled"]
            print(f"{m:9s} seed {seed} ratio {rr:.2f} intact z {res['z_depth']:+.2f} (B {res['depth_excess']:+.4f}, star sd {res['excessB_star_sd']:.4f}) | decoupled z mean {np.mean(zs):+.2f} cert {np.mean(np.array(zs) <= -2):.1f} ({time.time() - t0:.0f}s)")
            pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Done part {part}")

def merge():
    parts = sorted(glob.glob(str(OUT / "expR81_deep_per_backbone.part_*.csv")))
    df = pd.concat([pd.read_csv(p) for p in parts]).drop_duplicates(subset=["model", "seed", "kind", "dec_seed"]).sort_values(["model", "seed", "kind", "dec_seed"]); df.to_csv(OUT / "expR81_deep_per_backbone.csv", index=False)
    it = df[df.kind == "intact"]; dc = df[df.kind == "decoupled"]; S = []
    for m in MODELS:
        a = it[it.model == m]; b = dc[dc.model == m]
        if not len(a): continue
        S.append(dict(model=m, family=FAMILY[m], ratio=float(a.ratio.iloc[0]), dim=int(a.dim.iloc[0]), n_seeds=len(a), power=float((a.z_depth <= -2).mean()), z_mean=float(a.z_depth.mean()),
                      dec_runs=len(b), dec_power=float((b.z_depth <= -2).mean()) if len(b) else np.nan, dec_z_mean=float(b.z_depth.mean()) if len(b) else np.nan,
                      B_intact=float(a.depth_excess.mean()), B_decoupled=float(b.depth_excess.mean()) if len(b) else np.nan, star_sd_intact=float(a.star_sd.mean()), star_sd_decoupled=float(b.star_sd.mean()) if len(b) else np.nan,
                      excessB_real_intact=float(a.excessB_real.mean()), excessB_real_decoupled=float(b.excessB_real.mean()) if len(b) else np.nan, excessB_star_intact=float(a.excessB_star.mean()), excessB_star_decoupled=float(b.excessB_star.mean()) if len(b) else np.nan))
    S = pd.DataFrame(S); S.to_csv(OUT / "expR81_deep_per_backbone_summary.csv", index=False)
    F = it.groupby("family").agg(n_backbones=("model", "nunique"), n_runs=("z_depth", "size"), power=("z_depth", lambda z: float((z <= -2).mean())), min_backbone_power=("model", lambda mm: float(min(S.set_index("model").loc[x, "power"] for x in set(mm))))).reset_index()
    F["power_ge_0_8"] = F.power >= 0.8; F.to_csv(OUT / "expR81_deep_per_backbone_families.csv", index=False)
    print(S.round(3).to_string()); print(F.round(3).to_string()); print("SUMMARY expR81: every family >= 0.8:", bool(F.power_ge_0_8.all()), "| backbones done", len(S), "of 12")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--part", default=None); ap.add_argument("--models", nargs="+", default=MODELS); ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4]); ap.add_argument("--merge", action="store_true"); A = ap.parse_args()
    if A.merge: merge()
    else: run(A.part or "_".join(A.models), A.models, A.seeds)
