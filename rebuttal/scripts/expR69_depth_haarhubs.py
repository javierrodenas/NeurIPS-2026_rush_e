#!/usr/bin/env python3
"""expR69 (final pass, A1): the depth test with spectrum-matched star hubs.

The matched star of Table B34 places its hubs as a Gaussian sample at the real hub RMS radius ('aniso'). Since the paper's own
thesis is that the spectrum moves delta, the star 'aniso_haarhubs' places the hubs as a Haar resample of the 30 real hubs
(exact hub spectrum, random orientation; calibrated_delta.matched_star, seed0 = 5000) with the within-cluster construction
unchanged (anisotropic Haar). Both stars run on the 12 ImageNet backbones (WordNet-30 frame, 10 star seeds, hub null 10 x 3)
with the per-seed star excesses kept, so the depth verdict is also read as a percentile rank over the 10 star seeds:
r_star = #{star seeds with excess B <= real excess B}, p_star = (1 + r_star) / 11, resolution 1/11. The new star also runs
on the R9b implanted clouds (expR64b implant_v2, rand6 partition) at s = 0 and s = 1, 5 implant seeds each: false alarms and power.

    python expR69_depth_haarhubs.py --part <model>     # ~12 depth tests per backbone
    python expR69_depth_haarhubs.py --merge            # -> expR69_depth_haarhubs.csv, expR69_depth_haarhubs_summary.csv
"""
import os, sys, time, argparse, json
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "ICLR2027" / "tool"))
import calibrated_delta as cd
from expR64_implanted_depth import centroids, MODELS, OUT
from expR64b_implanted_depth_v2 import frame, implant_v2

def run_part(m):
    f = OUT / f"expR69_depth_haarhubs.part_{m}.csv"; rows = pd.read_csv(f).to_dict("records") if f.exists() else []
    done = {(r["cloud"], r["star"], int(r["seed"])) for r in rows}
    sup, rand6 = frame("wn30"); C = centroids(m)
    jobs = [("real", "aniso", 0), ("real", "aniso_haarhubs", 0)] + [(f"implant_s{s:g}", "aniso_haarhubs", sd) for s in (0.0, 1.0) for sd in range(5)]
    for cloud, star, seed in jobs:
        if (cloud, star, seed) in done: continue
        t0 = time.time(); Cs = C if cloud == "real" else implant_v2(C, sup, rand6, float(cloud.split("_s")[1]), seed)
        r = cd.depth_test(Cs, sup, star, 10)
        rows.append(dict(model=m, cloud=cloud, star=star, seed=seed, excessB_real=r["excessB_real"], excessB_star=r["excessB_star"], excessB_star_sd=r["excessB_star_sd"],
                         depth=r["depth_excess"], z=r["z_depth"], r_star=r["r_star"], p_star=r["p_star"], star_excesses=json.dumps(r["star_excesses"]), time_s=time.time() - t0))
        pd.DataFrame(rows).to_csv(f, index=False)
        print(f"{m:9s} {cloud:12s} {star:15s} seed {seed}: real {r['excessB_real']:+.4f} star {r['excessB_star']:+.4f} depth {r['depth_excess']:+.4f} z {r['z_depth']:+.2f} r_star {r['r_star']}/10 ({time.time()-t0:.0f}s)")
    print(f"PART DONE {m}")

def merge():
    df = pd.concat([pd.read_csv(p) for p in sorted(OUT.glob("expR69_depth_haarhubs.part_*.csv"))], ignore_index=True)
    df["order"] = df.model.map({mm: i for i, mm in enumerate(MODELS)}); df = df.sort_values(["order", "cloud", "star", "seed"]).drop(columns="order"); df.to_csv(OUT / "expR69_depth_haarhubs.csv", index=False)
    out = []
    for m in MODELS:
        g = df[df.model == m]; a = g[(g.cloud == "real") & (g.star == "aniso")].iloc[0]; h = g[(g.cloud == "real") & (g.star == "aniso_haarhubs")].iloc[0]
        s0 = g[(g.cloud == "implant_s0") & (g.star == "aniso_haarhubs")]; s1 = g[(g.cloud == "implant_s1") & (g.star == "aniso_haarhubs")]
        out.append(dict(model=m, real_gauss=a.excessB_real, star_gauss=a.excessB_star, depth_gauss=a.depth, z_gauss=a.z, r_star_gauss=int(a.r_star), p_star_gauss=a.p_star,
                        real_haar=h.excessB_real, star_haar=h.excessB_star, depth_haar=h.depth, z_haar=h.z, r_star_haar=int(h.r_star), p_star_haar=h.p_star,
                        cert_gauss=bool(a.z <= -2), cert_haar=bool(h.z <= -2), fa_s0_haar=int((s0.z <= -2).sum()), n_s0=len(s0), hits_s1_haar=int((s1.z <= -2).sum()), n_s1=len(s1), z_mean_s1_haar=float(s1.z.mean())))
    S = pd.DataFrame(out); S.to_csv(OUT / "expR69_depth_haarhubs_summary.csv", index=False)
    print(S.round(3).to_string())
    print(f"certified under Gaussian star: {int(S.cert_gauss.sum())}/12; under Haar-hub star: {int(S.cert_haar.sum())}/12; intersection: {list(S[S.cert_gauss & S.cert_haar].model)}")
    print(f"Haar-hub star on implants: false alarms at s=0 {int(S.fa_s0_haar.sum())}/{int(S.n_s0.sum())}; power at s=1 {S.hits_s1_haar.sum()/max(1,S.n_s1.sum()):.2f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--part"); ap.add_argument("--merge", action="store_true"); A = ap.parse_args()
    merge() if A.merge else run_part(A.part)
