#!/usr/bin/env python3
"""expR82 (eighth review, priority 1e, CPU): is the certified 'alignment of each cluster with its hub' the spread of feature norms?

The depth test of record (WordNet-30 frame, 10 star seeds) on the 12 ImageNet centroid clouds after two transforms that remove the
radial information: (a) every centroid L2-normalized; (b) the radial component of each offset removed, i.e. x_i - h_k(i) minus its
projection onto the direction of h_k(i), the hub kept. Both matched stars ('aniso' = Gaussian hubs at the real hub radius with the
anisotropic within-cluster Haar sample; 'aniso_haarhubs' = the hubs as a Haar resample of the real hubs), and the decoupling control
of expR74 (real hubs kept, offsets Haar-rotated, 10 seeds, 'aniso' star) on each transformed cloud.
Decision rule fixed by the brief: if the four certified backbones (ViT-S, ViT-B, ViT-L, DINOv2-L) keep z <= -2 under (a) or (b), the
paper keeps 'alignment of each cluster with its hub'; otherwise 'alignment' becomes 'the radial spread of feature norms within each
superclass'. --merge writes expR82_radial_control.csv, expR82_radial_control_summary.csv (per backbone and transform: z under both
stars, decoupled z mean and certified fraction) and expR82_decision.csv (the rule, per transform and combined).
Usage: python expR82_radial_control.py --part <name> --models m1 m2 ; python expR82_radial_control.py --merge"""
import os, sys, time, argparse, glob
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "ICLR2027" / "tool"))
import calibrated_delta as cd, expR74_decoupling as R74
from expR64_implanted_depth import centroids, MODELS, OUT
from expR64b_implanted_depth_v2 import frame
CERTIFIED = ["i21k_s", "i21k_b", "i21k_l", "dinov2_l"]; N_DEC = 10; STARS = ("aniso", "aniso_haarhubs")

def l2norm(C):
    return (C / np.linalg.norm(C, axis=1, keepdims=True)).astype(np.float32)

def deradial(C, sup):
    """Remove from each offset x_i - h_k its projection onto the hub direction h_k / |h_k|; hubs unchanged."""
    X = np.empty_like(C); K = sup.max() + 1
    for k in range(K):
        m = sup == k; h = C[m].mean(0); u = h / np.linalg.norm(h); off = C[m] - h
        X[m] = h + off - np.outer(off @ u, u)
    return X.astype(np.float32)

def run(part, models):
    f = OUT / f"expR82_radial_control.part_{part}.csv"; rows = pd.read_csv(f).to_dict("records") if f.exists() else []
    done = {(r["model"], r["transform"], r["star"], int(r["dec_seed"])) for r in rows}
    sup, _ = frame("wn30")
    for m in models:
        C = centroids(m)
        for tname, Ct in (("l2norm", l2norm(C)), ("deradial", deradial(C, sup))):
            assert abs(float(np.abs(np.linalg.norm(Ct, axis=1) - 1).max())) < 1e-4 if tname == "l2norm" else True
            for star in STARS:
                if (m, tname, star, -1) in done: continue
                t0 = time.time(); r = cd.depth_test(Ct, sup, star, 10)
                rows.append(dict(model=m, transform=tname, star=star, dec_seed=-1, excessB_real=r["excessB_real"], excessB_star=r["excessB_star"], star_sd=r["excessB_star_sd"], depth=r["depth_excess"], z=r["z_depth"], r_star=r["r_star"], p_star=r["p_star"], time_s=time.time() - t0))
                print(f"{m:9s} {tname:9s} {star:15s} intact: depth {r['depth_excess']:+.4f} z {r['z_depth']:+.2f} ({time.time() - t0:.0f}s)"); pd.DataFrame(rows).to_csv(f, index=False)
            for s in range(N_DEC):
                if (m, tname, "aniso_decoupled", s) in done: continue
                t0 = time.time(); r = cd.depth_test(R74.decouple(Ct, sup, s), sup, "aniso", 10)
                rows.append(dict(model=m, transform=tname, star="aniso_decoupled", dec_seed=s, excessB_real=r["excessB_real"], excessB_star=r["excessB_star"], star_sd=r["excessB_star_sd"], depth=r["depth_excess"], z=r["z_depth"], r_star=r["r_star"], p_star=r["p_star"], time_s=time.time() - t0))
                pd.DataFrame(rows).to_csv(f, index=False)
            zs = [r["z"] for r in rows if r["model"] == m and r["transform"] == tname and r["star"] == "aniso_decoupled"]
            print(f"{m:9s} {tname:9s} decoupled: z mean {np.mean(zs):+.2f} cert {np.mean(np.array(zs) <= -2):.1f}")
    print(f"PART DONE {part}")

def merge():
    parts = sorted(glob.glob(str(OUT / "expR82_radial_control.part_*.csv")))
    df = pd.concat([pd.read_csv(p) for p in parts]).drop_duplicates(subset=["model", "transform", "star", "dec_seed"]).sort_values(["model", "transform", "star", "dec_seed"]); df.to_csv(OUT / "expR82_radial_control.csv", index=False)
    real = pd.read_csv(OUT / "expR56_depth_variants.csv") if (OUT / "expR56_depth_variants.csv").exists() else None
    S = []
    for m in MODELS:
        for tname in ("l2norm", "deradial"):
            a = df[(df.model == m) & (df['transform'] == tname)]
            if not len(a): continue
            za = a[a.star == "aniso"].z; zh = a[a.star == "aniso_haarhubs"].z; zd = a[a.star == "aniso_decoupled"].z
            S.append(dict(model=m, transform=tname, certified_real=m in CERTIFIED, z_aniso=float(za.iloc[0]) if len(za) else np.nan, z_haarhubs=float(zh.iloc[0]) if len(zh) else np.nan,
                          keeps_aniso=bool(len(za) and za.iloc[0] <= -2), keeps_haarhubs=bool(len(zh) and zh.iloc[0] <= -2), dec_runs=len(zd), dec_z_mean=float(zd.mean()) if len(zd) else np.nan, dec_frac_certified=float((zd <= -2).mean()) if len(zd) else np.nan))
    S = pd.DataFrame(S); S.to_csv(OUT / "expR82_radial_control_summary.csv", index=False)
    D = []
    for tname in ("l2norm", "deradial"):
        c = S[(S['transform'] == tname) & S.certified_real]
        D.append(dict(transform=tname, n_certified_done=len(c), all_keep_aniso=bool(len(c) == 4 and c.keeps_aniso.all()), all_keep_haarhubs=bool(len(c) == 4 and c.keeps_haarhubs.all()), n_keep_aniso=int(c.keeps_aniso.sum()), n_keep_haarhubs=int(c.keeps_haarhubs.sum())))
    D = pd.DataFrame(D); D["rule_alignment_kept"] = bool(((D.all_keep_aniso) | (D.all_keep_haarhubs)).any()) if len(D) else False
    D.to_csv(OUT / "expR82_decision.csv", index=False)
    print(S.round(2).to_string()); print(D.to_string()); print("SUMMARY expR82: alignment kept under (a) or (b):", bool(D.rule_alignment_kept.iloc[0]) if len(D) else None, "| backbones done", S.model.nunique(), "of 12")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--part", default=None); ap.add_argument("--models", nargs="+", default=MODELS); ap.add_argument("--merge", action="store_true"); A = ap.parse_args()
    if A.merge: merge()
    else: run(A.part or "_".join(A.models), A.models)
