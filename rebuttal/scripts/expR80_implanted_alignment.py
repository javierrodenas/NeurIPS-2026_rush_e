#!/usr/bin/env python3
"""expR80 (parallel track, priority 1c, CPU; reuses expR74): implanted hub alignment on the real ImageNet clouds.

For each backbone: start from the decoupled cloud of expR74 (real hubs kept, each cluster's offsets rotated by an independent Haar
rotation, so the cluster orientations are random) and rotate each cluster's principal axis toward its hub direction (the hub minus the
global mean) with strength s in {0, 0.25, 0.5, 0.75, 1}: the rotation acts in the plane spanned by the principal axis and the hub
direction by s times the angle between them, so s = 0 is the decoupled cloud and s = 1 aligns every cluster's principal axis with its
hub direction exactly; the within-cluster spectrum is unchanged. 5 seeds (decoupling seed = implant seed). Depth test of the record
(anisotropic matched star, WordNet-30 frame, 10 star seeds) at every s; detection = z <= -2. Output: expR80_implanted_alignment.csv
(rows: model x seed x s) and expR80_implanted_alignment_summary.csv (detection rate per s, false alarms at s = 0, power at s = 1, the
decision rule of the brief: power >= 0.8 at s = 1 with false alarms <= 0.05 at s = 0).
Usage: python expR80_implanted_alignment.py --part <name> --models m1 m2 ...; then --merge"""
import os, sys, time, argparse, glob
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import expR56_depth_variants as R56, expR74_decoupling as R74
OUT = R56.OUT; MODELS = R56.MODELS; STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0); N_SEEDS = 5

def implant_alignment(X, sup, s):
    """Rotate each cluster's principal axis toward its hub direction by a fraction s of the angle between them."""
    Y = X.copy(); mu = X.mean(0); K = sup.max() + 1
    for k in range(K):
        m = sup == k; hub = X[m].mean(0); off = X[m] - hub; u = hub - mu; u /= max(np.linalg.norm(u), 1e-12)
        _, _, Vt = np.linalg.svd(off, full_matrices=False); v = Vt[0]
        if v @ u < 0: v = -v                      # the principal axis has no sign: take the half pointing toward the hub direction
        c = float(np.clip(v @ u, -1.0, 1.0)); theta = np.arccos(c)
        if theta < 1e-8 or s == 0.0: Y[m] = hub + off; continue
        w = u - c * v; w /= np.linalg.norm(w)      # orthonormal basis (v, w) of the plane spanned by v and u
        a = s * theta; ca, sa = np.cos(a), np.sin(a)
        pv = off @ v; pw = off @ w                  # coordinates in the plane; rotate them, keep the orthogonal complement
        off_rot = off + np.outer(pv * (ca - 1) - pw * sa, v) + np.outer(pv * sa + pw * (ca - 1), w)
        Y[m] = hub + off_rot
    return Y.astype(np.float32)

def run(part, models):
    csv_path = OUT / f"expR80_implanted_alignment.part_{part}.csv"
    rows = pd.read_csv(csv_path).to_dict("records") if csv_path.exists() else []
    done = {(r["model"], r["seed"], float(r["s"])) for r in rows}
    _, inet = R56.frames(); sup = np.asarray(inet[30]).astype(int)
    for m in models:
        C = R56.cents(m, "imagenet")
        for seed in range(N_SEEDS):
            D = R74.decouple(C, sup, seed)
            for s in STRENGTHS:
                if (m, seed, s) in done: continue
                t0 = time.time(); res = R56.depth_test(implant_alignment(D, sup, s), sup, "aniso")
                rows.append(dict(model=m, seed=seed, s=s, K=30, **res, time_s=time.time() - t0))
                print(f"{m:9s} seed {seed} s {s:.2f} depth {res['depth_excess']:+.4f} z {res['z_depth']:+.2f} ({rows[-1]['time_s']:.0f}s)")
                pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Done part {part}")

def merge():
    parts = sorted(glob.glob(str(OUT / "expR80_implanted_alignment.part_*.csv")))
    df = pd.concat([pd.read_csv(p) for p in parts]).drop_duplicates(subset=["model", "seed", "s"]).sort_values(["model", "seed", "s"]); df.to_csv(OUT / "expR80_implanted_alignment.csv", index=False)
    df["hit"] = df.z_depth <= -2; rate = df.groupby("s").hit.mean(); per_model = df.groupby(["model", "s"]).hit.mean().unstack()
    fa, pw = float(rate.get(0.0, np.nan)), float(rate.get(1.0, np.nan)); rule = bool(pw >= 0.8 and fa <= 0.05)
    S = pd.DataFrame([dict(s=s, detection_rate=float(rate[s]), n_runs=int((df.s == s).sum())) for s in sorted(rate.index)]); S.to_csv(OUT / "expR80_implanted_alignment_summary.csv", index=False)
    per_model.to_csv(OUT / "expR80_implanted_alignment_per_model.csv")
    pd.DataFrame([dict(false_alarms_s0=fa, power_s1=pw, n_models=df.model.nunique(), n_seeds=df.seed.nunique(), rule_power_ge_0_8_fa_le_0_05=rule)]).to_csv(OUT / "expR80_decision.csv", index=False)
    print(f"SUMMARY expR80: detection by s {dict((float(k), round(float(v), 3)) for k, v in rate.items())} | false alarms at s=0 {fa:.3f}, power at s=1 {pw:.3f} | rule met: {rule}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--part", default=None); ap.add_argument("--models", nargs="+", default=MODELS); ap.add_argument("--merge", action="store_true"); A = ap.parse_args()
    if A.merge: merge()
    else:
        assert A.part; run(A.part, A.models)
