#!/usr/bin/env python3
"""expR64 (positive-control pass, R9): implanted depth on real ImageNet centroids.

For each of the 12 backbones on ImageNet (n = 1000 census centroids, WordNet frame K = 30, the validated regime) the
real cloud is decomposed as the depth test decomposes it: hubs h_k (superclass means) and within-cluster offsets
o_i = x_i - h_k(i). The offsets are kept untouched; the hub configuration is replaced by an implanted two-level tree
of known depth and graded strength s: 6 super-hubs of 5 hubs (fixed random partition, RandomState(0); the WordNet
6-cut over the same hubs exists and is nested but unbalanced, 10/5/11/2/1/1, so it is run as a secondary partition
'wn6' for implant seed 0 only), super-hubs a Gaussian sample with the real hubs' RMS radius, each hub at
s * superhub + (1 - s) * own Gaussian draw, the hub cloud rescaled to the real RMS radius at every s;
s in {0, 0.25, 0.5, 0.75, 1}: s = 0 is a star with the real clusters, s = 1 a clean two-level tree with the real clusters.
x_i' = h'_k(i) + o_i. The depth test of expR56 / Table B34 runs unchanged (calibrated_delta.depth_test: anisotropic
matched star, 10 star seeds, hub-randomizing Haar null, 10 replicates x 3 seeds), 5 implant seeds per s; the census
excess of record (Haar x p99.9 x 200) runs on implant seed 0 at every s (the star caveat on real clouds).
Row s = 'real' is the untouched cloud through the same code.

    python expR64_implanted_depth.py --part <model>      # one backbone (~45 min on one core)
    python expR64_implanted_depth.py --merge             # -> expR64_implanted_depth.csv, expR64_implanted_depth_summary.csv, figure
"""
import os, sys, time, argparse
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "ICLR2027" / "tool"))
import calibrated_delta as cd
ROOT = Path(os.environ.get("PLATONIC_ROOT", "/media/HDD_4TB_2/javi/Platonic")); CACHE = ROOT / "results/practical_tasks_cache"
OUT = Path(os.environ.get("PLATONIC_RESULTS", str(HERE.parents[1] / "rebuttal/results")))
MODELS = ["i21k_t", "i21k_s", "i21k_b", "i21k_l", "dinov1_b", "dinov2_s", "dinov2_b", "dinov2_l", "dinov2_g", "clip_b", "clip_l", "siglip_b"]
S_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]; N_SEEDS = 5; K = 30; G = 6

def frames():
    from sklearn.cluster import AgglomerativeClustering
    WN = np.load(ROOT / "results/features/imagenet/imagenet1k_wordnet_dist.npy")
    sup = AgglomerativeClustering(n_clusters=K, metric="precomputed", linkage="average").fit_predict(WN)
    c6 = AgglomerativeClustering(n_clusters=G, metric="precomputed", linkage="average").fit_predict(WN)
    wn6 = np.array([np.bincount(c6[sup == k]).argmax() for k in range(K)])          # parent of each hub in the WordNet 6-cut (nested)
    rand6 = np.empty(K, dtype=int); rand6[np.random.RandomState(0).permutation(K)] = np.repeat(np.arange(G), K // G)   # 6 groups of 5
    return sup, {"rand6": rand6, "wn6": wn6}

def centroids(m):
    d = np.load(CACHE / f"{m}_imagenet_train.npz"); X = d["features"].astype(np.float32); y = d["labels"].astype(np.int64)
    return np.stack([X[y == c].mean(0) for c in range(int(y.max()) + 1)])

def rms(H): return float(np.sqrt(((H - H.mean(0)) ** 2).sum(1).mean()))

def implant(C, sup, parent, s, seed):
    """Replace the hub configuration by a two-level tree of strength s; keep every within-cluster offset."""
    H = np.stack([C[sup == k].mean(0) for k in range(K)]); R = rms(H); O = C - H[sup]; d = C.shape[1]
    rng = np.random.RandomState(1000 * seed + int(round(100 * s)))
    S = rng.randn(G, d).astype(np.float32); S *= R / rms(S)
    Gk = rng.randn(K, d).astype(np.float32); Gk *= R / rms(Gk)
    Hn = s * S[parent] + (1.0 - s) * Gk; Hn = Hn - Hn.mean(0); Hn *= R / rms(Hn); Hn = Hn + H.mean(0)
    return (Hn[sup] + O).astype(np.float32)

def ratio(C, sup):
    """within/between spread: RMS of the within-cluster offsets over the point-weighted RMS of the hub displacements."""
    H = np.stack([C[sup == k].mean(0) for k in range(K)]); O = C - H[sup]
    return float(np.sqrt((O ** 2).sum(1).mean()) / np.sqrt(((H[sup] - C.mean(0)) ** 2).sum(1).mean()))

def tighten(C, sup, target=0.6):
    """Shrink every within-cluster offset by one factor so that ratio(C, sup) = target (the upper edge of the validated regime)."""
    H = np.stack([C[sup == k].mean(0) for k in range(K)]); O = C - H[sup]; f = target / ratio(C, sup)
    return (H[sup] + f * O).astype(np.float32)

def run_part(m, tight=False):
    f = OUT / f"expR64_implanted_depth.part_{m}.csv"; rows = pd.read_csv(f).to_dict("records") if f.exists() else []
    done = {(r["partition"], str(r["s"]), int(r["seed"]), r["kind"]) for r in rows}
    sup, parts = frames(); C = centroids(m)
    def add(row): rows.append(row); pd.DataFrame(rows).to_csv(f, index=False)
    jobs = [("real", "real", 0, "depth")] + [("rand6", s, sd, "depth") for s in S_GRID for sd in range(N_SEEDS)] \
         + [("wn6", s, 0, "depth") for s in S_GRID] + [("rand6", s, 0, "census") for s in S_GRID]
    if tight:   # secondary variant: real clusters shrunk to within/between = 0.6, then the same implant; s in {0, 0.5, 1}, 2 seeds
        jobs = [("tight06", "real", 0, "depth")] + [("rand6_t06", s, sd, "depth") for s in (0.0, 0.5, 1.0) for sd in range(2)]
    for partition, s, seed, kind in jobs:
        if (partition, str(s), seed, kind) in done: continue
        t0 = time.time(); base = tighten(C, sup) if partition in ("tight06", "rand6_t06") else C
        Cs = base if s == "real" else implant(base, sup, parts["rand6" if partition.startswith("rand6") else partition], float(s), seed)
        row = dict(model=m, partition=partition, s=s, seed=seed, kind=kind, n=int(len(Cs)), d=int(Cs.shape[1]), hub_rms=rms(np.stack([Cs[sup == k].mean(0) for k in range(K)])), ratio=ratio(Cs, sup))
        if kind == "depth":
            r = cd.depth_test(Cs, sup, "aniso", 10)
            row.update(excessB_real=r["excessB_real"], excessB_star=r["excessB_star"], excessB_star_sd=r["excessB_star_sd"], depth=r["depth_excess"], z=r["z_depth"])
            print(f"{m:9s} {partition:5s} s={s!s:5} seed {seed} depth: real {r['excessB_real']:+.4f} star {r['excessB_star']:+.4f} depth {r['depth_excess']:+.4f} z {r['z_depth']:+.2f} ({time.time()-t0:.0f}s)")
        else:
            c = cd.census(Cs, 200, "haar", "p999")
            row.update(delta_999=c["delta"], delta_sd=c["delta_sd"], null_mean=c["null_mean"], null_sd=c["null_sd"], excess=c["excess"], r_above=c["r_above"], p_left=c["p_left"])
            print(f"{m:9s} {partition:5s} s={s!s:5} seed {seed} census: d999 {c['delta']:.4f} null {c['null_mean']:.4f} exc {c['excess']:+.4f} r={c['r_above']} ({time.time()-t0:.0f}s)")
        row["time_s"] = time.time() - t0; add(row)
    print(f"PART DONE {m}")

def merge():
    df = pd.concat([pd.read_csv(p) for p in sorted(OUT.glob("expR64_implanted_depth.part_*.csv"))], ignore_index=True)
    df["order"] = df.model.map({m: i for i, m in enumerate(MODELS)}); df = df.sort_values(["order", "kind", "partition", "s", "seed"]).drop(columns="order")
    df.to_csv(OUT / "expR64_implanted_depth.csv", index=False)
    dep = df[(df.kind == "depth") & (df.partition == "rand6") & (df.s != "real")].copy(); dep["s"] = dep.s.astype(float)
    cen = df[(df.kind == "census")].copy(); cen["s"] = cen.s.astype(float)
    real = df[(df.kind == "depth") & (df.partition == "real")].set_index("model"); sup_, _ = frames()
    b34 = pd.read_csv(OUT / "expR56_depth_variants.csv"); b34 = b34[(b34.dataset == "imagenet") & (b34.K == 30) & (b34.variant == "aniso")].set_index("model")
    out = []
    for m in MODELS:
        g = dep[dep.model == m]
        hits = {s: int((g[g.s == s].z <= -2).sum()) for s in S_GRID}; n = {s: int((g.s == s).sum()) for s in S_GRID}
        s_star = next((s for s in S_GRID if n[s] and hits[s] >= 4), None)
        c = cen[cen.model == m].set_index("s")
        out.append(dict(model=m, n_runs=len(g), real_z_here=float(real.loc[m, "z"]) if m in real.index else np.nan, real_z_B34=float(b34.loc[m, "z_depth"]),
                        hits_s0=hits[0.0], hits_s025=hits[0.25], hits_s05=hits[0.5], hits_s075=hits[0.75], hits_s1=hits[1.0], s_star=s_star,
                        z_mean_s0=float(g[g.s == 0].z.mean()), z_mean_s1=float(g[g.s == 1].z.mean()), z_max_s0=float(g[g.s == 0].z.max()),
                        census_exc_s0=float(c.loc[0.0, "excess"]) if 0.0 in c.index else np.nan, census_exc_s1=float(c.loc[1.0, "excess"]) if 1.0 in c.index else np.nan,
                        census_exc_spread=float(c.excess.max() - c.excess.min()) if len(c) else np.nan, census_null_sd=float(c.null_sd.mean()) if len(c) else np.nan,
                        census_r_min=int(c.r_above.min()) if len(c) else -1, wn6_s1_z=float(df[(df.model == m) & (df.partition == "wn6") & (df.s == "1.0")].z.mean()) if ((df.partition == "wn6") & (df.model == m)).any() else np.nan))
    tg = df[(df.kind == "depth") & (df.partition == "rand6_t06")].copy()
    if len(tg):
        tg["s"] = tg.s.astype(float)
        for r in out:
            g = tg[tg.model == r["model"]]
            r.update(tight_z_s0=float(g[g.s == 0].z.mean()) if (g.s == 0).any() else np.nan, tight_z_s05=float(g[g.s == 0.5].z.mean()) if (g.s == 0.5).any() else np.nan,
                     tight_z_s1=float(g[g.s == 1].z.mean()) if (g.s == 1).any() else np.nan, tight_hits_s1=int((g[g.s == 1].z <= -2).sum()), tight_hits_s0=int((g[g.s == 0].z <= -2).sum()))
    for r in out: r["ratio_real"] = ratio(centroids(r["model"]), sup_)
    S = pd.DataFrame(out); S.to_csv(OUT / "expR64_implanted_depth_summary.csv", index=False)
    print(S.to_string()); print(f"s=0 declared hierarchical (any seed): {int((S.hits_s0 > 0).sum())} backbones; s* per backbone: {dict(zip(S.model, S.s_star))}")
    figure(dep, cen)

def figure(dep, cen):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    FIG = HERE.parents[1] / "ICLR2027" / "figures"; sys.path.insert(0, str(FIG)); plt.style.use(str(FIG / "style.mplstyle"))
    from palette import color as fam_color
    NM = {"i21k_t":"ViT-T","i21k_s":"ViT-S","i21k_b":"ViT-B","i21k_l":"ViT-L","dinov1_b":"DINO-B","dinov2_s":"Dv2-S","dinov2_b":"Dv2-B","dinov2_l":"Dv2-L","dinov2_g":"Dv2-G","clip_b":"CLIP-B","clip_l":"CLIP-L","siglip_b":"SigLIP"}
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.1), gridspec_kw={"width_ratios": [1.3, 1]})
    for m in MODELS:
        g = dep[dep.model == m].groupby("s").z; c = fam_color(m)
        axes[0].plot(g.mean().index, g.mean().values, "-o", color=c, ms=2.5, lw=0.9, label=NM[m])
        axes[0].fill_between(g.min().index, g.min().values, g.max().values, color=c, alpha=0.12, lw=0)
        cc = cen[cen.model == m].sort_values("s"); axes[1].plot(cc.s, cc.excess, "-o", color=c, ms=2.5, lw=0.9)
    axes[0].axhline(-2, color="k", lw=0.7, ls="--"); axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_xlabel("implant strength $s$"); axes[0].set_ylabel("depth test $z$"); axes[0].set_title("(a) depth test on implanted hubs, real clusters")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=5.2, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.02), handlelength=1.2, columnspacing=0.9, handletextpad=0.4)
    axes[1].axhline(0, color="k", lw=0.5); axes[1].set_xlabel("implant strength $s$"); axes[1].set_ylabel("census excess"); axes[1].set_title("(b) census excess, same clouds")
    fig.tight_layout(w_pad=0.8, rect=[0, 0.13, 1, 1])
    for o in (FIG, HERE.parents[1] / "ICLR2027" / "iclr2027" / "figures"): fig.savefig(o / "fig_implanted_depth.pdf"); fig.savefig(o / "fig_implanted_depth.png", dpi=200)
    print("fig_implanted_depth written")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--part"); ap.add_argument("--tight", action="store_true"); ap.add_argument("--merge", action="store_true"); A = ap.parse_args()
    merge() if A.merge else run_part(A.part, A.tight)
