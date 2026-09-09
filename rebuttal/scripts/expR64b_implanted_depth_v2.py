#!/usr/bin/env python3
"""expR64b (positive-control pass, R9b) and expR67 (R12): the implanted-depth control with the two-level implant corrected
so that sibling hubs never coincide, on the WordNet-30 frame of record (R9b) and on a balanced WordNet frame chosen by rule
before any depth run (R12).

Implant (v2). Hubs h_k of the real cloud (superclass means) and within-cluster offsets o_i = x_i - h_k(i) as in the depth
test; offsets untouched. Six super-hubs S_g (fixed random partition of the K = 30 hubs into 6 groups of 5, RandomState(0)),
Gaussian with the real hubs' RMS radius; own draws G_k, Gaussian with the same radius; at strength s the hub is
    h'_k = s * S_g(k) + (1 - s + s * LAMBDA) * G_k,   LAMBDA = 0.4,
i.e. s = 0 is a star (h' = G_k) and s = 1 the two-level tree S_g(k) + 0.4 * G_k with separated siblings; the hub cloud is
rescaled to the real RMS radius at every s. x_i' = h'_k(i) + o_i. Depth test exactly as Table B34 (calibrated_delta.depth_test:
anisotropic matched star, 10 star seeds, hub-randomizing Haar null 10 x 3), s in {0, 0.25, 0.5, 0.75, 1}, 5 implant seeds
(RandomState(1000*seed + 100*s), the seeds of expR64). Census of record (Haar x p99.9 x 200) on implant seed 0 at every s.
Tight variant (noise diagnostic): the real offsets shrunk to within/between = 0.6, then the same implant, s in {0, 0.5, 1},
2 seeds. Row s = 'real' = the untouched cloud.

Frames. 'wn30' = the WordNet-30 cut of record (average linkage on the WordNet distance matrix). 'wn30bal' (R12,
pre-registered): among the K = 30 agglomerative cuts of the WordNet distance matrix with linkage in {average, complete,
single}, the one with the smallest variance of cluster sizes, chosen and logged (expR67_frame_choice.json) before any depth
run; decision rule fixed in advance: power at s = 1 >= 0.8 with zero hits at s = 0 -> frame of record (both reported),
otherwise appendix as run.

    python expR64b_implanted_depth_v2.py --frame wn30    --part <model>      # R9b, one backbone
    python expR64b_implanted_depth_v2.py --frame wn30bal --part <model>      # R12, one backbone (no census, no tight)
    python expR64b_implanted_depth_v2.py --frame <f> --merge                 # -> expR64b_<f>.csv, _summary.csv, figure (wn30)
"""
import os, sys, time, argparse, json
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "ICLR2027" / "tool"))
import calibrated_delta as cd
from expR64_implanted_depth import centroids, rms, ratio, tighten, MODELS, ROOT, OUT, K, G, S_GRID, N_SEEDS
LAMBDA = 0.4

def wn_cuts():
    from sklearn.cluster import AgglomerativeClustering
    WN = np.load(ROOT / "results/features/imagenet/imagenet1k_wordnet_dist.npy")
    return {lk: AgglomerativeClustering(n_clusters=K, metric="precomputed", linkage=lk).fit_predict(WN) for lk in ("average", "complete", "single")}

def frame(name):
    cuts = wn_cuts()
    if name == "wn30": sup = cuts["average"]
    else:
        f = OUT / "expR67_frame_choice.json"
        if f.exists(): choice = json.load(open(f))
        else:
            var = {lk: float(np.var(np.bincount(c))) for lk, c in cuts.items()}; best = min(var, key=var.get)
            choice = dict(rule="K=30 agglomerative cut of the WordNet distance matrix, linkage in {average, complete, single}, minimum variance of cluster sizes", variances=var,
                          sizes={lk: sorted(np.bincount(c).tolist()) for lk, c in cuts.items()}, chosen=best, chosen_sizes=sorted(np.bincount(cuts[best]).tolist()),
                          decision_rule="power at s=1 >= 0.8 with zero hits at s=0 -> frame of record; otherwise appendix", chosen_at=time.strftime("%Y-%m-%d %H:%M:%S"))
            json.dump(choice, open(f, "w"), indent=1); print("frame choice logged:", choice["chosen"], choice["chosen_sizes"])
        sup = cuts[choice["chosen"]]
    rand6 = np.empty(K, dtype=int); rand6[np.random.RandomState(0).permutation(K)] = np.repeat(np.arange(G), K // G)
    return sup, rand6

def implant_v2(C, sup, parent, s, seed):
    H = np.stack([C[sup == k].mean(0) for k in range(K)]); R = rms(H); O = C - H[sup]; d = C.shape[1]
    rng = np.random.RandomState(1000 * seed + int(round(100 * s)))
    S = rng.randn(G, d).astype(np.float32); S *= R / rms(S)
    Gk = rng.randn(K, d).astype(np.float32); Gk *= R / rms(Gk)
    Hn = s * S[parent] + (1.0 - s + s * LAMBDA) * Gk; Hn = Hn - Hn.mean(0); Hn *= R / rms(Hn); Hn = Hn + H.mean(0)
    return (Hn[sup] + O).astype(np.float32)

def run_part(fr, m):
    f = OUT / f"expR64b_{fr}.part_{m}.csv"; rows = pd.read_csv(f).to_dict("records") if f.exists() else []
    done = {(r["partition"], str(r["s"]), int(r["seed"]), r["kind"]) for r in rows}
    sup, rand6 = frame(fr); C = centroids(m)
    def add(row): rows.append(row); pd.DataFrame(rows).to_csv(f, index=False)
    jobs = [("real", "real", 0, "depth")] + [("rand6", s, sd, "depth") for s in S_GRID for sd in range(N_SEEDS)]
    if fr == "wn30": jobs += [("rand6", s, 0, "census") for s in S_GRID] + [("tight06", "real", 0, "depth")] + [("rand6_t06", s, sd, "depth") for s in (0.0, 0.5, 1.0) for sd in range(2)]
    for partition, s, seed, kind in jobs:
        if (partition, str(s), seed, kind) in done: continue
        t0 = time.time(); base = tighten(C, sup) if partition in ("tight06", "rand6_t06") else C
        Cs = base if s == "real" else implant_v2(base, sup, rand6, float(s), seed)
        row = dict(model=m, frame=fr, partition=partition, s=s, seed=seed, kind=kind, n=int(len(Cs)), d=int(Cs.shape[1]), hub_rms=rms(np.stack([Cs[sup == k].mean(0) for k in range(K)])), ratio=ratio(Cs, sup))
        if kind == "depth":
            r = cd.depth_test(Cs, sup, "aniso", 10)
            row.update(excessB_real=r["excessB_real"], excessB_star=r["excessB_star"], excessB_star_sd=r["excessB_star_sd"], depth=r["depth_excess"], z=r["z_depth"])
            print(f"{m:9s} {fr} {partition:9s} s={s!s:5} seed {seed} depth: real {r['excessB_real']:+.4f} star {r['excessB_star']:+.4f} depth {r['depth_excess']:+.4f} z {r['z_depth']:+.2f} ({time.time()-t0:.0f}s)")
        else:
            c = cd.census(Cs, 200, "haar", "p999")
            row.update(delta_999=c["delta"], delta_sd=c["delta_sd"], null_mean=c["null_mean"], null_sd=c["null_sd"], excess=c["excess"], r_above=c["r_above"], p_left=c["p_left"])
            print(f"{m:9s} {fr} {partition:9s} s={s!s:5} seed {seed} census: exc {c['excess']:+.4f} r={c['r_above']} ({time.time()-t0:.0f}s)")
        row["time_s"] = time.time() - t0; add(row)
    print(f"PART DONE {fr} {m}")

def merge(fr):
    df = pd.concat([pd.read_csv(p) for p in sorted(OUT.glob(f"expR64b_{fr}.part_*.csv"))], ignore_index=True)
    df["order"] = df.model.map({m: i for i, m in enumerate(MODELS)}); df = df.sort_values(["order", "kind", "partition", "s", "seed"]).drop(columns="order"); df.to_csv(OUT / f"expR64b_{fr}.csv", index=False)
    dep = df[(df.kind == "depth") & (df.partition == "rand6") & (df.s != "real")].copy(); dep["s"] = dep.s.astype(float)
    cen = df[df.kind == "census"].copy(); tg = df[(df.kind == "depth") & (df.partition == "rand6_t06")].copy()
    if len(cen): cen["s"] = cen.s.astype(float)
    if len(tg): tg["s"] = tg.s.astype(float)
    real = df[(df.kind == "depth") & (df.partition == "real")].set_index("model")
    out = []
    for m in MODELS:
        g = dep[dep.model == m]; hits = {s: int((g[g.s == s].z <= -2).sum()) for s in S_GRID}; n = {s: int((g.s == s).sum()) for s in S_GRID}
        r = dict(model=m, frame=fr, n_runs=len(g), real_z=float(real.loc[m, "z"]), real_depth=float(real.loc[m, "depth"]), ratio_real=float(real.loc[m, "ratio"]),
                 hits_s0=hits[0.0], hits_s025=hits[0.25], hits_s05=hits[0.5], hits_s075=hits[0.75], hits_s1=hits[1.0], s_star=next((s for s in S_GRID if n[s] and hits[s] >= 4), None),
                 z_mean_s0=float(g[g.s == 0].z.mean()), z_max_s0=float(g[g.s == 0].z.max()), z_mean_s05=float(g[g.s == 0.5].z.mean()), z_mean_s1=float(g[g.s == 1].z.mean()), z_max_s1=float(g[g.s == 1].z.max()))
        if len(cen):
            c = cen[cen.model == m].set_index("s"); r.update(census_exc_s0=float(c.loc[0.0, "excess"]), census_exc_s1=float(c.loc[1.0, "excess"]), census_exc_spread=float(c.excess.max() - c.excess.min()), census_null_sd=float(c.null_sd.mean()), census_r_min=int(c.r_above.min()))
        if len(tg):
            t = tg[tg.model == m]; r.update(tight_z_s0=float(t[t.s == 0].z.mean()), tight_z_s05=float(t[t.s == 0.5].z.mean()), tight_z_s1=float(t[t.s == 1].z.mean()), tight_hits_s1=int((t[t.s == 1].z <= -2).sum()), tight_hits_s0=int((t[t.s == 0].z <= -2).sum()))
        out.append(r)
    S = pd.DataFrame(out); S.to_csv(OUT / f"expR64b_{fr}_summary.csv", index=False)
    power = {s: float((dep[dep.s == s].z <= -2).mean()) for s in S_GRID}
    print(S.to_string()); print(f"[{fr}] power by s: {power}; s=0 hits total {int((dep[dep.s == 0].z <= -2).sum())}/{int((dep.s == 0).sum())}; real z<=-2: {int((S.real_z <= -2).sum())}/12")
    if fr == "wn30": figure(dep, cen, tg)

def figure(dep, cen, tg):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    FIG = HERE.parents[1] / "ICLR2027" / "figures"; sys.path.insert(0, str(FIG)); plt.style.use(str(FIG / "style.mplstyle"))
    from palette import color as fam_color
    NM = {"i21k_t":"ViT-T","i21k_s":"ViT-S","i21k_b":"ViT-B","i21k_l":"ViT-L","dinov1_b":"DINO-B","dinov2_s":"Dv2-S","dinov2_b":"Dv2-B","dinov2_l":"Dv2-L","dinov2_g":"Dv2-G","clip_b":"CLIP-B","clip_l":"CLIP-L","siglip_b":"SigLIP"}
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.1), gridspec_kw={"width_ratios": [1.3, 1]})
    for m in MODELS:
        g = dep[dep.model == m].groupby("s").z; c = fam_color(m)
        axes[0].plot(g.mean().index, g.mean().values, "-o", color=c, ms=2.5, lw=0.9, label=NM[m])
        axes[0].fill_between(g.min().index, g.min().values, g.max().values, color=c, alpha=0.12, lw=0)
        if len(tg):
            t = tg[tg.model == m].groupby("s").z.mean(); axes[0].plot(t.index, t.values, "--", color=c, lw=0.7, alpha=0.8)
        if len(cen):
            cc = cen[cen.model == m].sort_values("s"); axes[1].plot(cc.s, cc.excess, "-o", color=c, ms=2.5, lw=0.9)
    axes[0].axhline(-2, color="k", lw=0.7, ls="--"); axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_xlabel("implant strength $s$"); axes[0].set_ylabel("depth test $z$"); axes[0].set_title("(a) depth test, implanted hubs on real clusters")
    axes[1].axhline(0, color="k", lw=0.5); axes[1].set_xlabel("implant strength $s$"); axes[1].set_ylabel("census excess"); axes[1].set_title("(b) census excess, same clouds")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=5.2, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.02), handlelength=1.2, columnspacing=0.9, handletextpad=0.4)
    fig.tight_layout(w_pad=0.8, rect=[0, 0.13, 1, 1])
    for o in (FIG, HERE.parents[1] / "ICLR2027" / "iclr2027" / "figures"): fig.savefig(o / "fig_implanted_depth_v2.pdf"); fig.savefig(o / "fig_implanted_depth_v2.png", dpi=200)
    print("fig_implanted_depth_v2 written")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--frame", default="wn30", choices=["wn30", "wn30bal"]); ap.add_argument("--part"); ap.add_argument("--merge", action="store_true"); A = ap.parse_args()
    merge(A.frame) if A.merge else run_part(A.frame, A.part)
