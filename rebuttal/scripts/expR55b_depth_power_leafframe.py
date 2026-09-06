#!/usr/bin/env python3
"""expR55b (Phase B decision): power of the matched-star depth test with the FRAME AT THE FINEST CLUSTER LEVEL.

expR55 gave the test the top-level (K super-cluster) frame: the hub null keeps every frame cluster intact and only
rearranges the K hubs, so hierarchy built BELOW the frame is invisible by construction (power 0.00). Here the frame is
the leaf-cluster labels (hier2: 5K leaves; hier3: 9K leaves; star: its K clusters), the matched star is ANISOTROPIC
(within each leaf cluster a Haar sample with the cluster's own covariance, as expR56 variant (b)) and it uses 10 star seeds.
Configurations with more leaves than points are infeasible (a leaf must hold >= 1 point) and are skipped and listed.
Everything else as expR55:

Generator extends expR42 (Table B25). Grid, fixed before the runs:
  level    : star (K clusters, no hierarchy) | hier2 (K super x 5 sub, sub offset 0.5) | hier3 (K super x 3 sub x 3 subsub, offsets 0.5 / 0.25)
  K        : {6, 12, 20, 30} top-level clusters (the frame given to the depth test)
  ratio    : within/between noise ratio {0.1, 0.3, 0.6}
  aniso    : iso (isotropic Gaussian noise) | aniso (per-cluster Haar-rotated decaying spectrum s_j ~ j^-1/2, same total variance)
  n        : {100, 1000}, d = 768, 5 seeds each. Cluster sizes unequal but non-empty (each cluster gets >= 1 point, rest random).
Depth test: exactly the Table B29 protocol (expR50): excess B under the Haar hub null (10 replicates x 3 quadruple
seeds), matched isotropic star with 3 seeds (5 real seeds, 5 replicates), z against the combined spread.
Power = fraction of hierarchy runs with z <= -2; false alarms = fraction of star runs with |z| >= 2 (either sign reported).
Output: expR55b_depth_power_leafframe.csv (one row per run); figures/fig_depth_power.pdf made by --figure (the top-level-frame
figure of expR55 is kept as fig_depth_power_topframe.pdf).
"""
import os, sys, time, argparse, glob
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
OUT = Path(os.environ.get("PLATONIC_RESULTS", str(Path(__file__).resolve().parents[1]/"results")))
D_DIM = 768

def delta_norm(X, n_seeds):
    D = squareform(pdist(X, "euclidean")); diam = D.max(); n = len(D); out = []
    for s in range(n_seeds):
        rng = np.random.RandomState(s); i, j, k, l = (rng.randint(0, n, 500_000) for _ in range(4))
        ok = (i!=j)&(i!=k)&(i!=l)&(j!=k)&(j!=l)&(k!=l); i,j,k,l = i[ok],j[ok],k[ok],l[ok]
        S = np.sort(np.stack([D[i,j]+D[k,l], D[i,k]+D[j,l], D[i,l]+D[j,k]], 1), 1); out.append(((S[:,2]-S[:,1])/2).max()/diam)
    return float(np.mean(out)), float(np.std(out))
def haar_sample(M, rep, seed0=700):
    mu = M.mean(0); Mc = M - mu; U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
    rng = np.random.RandomState(seed0+rep); Z = rng.randn(len(M), len(M)); Q, R = np.linalg.qr(Z); Q = Q*np.sign(np.diag(R))
    return ((Q[:, :len(S)]*S)@Vt + mu).astype(np.float32)
def flatnull(C, sup, rep):
    hubs = np.stack([C[sup==s].mean(0) for s in range(sup.max()+1)]); return C - hubs[sup] + haar_sample(hubs, rep)[sup]
def excessB(C, sup, n_real=10, n_rep=10):
    dr, dr_sd = delta_norm(C, n_real); nB = [delta_norm(flatnull(C, sup, r), 3)[0] for r in range(n_rep)]
    return dr, dr_sd, float(dr-np.mean(nB)), float(np.std(nB, ddof=1))
def matched_star(C, sup, seed):
    """Anisotropic matched star (expR56 variant b): Gaussian hubs with the real hub RMS; within each cluster a centred
    Haar sample with the cluster's own covariance spectrum and directions (singleton clusters get no offset)."""
    rng = np.random.RandomState(seed); K = sup.max()+1; n, d = C.shape
    hubs = np.stack([C[sup==s].mean(0) for s in range(K)]); hub_rms = np.sqrt(((hubs-hubs.mean(0))**2).sum(1).mean())
    H = rng.randn(K, d); H *= hub_rms/np.sqrt((H**2).sum(1).mean())
    Z = np.zeros_like(C)
    for s in range(K):
        m = sup == s
        if m.sum() < 2: continue
        Zs = haar_sample(C[m], 0, seed0=10_000*seed + s); Z[m] = Zs - Zs.mean(0)
    return (H[sup] + Z).astype(np.float32)
def depth_test(C, sup, n_star=10):
    dr, dr_sd, exB, nBsd = excessB(C, sup)
    stars = [excessB(matched_star(C, sup, s), sup, n_real=5, n_rep=5) for s in range(n_star)]
    exS = float(np.mean([x[2] for x in stars])); exS_sd = float(np.std([x[2] for x in stars], ddof=1))
    depth = exB - exS; z = depth/max(np.sqrt(exS_sd**2 + nBsd**2 + dr_sd**2), 1e-9)
    return exB, exS, depth, z

# ---------------- generator (extends expR42) ----------------
def labels(rng, K, n):
    lab = np.concatenate([np.arange(K), rng.randint(0, K, n-K)]); rng.shuffle(lab); return lab
def noise(rng, n, d, aniso, groups):
    """iso: N(0, I). aniso: per group a Haar-rotated diagonal spectrum s_j ~ j^-1/2 (mean s^2 = 1), i.e. anisotropic, unequal clusters."""
    Z = rng.randn(n, d)
    if aniso == "iso": return Z
    s = np.arange(1, d+1)**-0.5; s = s/np.sqrt((s**2).mean())
    out = np.empty_like(Z)
    for g in np.unique(groups):
        Q, R = np.linalg.qr(rng.randn(d, d)); Q = Q*np.sign(np.diag(R))
        out[groups==g] = Z[groups==g] @ (Q*s)          # rows ~ N(0, Q diag(s^2) Q^T)
    return out
def n_leaves(level, K): return K if level == "star" else (5*K if level == "hier2" else 9*K)
def make(level, K, ratio, aniso, n, seed):
    """Returns the cloud and the LEAF-cluster labels (the frame). Leaves are assigned so that every leaf holds >= 1 point."""
    rng = np.random.RandomState(seed); d = D_DIM
    sup = rng.randn(K, d); L = n_leaves(level, K); leaf = labels(rng, L, n)
    if level == "star":
        centers = sup[leaf]
    elif level == "hier2":
        sub = (sup[:, None, :] + 0.5*rng.randn(K, 5, d)).reshape(K*5, d); centers = sub[leaf]
    else:
        sub = sup[:, None, :] + 0.5*rng.randn(K, 3, d); sub2 = (sub[:, :, None, :] + 0.25*rng.randn(K, 3, 3, d)).reshape(K*9, d); centers = sub2[leaf]
    return (centers + ratio*noise(rng, n, d, aniso, leaf)).astype(np.float32), leaf

GRID = [(lv, K, r, a, n, s) for lv in ("star","hier2","hier3") for K in (6,12,20,30) for r in (0.1,0.3,0.6)
        for a in ("iso","aniso") for n in (100,1000) for s in range(5)]

def run(part, Ks):
    csv_path = OUT/f"expR55b_depth_power_leafframe.part_{part}.csv"
    rows = pd.read_csv(csv_path).to_dict("records") if csv_path.exists() else []
    done = {(r["level"],r["K"],r["ratio"],r["aniso"],r["n"],r["seed"]) for r in rows}
    for lv, K, r, a, n, s in GRID:
        if K not in Ks or (lv,K,r,a,n,s) in done: continue
        if n_leaves(lv, K) > n: print(f"{lv} K{K} n{n}: infeasible ({n_leaves(lv,K)} leaves > {n} points), skipped"); continue
        t0 = time.time(); C, sup = make(lv, K, r, a, n, s)
        exB, exS, depth, z = depth_test(C, sup)
        rows.append(dict(level=lv, K=K, ratio=r, aniso=a, n=n, seed=s, excessB_real=exB, excessB_star=exS, depth=depth, z=z, time_s=time.time()-t0))
        print(f"{lv:5s} K{K:2d} r{r} {a:5s} n{n:4d} s{s}: depth {depth:+.4f} z {z:+.1f} ({rows[-1]['time_s']:.0f}s)")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Done part {part}")

def merge():
    parts = sorted(glob.glob(str(OUT/"expR55b_depth_power_leafframe.part_*.csv")))
    df = pd.concat([pd.read_csv(p) for p in parts]).drop_duplicates(subset=["level","K","ratio","aniso","n","seed"])
    df.to_csv(OUT/"expR55b_depth_power_leafframe.csv", index=False)
    h = df[df.level!="star"]; st = df[df.level=="star"]
    print(f"merged {len(df)} runs | power (hier, z<=-2): {(h.z<=-2).mean():.2f} | star false alarms |z|>=2: {(st.z.abs()>=2).mean():.2f} (z>=+2: {(st.z>=2).mean():.2f}, z<=-2: {(st.z<=-2).mean():.2f})")
    for n in (100,1000):
        for K in (6,12,20,30):
            hh = h[(h.n==n)&(h.K==K)]; ss = st[(st.n==n)&(st.K==K)]
            print(f"  n={n} K={K}: power {[(r, round((hh[hh.ratio==r].z<=-2).mean(),2)) for r in (0.1,0.3,0.6)]} | star |z|>=2 {(ss.z.abs()>=2).mean():.2f} | hier2 {(hh[hh.level=='hier2'].z<=-2).mean():.2f} hier3 {(hh[hh.level=='hier3'].z<=-2).mean():.2f} | iso {(hh[hh.aniso=='iso'].z<=-2).mean():.2f} aniso {(hh[hh.aniso=='aniso'].z<=-2).mean():.2f}")

def figure():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    FIG = Path(__file__).resolve().parents[2]/"iclr2027"/"figures"
    try: plt.style.use(str(FIG/"style.mplstyle"))
    except Exception: pass
    df = pd.read_csv(OUT/"expR55b_depth_power_leafframe.csv"); h = df[df.level!="star"]; st = df[df.level=="star"]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.9), sharey=True)
    cols = {6:"#4C72B0", 12:"#55A868", 20:"#DD8452", 30:"#8172B2"}
    for ax, n in zip(axes, (100, 1000)):
        for K in (6,12,20,30):
            hh = h[(h.n==n)&(h.K==K)]; ss = st[(st.n==n)&(st.K==K)]
            ax.plot([0.1,0.3,0.6], [(hh[hh.ratio==r].z<=-2).mean() for r in (0.1,0.3,0.6)], "-o", ms=3.5, color=cols[K], label=f"K={K}")
            ax.plot([0.1,0.3,0.6], [(ss[ss.ratio==r].z.abs()>=2).mean() for r in (0.1,0.3,0.6)], ":", color=cols[K], lw=1)
        ax.set_title(f"n = {n}"); ax.set_xlabel("within/between noise ratio"); ax.set_ylim(-0.03, 1.03); ax.set_xticks([0.1,0.3,0.6])
    axes[0].set_ylabel("power (z $\\leq$ $-$2 on hierarchies)")
    axes[0].plot([], [], ":", color="gray", label="star false alarms (|z| $\\geq$ 2)"); axes[0].legend(frameon=False, ncol=1, fontsize=6.5, loc="upper right")
    fig.tight_layout()
    for o in (FIG, FIG.parent/"iclr2027"/"figures"): fig.savefig(o/"fig_depth_power.pdf"); fig.savefig(o/"fig_depth_power.png", dpi=200)
    print("fig_depth_power written")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--part", default=None); ap.add_argument("--K", nargs="+", type=int, default=[6,12,20,30])
    ap.add_argument("--merge", action="store_true"); ap.add_argument("--figure", action="store_true"); A = ap.parse_args()
    if A.merge: merge()
    elif A.figure: figure()
    else:
        assert A.part; run(A.part, set(A.K))
