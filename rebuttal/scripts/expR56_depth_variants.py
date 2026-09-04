#!/usr/bin/env python3
"""expR56 (review-response, A2): the matched-star depth test on the real backbones, with
  (a) the isotropic star of Table B29 (expR50) and (b) an ANISOTROPIC star whose within-cluster cloud is
      a Haar-rotated sample with the real superclass cloud's covariance spectrum (haar_sample applied
      within each cluster), hubs Gaussian with matched RMS in both; 10 star seeds instead of 3 (A5);
  K sweep: CIFAR-100 K in {20, 10, 5}; ImageNet WordNet cuts at {30, 10, 60} superclasses.
Frames: CIFAR-100 K=20 = the coarse labels; K=10/5 = average-linkage agglomerative clustering of the 20
coarse superclasses on the model-averaged (12 backbones, each normalized by its mean) hub-distance matrix,
fixed once and shared by all models (written to expR56_frames_cifar100.csv). ImageNet cuts: agglomerative
clustering (average, precomputed) of the WordNet distance matrix, as the 30-cut of expR50.
Centroids: census cache for every dataset (ImageNet included; expR50 read ImageNet from the store).
Everything else (hub null seeds 700+rep, 10 replicates x 3 seeds, star runs with 5 real seeds x 5 replicates)
as in expR50. Output: expR56_depth_variants.csv.
"""
import os, sys, time, argparse, glob
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
ROOT = Path(os.environ.get("PLATONIC_ROOT", "/media/HDD_4TB_2/javi/Platonic"))
OUT = Path(os.environ.get("PLATONIC_RESULTS", str(Path(__file__).resolve().parents[1]/"results")))
CACHE = ROOT/"results/practical_tasks_cache"
MODELS = ["i21k_t","i21k_s","i21k_b","i21k_l","dinov1_b","dinov2_s","dinov2_b","dinov2_l","dinov2_g","clip_b","clip_l","siglip_b"]
COARSE = {0:[4,30,55,72,95],1:[1,32,67,73,91],2:[54,62,70,82,92],3:[9,10,16,28,61],4:[0,51,53,57,83],5:[22,39,40,86,87],6:[5,20,25,84,94],7:[6,7,14,18,24],8:[3,42,43,88,97],9:[12,17,37,68,76],10:[23,33,49,60,71],11:[15,19,21,31,38],12:[34,63,64,66,75],13:[26,45,77,79,99],14:[2,11,35,46,98],15:[27,29,44,78,93],16:[36,50,65,74,80],17:[47,52,56,59,96],18:[8,13,48,58,90],19:[41,69,81,85,89]}
SUP100 = np.zeros(100, dtype=int)
for s, cls in COARSE.items():
    for c in cls: SUP100[c] = s
N_STAR = 10

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
def matched_star(C, sup, seed, variant):
    rng = np.random.RandomState(seed); K = sup.max()+1; n, d = C.shape
    hubs = np.stack([C[sup==s].mean(0) for s in range(K)]); hub_rms = np.sqrt(((hubs-hubs.mean(0))**2).sum(1).mean())
    off = C - hubs[sup]
    H = rng.randn(K, d); H *= hub_rms/np.sqrt((H**2).sum(1).mean())
    if variant == "iso":
        wr = np.array([np.sqrt((off[sup==s]**2).sum(1).mean()) for s in range(K)])
        Z = rng.randn(n, d); Z *= (wr[sup]/np.sqrt(d))[:, None]
    else:   # aniso: within-cluster Haar sample with the cluster's own covariance spectrum (and directions), centered
        Z = np.empty_like(C)
        for s in range(K):
            m = sup == s; Cs = C[m]
            if m.sum() < 2: Z[m] = 0.0; continue
            Zs = haar_sample(Cs, 0, seed0=10_000*seed + s); Z[m] = Zs - Zs.mean(0)
    return (H[sup] + Z).astype(np.float32)
def depth_test(C, sup, variant, n_star=N_STAR):
    dr, dr_sd, exB, nBsd = excessB(C, sup)
    stars = [excessB(matched_star(C, sup, s, variant), sup, n_real=5, n_rep=5) for s in range(n_star)]
    exS = float(np.mean([x[2] for x in stars])); exS_sd = float(np.std([x[2] for x in stars], ddof=1))
    depth = exB - exS; z = depth/max(np.sqrt(exS_sd**2 + nBsd**2 + dr_sd**2), 1e-9)
    return dict(excessB_real=exB, excessB_star=exS, excessB_star_sd=exS_sd, depth_excess=depth, z_depth=z, delta=dr, delta_sd=dr_sd)

def cents(m, ds):
    d = np.load(CACHE/f"{m}_{ds}_train.npz"); X = d["features"].astype(np.float32); y = d["labels"].astype(np.int64)
    return np.stack([X[y==c].mean(0) for c in range(int(y.max())+1)])

def frames():
    """CIFAR-100: K=20 coarse; K=10/5 consensus clustering of the 20 coarse hubs. ImageNet: WordNet cuts 30/10/60."""
    f = OUT/"expR56_frames_cifar100.csv"
    if f.exists():
        fr = pd.read_csv(f); c100 = {int(k): fr[f"K{k}"].values.astype(int) for k in (20, 10, 5)}
    else:
        Dsum = np.zeros((20, 20))
        for m in MODELS:
            C = cents(m, "cifar100"); hubs = np.stack([C[SUP100==s].mean(0) for s in range(20)])
            D = squareform(pdist(hubs)); Dsum += D/D.mean()
        Dsum /= len(MODELS)
        c100 = {20: SUP100.copy()}
        for K in (10, 5):
            g = AgglomerativeClustering(n_clusters=K, metric="precomputed", linkage="average").fit_predict(Dsum)
            c100[K] = g[SUP100]
        pd.DataFrame({"cls": np.arange(100), **{f"K{k}": v for k, v in c100.items()}}).to_csv(f, index=False)
    WN = np.load(ROOT/"results/features/imagenet/imagenet1k_wordnet_dist.npy")
    inet = {K: AgglomerativeClustering(n_clusters=K, metric="precomputed", linkage="average").fit_predict(WN) for K in (30, 10, 60)}
    return c100, inet

def run(part, models):
    c100, inet = frames()
    csv_path = OUT/f"expR56_depth_variants.part_{part}.csv"
    rows = pd.read_csv(csv_path).to_dict("records") if csv_path.exists() else []
    done = {(r["model"], r["dataset"], r["K"], r["variant"]) for r in rows}
    for ds, fr in (("cifar100", c100), ("imagenet", inet)):
        for m in models:
            C = cents(m, ds)
            for K, sup in fr.items():
                for variant in ("iso", "aniso"):
                    if (m, ds, K, variant) in done: continue
                    t0 = time.time(); res = depth_test(C, np.asarray(sup).astype(int), variant)
                    rows.append(dict(model=m, dataset=ds, K=K, variant=variant, n_star=N_STAR, **res, time_s=time.time()-t0))
                    print(f"{m:9s} {ds:9s} K{K:2d} {variant:5s} real {res['excessB_real']:+.4f} star {res['excessB_star']:+.4f} depth {res['depth_excess']:+.4f} z {res['z_depth']:+.1f} ({rows[-1]['time_s']:.0f}s)")
                    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Done part {part}")

def merge():
    parts = sorted(glob.glob(str(OUT/"expR56_depth_variants.part_*.csv")))
    df = pd.concat([pd.read_csv(p) for p in parts]).drop_duplicates(subset=["model","dataset","K","variant"])
    df.to_csv(OUT/"expR56_depth_variants.csv", index=False)
    for (ds, K, v), g in df.groupby(["dataset","K","variant"]):
        print(f"{ds:9s} K{K:2d} {v:5s}: depth mean {g.depth_excess.mean():+.4f} | z<=-2 {int((g.z_depth<=-2).sum())}/{len(g)} | z>=+2 {int((g.z_depth>=2).sum())}/{len(g)} | max z {g.z_depth.max():+.1f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--part", default=None); ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--merge", action="store_true"); A = ap.parse_args()
    if A.merge: merge()
    else:
        assert A.part; run(A.part, A.models)
