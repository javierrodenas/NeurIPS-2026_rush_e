#!/usr/bin/env python3
"""calibrated_delta.py -- the paper's instrument as a single script.

Input : a centroid matrix (.npy, n x d; one row per class) and, optionally, integer superclass
        labels (.npy, length n).
Output: the raw four-point statistic (default: the 99.9th percentile of the four-point defect over 5e5
        sampled quadruples per seed, mean +- s.d. over 10 seeds; --stat sup gives the supremum), its
        excess over the mean of N spectrum-matched null replicates (default: the Haar construction,
        exact sample spectrum; --null gauss gives Gaussian coefficients), the percentile rank r/N and
        the left-tail p-value p = (1 + #{null <= real}) / (N+1); in the paper "genuine" is a
        Benjamini-Hochberg-corrected p <= 0.05 across a census, which a single cell cannot compute, so
        the tool reports the uncorrected p. With labels: the matched-star depth test of Table B34
        (excess B under the Haar hub null for the real centroids and for a matched star; default: the
        anisotropic star, --star iso for the isotropic one; 10 star seeds).

Estimator, null constructions and seed scheme are copied verbatim from the scripts that produced the
paper's tables: expR52_census_haar_p999_200.py (census of record) and expR56_depth_variants.py
(depth test). Deterministic: the same matrix always gives the same numbers.

    python calibrated_delta.py centroids.npy [--labels sup.npy] [--reps 200] [--null haar|gauss] [--stat p999|sup] [--star aniso|iso] [--json out.json]
"""
import argparse, json, sys, time
import numpy as np
from scipy.spatial.distance import pdist, squareform

N_QUADS = 500_000

# ---------------- census (expR52: Haar null x p99.9 statistic is the record) ----------------
def delta_stat(X, n_seeds, stat="p999", n_quads=None):
    n_quads = n_quads or N_QUADS
    D = squareform(pdist(X, "euclidean")); diam = D.max(); n = len(D)
    out = []
    for s in range(n_seeds):
        rng = np.random.RandomState(s)
        i, j, k, l = (rng.randint(0, n, n_quads) for _ in range(4))
        ok = (i!=j)&(i!=k)&(i!=l)&(j!=k)&(j!=l)&(k!=l); i,j,k,l = i[ok],j[ok],k[ok],l[ok]
        S = np.sort(np.stack([D[i,j]+D[k,l], D[i,k]+D[j,l], D[i,l]+D[j,k]], 1), 1)
        dfc = (S[:,2]-S[:,1])/2
        out.append((dfc.max() if stat == "sup" else np.percentile(dfc, 99.9))/diam)
    return float(np.mean(out)), float(np.std(out))
def delta_norm(X, n_seeds): return delta_stat(X, n_seeds, "sup")   # the supremum (used by the depth test, as in expR56)

def haarnull(C, rep):
    """Spectrum-matched Haar null: random orthogonal coefficients, exact sample spectrum (the record)."""
    mu = C.mean(0); U, S, Vt = np.linalg.svd(C-mu, full_matrices=False)
    rng = np.random.RandomState(300+rep); Z = rng.randn(len(C), len(C)); Q, R = np.linalg.qr(Z); Q = Q*np.sign(np.diag(R))
    return ((Q[:, :len(S)]*S)@Vt + mu).astype(np.float32)

def specnull(C, rep):
    """Spectrum-matched Gaussian null: Gaussian coefficients recombined with the real singular values."""
    mu = C.mean(0); Cc = C - mu
    U, S, Vt = np.linalg.svd(Cc, full_matrices=False)
    rng = np.random.RandomState(300+rep)
    G = rng.randn(len(C), len(S)).astype(np.float32)
    G /= G.std(0, keepdims=True) * np.sqrt(len(C))
    return (G*S)@Vt + mu

def census(C, n_rep=200, null="haar", stat="p999", n_quads=None):
    nf = haarnull if null == "haar" else specnull
    dr, dr_sd = delta_stat(C, 10, stat, n_quads)
    nulls = np.array([delta_stat(nf(C, r), 5, stat, n_quads)[0] for r in range(n_rep)])
    nm, nsd = float(nulls.mean()), float(nulls.std(ddof=1))
    r_above = int((nulls > dr).sum()); p_left = (1 + int((nulls <= dr).sum())) / (n_rep + 1)
    return dict(n=int(C.shape[0]), d=int(C.shape[1]), null=null, stat=stat, delta=dr, delta_sd=dr_sd, null_mean=nm, null_sd=nsd,
                excess=dr-nm, z=(dr-nm)/max(np.sqrt(nsd**2+dr_sd**2), 1e-9), n_rep=n_rep,
                r_above=r_above, p_left=p_left, genuine_uncorrected=bool(p_left <= 0.05), n_quads=int(n_quads or N_QUADS))

# ---------------- matched-star depth test (expR56: anisotropic star, 10 seeds) ----------------
def haar_sample(M, rep, seed0=700):
    mu = M.mean(0); Mc = M - mu; U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
    rng = np.random.RandomState(seed0+rep); Z = rng.randn(len(M), len(M)); Q, R = np.linalg.qr(Z); Q = Q*np.sign(np.diag(R))
    return ((Q[:, :len(S)]*S)@Vt + mu).astype(np.float32)

def flatnull(C, sup, rep):
    hubs = np.stack([C[sup==s].mean(0) for s in range(sup.max()+1)]); return C - hubs[sup] + haar_sample(hubs, rep)[sup]

def excessB(C, sup, n_real=10, n_rep=10):
    dr, dr_sd = delta_norm(C, n_real); nB = [delta_norm(flatnull(C, sup, r), 3)[0] for r in range(n_rep)]
    return dr, dr_sd, float(dr-np.mean(nB)), float(np.std(nB, ddof=1))

def matched_star(C, sup, seed, variant="aniso"):
    rng = np.random.RandomState(seed); K = sup.max()+1; n, d = C.shape
    hubs = np.stack([C[sup==s].mean(0) for s in range(K)]); hub_rms = np.sqrt(((hubs-hubs.mean(0))**2).sum(1).mean())
    off = C - hubs[sup]
    H = rng.randn(K, d); H *= hub_rms/np.sqrt((H**2).sum(1).mean())
    if variant == "aniso_haarhubs":   # spectrum-matched hubs: a Haar resample of the real hubs (exact hub spectrum, random orientation)
        H = haar_sample(hubs, seed, seed0=5000) - hubs.mean(0)
    if variant == "iso":
        wr = np.array([np.sqrt((off[sup==s]**2).sum(1).mean()) for s in range(K)])
        Z = rng.randn(n, d); Z *= (wr[sup]/np.sqrt(d))[:, None]
    else:   # anisotropic: within each cluster a centred Haar sample with the cluster's own covariance
        Z = np.zeros_like(C)
        for s in range(K):
            m = sup == s
            if m.sum() < 2: continue
            Zs = haar_sample(C[m], 0, seed0=10_000*seed + s); Z[m] = Zs - Zs.mean(0)
    return (H[sup] + Z).astype(np.float32)

def depth_test(C, sup, variant="aniso", n_star=10):
    dr, dr_sd, exB, nBsd = excessB(C, sup)
    stars = [excessB(matched_star(C, sup, s, variant), sup, n_real=5, n_rep=5) for s in range(n_star)]
    exS = float(np.mean([x[2] for x in stars])); exS_sd = float(np.std([x[2] for x in stars], ddof=1))
    depth = exB - exS; z = depth/max(np.sqrt(exS_sd**2 + nBsd**2 + dr_sd**2), 1e-9)
    star_vals = [float(x[2]) for x in stars]; r_star = int(sum(v <= exB for v in star_vals))   # star seeds at least as hierarchical as the real cloud
    return dict(K=int(sup.max()+1), star=variant, n_star=n_star, excessB_real=exB, excessB_star=exS, excessB_star_sd=exS_sd,
                depth_excess=depth, z_depth=z, star_excesses=star_vals, r_star=r_star, p_star=(1 + r_star) / (n_star + 1))

def run(C, sup=None, n_rep=200, null="haar", stat="p999", star="aniso"):
    C = np.asarray(C, dtype=np.float32)
    out = {"census": census(C, n_rep, null, stat)}
    if sup is not None:
        sup = np.asarray(sup).astype(int); assert len(sup) == len(C), "labels must have one entry per centroid"
        out["depth"] = depth_test(C, sup, star)
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("centroids", help=".npy, n x d")
    ap.add_argument("--labels", default=None, help=".npy, integer superclass label per centroid (enables the depth test)")
    ap.add_argument("--reps", type=int, default=200, help="spectrum-null replicates (default 200)")
    ap.add_argument("--null", default="haar", choices=["haar", "gauss"]); ap.add_argument("--stat", default="p999", choices=["p999", "sup"])
    ap.add_argument("--star", default="aniso", choices=["aniso", "iso", "aniso_haarhubs"], help="matched-star variant for the depth test (aniso_haarhubs: Haar-resampled real hubs)")
    ap.add_argument("--json", default=None, help="write the results to this file")
    A = ap.parse_args()
    t0 = time.time()
    out = run(np.load(A.centroids), np.load(A.labels) if A.labels else None, A.reps, A.null, A.stat, A.star)
    c = out["census"]
    print(f"n={c['n']} d={c['d']}  raw {c['stat']} statistic {c['delta']:.4f} +- {c['delta_sd']:.4f}  ({c['null']} null)")
    print(f"spectrum-matched null: mean {c['null_mean']:.4f} (sd {c['null_sd']:.4f}, {c['n_rep']} replicates)")
    print(f"excess {c['excess']:+.4f}  r = {c['r_above']}/{c['n_rep']}  p = {c['p_left']:.4f}  -> {'below the null (uncorrected p <= 0.05)' if c['genuine_uncorrected'] else 'not below the null'}")
    if "depth" in out:
        dd = out["depth"]
        print(f"depth test (K={dd['K']} clusters, {dd['star']} star, {dd['n_star']} seeds): excess B real {dd['excessB_real']:+.4f}, matched star {dd['excessB_star']:+.4f}, "
              f"depth {dd['depth_excess']:+.4f} (z {dd['z_depth']:+.2f}; negative = hierarchy above the labelled clusters beyond the star)")
    print(f"({time.time()-t0:.0f}s)")
    if A.json: json.dump(out, open(A.json, "w"), indent=1)

if __name__ == "__main__":
    main()
