#!/usr/bin/env python3
"""calibrated_delta.py -- the paper's instrument as a single script.

Input : a centroid matrix (.npy, n x d; one row per class) and, optionally, integer superclass
        labels (.npy, length n).
Output: raw delta_norm (mean +- s.d. over 10 quadruple seeds), the spectrum-matched excess with its
        percentile rank r/N and left-tail p-value p = (1 + #{null <= real}) / (N+1) over N = 200
        spectrum-null replicates (genuine := p <= 0.05), and, when labels are given, the matched-star
        depth test of Table B29 (excess B under the Haar hub null, for the real centroids and for a
        matched star, and their difference with a z against the combined spread).

Estimator, null constructions and seed scheme are copied verbatim from the scripts that produced the
paper's tables: expR39c_census200_cache.py (census: 5e5 quadruples per seed, 10 real seeds, null
seeds 300+rep with 5 quadruple seeds each) and expR50_depth_test.py (depth test: hub null seeds
700+rep, 10 replicates x 3 seeds; matched stars seeds 0..2 with 5 real seeds and 5 replicates).
Deterministic: the same matrix always gives the same numbers.

    python calibrated_delta.py centroids.npy [--labels sup.npy] [--reps 200] [--json out.json]
"""
import argparse, json, sys, time
import numpy as np
from scipy.spatial.distance import pdist, squareform

N_QUADS = 500_000

# ---------------- census (expR39c) ----------------
def delta_norm(X, n_seeds):
    D = squareform(pdist(X, "euclidean")); diam = D.max(); n = len(D)
    out = []
    for s in range(n_seeds):
        rng = np.random.RandomState(s)
        i, j, k, l = (rng.randint(0, n, N_QUADS) for _ in range(4))
        ok = (i!=j)&(i!=k)&(i!=l)&(j!=k)&(j!=l)&(k!=l); i,j,k,l = i[ok],j[ok],k[ok],l[ok]
        S = np.sort(np.stack([D[i,j]+D[k,l], D[i,k]+D[j,l], D[i,l]+D[j,k]], 1), 1)
        out.append(((S[:,2]-S[:,1])/2).max()/diam)
    return float(np.mean(out)), float(np.std(out))

def specnull(C, rep):
    """Spectrum-matched Gaussian null: Gaussian coefficients recombined with the real singular values."""
    mu = C.mean(0); Cc = C - mu
    U, S, Vt = np.linalg.svd(Cc, full_matrices=False)
    rng = np.random.RandomState(300+rep)
    G = rng.randn(len(C), len(S)).astype(np.float32)
    G /= G.std(0, keepdims=True) * np.sqrt(len(C))
    return (G*S)@Vt + mu

def census(C, n_rep=200):
    dr, dr_sd = delta_norm(C, 10)
    nulls = np.array([delta_norm(specnull(C, r), 5)[0] for r in range(n_rep)])
    nm, nsd = float(nulls.mean()), float(nulls.std(ddof=1))
    r_above = int((nulls > dr).sum()); p_left = (1 + int((nulls <= dr).sum())) / (n_rep + 1)
    return dict(n=int(C.shape[0]), d=int(C.shape[1]), delta=dr, delta_sd=dr_sd, null_mean=nm, null_sd=nsd,
                excess=dr-nm, z=(dr-nm)/max(np.sqrt(nsd**2+dr_sd**2), 1e-9), n_rep=n_rep,
                r_above=r_above, p_left=p_left, genuine=bool(p_left <= 0.05))

# ---------------- matched-star depth test (expR50) ----------------
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
    rng = np.random.RandomState(seed); K = sup.max()+1; n, d = C.shape
    hubs = np.stack([C[sup==s].mean(0) for s in range(K)]); hub_rms = np.sqrt(((hubs-hubs.mean(0))**2).sum(1).mean())
    off = C - hubs[sup]; wr = np.array([np.sqrt((off[sup==s]**2).sum(1).mean()) for s in range(K)])
    H = rng.randn(K, d); H *= hub_rms/np.sqrt((H**2).sum(1).mean())
    Z = rng.randn(n, d); Z *= (wr[sup]/np.sqrt(d))[:, None]
    return (H[sup] + Z).astype(np.float32)

def depth_test(C, sup):
    dr, dr_sd, exB, nBsd = excessB(C, sup)
    stars = [excessB(matched_star(C, sup, s), sup, n_real=5, n_rep=5) for s in range(3)]
    exS = float(np.mean([x[2] for x in stars])); exS_sd = float(np.std([x[2] for x in stars], ddof=1))
    depth = exB - exS; z = depth/max(np.sqrt(exS_sd**2 + nBsd**2 + dr_sd**2), 1e-9)
    return dict(K=int(sup.max()+1), excessB_real=exB, excessB_star=exS, excessB_star_sd=exS_sd,
                depth_excess=depth, z_depth=z)

def run(C, sup=None, n_rep=200):
    C = np.asarray(C, dtype=np.float32)
    out = {"census": census(C, n_rep)}
    if sup is not None:
        sup = np.asarray(sup).astype(int); assert len(sup) == len(C), "labels must have one entry per centroid"
        out["depth"] = depth_test(C, sup)
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("centroids", help=".npy, n x d")
    ap.add_argument("--labels", default=None, help=".npy, integer superclass label per centroid (enables the depth test)")
    ap.add_argument("--reps", type=int, default=200, help="spectrum-null replicates (default 200)")
    ap.add_argument("--json", default=None, help="write the results to this file")
    A = ap.parse_args()
    t0 = time.time()
    out = run(np.load(A.centroids), np.load(A.labels) if A.labels else None, A.reps)
    c = out["census"]
    print(f"n={c['n']} d={c['d']}  raw delta_norm {c['delta']:.4f} +- {c['delta_sd']:.4f}")
    print(f"spectrum-matched null: mean {c['null_mean']:.4f} (sd {c['null_sd']:.4f}, {c['n_rep']} replicates)")
    print(f"excess {c['excess']:+.4f}  r = {c['r_above']}/{c['n_rep']}  p = {c['p_left']:.4f}  -> {'GENUINE (p <= 0.05)' if c['genuine'] else 'not genuine'}")
    if "depth" in out:
        dd = out["depth"]
        print(f"depth test (K={dd['K']} superclasses): excess B real {dd['excessB_real']:+.4f}, matched star {dd['excessB_star']:+.4f}, "
              f"depth {dd['depth_excess']:+.4f} (z {dd['z_depth']:+.2f}; negative = more tree-like than a matched star)")
    print(f"({time.time()-t0:.0f}s)")
    if A.json: json.dump(out, open(A.json, "w"), indent=1)

if __name__ == "__main__":
    main()
