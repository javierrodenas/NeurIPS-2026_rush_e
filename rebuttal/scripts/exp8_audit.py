#!/usr/bin/env python3
"""
REBUTTAL EXP 8 — audit/robustness pass over the rebuttal experiments,
focused on the anomalous DINOv2 results.

P1  CIFAR-100 superclass recovery: is the ARI~0.00 of DINOv2-B/L/G an
    average-linkage chaining artifact?  Methods {average, complete, ward,
    kmeans} x spaces {Euclid, cosine, Poincare}; report ARI/NMI and the
    max cluster size at the 20-cut (degeneracy detector).
P2  WordNet alignment robustness: Spearman on Euclid (reproduces exp3?),
    Spearman on cosine distances, and scale-free triplet agreement
    (Euclid + cosine), all 12 models.
P3  ORC bridge with INTRINSIC clusters: split kNN-graph edges by the
    model's OWN Ward-30 clusters (not WordNet-30).  Does DINOv2 now show
    the within>across gap?
P4  Dimension-matching robustness: delta at PCA-192 / PCA-384 (+ %var
    retained) and Gaussian random projection to 192 (3 seeds).
P5  PC-permutation null (exact spectrum + marginals): independent check
    of the exp1c spectrum-null conclusion.
P6  Pooling curve: WordNet rho for m in {1,10,100} imgs/class,
    representative models.

Output: rebuttal/results/exp8_{p1..p6}.csv
"""
import os, sys, time
sys.stdout.reconfigure(line_buffering=True)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path("/home/javi/Platonic"); CACHE = ROOT/"results/practical_tasks_cache"
OUT = ROOT/"rebuttal/results"
WN = np.load(ROOT/"results/features/imagenet/imagenet1k_wordnet_dist.npy")
IU = np.triu_indices(1000, 1)
PARADIGMS = {"i21k_t":"Supervised","i21k_s":"Supervised","i21k_b":"Supervised","i21k_l":"Supervised",
 "dinov1_b":"SSL","dinov2_s":"SSL","dinov2_b":"SSL","dinov2_l":"SSL","dinov2_g":"SSL",
 "clip_b":"Contrastive","clip_l":"Contrastive","siglip_b":"Contrastive"}
COARSE = {0:[4,30,55,72,95],1:[1,32,67,73,91],2:[54,62,70,82,92],3:[9,10,16,28,61],4:[0,51,53,57,83],
 5:[22,39,40,86,87],6:[5,20,25,84,94],7:[6,7,14,18,24],8:[3,42,43,88,97],9:[12,17,37,68,76],
 10:[23,33,49,60,71],11:[15,19,21,31,38],12:[34,63,64,66,75],13:[26,45,77,79,99],
 14:[2,11,35,46,98],15:[27,29,44,78,93],16:[36,50,65,74,80],17:[47,52,56,59,96],
 18:[8,13,48,58,90],19:[41,69,81,85,89]}
SUP = np.zeros(100, dtype=int)
for c, fs in COARSE.items():
    for f in fs: SUP[f] = c
TARGET = 0.70710678

def s4(X):
    mu = X.mean(0); Xc = X-mu
    p95 = np.percentile(np.linalg.norm(Xc,axis=1),95)
    s = 2*np.arctanh(TARGET)/max(p95,1e-7); Xs = Xc*s
    nrm = np.linalg.norm(Xs,axis=1,keepdims=True).clip(min=1e-7)
    Y = np.tanh(nrm/2)/nrm*Xs
    cur = np.linalg.norm(Y,axis=1,keepdims=True).clip(min=1e-7)
    return Y*np.clip((1-1e-3)/cur, None, 1.0)

def poinc_D(Y):
    sq = (Y**2).sum(1)
    d2 = np.maximum(sq[:,None]+sq[None,:]-2*Y@Y.T,0)
    den = np.maximum((1-sq[:,None])*(1-sq[None,:]),1e-12)
    return np.arccosh(np.maximum(1+2*d2/den,1+1e-12))

def cos_D(X):
    Xn = X/np.linalg.norm(X,axis=1,keepdims=True).clip(min=1e-12)
    return 1.0 - Xn@Xn.T

def delta_max(X=None, D=None, n_quads=500_000, n_seeds=10):
    if D is None: D = squareform(pdist(X,'euclidean'))
    diam = D.max(); n = len(D); out=[]
    for s in range(n_seeds):
        rng = np.random.RandomState(s)
        i,j,k,l = (rng.randint(0,n,n_quads) for _ in range(4))
        ok=(i!=j)&(i!=k)&(i!=l)&(j!=k)&(j!=l)&(k!=l); i,j,k,l=i[ok],j[ok],k[ok],l[ok]
        S = np.sort(np.stack([D[i,j]+D[k,l],D[i,k]+D[j,l],D[i,l]+D[j,k]],1),1)
        out.append(((S[:,2]-S[:,1])/2).max()/diam)
    return float(np.mean(out)), float(np.std(out))

def orc_edges(D, k=10):
    n = len(D)
    nn = np.argsort(D, axis=1)[:, 1:k+1]
    edges = set()
    for u in range(n):
        for v in nn[u]: edges.add((min(u,int(v)), max(u,int(v))))
    out = []
    for (u, v) in edges:
        M = D[np.ix_(nn[u], nn[v])]
        r, c = linear_sum_assignment(M)
        out.append((u, v, 1.0 - M[r,c].mean()/max(D[u,v],1e-12)))
    return out

def cents_from(model, ds):
    d = np.load(CACHE/f"{model}_{ds}_train.npz")
    X = d["features"].astype(np.float32); y = d["labels"].astype(np.int64)
    n_classes = int(y.max())+1
    return np.stack([X[y==c].mean(0) for c in range(n_classes)]), X, y

def cut_score(Z, true, n=20):
    cut = fcluster(Z, t=n, criterion="maxclust")
    return (adjusted_rand_score(true, cut), normalized_mutual_info_score(true, cut),
            int(np.bincount(cut).max()))

# ─── P1: CIFAR-100 recovery robustness ────────────────────────────────────
rows1 = []
for m in PARADIGMS:
    C, _, _ = cents_from(m, "cifar100")
    D_r = squareform(pdist(C)); D_c = cos_D(C); D_h = poinc_D(s4(C))
    variants = {}
    for sp, D in [("R", D_r), ("COS", D_c), ("H", D_h)]:
        cd = squareform(D, checks=False)
        for meth in ["average", "complete"]:
            variants[f"{meth}_{sp}"] = cut_score(linkage(cd, method=meth), SUP)
    variants["ward_R"] = cut_score(linkage(C, method="ward"), SUP)
    km = KMeans(n_clusters=20, n_init=10, random_state=0).fit_predict(C)
    variants["kmeans_R"] = (adjusted_rand_score(SUP, km),
                            normalized_mutual_info_score(SUP, km),
                            int(np.bincount(km).max()))
    for k, (ari, nmi, mx) in variants.items():
        meth, sp = k.rsplit("_", 1)
        rows1.append(dict(model=m, paradigm=PARADIGMS[m], method=meth, space=sp,
                          ari=ari, nmi=nmi, max_cluster=mx))
    pd.DataFrame(rows1).to_csv(OUT/"exp8_p1_recovery.csv", index=False)
    best = max(variants.items(), key=lambda kv: kv[1][0])
    print(f"P1 {m:10s} avg_R ari={variants['average_R'][0]:.3f}(mx{variants['average_R'][2]}) "
          f"ward_R={variants['ward_R'][0]:.3f} kmeans={variants['kmeans_R'][0]:.3f} "
          f"best={best[0]}:{best[1][0]:.3f}", flush=True)

# ─── P2: WordNet alignment robustness ─────────────────────────────────────
def triplet_agreement(D, Dw, n_trip=200_000, seed=0):
    rng = np.random.RandomState(seed); n = len(D)
    i, j, k = rng.randint(0, n, (3, n_trip))
    ok = (i!=j)&(i!=k)&(j!=k); i,j,k = i[ok],j[ok],k[ok]
    w_closer = np.sign(Dw[i,k]-Dw[i,j]); keep = w_closer != 0
    i,j,k,w = i[keep],j[keep],k[keep],w_closer[keep]
    f_closer = np.sign(D[i,k]-D[i,j])
    return float((f_closer == w).mean())

rows2 = []
for m in PARADIGMS:
    C, _, _ = cents_from(m, "imagenet")
    D_r = squareform(pdist(C)); D_c = cos_D(C)
    r_eu = spearmanr(D_r[IU], WN[IU]).statistic
    r_co = spearmanr(D_c[IU], WN[IU]).statistic
    t_eu = triplet_agreement(D_r, WN); t_co = triplet_agreement(D_c, WN)
    rows2.append(dict(model=m, paradigm=PARADIGMS[m], rho_eucl=r_eu, rho_cos=r_co,
                      triplet_eucl=t_eu, triplet_cos=t_co))
    pd.DataFrame(rows2).to_csv(OUT/"exp8_p2_alignment.csv", index=False)
    print(f"P2 {m:10s} rho_eu={r_eu:+.3f} rho_cos={r_co:+.3f} "
          f"trip_eu={t_eu:.3f} trip_cos={t_co:.3f}", flush=True)

# ─── P3: ORC bridge with intrinsic (own) clusters ─────────────────────────
rows3 = []
for m in PARADIGMS:
    C, _, _ = cents_from(m, "imagenet")
    D = squareform(pdist(C))
    own = AgglomerativeClustering(n_clusters=30, metric="precomputed",
                                  linkage="average").fit_predict(D)
    ed = orc_edges(D)
    kap = np.array([e[2] for e in ed])
    within = np.array([own[e[0]] == own[e[1]] for e in ed])
    rows3.append(dict(model=m, paradigm=PARADIGMS[m], n_edges=len(ed),
                      frac_within=float(within.mean()),
                      orc_within=float(kap[within].mean()),
                      orc_across=float(kap[~within].mean()) if (~within).sum() else np.nan,
                      fneg_within=float((kap[within]<0).mean()),
                      fneg_across=float((kap[~within]<0).mean()) if (~within).sum() else np.nan))
    pd.DataFrame(rows3).to_csv(OUT/"exp8_p3_orc_intrinsic.csv", index=False)
    r = rows3[-1]
    print(f"P3 {m:10s} own-clusters orc w/a={r['orc_within']:+.3f}/{r['orc_across']:+.3f} "
          f"fneg w/a={r['fneg_within']:.3f}/{r['fneg_across']:.3f}", flush=True)

# ─── P4: dimension-matching robustness ────────────────────────────────────
rows4 = []
for m in PARADIGMS:
    C, _, _ = cents_from(m, "imagenet")
    Cc = C - C.mean(0)
    U, S, Vt = np.linalg.svd(Cc, full_matrices=False)
    tot = (S**2).sum()
    for dc in [192, 384]:
        if C.shape[1] < dc:
            continue
        Cp = U[:, :dc]*S[:dc]
        dm, dsd = delta_max(Cp)
        rows4.append(dict(model=m, paradigm=PARADIGMS[m], variant=f"pca{dc}",
                          delta=dm, delta_std=dsd,
                          var_retained=float((S[:dc]**2).sum()/tot)))
    rps = []
    for sd in range(3):
        rng = np.random.RandomState(sd)
        P = rng.randn(C.shape[1], 192).astype(np.float32)/np.sqrt(192)
        rps.append(delta_max(Cc@P, n_seeds=5)[0])
    rows4.append(dict(model=m, paradigm=PARADIGMS[m], variant="rp192",
                      delta=float(np.mean(rps)), delta_std=float(np.std(rps)),
                      var_retained=np.nan))
    pd.DataFrame(rows4).to_csv(OUT/"exp8_p4_dimmatch.csv", index=False)
    pr = [r for r in rows4 if r["model"]==m]
    print(f"P4 {m:10s} " + "  ".join(f"{r['variant']}={r['delta']:.4f}" for r in pr), flush=True)

# ─── P5: PC-permutation null (spectrum + marginals preserved) ─────────────
rows5 = []
for m in PARADIGMS:
    C, _, _ = cents_from(m, "imagenet")
    Cc = C - C.mean(0)
    U, S, Vt = np.linalg.svd(Cc, full_matrices=False)
    US = U*S
    d_real, _ = delta_max(C)
    nulls = []
    for rep in range(3):
        rng = np.random.RandomState(200+rep)
        USp = np.stack([US[rng.permutation(len(US)), c] for c in range(US.shape[1])], 1)
        nulls.append(delta_max(USp@Vt, n_seeds=5)[0])
    rows5.append(dict(model=m, paradigm=PARADIGMS[m], delta_real=d_real,
                      delta_permnull=float(np.mean(nulls)),
                      permnull_std=float(np.std(nulls)),
                      excess=d_real-float(np.mean(nulls))))
    pd.DataFrame(rows5).to_csv(OUT/"exp8_p5_permnull.csv", index=False)
    r = rows5[-1]
    print(f"P5 {m:10s} real={r['delta_real']:.4f} permnull={r['delta_permnull']:.4f} "
          f"excess={r['excess']:+.4f}", flush=True)

# ─── P6: pooling curve ────────────────────────────────────────────────────
rows6 = []
for m in ["i21k_l", "dinov2_l", "dinov2_g", "clip_l"]:
    _, X, y = cents_from(m, "imagenet")
    for mm in [1, 10, 100]:
        rhos = []
        draws = 3 if mm < 100 else 1
        for sd in range(draws):
            rng = np.random.RandomState(sd)
            cents = []
            for c in range(1000):
                ic = np.where(y==c)[0]
                pick = ic if mm >= len(ic) else rng.choice(ic, mm, replace=False)
                cents.append(X[pick].mean(0))
            Dm = squareform(pdist(np.stack(cents)))
            rhos.append(spearmanr(Dm[IU], WN[IU]).statistic)
        rows6.append(dict(model=m, m_imgs=mm, rho_mean=float(np.mean(rhos)),
                          rho_std=float(np.std(rhos))))
        pd.DataFrame(rows6).to_csv(OUT/"exp8_p6_pooling.csv", index=False)
        print(f"P6 {m:10s} m={mm:3d} rho={np.mean(rhos):+.3f}±{np.std(rhos):.3f}", flush=True)

print("Done")
