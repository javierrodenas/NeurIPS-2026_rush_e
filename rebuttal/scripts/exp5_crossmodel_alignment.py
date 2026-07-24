#!/usr/bin/env python3
"""
REBUTTAL EXP 5 — cross-model alignment, Euclidean vs hyperbolic
(answers the PRH-connection concern of RJje-minor and Xbn5 W2/Q4).

PRH's mutual-kNN alignment metric between every pair of the 12 vision panel
models, computed on the shared 1000 ImageNet class centroids:

  align(A, B) = mean_j |kNN_A(j) ∩ kNN_B(j)| / k        (k = 10)

once with Euclidean kNN on raw centroids (R) and once with Poincare kNN
after the paper's parameter-free projection (H).  If H-alignment > R-alignment,
the cross-model shared structure that PRH measures is *more* visible in the
hyperbolic metric — a direct, quantitative bridge from our within-model
geometry to PRH's cross-model convergence.

Output: rebuttal/results/exp5_crossmodel_alignment.csv
"""
import os, sys, time
sys.stdout.reconfigure(line_buffering=True)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist, squareform

ROOT = Path("/home/javi/Platonic")
CACHE = ROOT / "results/practical_tasks_cache"
OUT = ROOT / "rebuttal/results"
OUT.mkdir(parents=True, exist_ok=True)

PARADIGMS = {
    "i21k_t":"Supervised","i21k_s":"Supervised","i21k_b":"Supervised","i21k_l":"Supervised",
    "dinov1_b":"SSL","dinov2_s":"SSL","dinov2_b":"SSL","dinov2_l":"SSL","dinov2_g":"SSL",
    "clip_b":"Contrastive","clip_l":"Contrastive","siglip_b":"Contrastive",
}
PER_CLASS = 100
K = 10
TARGET = 0.70710678


def load_centroids(model):
    d = np.load(CACHE / f"{model}_imagenet_fulltrain.npz")
    X_full = d["features"].astype(np.float32); y_full = d["labels"].astype(np.int64)
    rng = np.random.RandomState(0)
    sel = []
    for c in range(int(y_full.max()) + 1):
        ic = np.where(y_full == c)[0]
        sel.extend(rng.choice(ic, min(PER_CLASS, len(ic)), replace=False).tolist())
    X = X_full[np.array(sel)]; y = y_full[np.array(sel)]
    n_classes = int(y.max()) + 1
    return np.stack([X[y == c].mean(0) for c in range(n_classes)])


def s4_project(X, target=TARGET):
    mu = X.mean(0)
    Xc = X - mu
    p95 = np.percentile(np.linalg.norm(Xc, axis=1), 95)
    s = 2 * np.arctanh(target) / max(p95, 1e-7)
    Xs = Xc * s
    nrm = np.linalg.norm(Xs, axis=1, keepdims=True).clip(min=1e-7)
    Y = np.tanh(nrm/2) / nrm * Xs
    cur = np.linalg.norm(Y, axis=1, keepdims=True).clip(min=1e-7)
    return Y * np.clip((1-1e-3)/cur, None, 1.0)


def poincare_D(Y):
    sq = (Y**2).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2*Y@Y.T, 0)
    denom = np.maximum((1-sq[:, None])*(1-sq[None, :]), 1e-12)
    return np.arccosh(np.maximum(1 + 2*d2/denom, 1+1e-12))


def knn_sets(D, k=K):
    nn = np.argsort(D, axis=1)[:, 1:k+1]
    return nn


def mutual_knn(nnA, nnB, k=K):
    inter = [len(set(nnA[j]).intersection(nnB[j])) for j in range(len(nnA))]
    return float(np.mean(inter) / k)


def main():
    models = list(PARADIGMS)
    nn_R, nn_H = {}, {}
    for m in models:
        t0 = time.time()
        C = load_centroids(m)
        nn_R[m] = knn_sets(squareform(pdist(C)))
        nn_H[m] = knn_sets(poincare_D(s4_project(C)))
        print(f"loaded {m} ({time.time()-t0:.0f}s)", flush=True)

    rows = []
    for i, a in enumerate(models):
        for b in models[i+1:]:
            aR = mutual_knn(nn_R[a], nn_R[b])
            aH = mutual_knn(nn_H[a], nn_H[b])
            rows.append(dict(model_a=a, model_b=b,
                             par_a=PARADIGMS[a], par_b=PARADIGMS[b],
                             align_R=aR, align_H=aH, gain=aH-aR))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "exp5_crossmodel_alignment.csv", index=False)
    print(df[["align_R","align_H","gain"]].mean().to_string())
    both_ssl = df[(df.par_a=="SSL") & (df.par_b=="SSL")]
    print("SSL-SSL pairs:", both_ssl[["align_R","align_H","gain"]].mean().to_string())
    print("Done ->", OUT / "exp5_crossmodel_alignment.csv")


if __name__ == "__main__":
    main()
