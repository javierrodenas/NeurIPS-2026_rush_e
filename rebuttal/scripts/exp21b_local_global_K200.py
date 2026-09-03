#!/usr/bin/env python3
"""exp21b (final pass, R2): the Groger calibration done Groger's way.

exp21_local_global_calibrated.py used 3 permutations and reported raw - null mean (the
"null-centered" variant of Groger et al.'s Appendix E.3). This re-run keeps the metrics
(mutual-kNN with k=10, identical implementation to exp5; linear CKA) and the 66 pairs, and
applies the calibration of Groger, Wen & Brbic (2026) verbatim, SCALAR calibration only
(one score per pair; no layer search):

  K = 200 permutations of the class correspondence (np.random.RandomState(0)),
  tau95   = the ceil((1-alpha)(K+1))-th order statistic of {raw} U {nulls}, alpha = 0.05  (eq. 9)
  p       = (1 + #{null >= raw}) / (K + 1)                                               (eq. 10)
  cal     = max((raw - tau95) / (s_max - tau95), 0), s_max = 1 for mKNN and CKA          (eq. 12)

plus the old null-centered raw - null_mean for continuity. Centroids: the paper's class
representation (Sec. 3), the mean of the 100 cached ImageNet training embeddings per class
({m}_imagenet_train.npz, the census cache regenerated locally by extract_imagenet_cache_local.py).
DEVIATION from exp21/exp5, which used a 100/class RandomState(0) subsample of the full train:
that subsample would need a second 100k-image extraction per model; the census cache is the
protocol the paper states, and every model is treated identically. Raw values therefore differ
slightly from exp21's (recorded in CHANGELOG_final.md). The analytic mKNN chance level
k/(n-1) = 10/999 is printed as a check.

Output: rebuttal/results/exp21b_local_global_K200.csv
"""
import os, sys, time, itertools, math
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist, squareform

ROOT = Path(os.environ.get("PLATONIC_ROOT", "/media/HDD_4TB_2/javi/Platonic"))
OUT = Path(os.environ.get("PLATONIC_RESULTS", str(Path(__file__).resolve().parents[1]/"results")))
CACHE = ROOT/"results/practical_tasks_cache"

MODELS = ["i21k_t","i21k_s","i21k_b","i21k_l","dinov1_b","dinov2_s","dinov2_b","dinov2_l","dinov2_g","clip_b","clip_l","siglip_b"]
K_NN = 10; K_PERM = 200; ALPHA = 0.05; TARGET = 0.70710678

def load_centroids(model):
    d = np.load(CACHE/f"{model}_imagenet_train.npz")   # census cache: the 100 cached training embeddings per class of Sec. 3
    X = d["features"].astype(np.float32); y = d["labels"].astype(np.int64)
    return np.stack([X[y == c].mean(0) for c in range(int(y.max()) + 1)])

def s4_project(X, target=TARGET):                     # exp5, verbatim
    mu = X.mean(0); Xc = X - mu
    p95 = np.percentile(np.linalg.norm(Xc, axis=1), 95)
    s = 2 * np.arctanh(target) / max(p95, 1e-7)
    Xs = Xc * s
    nrm = np.linalg.norm(Xs, axis=1, keepdims=True).clip(min=1e-7)
    Y = np.tanh(nrm/2) / nrm * Xs
    cur = np.linalg.norm(Y, axis=1, keepdims=True).clip(min=1e-7)
    return Y * np.clip((1-1e-3)/cur, None, 1.0)

def poincare_D(Y):                                    # exp5, verbatim
    sq = (Y**2).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2*Y@Y.T, 0)
    denom = np.maximum((1-sq[:, None])*(1-sq[None, :]), 1e-12)
    return np.arccosh(np.maximum(1 + 2*d2/denom, 1+1e-12))

def knn_sets(D, k=K_NN):
    return np.argsort(D, axis=1)[:, 1:k+1]

def mutual_knn(nnA, nnB, k=K_NN):
    inter = [len(set(nnA[j]).intersection(nnB[j])) for j in range(len(nnA))]
    return float(np.mean(inter) / k)

def perm_knn(nnB, p):
    """exp21's null construction: relabel model b's classes by permutation p."""
    return p[np.asarray(nnB)[np.argsort(p)]]

def cka_linear(X, Y):                                 # exp21, verbatim
    Xc = X - X.mean(0); Yc = Y - Y.mean(0)
    hsic = np.linalg.norm(Xc.T @ Yc, "fro")**2
    return float(hsic / (np.linalg.norm(Xc.T @ Xc, "fro") * np.linalg.norm(Yc.T @ Yc, "fro")))

def calibrate(raw, nulls, s_max=1.0, alpha=ALPHA):
    K = len(nulls)
    tau = float(np.sort(np.concatenate([[raw], nulls]))[math.ceil((1-alpha)*(K+1)) - 1])  # eq. 9
    p = float((1 + np.sum(np.asarray(nulls) >= raw)) / (K + 1))                            # eq. 10
    cal = max((raw - tau) / (s_max - tau), 0.0) if s_max > tau else 0.0                    # eq. 12
    return dict(null_mean=float(np.mean(nulls)), null_sd=float(np.std(nulls, ddof=1)),
                tau95=tau, p=p, cal=cal, nullcentered=float(raw - np.mean(nulls)))

def main():
    feats_R, feats_H, nn_R, nn_H = {}, {}, {}, {}
    for m in MODELS:
        C = load_centroids(m); P = s4_project(C)
        feats_R[m], feats_H[m] = C, P
        nn_R[m] = knn_sets(squareform(pdist(C))); nn_H[m] = knn_sets(poincare_D(P))
        print(f"prepared {m} (d={C.shape[1]})", flush=True)
    rng = np.random.RandomState(0)
    perms = [rng.permutation(1000) for _ in range(K_PERM)]
    print(f"analytic mKNN chance k/(n-1) = {K_NN/999:.4f}")
    rows = []
    for a, b in itertools.combinations(MODELS, 2):
        t0 = time.time(); row = dict(model_a=a, model_b=b)
        for tag, raw, nulls in [
            ("knn_R", mutual_knn(nn_R[a], nn_R[b]),
             [mutual_knn(nn_R[a], perm_knn(nn_R[b], p)) for p in perms]),
            ("knn_H", mutual_knn(nn_H[a], nn_H[b]),
             [mutual_knn(nn_H[a], perm_knn(nn_H[b], p)) for p in perms]),
            ("cka_R", cka_linear(feats_R[a], feats_R[b]),
             [cka_linear(feats_R[a], feats_R[b][p]) for p in perms]),
            ("cka_H", cka_linear(feats_H[a], feats_H[b]),
             [cka_linear(feats_H[a], feats_H[b][p]) for p in perms]),
        ]:
            row[f"{tag}_raw"] = raw
            row.update({f"{tag}_{k}": v for k, v in calibrate(raw, nulls).items()})
        row["time_s"] = time.time() - t0
        rows.append(row)
        print(f"{a}-{b}: kNN {row['knn_R_raw']:.3f} cal {row['knn_R_cal']:.3f} p {row['knn_R_p']:.4f} | "
              f"CKA {row['cka_R_raw']:.3f} cal {row['cka_R_cal']:.3f} p {row['cka_R_p']:.4f} ({row['time_s']:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(OUT/"exp21b_local_global_K200.csv", index=False)
    df = pd.DataFrame(rows)
    print("\n== MEANS over 66 pairs ==")
    for tag in ["knn_R", "knn_H", "cka_R", "cka_H"]:
        print(f"  {tag}: raw {df[f'{tag}_raw'].mean():.4f} | cal(eq.12) {df[f'{tag}_cal'].mean():.4f} | "
              f"null-centered {df[f'{tag}_nullcentered'].mean():.4f} | null mean {df[f'{tag}_null_mean'].mean():.4f} | "
              f"frac p<0.05 {(df[f'{tag}_p'] < 0.05).mean():.3f}")
    print("Done exp21b")

if __name__ == "__main__":
    main()
