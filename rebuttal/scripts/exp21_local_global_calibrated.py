#!/usr/bin/env python3
"""
ICLR EXP 21 — the Platonic->Aristotelian arc, measured with PUBLISHED metrics only.

Applies the permutation-calibration framework of Groger et al. (Aristotelian
View) to our cross-model geometry, using exactly two established metrics:
  LOCAL : mutual-kNN alignment (PRH's metric; identical implementation to our
          exp5, which is already in the OpenReview record: 0.474 R / 0.427 H).
  GLOBAL: linear CKA (Kornblith et al. 2019) between the two models' shared
          1000 ImageNet centroids.
Calibration (their move, verbatim): permutation null = shuffle the class
correspondence of one model (3 permutations); calibrated score = raw - null.

Prediction of the form-not-content story: calibrated LOCAL agreement survives,
calibrated GLOBAL agreement drops toward its null, and the Poincare arm
improves neither.

Output: rebuttal/results/exp21_local_global.csv + exp21.log
"""
import os, sys, time, itertools
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist, squareform

ROOT = Path("/home/javi/Platonic"); OUT = ROOT/"rebuttal/results"
sys.path.insert(0, str(ROOT/"rebuttal/scripts"))
from exp5_crossmodel_alignment import (load_centroids, s4_project, poincare_D,
                                       knn_sets, mutual_knn, PARADIGMS)

def cka_linear(X, Y):
    """Kornblith et al. 2019, linear CKA on centered feature matrices."""
    Xc = X - X.mean(0); Yc = Y - Y.mean(0)
    hsic = np.linalg.norm(Xc.T @ Yc, "fro")**2
    return float(hsic / (np.linalg.norm(Xc.T @ Xc, "fro") * np.linalg.norm(Yc.T @ Yc, "fro")))

def main():
    models = list(PARADIGMS)
    feats_R, feats_H, nn_R, nn_H = {}, {}, {}, {}
    for m in models:
        C = load_centroids(m)
        P = s4_project(C)
        feats_R[m], feats_H[m] = C, P
        nn_R[m] = knn_sets(squareform(pdist(C)))
        nn_H[m] = knn_sets(poincare_D(P))
        print(f"prepared {m}")

    rng = np.random.RandomState(0)
    perms = [rng.permutation(1000) for _ in range(3)]
    rows = []
    for a, b in itertools.combinations(models, 2):
        t0 = time.time()
        # LOCAL: mutual-kNN, raw + permutation null (relabel model b's classes)
        knn_raw_R = mutual_knn(nn_R[a], nn_R[b])
        knn_raw_H = mutual_knn(nn_H[a], nn_H[b])
        knn_null = float(np.mean([
            mutual_knn(nn_R[a], [ [int(p[j]) for j in row] for row in
                                  np.array(nn_R[b], dtype=int)[np.argsort(p)] ])
            for p in perms]))
        # GLOBAL: linear CKA, raw + permutation null
        cka_raw_R = cka_linear(feats_R[a], feats_R[b])
        cka_raw_H = cka_linear(feats_H[a], feats_H[b])
        cka_null = float(np.mean([cka_linear(feats_R[a], feats_R[b][p]) for p in perms]))
        cka_null_H = float(np.mean([cka_linear(feats_H[a], feats_H[b][p]) for p in perms]))
        rows.append(dict(model_a=a, model_b=b,
                         knn_R=knn_raw_R, knn_H=knn_raw_H, knn_null=knn_null,
                         knn_cal_R=knn_raw_R-knn_null, knn_cal_H=knn_raw_H-knn_null,
                         cka_R=cka_raw_R, cka_H=cka_raw_H,
                         cka_null=cka_null, cka_null_H=cka_null_H,
                         cka_cal_R=cka_raw_R-cka_null, cka_cal_H=cka_raw_H-cka_null_H,
                         time_s=time.time()-t0))
        r = rows[-1]
        print(f"{a}-{b}: kNN {knn_raw_R:.3f}/{knn_raw_H:.3f} (null {knn_null:.3f}) | "
              f"CKA {cka_raw_R:.3f}/{cka_raw_H:.3f} (null {cka_null:.3f})")
        pd.DataFrame(rows).to_csv(OUT/"exp21_local_global.csv", index=False)

    df = pd.DataFrame(rows)
    print("\n== MEANS over 66 pairs ==")
    for c in ["knn_R","knn_H","knn_null","knn_cal_R","cka_R","cka_H","cka_null","cka_cal_R","cka_cal_H"]:
        print(f"  {c:10s} {df[c].mean():.4f}")
    print("Done")

if __name__ == "__main__":
    main()
