#!/usr/bin/env python3
"""Reproduction check for calibrated_delta.py against the paper's tables (used by sweep_freeze.py).

Two cells: ViT-L / CIFAR-100 and DINOv2-L / ImageNet.
  * Table 1 (census of record, expR52_census_haar_p999_200.csv: Haar null x p99.9 statistic): centroids from
    the census cache ({m}_{ds}_train.npz); the tool must reproduce excess (3 dp), r and p exactly.
  * Table B34 (expR56_depth_variants.csv, anisotropic star, 10 seeds, census cache for both datasets):
    same frames (CIFAR-100 coarse-20; ImageNet WordNet-30); depth (3 dp) and z (1 dp) must match.
Writes rebuttal/results/tool_check.json; sweep_freeze.py compares it with the CSVs and re-runs this
script only if the JSON is missing (the 200-replicate ImageNet cell takes ~10 CPU-minutes).
"""
import os, sys, json
from pathlib import Path
import numpy as np
from sklearn.cluster import AgglomerativeClustering
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import calibrated_delta as cd
ROOT = Path(os.environ.get("PLATONIC_ROOT", "/media/HDD_4TB_2/javi/Platonic"))
OUT = Path(os.environ.get("PLATONIC_RESULTS", str(HERE.parents[1]/"rebuttal/results")))
CACHE = ROOT/"results/practical_tasks_cache"
COARSE = {0:[4,30,55,72,95],1:[1,32,67,73,91],2:[54,62,70,82,92],3:[9,10,16,28,61],4:[0,51,53,57,83],5:[22,39,40,86,87],6:[5,20,25,84,94],7:[6,7,14,18,24],8:[3,42,43,88,97],9:[12,17,37,68,76],10:[23,33,49,60,71],11:[15,19,21,31,38],12:[34,63,64,66,75],13:[26,45,77,79,99],14:[2,11,35,46,98],15:[27,29,44,78,93],16:[36,50,65,74,80],17:[47,52,56,59,96],18:[8,13,48,58,90],19:[41,69,81,85,89]}
SUP100 = np.zeros(100, dtype=int)
for s, cls in COARSE.items():
    for c in cls: SUP100[c] = s

def cache_centroids(m, ds):
    d = np.load(CACHE/f"{m}_{ds}_train.npz"); X = d["features"].astype(np.float32); y = d["labels"].astype(np.int64)
    return np.stack([X[y==c].mean(0) for c in range(int(y.max())+1)])

def main():
    WN = np.load(ROOT/"results/features/imagenet/imagenet1k_wordnet_dist.npy")
    WN30 = AgglomerativeClustering(n_clusters=30, metric="precomputed", linkage="average").fit_predict(WN)
    np.save(HERE/"example_labels_cifar100_coarse20.npy", SUP100); np.save(HERE/"example_labels_imagenet_wn30.npy", WN30)
    res = {}
    C = cache_centroids("i21k_l", "cifar100")
    res["i21k_l/cifar100"] = cd.run(C, SUP100)                                 # Table 1 + B34 (cache)
    C = cache_centroids("dinov2_l", "imagenet")
    res["dinov2_l/imagenet"] = cd.run(C, WN30)                                 # Table 1 + B34 (cache)
    json.dump(res, open(OUT/"tool_check.json", "w"), indent=1)
    for k, v in res.items(): print(k, json.dumps(v))

if __name__ == "__main__":
    main()
