#!/usr/bin/env python3
"""expR70 (final pass, A2): ViTs supervised on ImageNet-1k leaf labels only, through the census protocol.

Backbones: deit_b = timm deit_base_patch16_224.fb_in1k (DeiT-B/16, supervised on IN-1k, no distillation) and, as the second
IN-1k-only backbone, vit_b_in1k = timm vit_base_patch16_224.augreg_in1k. Stage 1 (GPU): ImageNet train centroids with the
census cache protocol (extract_imagenet_cache_local: first 100 files per class in sorted order, the model's own eval
transform, pooled embedding; file {m}_imagenet_train.npz) and CIFAR-100 train features (50k images, same transform;
{m}_cifar100_train.npz). Stage 2 (CPU): census of record (Haar x p99.9 x 200) on ImageNet and CIFAR-100 centroids; depth test
at K = 30 (WordNet frame) with the current star and the Haar-hub star of expR69; WordNet Spearman alignment with the shuffle
control (exp3 protocol: upper-triangle Spearman between the 1000 x 1000 centroid distances and WordNet distances); CIFAR-100
superclass recovery (ARI of the 20-cluster cut, configurations of Table B11: euclid-average/ward, cosine-average/complete/ward).
The same stage-2 numbers are computed for i21k_b from its cache for the side-by-side. Output: expR70_inet1k_supervised.csv.

    python expR70_inet1k_supervised.py --extract [--device cuda:0]      # stage 1
    python expR70_inet1k_supervised.py --analyse                        # stage 2
"""
import os, sys, time, argparse
os.environ.setdefault("HF_HOME", "/media/HDD_4TB_2/javi/hf_cache")
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "ICLR2027" / "tool"))
ROOT = Path(os.environ.get("PLATONIC_ROOT", "/media/HDD_4TB_2/javi/Platonic")); CACHE = ROOT / "results/practical_tasks_cache"
OUT = Path(os.environ.get("PLATONIC_RESULTS", str(HERE.parents[1] / "rebuttal/results")))
NEW = {"deit_b": ("timm", "deit_base_patch16_224.fb_in1k", 128), "vit_b_in1k": ("timm", "vit_base_patch16_224.augreg_in1k", 128)}

def extract(device):
    import torch, timm, torchvision
    import extract_imagenet_cache_local as E
    E.MODEL_DEFS.update(NEW)
    train_files, _ = E.build_indices(); union = sorted({p for p, _ in train_files}, key=str)
    torch.cuda.set_device(device)
    for m in NEW:
        E.extract_union(m, union, device, sets=("train",))
        out = CACHE / f"{m}_cifar100_train.npz"
        if out.exists(): print("  CACHED", out.name); continue
        model, transform, batch = E.load_model(m, device)
        ds = torchvision.datasets.CIFAR100(str(ROOT / "data"), train=True, transform=transform, download=False)
        loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False, num_workers=6)
        feats, labels = [], []; t0 = time.time()
        with torch.no_grad():
            for x, y in loader:
                f = model(x.to(device));
                if f.dim() == 3: f = f[:, 0]
                feats.append(f.float().cpu().numpy()); labels.append(y.numpy())
        np.savez_compressed(out, features=np.concatenate(feats), labels=np.concatenate(labels)); print(f"  saved {out.name} ({time.time()-t0:.0f}s)")
        del model; torch.cuda.empty_cache()
    print("EXTRACT DONE")

def analyse():
    import calibrated_delta as cd
    from scipy.stats import spearmanr
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import linkage, fcluster
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import adjusted_rand_score
    from expR64_implanted_depth import centroids, K
    WN = np.load(ROOT / "results/features/imagenet/imagenet1k_wordnet_dist.npy"); sup = AgglomerativeClustering(n_clusters=K, metric="precomputed", linkage="average").fit_predict(WN); iu = np.triu_indices(1000, 1)
    COARSE = {0:[4,30,55,72,95],1:[1,32,67,73,91],2:[54,62,70,82,92],3:[9,10,16,28,61],4:[0,51,53,57,83],5:[22,39,40,86,87],6:[5,20,25,84,94],7:[6,7,14,18,24],8:[3,42,43,88,97],9:[12,17,37,68,76],10:[23,33,49,60,71],11:[15,19,21,31,38],12:[34,63,64,66,75],13:[26,45,77,79,99],14:[2,11,35,46,98],15:[27,29,44,78,93],16:[36,50,65,74,80],17:[47,52,56,59,96],18:[8,13,48,58,90],19:[41,69,81,85,89]}
    SUP100 = np.zeros(100, dtype=int)
    for s, cls in COARSE.items():
        for c in cls: SUP100[c] = s
    def cents(m, ds):
        d = np.load(CACHE / f"{m}_{ds}_train.npz"); X = d["features"].astype(np.float32); y = d["labels"].astype(np.int64); return np.stack([X[y == c].mean(0) for c in range(int(y.max()) + 1)])
    def recovery(C):
        out = {}
        for metric in ("euclid", "cosine"):
            Xn = C / np.linalg.norm(C, axis=1, keepdims=True) if metric == "cosine" else C
            for link in (("average", "ward") if metric == "euclid" else ("average", "complete", "ward")):
                Z = linkage(Xn, method=link, metric="euclidean"); cut = fcluster(Z, 20, criterion="maxclust"); out[f"{metric}-{link}"] = adjusted_rand_score(SUP100, cut)
        return out
    rows = []
    for m in list(NEW) + ["i21k_b"]:
        t0 = time.time(); C = cents(m, "imagenet"); D = squareform(pdist(C)); rng = np.random.RandomState(0); perm = rng.permutation(1000)
        r_wn = spearmanr(D[iu], WN[iu]).statistic; r_shuf = spearmanr(D[np.ix_(perm, perm)][iu], WN[iu]).statistic
        c_in = cd.census(C, 200, "haar", "p999"); d_g = cd.depth_test(C, sup, "aniso", 10); d_h = cd.depth_test(C, sup, "aniso_haarhubs", 10)
        C100 = cents(m, "cifar100"); c_c100 = cd.census(C100, 200, "haar", "p999"); rec = recovery(C100)
        rows.append(dict(model=m, d=int(C.shape[1]), excess_in=c_in["excess"], r_in=c_in["r_above"], p_in=c_in["p_left"], null_in=c_in["null_mean"], excess_c100=c_c100["excess"], r_c100=c_c100["r_above"], p_c100=c_c100["p_left"], null_c100=c_c100["null_mean"],
                         depth_gauss=d_g["depth_excess"], z_gauss=d_g["z_depth"], r_star_gauss=d_g["r_star"], depth_haar=d_h["depth_excess"], z_haar=d_h["z_depth"], r_star_haar=d_h["r_star"],
                         spearman_wn=r_wn, spearman_shuf=r_shuf, **{f"ari_{k}": v for k, v in rec.items()}, ari_max=max(rec.values()), time_s=time.time() - t0))
        pd.DataFrame(rows).to_csv(OUT / "expR70_inet1k_supervised.csv", index=False)
        print(f"{m:10s}: IN exc {c_in['excess']:+.4f} (r={c_in['r_above']}) | C100 exc {c_c100['excess']:+.4f} (r={c_c100['r_above']}) | depth z gauss {d_g['z_depth']:+.2f} haar {d_h['z_depth']:+.2f} | rho_WN {r_wn:+.3f} (shuf {r_shuf:+.3f}) | ARI max {max(rec.values()):.2f} ({time.time()-t0:.0f}s)")
    print("ANALYSE DONE")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--extract", action="store_true"); ap.add_argument("--analyse", action="store_true"); ap.add_argument("--device", default="cuda:0"); A = ap.parse_args()
    if A.extract: extract(A.device)
    if A.analyse: analyse()
