#!/usr/bin/env python3
"""expR79 (parallel track, priority 1b, CPU): does the depth test fire on structures that are hierarchical by construction?

(a) Synthetic deep hierarchy with ViT-L's real ImageNet spectrum: a three-level tree over the 30 hubs of the WordNet frame of record
    (the nested 30 / 6 / 2 cuts of expR76_frames.json: hub = super-hub + sub-hub offset + hub offset), 1000 leaf centroids as hubs plus
    within-cluster offsets, the within/between spread ratio set to the real one of ViT-L (expR64b_wn30_summary.csv, ratio_real), and
    the whole cloud re-spectred to ViT-L's centered singular values (U of the synthetic cloud, Sigma of the real one); 5 seeds.
    Control: the same construction with a flat two-level tree (hubs iid, no super-hubs).
(b) The WordNet Poincare embeddings of Nickel & Kiela (2017): trained here with gensim's PoincareModel on the transitive closure of the
    hypernym tree that spans the 1000 ImageNet leaf synsets (dims 10 and 50, 50 epochs, burn-in 10), the leaf synsets' coordinates read
    as the 1000 class centroids (Euclidean reading of the ball coordinates, as the instrument reads any cloud).
Every cloud goes through the census excess (centered Haar null x 200), the depth test with the matched anisotropic star (K = 30 frame of
record, 10 star seeds) and the decoupling control (10 seeds). Output: expR79_synthetic_deep_poincare.csv.
Usage: python expR79_synthetic_deep_poincare.py [--part synthetic|poincare|all] [--seeds 5]"""
import os, sys, time, json, argparse
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "ICLR2027" / "tool"))
import calibrated_delta as cd, expR56_depth_variants as R56, expR74_decoupling as R74, expR75_census_centered_haar as R75
ROOT = R56.ROOT; OUT = R56.OUT; CACHE = R56.CACHE
cd.haarnull = R75.null_haar_centered

def tests(C, sup, n_star=10, n_dec=10):
    cen = cd.census(C, 200, "haar", "p999"); res = R56.depth_test(C, sup, "aniso", n_star=n_star)
    zs = [R56.depth_test(R74.decouple(C, sup, s), sup, "aniso", n_star=n_star)["z_depth"] for s in range(n_dec)]
    return dict(delta=cen["delta"], excess=cen["excess"], r_above=cen["r_above"], p_left=cen["p_left"], depth=res["depth_excess"], z=res["z_depth"], zdec_mean=float(np.mean(zs)), zdec_sd=float(np.std(zs, ddof=1)), dec_frac_cert=float(np.mean(np.array(zs) <= -2)))

def real_vitl():
    d = np.load(CACHE / "i21k_l_imagenet_train.npz"); X = d["features"].astype(np.float32); y = d["labels"]
    return np.stack([X[y == c].mean(0) for c in range(1000)])

def synthetic(C_real, cuts, ratio, seed, deep=True):
    """Hubs at three nested levels (2 -> 6 -> 30) or flat (30 iid hubs); offsets iid; ratio = within-cluster spread / between-hub spread; then re-spectred to C_real."""
    rng = np.random.RandomState(seed); n, d = C_real.shape; sup30, sup6, sup2 = cuts[30], cuts[6], cuts[2]
    H2 = rng.randn(2, d); H6 = rng.randn(6, d); H30 = rng.randn(30, d)
    par6 = np.array([int(np.bincount(sup2[sup6 == k]).argmax()) for k in range(6)]); par30 = np.array([int(np.bincount(sup6[sup30 == k]).argmax()) for k in range(30)])
    if deep: hubs = 1.0 * H2[par6[par30]] + 0.7 * H6[par30] + 0.5 * H30      # level contributions: super-hub, sub-hub, hub
    else: hubs = H30 * np.sqrt(1.0 + 0.49 + 0.25)
    between = np.sqrt(((hubs - hubs.mean(0)) ** 2).sum(1).mean()); off = rng.randn(n, d); off *= ratio * between / np.sqrt((off ** 2).sum(1).mean())
    X = hubs[sup30] + off
    mu_r, Ur, Sr, Vtr = R56.svd(C_real) if hasattr(R56, "svd") else (C_real.mean(0),) + np.linalg.svd(C_real - C_real.mean(0), full_matrices=False)
    U, S, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    return ((U * Sr[:len(S)]) @ Vtr[:len(S)] + mu_r).astype(np.float32)   # the synthetic cloud's coefficients, the real cloud's spectrum and directions

def poincare(dim, seed=0, epochs=50):
    from nltk.corpus import wordnet as wn
    from gensim.models.poincare import PoincareModel
    wnids = sorted(os.listdir("/media/HDD_4TB_1/javi/ILSVRC2012_img_train")); leaves = [wn.synset_from_pos_and_offset("n", int(w[1:])) for w in wnids]
    rel = set()
    for s in leaves:
        for path in s.hypernym_paths():
            for a, b in zip(path[1:], path[:-1]): rel.add((a.name(), b.name()))   # (child, parent) edges of the spanning tree
    closure = set()
    for s in leaves:
        for path in s.hypernym_paths():
            names = [p.name() for p in path]
            for i in range(len(names)):
                for j in range(i): closure.add((names[i], names[j]))        # transitive closure, as Nickel & Kiela train on
    model = PoincareModel(sorted(closure), size=dim, negative=10, seed=seed, burn_in=10); model.train(epochs=epochs, print_every=1000)
    return np.stack([model.kv[s.name()] for s in leaves]).astype(np.float32), len(closure)

def main(A):
    cuts = {int(k): np.asarray(v) for k, v in json.load(open(OUT / "expR76_frames.json")).items()} if (OUT / "expR76_frames.json").exists() else None
    if cuts is None:
        sys.path.insert(0, str(HERE)); import expR76_hier_finetune_full as R76; cuts = R76.frames()
    sup = cuts[30]; C_real = real_vitl(); ratio = float(pd.read_csv(OUT / "expR64b_wn30_summary.csv").set_index("model").loc["i21k_l", "ratio_real"])
    rows = []
    if A.part in ("synthetic", "all"):
        for deep in (True, False):
            for s in range(A.seeds):
                t0 = time.time(); r = tests(synthetic(C_real, cuts, ratio, s, deep), sup); rows.append(dict(cloud="synthetic_deep_vitl_spectrum" if deep else "synthetic_flat_vitl_spectrum", seed=s, ratio=ratio, dim=C_real.shape[1], **r, time_s=time.time() - t0))
                print(f"{rows[-1]['cloud']:30s} seed {s} excess {r['excess']:+.4f} (r {r['r_above']}) z {r['z']:+.2f} decoupled {r['zdec_mean']:+.2f} ({rows[-1]['time_s']:.0f}s)")
                pd.DataFrame(rows).to_csv(OUT / "expR79_synthetic_deep_poincare.csv", index=False)
    if A.part in ("poincare", "all"):
        for dim in (10, 50):
            t0 = time.time(); X, ncl = poincare(dim); r = tests(X, sup); rows.append(dict(cloud=f"wordnet_poincare_d{dim}", seed=0, ratio=float("nan"), dim=dim, n_closure=ncl, **r, time_s=time.time() - t0))
            print(f"wordnet_poincare_d{dim:<3d} ({ncl} closure pairs) excess {r['excess']:+.4f} (r {r['r_above']}) z {r['z']:+.2f} decoupled {r['zdec_mean']:+.2f} ({rows[-1]['time_s']:.0f}s)")
            pd.DataFrame(rows).to_csv(OUT / "expR79_synthetic_deep_poincare.csv", index=False)
    print("DONE expR79")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--part", default="all", choices=["synthetic", "poincare", "all"]); ap.add_argument("--seeds", type=int, default=5); main(ap.parse_args())
