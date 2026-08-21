#!/usr/bin/env python3
"""
ICLR EXP 27 — tree map on a non-WordNet concept set (closes the circularity
door; round-4 minor). DBpedia Classes: 219 leaf classes with a true 3-level
hierarchy that does not derive from WordNet.

Embedders BGE/E5/GTE (exp14 protocol, class centroids over 100 texts/class):
dendrograms under all six configurations, ARI of the 70-cluster cut vs the
true level-2 partition and 9-cluster cut vs level-1, cross-model ARI, and the
same degeneracy + CPCC diagnostics as the vision map.

Output: exp27_dbpedia_treemap.json + log.
"""
import os, sys, json, itertools, time
os.environ.setdefault("HF_HOME", "/home/javi/Platonic/.hf_cache")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd, torch
from pathlib import Path
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from sklearn.metrics import adjusted_rand_score

ROOT = Path("/home/javi/Platonic"); OUT = ROOT/"rebuttal/results"
sys.path.insert(0, str(ROOT/"rebuttal/scripts"))
from exp14_dbpedia_text import load_dbpedia, encode, MODELS

def main():
    texts_tr, y_tr, _, _, sup, n_cls = load_dbpedia()
    # level-1 partition: recover from raw csv
    from huggingface_hub import hf_hub_download
    tr = pd.read_csv(hf_hub_download("DeveloperOats/DBPedia_Classes", "DBPEDIA_train.csv", repo_type="dataset"))
    l3 = sorted(tr.l3.unique()); l3i = {c:i for i,c in enumerate(l3)}
    l1_of = tr.drop_duplicates("l3").set_index("l3").l1.to_dict()
    l1 = np.array([hash(l1_of[c]) for c in l3]); _, l1 = np.unique(l1, return_inverse=True)

    cents = {}
    for short, hf, pool, prefix in MODELS:
        t0 = time.time()
        X = encode(hf, pool, prefix, texts_tr)
        C = np.stack([X[np.array(y_tr)==c].mean(0) for c in range(n_cls)])
        cents[short] = C
        print(f"encoded {short} ({time.time()-t0:.0f}s)")

    res = {}
    for metric in ["euclid","cosine"]:
        for link in ["average","complete","ward"]:
            trees, cpccs, maxfr = {}, [], []
            for m, C in cents.items():
                if link == "ward":
                    Xf = C/np.linalg.norm(C,axis=1,keepdims=True) if metric=="cosine" else C
                    D = pdist(Xf); Z = linkage(Xf, "ward")
                else:
                    D = pdist(C, "cosine" if metric=="cosine" else "euclidean")
                    Z = linkage(D, link)
                trees[m] = Z
                cpccs.append(float(cophenet(Z, D)[0]))
                cut = fcluster(Z, 70, "maxclust")
                _, cnt = np.unique(cut, return_counts=True)
                maxfr.append(float(cnt.max()/n_cls))
            ari_l2 = {m: float(adjusted_rand_score(fcluster(trees[m],70,"maxclust"), sup)) for m in cents}
            ari_l1 = {m: float(adjusted_rand_score(fcluster(trees[m],9,"maxclust"), l1)) for m in cents}
            cross = float(np.mean([adjusted_rand_score(fcluster(trees[a],70,"maxclust"),
                                                       fcluster(trees[b],70,"maxclust"))
                                   for a,b in itertools.combinations(cents,2)]))
            res[f"{metric}-{link}"] = dict(cpcc=float(np.mean(cpccs)), maxfrac=float(np.max(maxfr)),
                                           ari_l2=ari_l2, ari_l1=ari_l1, cross_model=cross)
            print(f"{metric}-{link}: CPCC {np.mean(cpccs):.3f} maxfrac {np.max(maxfr):.2f} | "
                  f"ARI vs L2 {['%.2f'%ari_l2[m] for m in cents]} | vs L1 {['%.2f'%ari_l1[m] for m in cents]} | "
                  f"cross {cross:.2f}")
    json.dump(res, open(OUT/"exp27_dbpedia_treemap.json","w"), indent=1)
    print("Done")

if __name__ == "__main__":
    main()
