#!/usr/bin/env python3
"""Fifth review (2026-09-18), S5.3: how many levels the WordNet hierarchy has above the 30 hubs of the ImageNet frame.
The frame is the K=30 average-linkage cut of the WordNet distance matrix used by expR56 (frames()['imagenet'][30]); each
cluster is mapped to the lowest common hypernym of its member synsets (ILSVRC wnids, sorted order = class index), and
h = the number of levels between the root (entity.n.01, depth 0) and the deepest such hub ancestor, i.e. the height of
the WordNet tree above the hubs. Writes rebuttal/results/final_wordnet_levels.json (h, per-cluster depths, hub synsets)."""
import sys, json, os
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import expR56_depth_variants as R56
from nltk.corpus import wordnet as wn
NUM = dict(enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()))
wnids = sorted(os.listdir("/media/HDD_4TB_1/javi/ILSVRC2012_img_train")); assert len(wnids) == 1000
syn = [wn.synset_from_pos_and_offset("n", int(w[1:])) for w in wnids]
_, inet = R56.frames(); sup = np.asarray(inet[30]).astype(int); assert sup.max() + 1 == 30
def lca(ss):
    cur = ss[0]
    for s in ss[1:]:
        h = cur.lowest_common_hypernyms(s); cur = min(h, key=lambda x: x.min_depth()) if h else wn.synset("entity.n.01")
    return cur
clusters = []
for k in range(30):
    members = [syn[i] for i in np.where(sup == k)[0]]; a = lca(members)
    clusters.append(dict(cluster=k, size=len(members), hub_synset=a.name(), depth=a.min_depth()))
depths = [c["depth"] for c in clusters]; h = int(max(depths))
out = dict(frame="expR56 ImageNet WordNet cut, K=30 (average linkage on imagenet1k_wordnet_dist.npy)", root="entity.n.01 at depth 0",
           definition="h = max over the 30 clusters of the depth of the lowest common hypernym of the cluster's classes = levels of the WordNet tree above the hubs",
           h=h, h_word=NUM.get(h, str(h)), depth_median=float(np.median(depths)), depth_min=int(min(depths)), clusters=clusters)
json.dump(out, open(R56.OUT/"final_wordnet_levels.json", "w"), indent=1)
print("h =", h, "(", out["h_word"], ") depths:", sorted(depths))
