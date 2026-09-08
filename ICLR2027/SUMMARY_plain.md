# The paper in ten lines (as it now reads)

1. Hyperbolic representation learning rests on a premise: standard models are already tree-like, because their features score a low Gromov δ.
2. That score is confounded: high dimension, a skewed spectrum and the supremum statistic all push δ down with no hierarchy present.
3. We build an instrument that reads δ as an excess over a random cloud with the same dimension and spectrum, ranked against 200 replicates, plus a depth test whose power and validity regime are measured.
4. Read on image features, where the premise is measured, the excess is within null noise in 14 of 24 cells and small (-0.023 at most) in the rest: the premise does not survive calibration as stated.
5. Read on class centroids, clustered structure beyond the null is present in 49 of 72 cells and every family, but a star already has such structure.
6. A validated matched-star test certifies hierarchy above the WordNet superclasses in only 4 of 12 ImageNet backbones; on small class sets the test is not validated.
7. A model trained in hyperbolic space (MERU) shows the same clustering as its Euclidean twin and no additional depth: imposing the geometry creates nothing at the class level.
8. Comparing the models' trees naively isolates the self-supervised family as an island; that island is an artifact of the clustering cut, and once controlled every recipe recovers the human taxonomy partially, more with supervision.
9. The self-supervised tree lives in angles rather than distances, which is why Euclidean cuts break it and why cosine collects it.
10. For practice: do not set a curvature from raw δ; measure excess, depth and where the structure lives, and let a zero-cost readout (cosine, or Poincaré for contrastive VLMs on prototype tasks) collect what is there.
