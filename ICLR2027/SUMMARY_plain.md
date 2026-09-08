# The paper in ten lines (as it now reads)

1. Hyperbolic representation learning rests on a premise, latent hyperbolicity: standard models are already tree-like because their features score a low Gromov δ.
2. That raw reading is confounded three times over: by dimension, by covariance spectrum and by the supremum statistic, and each confound pushes δ down with no hierarchy present.
3. We build an instrument that reads δ as an excess over a random cloud of the same dimension and spectrum, ranks that excess against matched replicates, and adds a depth test whose power and validity regime are measured.
4. Read on image features, where the premise is read, the excess sits within null noise in most cells and is small where it survives: the premise does not survive calibration as stated.
5. Read on class centroids, clustered structure is genuine in 49 of 72 cells and in every family, but a pure star already produces such an excess, so clustered does not mean deep.
6. The depth test, validated at ImageNet's class count, certifies hierarchy above the superclasses in 4 of 12 backbones and never finds a backbone less hierarchical than a star.
7. A backbone trained in hyperbolic space reads the same clustering as its Euclidean twin and no additional depth: imposing the geometry creates nothing at the class level.
8. Comparing the models' trees naively isolates the self-supervised family as an island; the island is an artifact of the clustering cut, and once the cut is controlled every recipe recovers the human taxonomy partially, more so with supervision.
9. The self-supervised structure is an angular tree: it lives in angles rather than distances, which is why Euclidean cuts break it and why cosine collects it.
10. Read correctly, foundation models organize classes into clustered structure that is occasionally hierarchical and moderately shared; they do not converge to one common tree, and their raw tree-likeness is not evidence for hyperbolic geometry. Measure the excess, the depth and where the structure lives before imposing curvature, and let a zero-cost readout collect what is there.
