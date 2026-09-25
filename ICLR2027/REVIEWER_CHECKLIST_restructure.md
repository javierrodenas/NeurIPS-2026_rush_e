# Reviewer-question checklist (restructured paper)

Answerable from the main text alone; section pointers refer to the restructured paper.

**1. What is the premise being tested, and who relies on it?**
Latent hyperbolicity: that standard models already organize concepts in a tree-like metric structure, read as a low Gromov δ on their
features. It is the stated motivation of hyperbolic representation learning in vision, vision–language and language (§1 first paragraph;
§2 "Latent hyperbolicity, the target claim"), read on sample-level features, uncalibrated, under a single construction of the statistic.

**2. Why does the sample-level reading fail?**
Three confounds push raw δ down without hierarchy: dimension, covariance spectrum and the supremum statistic (§1, §3). Read with the
instrument on ≈1000 stratified images per cell, the raw supremum sits in the literature's low band (0.087–0.145) and the
calibrated excess is not genuine in 14 of 24 cells; where it is genuine (10 cells, DINOv2 and SigLIP-B on CIFAR-100, ViT-S/B/L and
DINOv2-L/G on DTD) it is at most 0.023 (§4.1, Table 1 last two columns, Figure 2b). A low raw δ on image features is what a random cloud
of that dimension and spectrum produces.

**3. What survives at the class level, and is it deep?**
Clustered structure beyond the null in 49 of 72 cells (18 of 24 on ImageNet and CIFAR-100), every family (§4.2, Table 1); a pure star already
has such structure (§4.3); a validated matched-star test (n = 1000) certifies hierarchy above the WordNet superclasses in 4 of 12 ImageNet
backbones and never less than a star; CIFAR-100 is in the unvalidated regime (§4.4, Figure 4).

**4. Does imposing hyperbolic geometry create depth?**
No. MERU and its Euclidean CLIP twin show the same clustering excess on ImageNet images (MERU -0.007 to -0.010, CLIP
-0.006 to -0.010), MERU's Lorentz-metric reading equals its Euclidean one to 0.0002, and under the validated depth
test neither is more hierarchical than a matched star (z -1.5 to -0.5 vs -1.8 to -1.0) (§4.5, Table B30).

**5. Whose tree is it, and where does it live?**
The naive map's island is an artifact of the clustering cut (§5.1–5.2): at the cut the gap is moderate (0.38 vs 0.48), in cophenetic
correlation it remains (0.48 vs 0.78), in triplet agreement under angular configurations it closes (0.77 vs 0.76). Every recipe recovers
the human taxonomy partially, more with supervision (11 of 12 paired comparisons); WordNet alignment follows supervision (ρ +0.57..+0.59
contrastive, +0.49..+0.53 supervised, +0.18..+0.22 DINOv2); the self-supervised tree lives in angles (sibling triplets 0.91 cosine vs 0.71
Euclidean; cosine census agrees in 65/72 cells) (§5.3–5.4); DBpedia rules out WordNet circularity (§5.5); text depends on recipe, scale and
probe (§5.6); local agreement survives permutation calibration (§5.7).

**6. What should a practitioner measure, and which metric should they use?**
Not raw δ: the Khrulkov rule c = (0.144/δ_rel)² would assign curvatures from 0.48 to 2.5 to structureless Gaussian clouds of d = 192 to 1536
(§6.1). Measure the excess and its rank, the matched-star depth (n ≈ 1000 regime) and where the structure lives (§6.2). Then pick the readout:
cosine collects the self-supervised structure; Poincaré adds +0.1 to +1.3 pp over cosine for the contrastive VLMs on prototype tasks and is
inconsistent elsewhere (−1.7 to +1.1 pp); nothing helps on flat labels; sample-level tasks never benefit (§6.3, Appendix A.10).
