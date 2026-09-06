# Reviewer-question checklist (Phase B)

Each question a reviewer of the NeurIPS version asked, or could ask, with the answer as it now stands in the main text
(section pointers; numbers from `rebuttal/results/*.csv` via the generators).

**1. Do 55/72 and 43/48 survive the Haar null and the p99.9 statistic?**
Not as stated, and the paper no longer states them. Under the census of record (Haar null × 99.9th percentile × BH, §3) the sign
count is 70/72, the genuine count 49/72 and 18/24 on ImageNet + CIFAR-100 (abstract, §1 (ii), §4 first paragraph, Table 1). The
full 2×2 {Gaussian, Haar} × {supremum, p99.9} verdicts are Table B21 (49 / 46 / 52 / 52 genuine); the §3 disclosure paragraph
says which two choices changed after pre-registration and why.

**2. Does the supremum converge on ImageNet?**
Yes for the purpose it is used: 30 class-bootstrap resamples per backbone give an excess s.d. ≤ 0.001 and 30/30 negative
resamples for all 12 backbones (§4, Table B36). What does not converge is the *rank* of a single extreme quadruple: the supremum
over 5·10⁵ quadruples of 1000 centroids is a one-quadruple extreme, which in DINOv2 sits above every null replicate while the
p99.9 sits below — the "DINOv2–ImageNet exception is a supremum artifact" paragraph in §4.

**3. What is the power of the depth test, and what do the CIFAR-100 z's mean?**
Power is 1.00 for two- and three-level synthetic hierarchies at noise ratios ≤ 0.3 for n = 100 and n = 1000 with the frame at the
leaf clusters (≥ 0.85 at ratio 0.6 for n = 1000), and pure stars are never declared less hierarchical than their match; but with
n = 100 and K ≥ 12 leaf clusters pure stars are declared more hierarchical (z ≤ −2) in 17% of runs (6.2% pooled, 0% at n = 1000),
so the pre-set bar (≤ 5% in each direction) is failed and the paper does not certify depth above the labelled clusters (§4 "Depth
above the labelled clusters is not certified", Table B35, figure in Appendix A.5). The CIFAR-100 z's (3/12 at z ≤ −2 with the
anisotropic star) fall in exactly that regime and are reported as not validated; the earlier isotropic z's up to +6.8 were an
artifact of a star blind to each cluster's covariance and are withdrawn. With the frame at the top level the test has zero power
by construction (the hub null keeps frame clusters intact), the scope limit stated in Appendix A.5.

**4. What does the cosine census say?**
It agrees with the Euclidean record in 65/72 cells (56/72 genuine, 22/24 on ImageNet + CIFAR-100; DINOv2 on ImageNet genuine
under both readings); §4 "The angular reading agrees", Table B32. It is reported as robustness only; the record stays Euclidean.
Under angular configurations DINOv2 is no longer an island in the tree map (triplet agreement 0.77 vs 0.76, §5).

**5. Why is DINOv2 at null under one reading and genuine under another, and which holds under the record?**
Genuine under the record: DINOv2-S/B/L/G on ImageNet have BH p ≤ 0.05 under both p99.9 constructions and sit above every null
replicate (r = 0) under both supremum constructions (Table 1 sup. column, Table B21). The mechanism is question 2; the text
calls the old reading a supremum artifact and never "at null".

**6. Is the DINOv2 island real?**
It is a property of the clustering cut and of the Euclidean configuration: without a cut it persists in Euclidean-average
dendrograms (cophenetic 0.36 vs 0.80 inside the block; triplets 0.47 vs 0.74) and disappears under cosine-average (0.77 vs 0.76);
§5, Table B33.

**7. Does scale deepen the structure?**
Only DINOv2 on CIFAR-10 (−0.071 → −0.130) and weakly on CIFAR-100 (−0.031 → −0.040); no general claim (§4).

**8. Is the text result robust?**
Robust at the larger scales, fragile at the smaller: GPT-2 L/XL genuine under all four constructions and under cosine; GPT-2 S
(p = 0.02) and M (p = 0.15) flip across readings; Pythia (3 sizes) and OLMo-1B genuine throughout; sentence embedders absent on
class names (+0.002..+0.004) and present on DBpedia (§4 text paragraph, Table B23).
