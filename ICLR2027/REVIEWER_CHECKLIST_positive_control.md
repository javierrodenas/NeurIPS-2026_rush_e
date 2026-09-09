# Reviewer checklist — positive-control pass

The three questions of the second review, each answered from the main text alone; line numbers are the ICLR margin numbers of the compiled PDF.

## Q1. The depth test has no positive control on real data; how should the reader interpret 4 of 12?

- **Section 4, 'On real clouds the test is conservative and weak'** (line 296): On the real clouds with the hub arrangement replaced by an implanted two-level tree the test raises no false alarm (none of 60 zero-strength runs) and detects a clean tree in 0 of 12 backbones; the rest are missed because the within-cluster spread of the real clouds is 1.3–3.9 times their between-hub spread while the synthetic sweep covered ratios below one, and shrinking the offsets into that range makes every backbone detectable. The test is conservative and weak at ImageNet's noise level: its positive verdicts stand, 4 of 12 is a lower bound, and the other eight are 'not detected', never 'not hierarchical'.
- **Section 4, 'Depth is certified on ImageNet and not on CIFAR-100'** (line 291): The certified set (ViT-S/B/L, DINOv2-L) is unchanged; the wording for the rest is 'not detected'.
- **Figure 4, panel (b)** (line 282): z against implant strength for every backbone, family colors; dashed: the same clouds with their within-cluster spread shrunk into the validated range.
- **Limitations (i)** (line 21): The limitation states the weakness and that the one trained positive control is inconclusive.

## Q2. Does the test's weakness change the abstract's claim about MERU and about depth?

- **Abstract** (line 23): 'No additional depth' became 'no detected depth' everywhere: MERU shows the same clustering and no detected depth; the test is described as conservative and weak.
- **Section 4, 'Imposing the geometry does not create depth'** (line 312): Neither MERU nor its twin is detected as more hierarchical than a matched star; training in hyperbolic space neither creates the clustering nor adds detectable depth.
- **Section 6, 'Measure before imposing'** (line 449): The practical reading of the test's asymmetry.

## Q3. How many of the 49 genuine cells survive when centroid-resampling and estimator noise are combined with the null?

- **Section 4, 'The count survives resampling'** (line 256): Folding the bootstrap s.d. and the estimator s.d. into the null, or repeating the census on 30 resampled centroid sets, leaves 42–47 of the 72 cells genuine and 17–18 of the 24 on ImageNet and CIFAR-100; the cells that move are on the transfer sets.
- **Table 1 caption** (line 233): The joint count in one clause, with the appendix table pointer.
