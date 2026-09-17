# Reviewer checklist — third review (final pass)

Each question of the third review, answered from the main text alone; line numbers are the ICLR margin numbers of the compiled PDF (`ICLR2027/main_iclr2027_final.pdf`).

## Q1. The matched star's hubs are Gaussian; a star with the real hubs' spectrum could pass the depth test too.

- **Abstract** (line 22): The abstract states that the four certified backbones are the same under a star whose hubs carry the real hubs' spectrum.
- **Section 4, 'A spectrum-matched star confirms the four'** (line 286): Hubs drawn as a Haar resample of the real hubs (exact hub spectrum, ten star seeds): the certified set is unchanged, z from -2.0 to -3.7; on the implanted clouds the new star raises 0 of 60 false alarms at s = 0 and detects one implant in sixty at s = 1; read as a rank over the star seeds (resolution 1/11) the verdict does not separate certified from uncertified backbones while z does. Table 6 has both stars per backbone.

## Q2. The certified ViTs were trained on the hierarchical IN-21k label set; the depth may be the label hierarchy.

- **Section 4, same paragraph** (line 293): Two ViT-B/16 supervised on ImageNet-1k leaf labels only: the augreg recipe is certified under both stars (z -3.03/-2.65) and DeiT-B is not (z -1.26/-1.21), so the IN-21k label hierarchy is not what makes the census ViTs deep.
- **Section 5, 'WordNet alignment follows supervision and recipe'** (line 375): The augreg IN-1k ViT-B is aligned with WordNet like the IN-21k ViTs (rho +0.52) while DeiT-B is nearly unaligned (+0.08) yet recovers the CIFAR-100 superclasses at ARI 0.56: alignment is recipe-dependent even among leaf-supervised ViTs (Table 10).
- **Limitations (iv)** (line 476): The limitation records that the objective-geometry link is correlational and that leaf-label supervision can produce depth and alignment without guaranteeing either.

## Q3. The corollary correlates the raw delta with the gains; the paper's own point is that the raw reading is confounded.

- **Section 6, 'The calibrated reading predicts the zero-cost gain'** (line 437): Recomputed on the record excess with Fisher-z CI95 and the family control: the nearest-centroid prediction survives on all four hierarchical datasets, the few-shot prediction on three (not ImageNet), and the depth verdict predicts no gain anywhere. Table 13 gives raw and calibrated correlations side by side; Figure 11 is drawn on the excess.

## Q4. MERU's embeddings may sit where the hyperboloid is essentially flat, so the control may not test curvature.

- **Section 4, 'Imposing the geometry does not create detected depth'** (line 312): Stated in the text: MERU's embeddings sit in the near-flat regime (learned c = 0.10, radius x sqrt(c) at the median 0.258-0.279, Lorentz-to-Euclidean distance ratio 0.997), so the control tests the training objective, not a curved geometry; the lead-in now says 'does not create detected depth'.
- **Abstract** (line 24): The abstract says so in one clause.

## Q5. 'Budget-stable' rested on a supremum-statistic sweep whose ImageNet magnitudes drifted by 0.02.

- **Section 3, 'Two choices changed after pre-specification'** (line 193): The sweep was rerun under the record (Haar null, p99.9, 200 replicates) at every budget: the ImageNet excess drifts by at most 0.38 of its own spread from 10^5 upward and never changes sign (Table 4, with the superseded supremum rows next to it).

## Q6. Effect sizes: the excesses are small in absolute terms; 'small' is not a size.

- **Section 4, 'The premise does not survive calibration'** (line 209): 'Small' is replaced by the fraction of the null reading: at the sample level the genuine cells remove 7 to 39 per cent of the null, the larger figure for DINOv2-G on CIFAR-100 images.
- **Section 6, 'The raw reading cannot select a curvature'** (line 435): On ImageNet centroids the excess removes 5 to 33 per cent of the null reading; the normalized column is in Table 3(b), Table 5 and Table 11.
- **Abstract** (line 19): The abstract bounds the surviving sample-level excess at 40% of the null (file: 39%).

## Q7. 'The island vanishes' / 'closes entirely' overstates: a gap remains under most configurations.

- **Section 5, 'The categorical island is an artifact, and a moderate gap remains'** (line 353): Under the selected configuration the DINOv2 family agrees with the block at 0.38 against 0.48; under the other admissible configurations about a third of the within-block agreement is missing (median over configurations and measures 0.34, Table 9); the triplet gap closes under cosine-average only. 'Vanishes' and 'closes entirely' are gone, Figure 5's caption included.
- **Abstract** (line 26): The abstract says a gap of about a third remains once the cut is controlled.

## Q8. The text census: Table 13 and Table 14 disagreed on GPT-2 S/M, and the prose followed the older table.

- **Section 5, 'In text, clustered structure depends on recipe, scale and probe'** (line 414): One census, the record (Table 11): GPT-2 S genuine at p = 0.020 and flipping under the supremum and under cosine, M not genuine under the record, L and XL genuine under every construction, Pythia at every scale. The 3-replicate original extraction table is removed and the template table's caption is rewritten from the record.
- **Section 1, 'The raw reading is confounded'** (line 63): The introduction's example now names GPT-2 M, the size that sits at its null under the record.

## Q9. Minor: 'pre-registered', the ordering trees < hyperbolic < spherical, the repeated sentence in A.3, the old illustration, bold in Table 1, the bridges and the number density.

- **Section 3** (line 184): 'Pre-registered' is 'pre-specified' throughout (no dated public record exists).
- **Section 2, 'Background'** (line 133): The ordering is stated for hyperbolic regions of radius >= 4, where it holds in Table 2.
- **Table 1 caption** (line 232): Bold is restored on the sign-positive cells.
- **Sections 3-5, last paragraphs** (line 199): Three bridges remain (end of Sections 3, 4 and 5, distinct wording); every other paragraph ends on its claim. The repeated sentence of A.3 and the old illustration (former Figure 12) are gone; the appendix is fourteen tables, one per question, numbered in citation order.
