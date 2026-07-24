We thank the reviewer for a review that identified precisely the two deepest interpretive risks in the paper — semantic correctness and centroid circularity. We ran new quantitative experiments targeting both (same frozen features, splits and estimators as the paper), and they materially sharpen the story. We report them below together with the honest answer on downstream scope.

## 1. Semantic correctness, quantitatively (W1, Q1)

The reviewer is right that Figure 3 is only visual evidence. We added two ground-truth-hierarchy measurements:

**(a) WordNet alignment on ImageNet.** Spearman correlation between the 1000×1000 inter-centroid distance matrix and the WordNet tree-distance matrix:

| family | $\rho$(centroid dist, WordNet dist) |
|---|---|
| CLIP-L / SigLIP-B / CLIP-B | +0.59 / +0.58 / +0.57 |
| ViT-T/S/B/L (i21k) | +0.52 / +0.50 / +0.49 / +0.53 |
| DINOv1-B | +0.36 |
| DINOv2-S/B/L/G | +0.22 / +0.18 / +0.18 / +0.20 |
| label-shuffle control | $\pm$0.03 |
| Gaussian control | $\approx$0.00 |

**(b) Superclass recovery on CIFAR-100.** Cutting the dendrogram of the 100 class centroids at 20 clusters and scoring against the *true* 20 superclasses (Ward linkage; we verified robustness over average/complete/k-means, since average linkage chains degenerately on the most compact models): ARI 0.61 (ViT-B), 0.56 (ViT-S), 0.55 (ViT-L), 0.52 (CLIP-L), 0.51 (DINOv2-S), 0.47 (DINOv2-B), vs ≈0.00 for permuted labels. So the recovered branches align with the human taxonomy far above chance — the mathematical tree is, to a quantifiable degree, the semantic tree. We will add both measurements, and we agree HierarCaps-style multi-level caption hierarchies are the natural next benchmark; we commit to including such an evaluation in the revision.

**An instructive decoupling.** Note the inversion: the family with the *strongest* tree-likeness by $\hat\delta$/ORC (DINOv2, trained with no labels and no language) has the *weakest* WordNet alignment, while language-supervised models align most. So "how tree-like the geometry is" and "whose tree it is" are distinct, separately measurable properties. A finer decomposition sharpens this further: DINOv2's semantic structure is concentrated in the *angular* component of its geometry — on CIFAR-100, sibling-vs-non-sibling triplet agreement for DINOv2-L/G is 0.91/0.85 under cosine distance but only 0.71/0.63 under raw Euclidean (supervised and CLIP models score 0.93–0.96 under both) — while its *global* cross-branch taxonomy alignment remains the weakest on ImageNet under either metric (coarse-level triplets 0.59–0.66 vs 0.83–0.85 for CLIP/SigLIP). In short: strong local-angular semantics, flat global tree. We will report this local/global and angular/radial decomposition explicitly rather than implying that low $\hat\delta$ means human-taxonomy alignment.

## 2. The centroid problem (W2, Q2)

The concern is legitimate and we now bound it with three measurements:

1. **Random vs semantic groupings.** If the space were merely "flexible enough to be binned into linguistic buckets", arbitrary binning should look as tree-like as semantic binning. It does not: merging the 1000 classes into 100 WordNet-coherent groups yields consistently lower $\hat\delta$ than 100 random groups (CLIP-B .126 vs .174; CLIP-L .117 vs .162; ViT-L .104 vs .117; 8/12 models, remaining 4 at floor). The measured hierarchy tracks semantic structure, not the pooling operation.
2. **The decoupling above.** If averaging with label-defined bins injected the human taxonomy into the geometry, the label-free model (DINOv2) should not be the *most* tree-like yet *least* WordNet-aligned family — the injection account predicts the opposite coupling.
3. **No pooling at all.** Using a single random image per class (no centroid averaging), the distance matrix still correlates with WordNet: $\rho$ = +0.34 (ViT-T), +0.27 (CLIP-L), +0.23 (SigLIP/CLIP-B), +0.20 (DINOv1), lower for DINOv2 (+0.03–0.15), averaged over 5 draws with small std. Pooling denoises (centroid $\rho$ is roughly 2× higher) but the hierarchical signal pre-exists at the single-sample level for most families; it is not manufactured by averaging.

What remains true — and we will say it plainly in the limitations — is that the *class inventory* (which 1000 concepts exist) is human-chosen, and for language models the prompts additionally inject class names; a related prompt-robustness analysis (reported to Reviewers Xbn5/RJje) leads us to narrow the causal-LM claims. For vision models, however, points 1–3 above show the measured tree is a property of the representation, probed — not created — by the label structure.

## 3. Overstated claims and downstream scope (W3, Q3)

We accept this criticism and will act on it in three ways.

- **Language.** "Universal geometry of foundation models" will be softened to the precise statement: tree-like inter-class metric structure appears across the 30 studied models with strength depending on objective and scale, and its absolute level must be read against matched random-cloud baselines (new control tables added following Reviewer RJje; several absolute-level claims are restated as excesses over matched nulls).
- **Honest negative results at sample level.** We had already run two of the deployment-style tasks the reviewer asks about, and the projection does *not* help there: kNN classification (mean advantage −2.6pp at the paper's radius) and sample-to-sample retrieval P@10 (−3.8pp fine / −3.0pp superclass, positive in 0/60 cells). The mechanism is consistent: the class hierarchy lives at the prototype level, while the radial map distorts local sample neighborhoods. The revision reports both and states the domain of validity — prototype-based metric tasks — explicitly, replacing any implication of general deployment gains.
- **A positive, sharper finding.** Adding cosine/spherical and matched radial-rescaling controls (full 60-cell study) shows the hyperbolic advantage over the *best* zero-cost alternative concentrates on contrastive VLMs: for CLIP, Poincaré distances beat cosine on few-shot by +0.9..+1.3pp even after L2 normalization (per-episode CI95 ≤ ±0.15pp), while for DINOv2 cosine suffices. For dense prediction and end-to-end hyperbolic training our results make no claim, and we will point to the training-time literature (MERU, HypLoRA) as the appropriate tools there.

We believe these additions turn the two conflations the reviewer identified (structural existence vs semantic correctness; representation hierarchy vs linguistic bucketing) into explicitly measured quantities, and we are happy to run further checks during the discussion phase.
