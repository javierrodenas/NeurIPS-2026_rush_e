We thank the reviewer for a review that identified precisely the two deepest interpretive risks in the paper — semantic correctness and centroid circularity. We ran new quantitative experiments targeting both (same frozen features, splits and estimators as the paper), reported below in Q→A form together with the honest answer on downstream scope.

> **Q1 (Semantic Correctness).** *"How can we quantitatively guarantee that the generated tree accurately mirrors the true semantic hierarchy of the concepts, rather than just an arbitrary branching structure? Have you considered evaluating the alignment against a dataset with strict, ground-truth hierarchical labels?"*

**A1.** The reviewer is right that Figure 3 is only visual evidence. Two new ground-truth measurements:

**(a) WordNet alignment on ImageNet** — Spearman correlation between the 1000×1000 inter-centroid distance matrix and the WordNet tree-distance matrix:

| family | $\rho$(centroid dist, WordNet dist) |
|---|---|
| CLIP-L / SigLIP-B / CLIP-B | +0.59 / +0.58 / +0.57 |
| ViT-T/S/B/L (i21k) | +0.52 / +0.50 / +0.49 / +0.53 |
| DINOv1-B | +0.36 |
| DINOv2-S/B/L/G | +0.22 / +0.18 / +0.18 / +0.20 |
| label-shuffle / Gaussian controls | $\pm$0.03 / $\approx$0.00 |

**(b) Superclass recovery on CIFAR-100** — cutting the centroid dendrogram at 20 clusters (Ward linkage; robustness verified over average/complete/k-means, since average linkage chains degenerately on the most compact models) and scoring against the *true* 20 superclasses: ARI 0.61 (ViT-B), 0.56 (ViT-S), 0.55 (ViT-L), 0.52 (CLIP-L), 0.51 (DINOv2-S), 0.47 (DINOv2-B), vs ≈0.00 for permuted labels. The recovered branches align with the human taxonomy far above chance. We agree HierarCaps-style multi-level hierarchies are the natural next benchmark and commit to including such an evaluation in the revision.

**An instructive decoupling.** The family with the *strongest* tree-likeness by $\hat\delta$/ORC (DINOv2, trained with no labels and no language) has the *weakest* WordNet alignment, while language-supervised models align most: "how tree-like the geometry is" and "whose tree it is" are distinct, separately measurable properties. A finer decomposition sharpens this: DINOv2's semantic structure is concentrated in the *angular* component of its geometry — on CIFAR-100, sibling-vs-non-sibling triplet agreement for DINOv2-L/G is 0.91/0.85 under cosine distance but only 0.71/0.63 under raw Euclidean (supervised and CLIP score 0.93–0.96 under both) — while its *global* cross-branch taxonomy alignment remains the weakest on ImageNet under either metric (coarse-level triplets 0.59–0.66 vs 0.83–0.85 for CLIP/SigLIP). In short: strong local-angular semantics, flat global tree. We will report this local/global and angular/radial decomposition explicitly rather than implying that low $\hat\delta$ means human-taxonomy alignment.

> **Q2 (The Centroid Methodology).** *"How do you disentangle the intrinsic visual hierarchy learned by the model from the linguistic hierarchy imposed by averaging the ImageNet classes? Could the tree-structure simply be an artifact of testing the models through a linguistically biased lens?"*

**A2.** Three measurements now bound this legitimate concern:

1. **Random vs semantic groupings.** If the space were merely "flexible enough to be binned into linguistic buckets", arbitrary binning should look as tree-like as semantic binning. It does not: 100 WordNet-coherent groups yield consistently lower $\hat\delta$ than 100 random groups (CLIP-B .126 vs .174; CLIP-L .117 vs .162; ViT-L .104 vs .117; 8/12 models, the rest at floor). The measured hierarchy tracks semantic structure, not the pooling operation.
2. **The decoupling above.** If label-defined averaging injected the human taxonomy into the geometry, the label-free model (DINOv2) should not be the *most* tree-like yet *least* WordNet-aligned family — the injection account predicts the opposite coupling.
3. **No pooling at all.** With a single random image per class (no centroids), the distance matrix still correlates with WordNet: $\rho$ = +0.34 (ViT-T), +0.27 (CLIP-L), +0.23 (SigLIP/CLIP-B), +0.20 (DINOv1), lower for DINOv2 (+0.03–0.15); 5 draws, small std. Pooling denoises (centroid $\rho$ is ~2× higher) but the hierarchical signal pre-exists at the single-sample level — it is not manufactured by averaging.

What remains true — and the limitations section will say it plainly — is that the *class inventory* is human-chosen, and for language models the prompts additionally inject class names (a prompt-robustness analysis reported to Reviewers Xbn5/RJje leads us to narrow the causal-LM claims). For vision models, points 1–3 show the measured tree is a property of the representation, probed — not created — by the label structure.

> **Q3 (Downstream Utility).** *"Given the modest performance gains on simple metric tasks, do you have any preliminary results or theoretical intuition on whether this post-hoc Poincaré projection yields benefits in more complex, modern deployment scenarios like dense segmentation or text-image retrieval?"* — and W3: *"the downstream applications must better reflect how foundation models are actually deployed."*

**A3.** We accept this criticism and act on it in three ways:

- **Language.** "Universal geometry of foundation models" will be softened to the precise statement: tree-like inter-class metric structure appears across the 30 studied models with strength depending on objective, scale **and dataset**, read against matched random-cloud baselines (new control tables following Reviewer RJje; run on all six datasets, they directly support the capacity×data framing — e.g., DINOv2's null-corrected tree structure is absent on ImageNet yet the strongest of the panel on the transfer datasets, deepening with scale).
- **Honest negative results at sample level.** We had already run two deployment-style tasks, and the projection does *not* help there: kNN classification (mean −2.6pp at the paper's radius) and sample-to-sample retrieval P@10 (−3.8pp fine / −3.0pp superclass; positive in 0/60 cells). Mechanism: the class hierarchy lives at the prototype level, while the radial map distorts local sample neighborhoods. The revision reports both and states the domain of validity — prototype-based metric tasks — explicitly.
- **A positive, sharper finding.** Adding cosine/spherical and matched radial-rescaling controls (full 60-cell study): the hyperbolic advantage over the *best* zero-cost alternative concentrates on contrastive VLMs — for CLIP, Poincaré beats cosine on few-shot by +0.9..+1.3pp even after L2 normalization (per-episode CI95 ≤ ±0.15pp), while for DINOv2 cosine suffices. For dense prediction and end-to-end hyperbolic training our results make no claim; we will point to training-time approaches (MERU, HypLoRA) as the appropriate tools there.

We believe these additions turn the two conflations the reviewer identified (structural existence vs semantic correctness; representation hierarchy vs linguistic bucketing) into explicitly measured quantities, and we are happy to run further checks during the discussion phase.
