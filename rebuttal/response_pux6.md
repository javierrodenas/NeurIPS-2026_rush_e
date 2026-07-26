We are grateful for this thoughtful review, which identified precisely the two deepest interpretive risks in the paper, semantic correctness and centroid circularity. We ran new quantitative experiments targeting both (same frozen features, splits and estimators as the paper); full tables will be in the revision.

> **Q1 (Semantic Correctness).** *"How can we quantitatively guarantee that the generated tree accurately mirrors the true semantic hierarchy of the concepts, rather than just an arbitrary branching structure? Have you considered evaluating the alignment against a dataset with strict, ground-truth hierarchical labels?"*

**A1. The reviewer is right that Figure 3 is only visual evidence. Two new ground-truth measurements:**

**(a) WordNet alignment on ImageNet**: Spearman correlation between the 1000×1000 inter-centroid distances and the WordNet tree distances:

| family | $\rho$ |
|---|---|
| CLIP-L / SigLIP-B / CLIP-B | +0.57..+0.59 |
| ViT-T/S/B/L (i21k) | +0.49..+0.53 |
| DINOv1-B | +0.36 |
| DINOv2-S/B/L/G | +0.18..+0.22 |
| shuffle / Gaussian controls | ±0.03 / ≈0.00 |

**(b) Superclass recovery on CIFAR-100**: cutting the centroid dendrogram at 20 clusters recovers the *true* 20 superclasses at ARI up to 0.61 (Ward; robustness verified over four clustering methods), vs ≈0.00 for permuted labels. We agree HierarCaps-style multi-level hierarchies are the natural next benchmark and commit to such an evaluation in the revision.

**An instructive decoupling.** The family with the *strongest* tree-likeness by $\hat\delta$/ORC (DINOv2, trained with no labels and no language) has the *weakest* WordNet alignment: "how tree-like" and "whose tree" are distinct, separately measurable properties. A finer decomposition:
- **Local, angular:** on CIFAR-100, sibling-vs-non-sibling triplet agreement (chance = 0.5) for DINOv2-L/G is 0.91/0.85 under cosine but only 0.71/0.63 under raw Euclidean: DINOv2's semantics are concentrated in the angular component.
- **Global:** its cross-branch taxonomy alignment remains the weakest on ImageNet under either metric.

In short: strong local-angular semantics, flat global tree. The revision reports this decomposition explicitly rather than implying that low $\hat\delta$ means human-taxonomy alignment.

> **Q2 (The Centroid Methodology).** *"How do you disentangle the intrinsic visual hierarchy learned by the model from the linguistic hierarchy imposed by averaging the ImageNet classes? Could the tree-structure simply be an artifact of testing the models through a linguistically biased lens?"*

**A2. Three measurements bound this legitimate concern:**

1. **Random vs semantic groupings.** If the space were merely "flexible enough to be binned into linguistic buckets", arbitrary binning would look as tree-like as semantic binning. It does not: WordNet-coherent groups yield lower $\hat\delta$ than random groups in 11/12 models (e.g. CLIP-B .126 vs .174).
2. **The decoupling above.** If label-defined averaging injected the human taxonomy, the label-free model (DINOv2) should not be the *most* tree-like yet *least* WordNet-aligned family; the injection account predicts the opposite coupling.
3. **No pooling at all.** With a single random image per class (no centroids), distances still correlate with WordNet ($\rho$ up to +0.34; lower for DINOv2). Pooling denoises, but the hierarchical signal pre-exists at the single-sample level; it is not manufactured by averaging.

What remains true, and the limitations section will say plainly: the *class inventory* is human-chosen, and for language models the prompts inject class names (a prompt-robustness analysis reported to Reviewers Xbn5/RJje narrows the causal-LM claims). For vision models, points 1–3 show the measured tree is a property of the representation, probed, not created, by the label structure.

> **Q3 (Downstream Utility).** *"Given the modest performance gains on simple metric tasks, do you have any preliminary results or theoretical intuition on whether this post-hoc Poincaré projection yields benefits in more complex, modern deployment scenarios like dense segmentation or text-image retrieval?"*, and W3: *"The authors claim to have uncovered the 'universal geometry of foundation models'. However ... the downstream applications must better reflect how foundation models are actually deployed."*

**A3. We accept this criticism and act on it in three ways:**

- **Language.** "Universal geometry" will be softened to the precise statement: tree-like inter-class structure appears across the 30 models with strength depending on objective, scale **and dataset**, read against matched random-cloud baselines (e.g. DINOv2's null-corrected structure is absent on ImageNet yet the strongest of the panel on transfer datasets).
- **Honest negatives at sample level.** We had already run two deployment-style tasks and the projection does *not* help there: kNN (−2.6pp mean) and sample-to-sample retrieval (positive in 0/60 cells). The class hierarchy lives at the prototype level; the radial map distorts local neighborhoods. The revision states the domain of validity, prototype-based metric tasks, explicitly.
- **A positive, sharper finding.** With cosine and matched rescaling controls added (full 60-cell study), the hyperbolic advantage over the *best* zero-cost alternative concentrates on contrastive VLMs: CLIP beats cosine on few-shot even after L2 normalization (+0.9..+1.3pp on CIFAR-100/10/DTD), while for DINOv2 cosine suffices. For dense prediction, cross-modal (text–image) retrieval, and end-to-end hyperbolic training our results make no claim; we will point to training-time approaches (MERU, HypLoRA) as the appropriate tools there.

We believe these additions turn the two conflations the reviewer identified (structural existence vs semantic correctness; representation hierarchy vs linguistic bucketing) into explicitly measured quantities, and we are happy to run further checks during the discussion phase. Thank you again for pushing us to quantify both; the final version will be a stronger paper for it.
