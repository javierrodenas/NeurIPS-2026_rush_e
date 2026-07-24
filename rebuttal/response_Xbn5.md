We thank the reviewer for the encouraging assessment and for four questions that are exactly the right stress-tests. We ran new experiments for each (same frozen features, splits and estimators as the paper; code available to the AC on request) and report everything we found, including the negatives — we would rather the reviewer reconsider significance on an honest basis than on a curated one.

> **Q1.** *"Can you validate the post-hoc projection on at least one additional downstream task, such as retrieval, clustering, or potentially a text-based task? If $\delta$ and ORC continue to predict when hyperbolic distances help beyond NC and FS vision classification, it would definitely make me reconsider the significance of the study."*

**A1.** We extended the evaluation in four directions; two are negative (now scoped accordingly), two are positive and sharpen the contribution:

- **Sample-level tasks (negative).** kNN (k=5) and sample-to-sample retrieval P@10 across the same 60 cells: the projection *hurts* both (kNN mean −2.6pp; retrieval −3.8pp fine / −3.0pp superclass; positive in 0/60 cells). Mechanism: the class hierarchy the diagnostics detect lives at the prototype level; the radial map distorts local sample neighborhoods. The revision restricts the claimed domain of validity to prototype-based metric tasks.
- **Hierarchy recovery (negative for H, informative overall).** Cutting centroid dendrograms of CIFAR-100 at 20 clusters vs the true superclasses (robustness over average/complete/Ward/k-means): Euclidean Ward recovers them well (ARI up to 0.61) and Poincaré-distance linkage does not improve on the best Euclidean/cosine variant for any model. Reported as-is.
- **Metric-controlled study on the two tasks (positive).** Adding cosine (≡ spherical for ranking) and a matched radial-rescaling control (same tanh map, Euclidean distance) to all 60 cells: (i) the rescaling control changes nothing (|RT−R| ≤ 0.25pp), so the paper's H−R gain is attributable to the metric itself; (ii) against cosine — the strongest zero-cost alternative — the hyperbolic advantage concentrates on contrastive VLMs: CLIP few-shot H−COS = +0.7pp raw, +0.9..+1.3pp after L2 normalization (Poincaré *stacks on top of* normalization; per-episode CI95 ≤ ±0.15pp), while for DINOv2 cosine suffices. $\delta$/ORC continue to predict the H−R contrast as in the paper; the practical protocol becomes a three-way decision (Euclidean / cosine / hyperbolic) in the revision.
- **Cross-model alignment (negative; see A4).**

A text-based downstream evaluation was not run in this period; we state it as future work in the revision rather than implying coverage.

> **Q2.** *"How robust are the text-model geometry results to prompt variation? ... an additional, simple validation using variations of the prompt 'a photo of a {object}' would provide more reassurance."*

**A2.** Six templates over the 1000 ImageNet class names ("a photo of a {}", "{}" name-only, "an image of a {}", "a close-up photo of a {}", "this is a photo of a {}", "a {} in the wild"):

| model | min–max $\hat\delta$ across templates | range |
|---|---|---|
| BGE-base | .115–.126 | .011 |
| E5-base | .114–.120 | .006 |
| GPT-2 | .080–.145 | .065 |
| GPT-2-M | .091–.134 | .043 |

Text **embedders are highly robust** (range ≤ .011; the paper's template reproduces our published values, e.g. BGE .124 vs .123). **Causal LMs are template-sensitive**, and under name-only prompts the LM-vs-embedder ordering attenuates or reverses. We will therefore report causal-LM geometry as mean±range over templates and narrow the corresponding claims; the vision results — the paper's core — involve no prompts. We thank the reviewer for prompting exactly the check that surfaced this.

> **Q3.** *"Given the sometimes very small improvements on downstream tasks (e.g., +0.80 pp on DTD NC), it would be useful to include an additional statistical measure, such as standard deviation over 3 or 5 seeds, to show that even small improvements are consistent."*

**A3.** Few-shot cells use 1000 episodes; we now compute per-episode *paired* confidence intervals for the advantage: across the 60 cells, CI95 ranges from ±0.06 to ±0.21pp. The headline few-shot gains exceed their cell CI by large factors (DINOv2-G ImageNet +0.9pp vs ±0.09; DINOv2-G CIFAR-100 +1.2pp vs ±0.10; CLIP-L DTD +1.0pp vs ±0.14). NC is deterministic given the frozen features (no seed variance); the revision adds McNemar tests on the paired predictions (e.g., the +2.1pp NC gain of DINOv2-L on CIFAR-100 corresponds to 210 test decisions on n=10,000). The DTD NC +0.8pp cell the reviewer cites is one of the smaller effects and will be presented with its uncertainty rather than as a headline.

> **Q4.** *"I would like the authors to clarify the main reasons why, and in what sense, this work is supposedly tightly connected to the Platonic Representation Hypothesis."*

**A4.** We agree the connection as framed was too strong, and we tested the natural quantitative bridge: PRH-style mutual-kNN alignment between all 66 pairs of the 12 vision models on the shared 1000 ImageNet class centroids, under Euclidean vs Poincaré kNN. Result: hyperbolic distances do **not** increase cross-model alignment (mean 0.474 Euclid vs 0.427 Poincaré; 6/66 pairs improve; within-SSL pairs flat at +0.02). The revision will therefore (i) reposition the contribution as characterizing the *within-model geometry* of representations — a property that stands regardless of whether cross-model convergence holds — (ii) soften title/abstract framing accordingly, and (iii) cite and discuss the recent critical work on PRH-style convergence raised by Reviewer RJje. The honest one-line answer to "why PRH?": PRH motivates asking *what the converged representation looks like*; our paper answers the shape question for inter-class structure, and makes no convergence claim.

We hope the prompt-robustness table, the paired CIs, and the metric-controlled study (with its family-level finding) constitute the "one additional set of results" the reviewer asked for — with the negatives included so the significance judgment can be made on the full picture. We are glad to run further checks during the discussion phase.
