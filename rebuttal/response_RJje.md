We sincerely appreciate the time and effort invested in this review. We ran **every requested control** on the same frozen features, splits and estimator as the paper (code on request). Each answer below leads with its verdict; full per-model tables will be in the revision.

> **Q1.** *"Can the authors directly test whether the observed structure is better explained by negative curvature than by positive curvature or sphere-like geometry?"*

**A1. Yes: spherical geometry cannot produce the observed values.** Calibrating the exact estimator $\hat\delta=\delta_{\max}/\mathrm{diam}$ (500K quadruples × 10 seeds) on reference geometries:

| space | $\hat\delta$ |
|---|---|
| balanced binary tree (depth 10) | 0.000 |
| $\mathbb{H}^2$ ($K{=}-1$), regions $R{=}2/4/8/16$ | 0.162 / 0.087 / 0.043 / 0.022 |
| uniform $S^{99}$ (chord / geodesic) | 0.143 / 0.179 |
| iid Gaussian, $n{=}1000$, $d{=}192/768/1536$ | 0.104 / 0.061 / 0.046 |

Hyperbolic spaces have bounded absolute $\delta$ ($\to\ln2$), so $\hat\delta\to0$ as the region grows; spheres are scale-invariant, stuck high. Table 1's ImageNet rows (.066–.123) all sit below the sphere band, and sphericizing each model's centroids *raises* $\hat\delta$ (geodesic; A2): low $\hat\delta$ is diagnostic of tree-like, not spherical, structure. The Gaussian row anticipates the reviewer's concentration point (correct): raw levels are only interpretable against matched nulls, addressed next.

> **Q2.** *"Can the authors include a spherical baseline in the main figures...?"* / *"... random Gaussian, random normalized, whitened, and dimension-matched controls."*

**A2. All requested controls run.** Representative rows (ImageNet centroids):

| model | real | Gauss null | L2-chord | L2-geodesic |
|---|---|---|---|---|
| ViT-L | .084 | .055 | .100 | .107 |
| DINOv2-L | .067 | .055 | .096 | .106 |
| CLIP-B | .122 | .074 | .121 | .123 |

- **Sphere:** sphericizing raises $\hat\delta$ for all 12 models under the geodesic; 10/12 under chord, the two exceptions (CLIP-B/L) within .002. **Whitening** collapses centroids to a near-regular simplex (a degenerate star), so we rely on the spectrum-matched null instead.
- **Concession, and the calibrated verdict.** Low absolute $\hat\delta$ alone is not evidence of hierarchy: iid clouds approach a star. Against **spectrum-matched nulls** (same covariance spectrum; confirmed by a PC-permutation null), supervised ViTs and CLIP/SigLIP sit clearly below their nulls (excesses −0.015..−0.047; null s.d. ≤ .007), genuine higher-order tree structure; DINOv2 sits at its null **on ImageNet only**.
- **Across all six datasets, 69/72 model×dataset cells beat their nulls.** The 3 exceptions are DINOv2-S/B/G on ImageNet; on the transfer datasets, where the paper's gains concentrate, DINOv2 shows the *strongest* excess of the panel, deepening with scale (S→G on CIFAR-10: −0.087→−0.146).
- **Dimension matching:** under random projection the family ordering survives; PCA truncation attenuates it. We will restate absolute and scaling claims (incl. the "45%") as excess over matched nulls; the within-architecture ablations are same-$d$ and unaffected. We also do **not** estimate curvature: descriptive claims become "tree-like inter-class metric structure".

> **Q3.** *"What happens if distances are computed after L2 normalization, with cosine distance, or with spherical distance?"*

**A3. Family-dependent; added to all 60 cells** (cosine ≡ spherical for ranking). Hierarchical-dataset means:

- **SSL (DINOv2):** cosine is the strongest zero-cost metric (H−COS −0.5pp).
- **Supervised ViTs:** H ≈ COS.
- **Contrastive VLMs: H beats COS**, +0.7pp raw and +0.9..+1.3pp after L2-normalizing first on CIFAR-100/10/DTD (ImageNet +0.1..+0.2; CI95 ≤ ±0.15pp): Poincaré stacks on top of normalization.

The practical protocol becomes three-way (Euclidean / cosine / hyperbolic); $\hat\delta$/ORC still predict the H−R contrast, unchanged.

> **Q4.** *"What happens if the same radial transformation is used, but distances remain Euclidean or spherical?"*

**A4. Nothing changes: the gain is the metric.** The identical tanh map with Euclidean distances (RT) is indistinguishable from raw (|RT−R| ≤ 0.25pp), while H−RT ≈ H−R (up to +1.2pp). The spherical half reduces to centered cosine, since the map preserves direction, also run (COSC): same picture as A3.

> **Q5.** *"Can the authors estimate curvature instead of fixing the Poincaré ball to curvature −1?"*

**A5.** We swept the effective curvature scale ($t\in[0.25,0.95]$; $c\in[0.25,4]$ on full ImageNet): the few-shot advantage is positive at every setting, peaking near $t≈0.6$–$0.7$; conclusions are insensitive to the −1 convention. Estimating intrinsic curvature is a different problem, stated as out of scope.

> **Q6.** *"How much of the measured tree-like structure remains if class labels are replaced by random class groupings? Or ... under different prompts...?"*

**A6.**
- **Groupings:** WordNet-coherent groups are more tree-like than random groups in **11/12 models** (e.g. CLIP-B .126 vs .174): the structure tracks semantics, not binning.
- **Anti-circularity:** centroid distances correlate with WordNet at $\rho$ up to +0.59 (CLIP-L) vs +0.18..+0.22 for label-free DINOv2 (shuffle ±0.03). The lowest-$\hat\delta$ family is the *least* WordNet-aligned, the opposite of what label injection would produce: the *amount* of tree-likeness and the *identity* of the tree are decoupled.
- **Prompts** (6 templates incl. name-only): embedders are highly robust (range ≤.011; the paper's template reproduces, BGE .124 vs .123); causal LMs are template-sensitive, and name-only attenuates or reverses their ordering vs embedders: **we will narrow the causal-LM claims**, discuss template dependence in the limitations, and add WordNet-definition and non-visual prompts in the revision. The core vision results involve no prompts.

> **W (ORC).** *"A confusing internal tension in the ORC argument ... highest positive ORC and sometimes 0 percent negative edges."*

**A7.** We apologize: our §3.2 sentence was misleading. An edge-type analysis (k=10 centroid kNN graph, 30 WordNet superclasses) resolves it: high mean ORC reflects tight *within*-superclass clusters, while negative edges concentrate on the *bridges* between them (2–4× higher fraction, e.g. CLIP-L .040 vs .084). DINOv2 on ImageNet is the exception, ≈0% negative even across its own clusters, consistent with its near-null geometry (A2). The ORC paragraph will be rewritten accordingly.

> **Minor.** *"The connection to the PRH is not fully established ... discuss recent work that questions PRH-style convergence [1,2]."*

**A8.** Agreed, and we ran the suggested test: mutual-kNN alignment across all 66 model pairs on the shared 1000 centroids, Euclidean vs Poincaré. Hyperbolic does **not** improve cross-model alignment (0.474 vs 0.427). This matches §2.4 of the submission ("a within-model property that stands on its own"); the revision promotes that framing into the title/abstract, softens "universal", and cites both suggested works. [2] we independently mirror: our matched-null recalibration is its calibration move applied to $\hat\delta$, and our conclusion is its within-model complement: what is shared is the *form* (tree-likeness, 69/72), the *content* is model-specific (each model its own tree).

**Committed revisions:** matched-null tables with claims restated as excesses; rewritten ORC paragraph; WordNet-alignment section; full cosine/RT/normalized comparison and three-way protocol; narrowed causal-LM claims; PRH discussion with both references; removal of curvature-estimation implications.

Thank you again for a review that has materially improved this work.
