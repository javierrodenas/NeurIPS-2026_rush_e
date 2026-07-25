We thank the reviewer for an unusually constructive review. We ran **every requested control** on the same frozen features, splits and estimator as the paper (code available to the AC on request). Some controls confirm the paper's mechanisms; two sharpen its claims — we state both explicitly, with revision commitments at the end.

> **Q1.** *"Can the authors directly test whether the observed structure is better explained by negative curvature than by positive curvature or sphere-like geometry?"*

**A1.** We calibrated the exact estimator $\hat\delta=\delta_{\max}/\mathrm{diam}$ (500K quadruples × 10 seeds) on reference geometries:

| space | $\hat\delta$ |
|---|---|
| balanced binary tree (depth 10; 1023 nodes ≈ centroid count) | 0.000 |
| $\mathbb{H}^2$ ($K{=}-1$), region radius $R{=}2/4/8/16$ | 0.162 / 0.087 / 0.043 / 0.022 |
| uniform $S^{99}$ (chord / geodesic) | 0.143 / 0.179 |
| iid Gaussian, $n{=}1000$, $d{=}192/768/1536$ | 0.104 / 0.061 / 0.046 |

Hyperbolic spaces have bounded absolute $\delta$ ($\to\ln2$), so $\hat\delta\to0$ as the region grows; spheres are scale-invariant, stuck at high $\hat\delta$ — they cannot produce low values at any radius. The 12 models' centroid values (.067–.123) all sit below the sphere band, and the data-side direct test agrees: forcing each model's centroids onto the sphere (L2) *raises* $\hat\delta$ in all 12 cases (table in A2). Low $\hat\delta$ is therefore diagnostic of tree-like rather than spherical structure — but the concentration concern is **correct**: iid Gaussians approach the equidistant (star-tree) limit as $d$ grows, so raw levels are only interpretable against matched nulls (A2).

> **Q2.** *"Can the authors include a spherical baseline in the main figures...?"* / *"...could also arise from positive curvature, sphere-like geometry, high-dimensional concentration, or clustered class separation."*

**A2.** Yes — new matched-$n,d$ controls on the ImageNet centroids:

| model | real | Gauss null | L2-chord | L2-geodesic | WordNet grp | random grp |
|---|---|---|---|---|---|---|
| ViT-L | .084 | .055 | .100 | .107 | .104 | .117 |
| DINOv2-G | .080 | .046 | .098 | .106 | .054 | .056 |
| CLIP-L | .118 | .061 | .117 | .119 | .117 | .162 |

(i) Forcing centroids onto the sphere *raises* $\hat\delta$ for all 12 models, and the cross-model ordering is preserved under the cosine metric — spherical geometry is disfavored. (ii) **Concession and calibration**: trained centroids sit above the iid null (a degenerate near-star), so low absolute $\hat\delta$ alone cannot be read as "semantically hierarchical". Against a **spectrum-matched Gaussian null** (all second-order structure equated; confirmed by an independent PC-permutation null), supervised ViTs (excess −0.015..−0.040, strongest for ViT-L) and CLIP/SigLIP (−0.017..−0.047) sit clearly below their nulls — genuine higher-order tree structure — whereas DINOv2 sits at its null **on ImageNet specifically** (−0.01..+0.02). Extending the nulls to all six datasets shows this is dataset-dependent, in line with the paper's capacity×data framing: on the transfer datasets, where the downstream gains concentrate, DINOv2 shows the *strongest* genuine excess of the panel (DINOv2-G: −0.078 CIFAR-100, −0.146 CIFAR-10), deepening with scale (S→G on CIFAR-10: −0.087→−0.146). Under random projection to matched $d{=}192$ the family ordering survives; PCA truncation attenuates it. All absolute-level and cross-dimension statements (incl. the "45%" headline) will be restated as excess over matched nulls; the within-architecture ablations are same-$d$ contrasts and unaffected. We also agree we do **not** estimate curvature (sign or value): descriptive sections will say "tree-like inter-class metric structure" instead.

> **Q3.** *"What happens if distances are computed after L2 normalization, with cosine distance, or with spherical distance?"*

**A3.** Added to all 60 cells (cosine ≡ spherical geodesic for ranking, by monotonicity). Honest summary: on SSL backbones cosine is the strongest zero-cost metric (FS COS−R +1.44pp; H−COS −0.5pp); on supervised ViTs H≈COS; on contrastive VLMs **H beats COS** (FS +0.7pp raw; after L2-normalizing first, +0.9..+1.3pp on CIFAR-100/10/DTD and +0.1..+0.2pp on ImageNet — Poincaré *stacks on top of* normalization for CLIP; per-episode CI95 ≤ ±0.15pp). We will report the full comparison and reframe the practical protocol as a three-way choice, with $\hat\delta$/ORC predicting the H−R contrast (unchanged) and paradigm predicting H-vs-cosine. We believe "hyperbolic beats spherical specifically on contrastive VLMs" is of independent interest.

> **Q4.** *"What happens if the same radial transformation is used, but distances remain Euclidean or spherical?"*

**A4.** Answered directly with an RT control: the *identical* radial tanh map, then Euclidean distances on the transformed points. RT−R ≈ 0 everywhere (|mean| ≤ 0.25pp), while H−RT ≈ H−R (SSL few-shot +0.89pp; DINOv2-G +1.20pp). For the *spherical* half of the question: since the radial map preserves direction, "same transform + spherical distance" reduces exactly to cosine on centered features — also run (COSC), giving the same picture as A3. The paper's H−R gain is attributable to the hyperbolic metric, not to radial rescaling.

> **Q5.** *"Can the authors estimate curvature instead of fixing the Poincaré ball to curvature −1?"*

**A5.** We swept the effective curvature scale (radius target $t\in[0.25,0.95]$; curvature $c\in[0.25,4]$ on full ImageNet): the few-shot advantage is positive at every setting, peaking near $t≈0.58$–$0.71$ — conclusions are insensitive to the −1 convention. Estimating intrinsic curvature is a different problem, which we will state as out of scope.

> **Q6.** *"How much of the measured tree-like structure remains if class labels are replaced by random class groupings? Or ... under different prompts, class-name-only prompts...?"*

**A6.** (i) *Groupings*: merging the 1000 classes into 100 WordNet-coherent groups yields lower $\hat\delta$ than 100 random groups in 8/12 models (CLIP-B .126 vs .174; ViT-L .104 vs .117; the largest DINOv2 at floor under both) — tree-likeness tracks semantics, not binning. (ii) *Anti-circularity*: Spearman between centroid distances and WordNet tree distances: CLIP-L +0.59, SigLIP +0.58, supervised +0.49..+0.53, DINOv1 +0.36, DINOv2 +0.18..+0.22 (shuffle ±0.03). The family with the lowest $\hat\delta$ (label-free DINOv2) is the *least* WordNet-aligned — if the centroid construction injected the taxonomy, the opposite pattern would appear: the *amount* of tree-likeness and the *identity* of the tree are decoupled. (iii) *Prompts* (6 templates incl. class-name-only): embedders are highly robust ($\hat\delta$ range ≤0.011; BGE .124 reproduces the paper's .123); causal LMs are template-sensitive (GPT-2-M .091–.134), and name-only attenuates/reverses the LM-vs-embedder ordering — **we will narrow the causal-LM claims accordingly**, and the limitations section will discuss template dependence explicitly. The two remaining variants the reviewer lists (WordNet-definition prompts and non-visual concepts) will be added in the revision. The paper's core vision results involve no prompts.

> **W (ORC).** *"There is a confusing internal tension in the ORC argument ... the most tree-like models ... have the highest positive ORC and sometimes 0 percent negative edges."*

**A7.** The reviewer is right that our §3.2 sentence was misleading. New edge-type analysis (k=10 centroid kNN graph, WordNet-30 superclasses): for supervised and contrastive models, within-superclass edges have mean ORC +0.34..+0.41 vs +0.26..+0.31 across, and the negative-edge fraction is 2–4× higher on across (bridge) edges (ViT-L: .021 vs .068; CLIP-L: .040 vs .084) — high mean ORC = tight leaf clusters; negative curvature concentrates on the bridges. DINOv2 is the exception on ImageNet: even its cross-cluster edges are positively curved (0% negative under WordNet-30 *and* under its own 30 intrinsic clusters), consistent with its ImageNet-specific near-null geometry (A2). We will rewrite the ORC paragraph accordingly.

> **Minor.** *"The connection to the PRH is not fully established ... should discuss recent work that questions PRH-style convergence [1,2]."*

**A8.** Agreed. We ran the suggested experiment: mutual-kNN alignment across all 66 model pairs on the shared 1000 centroids, Euclidean vs Poincaré — hyperbolic does **not** improve cross-model alignment (0.474 vs 0.427; 6/66 pairs improve). We will reposition the contribution as within-model geometry, soften "universal", and cite and discuss both suggested works (our within-model results require no cross-model alignment and are compatible with their critiques).

**Committed revisions.** 1. Matched-null calibration tables + absolute/scaling claims restated as excesses; 2. rewritten ORC paragraph + bridge analysis; 3. WordNet-alignment section; 4. full cosine/RT/normalized metric comparison + three-way protocol; 5. narrowed causal-LM claims with template ranges; 6. PRH discussion incl. both references; 7. removal of curvature-estimation implications.

We are happy to run further controls in the discussion phase.
