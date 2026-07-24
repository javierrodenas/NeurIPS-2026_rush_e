We thank the reviewer for an unusually constructive review. Rather than answering rhetorically, we ran **every control the reviewer requested** on the same frozen features, splits and estimator as the paper (code will be shared with the AC upon request). Some controls confirm the paper's mechanisms; two genuinely sharpen its claims, and we say so explicitly below. We believe the resulting paper is stronger and more precise, and we commit to concrete revisions (list at the end).

## 1. Spherical / Gaussian / whitened / dimension-matched baselines (W2, Q1–Q3)

We first calibrated the normalized estimator $\hat\delta=\delta_{\max}/\mathrm{diam}$ on reference geometries (same 500K-quadruple, 10-seed protocol):

| space | $\hat\delta$ |
|---|---|
| balanced binary tree (depth 9) | 0.000 |
| $\mathbb{H}^2$ ($K{=}-1$), region radius $R{=}2/4/8/16$ | 0.162 / 0.087 / 0.043 / 0.022 |
| uniform $S^{99}$ (chord / geodesic) | 0.143 / 0.179 |
| iid Gaussian, $n{=}1000$, $d{=}192/768/1536$ | 0.104 / 0.061 / 0.046 |

Two structural facts emerge. (i) Hyperbolic spaces have **bounded absolute** $\delta$ ($\to\ln 2$ for the four-point condition), so $\hat\delta\to 0$ as the region grows, while spheres are scale-invariant at high $\hat\delta$: low $\hat\delta$ *is* diagnostic of tree-like/hyperbolic-like rather than spherical geometry. (ii) The reviewer's concentration concern is **correct**: iid Gaussian clouds approach the equidistant (star-tree) limit as $d$ grows, so raw $\hat\delta$ levels are only interpretable against a dimension-matched null. We now report matched controls (ImageNet 1000-class centroids; per-model matched $n,d$):

| model | real | Gauss null | L2-chord | L2-geodesic | WordNet groups | random groups |
|---|---|---|---|---|---|---|
| ViT-L | .084 | .055 | .100 | .107 | .104 | .117 |
| DINOv2-L | .067 | .055 | .096 | .106 | .060 | .058 |
| DINOv2-G | .080 | .046 | .098 | .106 | .054 | .056 |
| CLIP-B | .122 | .074 | .121 | .123 | .126 | .174 |
| CLIP-L | .118 | .061 | .117 | .119 | .117 | .162 |

What survives, and what we will restate:

- **Spherical geometry is disfavored**: forcing the centroids onto the sphere (L2 + chord or geodesic) *raises* $\hat\delta$ for every one of the 12 models. The cross-model ordering is preserved under the cosine metric (DINOv2-G .098 lowest, ViT-T .147 highest), so the descriptive ranking is not an artifact of raw Euclidean scale.
- **Semantic-grouping contrast** (Q6): merging the 1000 classes into 100 WordNet-coherent groups yields lower $\hat\delta$ than 100 random groups in 8/12 models (CLIP-B .126 vs .174; ViT-L .104 vs .117); the largest DINOv2 models are at floor under both. Tree-likeness tracks semantic structure, not arbitrary binning.
- **Concession and fix**: trained centroids sit *above* the iid-Gaussian null, i.e., low absolute $\hat\delta$ alone cannot be read as "semantically hierarchical" (the null is a degenerate star). The most informative control is a **spectrum-matched Gaussian null** (identical covariance spectrum, so all second-order structure is equated): supervised ViTs sit clearly *below* it (excess −0.019/−0.015/−0.023/−0.031 for T/S/B/L — genuine higher-order tree structure, growing with scale), as do CLIP/SigLIP (−0.021/−0.020/−0.017), whereas DINOv2 sits essentially *at* its null (+0.011..−0.011): its very low raw $\hat\delta$ is largely carried by its covariance spectrum and extreme cluster compactness. This concords with the WordNet-alignment inversion in §3 below — two independent controls giving the same refined picture. Under PCA to a common $d{=}192$, DINOv2-L remains the most tree-like (.085, below the .104 null) but the monotone S$\to$G trend attenuates. In the revision, all absolute-level and cross-dimension statements (including the "45%" scaling headline) will be restated as **excess over matched nulls**, with these tables added. The within-architecture ablations (random weights, fine-tuning, layer-wise) are unaffected (same-$d$ contrasts).
- We agree we **do not estimate curvature** (sign or value), and never intended to claim so; we will systematically replace "hyperbolic/negative curvature" language by "tree-like inter-class metric structure" in the descriptive sections, reserving hyperbolicity for the (empirically tested) projection.

## 2. The ORC "internal tension" (W3)

The reviewer is right that our §3.2 sentence was misleading. New edge-type analysis (k=10 centroid kNN graph, edges split by WordNet-30 superclass): for supervised and contrastive models, within-superclass edges have mean ORC +0.34..+0.41 vs +0.26..+0.31 across, and the **negative-edge fraction is 2–4× higher on across (bridge) edges** (ViT-L: .021 vs .068; CLIP-L: .040 vs .084). So high mean ORC = tight leaf clusters; the sparse low/negative-κ edges are precisely the inter-cluster bridges. DINOv2's ~0% negative edges arise because its clusters are so tight that k=10 neighborhoods almost never cross groups. We will rewrite the ORC paragraph accordingly ("high mean ORC reflects leaf compactness; negative curvature concentrates on bridges when the graph spans them").

## 3. Label / prompt confounds (W4, W5, Q6)

- **Quantitative anti-circularity result**: Spearman correlation between centroid distances and WordNet tree distances (1000 classes): CLIP-L +0.59, SigLIP +0.58, CLIP-B +0.57, supervised ViTs +0.49..+0.53, DINOv1 +0.36, DINOv2 +0.18..+0.22 (label-shuffle control ±0.03; Gaussian ≈0). Crucially, the family with the *lowest* $\hat\delta$ (DINOv2, trained without labels or language) is the *least* WordNet-aligned, while language-supervised models are the most aligned. If the centroid construction injected the human taxonomy into $\hat\delta$, the opposite pattern would appear. The *amount* of tree-likeness and the *identity* of the tree are decoupled, and we will make this explicit — it directly bounds what the centroid methodology can and cannot inject.
- **Prompts**: across 6 templates (incl. class-name-only): text embedders are highly robust ($\hat\delta$ range ≤0.011: BGE .115–.126, E5 .114–.120; template "a photo of a" reproduces the paper: BGE .124 vs .123). Causal LMs are template-sensitive (GPT-2-M .091–.134), and under class-name-only the LM-vs-embedder ordering attenuates/reverses. **We will narrow the causal-LM claims accordingly** (report mean±range over templates; remove cross-family conclusions that do not survive name-only prompts). The paper's core vision results involve no prompts.

## 4. Radial rescaling vs hyperbolic metric; cosine baselines (W6, Q3–Q5)

- **Q4 answered directly (RT control)**: applying the *identical* radial tanh map and then using **Euclidean** distances on the transformed points changes nothing (|RT−R| ≤ 0.25pp everywhere), while H−RT ≈ H−R (SSL few-shot +0.89pp; DINOv2-G +1.20pp). The paper's H−R gain is attributable to the hyperbolic metric, not to nonlinear radial rescaling.
- **Q3 (cosine/spherical downstream)**: we added COS (raw and centered; equivalent ranking to spherical geodesic by monotonicity) to all 60 cells. Honest summary: on SSL backbones cosine is the strongest zero-cost metric (FS COS−R +1.44pp; H−COS −0.5pp); on supervised ViTs H≈COS; on contrastive VLMs **H beats COS** (FS +0.69pp; and after L2-normalizing first, Poincaré-on-normalized beats cosine by +0.9..+1.3pp, per-episode CI95 ≤ ±0.15pp — the hyperbolic metric *stacks on top of* normalization for CLIP). We will report the full metric comparison and reframe the practical protocol as a three-way choice, with $\hat\delta$/ORC predicting the H−R contrast (unchanged) and paradigm predicting H-vs-cosine. We believe "hyperbolic beats spherical specifically on contrastive VLMs" is a finding of independent interest.
- **Q5 (estimate curvature rather than fix −1)**: we swept the effective curvature scale two ways (ball-radius target t ∈ [0.25, 0.95]; curvature c ∈ [0.25, 4] on full ImageNet): the few-shot advantage is positive at every setting and peaks near t≈0.58–0.71; conclusions are insensitive to the −1 convention. An intrinsic curvature estimate is indeed a different problem, which we will state explicitly as out of scope.

## 5. PRH connection and missing references (minor)

We agree the framing overstated the link. Our results are a *within-model* characterization of the geometry of the (putatively convergent) representation; they neither require nor evidence cross-model convergence, and are compatible with the critiques in the two suggested works, which we will cite and discuss (both question cross-model/cross-modal alignment, not within-model structure). We also ran the reviewer's suggested experiment: mutual-kNN alignment of the 1000 shared class centroids across all 66 pairs of the 12 vision models, under Euclidean vs Poincaré kNN. Hyperbolic distances do **not** improve cross-model alignment (mean 0.474 vs 0.427; 6/66 pairs improve) — so we will moderate the PRH framing on quantitative grounds rather than by re-argument, and soften "universal" in title/abstract language as the AC suggests.

## Committed revisions

1. Matched-null calibration tables (Gaussian, spectrum-matched, sphere, dimension-matched) + restated absolute/scaling claims as excesses; 2. rewritten ORC paragraph + bridge analysis; 3. WordNet-alignment section (decoupling tree-likeness from taxonomy identity); 4. full cosine/RT/normalized metric comparison + three-way practical protocol; 5. narrowed causal-LM claims with template ranges; 6. PRH discussion incl. both suggested references; 7. removal of curvature-estimation implications.

We hope this addresses the substance of the concerns and are happy to run further controls in the discussion phase.
