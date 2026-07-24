We thank the reviewer for the careful reading and for stating openly which parts were unclear — the questions raised are legitimate, and several of them uncovered real presentation defects that we commit to fixing. Below we answer every technical question directly (with new calibration measurements where useful) and then list the concrete writing revisions.

## Direct answers to the questions

**Q1: "What is the unit hyperbolic plane?"** Yes: $\mathbb{H}^2$ with constant curvature $K=-1$ (equivalently, the Poincaré disk model with $c=-1$). We will define it explicitly.

**Q1/Q2: the constants $\log(1+\sqrt2)$ and $0.127$, and why the sphere value can be *smaller* than the hyperbolic one.** This apparent paradox is a units problem in our text, and the reviewer is right to flag it: the two numbers are not on the same scale. $\ln(1+\sqrt2)=\operatorname{arccosh}(\sqrt2)\approx 0.881$ (natural log) is the **absolute** thin-triangles constant of $\mathbb{H}^2$ (standard result, e.g. Bridson–Haefliger, Ch. III.H), while $0.127$ for $S^{99}$ is a **normalized** value $\hat\delta=\delta_{\max}/\mathrm{diam}$ from our estimator. To make this concrete we calibrated our exact estimator (500K quadruples, multi-seed) on reference geometries:

| space | absolute $\delta$ | $\hat\delta=\delta/\mathrm{diam}$ |
|---|---|---|
| balanced binary tree (depth 9) | 0.000 | 0.000 |
| $\mathbb{H}^2$ ($K{=}-1$), region of radius $R=2/4/8/16$ | 0.65/0.69/0.69/0.69 | 0.162/0.087/0.043/0.022 |
| uniform $S^{99}$ (chord / geodesic) | 0.25/0.37 | 0.143/0.179 |
| iid Gaussian cloud, $n{=}1000$, $d{=}192/768/1536$ | — | 0.104/0.061/0.046 |

The key structural fact: hyperbolic spaces have **bounded absolute** $\delta$ (our four-point measurements converge to $\approx\ln 2\approx0.693$), so as the sampled region grows, $\hat\delta\to0$ — hyperbolic geometry becomes tree-like "in the large". Spheres instead are scale-invariant: $\hat\delta$ stays high no matter the radius. So on the normalized scale that the whole paper uses, the ordering is exactly the intuitive one: trees $0 <$ hyperbolic regions $<$ spheres, and there is no contradiction. The $S^{99}$ number is our own numerical calibration (not from Gromov 1987) — the revision will state its provenance, protocol, and the absolute-vs-normalized distinction explicitly, and will not mix the two scales in one sentence again.

**Q3: "What is a low enough $\delta$?"** Following this question (and a related one by Reviewer RJje) we now give $\hat\delta$ an operational zero point rather than an absolute threshold: (i) *calibration*, as in the table above; (ii) *matched nulls* — because random high-dimensional clouds concentrate toward equidistance (a degenerate star tree), the informative quantity is $\hat\delta$ relative to a null with the same $n$, $d$ and covariance spectrum (we added these nulls for all 12 vision models); and (iii) *behavioral validation* — the paper's empirical rule ($\hat\delta\lesssim0.10$ on centroids together with a semantically deep label set predicts a positive Poincaré-projection gain) is a correlation validated on 60 model×dataset cells, not a geometric axiom. The revision reframes §4's "low $\delta$" statements accordingly.

**"The authors normalize $\delta$, then use normalized figures to say low $\delta$ = tree-like — is that step correct?"** Normalization by the diameter is what makes the comparison scale-free (an isometric rescaling of the same geometry should not change how "tree-like" it is; unnormalized $\delta$ would). Trees have $\delta=0$ on both scales; the calibration above shows that on the normalized scale hyperbolic-like structure sits low and spherical structure sits high. What normalization *cannot* fix by itself is the dimension effect of point clouds, which is why we added the matched nulls of point (ii) above.

**[L214] "collapse toward zero" on MNIST/FMNIST.** The reviewer is right and our wording was imprecise: on MNIST the NC advantages are mostly *negative* (Euclidean is better), while few-shot advantages hover near zero. The correct statement — which the revision will make — is that on flat-hierarchy datasets the hyperbolic projection gives no benefit and can hurt, which is the prediction of the mechanism (no tree structure to exploit). We will replace "collapse toward zero" with the signed description and point to the per-dataset appendix figures.

**[L170] the projection formula.** The formula is the exponential map of the Poincaré ball at the origin applied to the centered, rescaled feature: $\phi(x)=\tanh(\tfrac12\|s(x-\mu)\|)\cdot\frac{s(x-\mu)}{\|s(x-\mu)\|}$, i.e., direction preserved, norm mapped through $\tanh$ so that the 95th-percentile feature lands at radius $1/\sqrt2$. We will rewrite this subsection with the geometric reading of each factor and a small diagram.

## Writing revisions we commit to

1. **A main-text results table.** We agree this is the paper's biggest presentation defect: the key numbers now live only in an appendix table that the main text never references. The revision adds a compact main-text table (per-family means for $\hat\delta$, ORC, NC/FS advantages, plus the headline cells), splits the appendix monolith into per-section tables, and moves the inline parenthetical number lists into those tables.
2. **Introduction**: merge the four fragments into two paragraphs; add references for the capacity/hierarchy claims (or mark them as hypotheses we test); fix the missing verb at L59.
3. **Related work**: restructure §2.3 as "Background" (it is indeed preliminaries, not related work), merge the PRH paragraph into §2.2, avoid symbols like $\mathbb{H}$ in prose.
4. **Methodology**: add citations in §3.1 (feature-extraction conventions); add the motivation for the choice of the two descriptors in §3.2 ($\hat\delta$: global, coordinate-free, tied to embeddability theorems; ORC: local, edge-level, complementary — plus the new calibration table); replace "argmin" phrasing and give the few-shot protocol as an explicit algorithm box.
5. **Experiments**: rewrite §4.1 to state its conclusion first and reference the (new) main-text table; enlarge Figure 2; fix the broken figure reference at L242; move "Practical protocol" next to the setup as suggested; expand A1–A4 from single-claim paragraphs into claim + evidence + interpretation (the new bridge/null analyses give each one a quantitative backing).

We hope these answers resolve the specific confusions (particularly the absolute-vs-normalized $\delta$ scales, which we regard as the root of several of the reviewer's concerns) and demonstrate that the underlying methodology is sound. We are happy to clarify anything else during the discussion phase.
