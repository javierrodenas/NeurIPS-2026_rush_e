We thank the reviewer for the careful reading and for stating openly which parts were unclear — several questions uncovered real presentation defects that we commit to fixing. Below we answer every technical question directly (with new calibration measurements where useful), then list the concrete writing revisions.

> **Q1.** *"[L145] What is the unit hyperbolic plane? Do the authors refer to the unit Poincaré ball (i.e., curvature=-1)?"*

**A1.** Yes: $\mathbb{H}^2$ with constant curvature $K=-1$ (equivalently the Poincaré disk with $c=-1$). We will define it explicitly.

> **Q2.** *"Why do the authors claim that the estimator is calibrated, looking at $S^{99}$? Where did they take this figure (please cite or explain)?"* — and the related remark: *"the 'unit hyperbolic plane' has $\delta=\log(1+\sqrt2)$ ... while the '99-dimensional sphere' has $\delta=0.127$. This is counterintuitive."*

**A2.** The $S^{99}$ value is our own numerical calibration (not from Gromov 1987) — the revision will state its provenance and protocol. The apparent paradox is a **units problem in our text**, and the reviewer is right to flag it: $\ln(1+\sqrt2)=\operatorname{arccosh}(\sqrt2)\approx0.881$ (natural log) is the **absolute** thin-triangles constant of $\mathbb{H}^2$ (e.g., Bridson–Haefliger III.H), while $0.127$ is a **normalized** value $\hat\delta=\delta_{\max}/\mathrm{diam}$. Calibrating our exact estimator (500K quadruples, multi-seed) on reference geometries:

| space | absolute $\delta$ | $\hat\delta=\delta/\mathrm{diam}$ |
|---|---|---|
| balanced binary tree (depth 9) | 0.000 | 0.000 |
| $\mathbb{H}^2$ ($K{=}-1$), region radius $R{=}2/4/8/16$ | 0.65/0.69/0.69/0.69 | 0.162/0.087/0.043/0.022 |
| uniform $S^{99}$ (chord / geodesic) | 0.25/0.37 | 0.143/0.179 |
| iid Gaussian, $n{=}1000$, $d{=}192/768/1536$ | — | 0.104/0.061/0.046 |

Hyperbolic spaces have **bounded absolute** $\delta$ (our four-point measurements converge to $\approx\ln2$), so $\hat\delta\to0$ as the sampled region grows — hyperbolic geometry becomes tree-like "in the large". Spheres are scale-invariant: $\hat\delta$ stays high at any radius. On the normalized scale the whole paper uses, the ordering is the intuitive one: trees $0<$ hyperbolic regions $<$ spheres. The revision will make the absolute-vs-normalized distinction explicit and never mix the two scales in one sentence.

> **Q3.** *"What would be a small enough value for $\delta$ to say that something is well-suited for Hyperbolic representations (and why)?"*

**A3.** Following this question (and Reviewer RJje's related one), $\hat\delta$ now gets an operational zero point rather than an absolute threshold: (i) the calibration above; (ii) **matched nulls** — random high-dimensional clouds concentrate toward equidistance (a degenerate star tree), so the informative quantity is $\hat\delta$ relative to a null with the same $n$, $d$ and covariance spectrum (added for all 12 vision models and all 6 datasets); (iii) behavioral validation — the paper's rule ($\hat\delta\lesssim0.10$ plus a semantically deep label set predicts a positive projection gain) is a correlation validated on 60 model×dataset cells, not a geometric axiom. §4's "low $\delta$" statements will be reframed accordingly.

> **W.** *"The authors normalize the $\delta$, but then they use the normalized figures to say that a low $\delta$ is synonymous with tree-like. I am unsure about the correctness of this step."*

**A4.** Normalization by the diameter is what makes the comparison scale-free: an isometric rescaling of the same geometry should not change how tree-like it is (unnormalized $\delta$ would change). Trees have $\delta=0$ on both scales, and the calibration above shows hyperbolic-like structure sits low and spherical structure sits high on the normalized scale. What normalization *cannot* fix is the dimension effect of point clouds — hence the matched nulls of A3(ii).

> **W.** *"[4.2, L214] Looking at Table 1 (Supp), they do not seem to 'collapse toward zero' as they are mostly negative (meaning there is an advantage in using Euclidean representations)."*

**A5.** The reviewer is right and our wording was imprecise: on MNIST the NC advantages are mostly *negative* (Euclidean is better), while few-shot advantages hover near zero. The correct statement — which the revision will make — is that on flat-hierarchy datasets the projection gives no benefit and can hurt, which is exactly the prediction of the mechanism (no tree structure to exploit).

> **W.** *"[3.3] The formula in L170 is unclear in both its meaning and calculation."*

**A6.** It is the exponential map of the Poincaré ball at the origin applied to the centered, rescaled feature: $\phi(x)=\tanh(\tfrac12\|s(x-\mu)\|)\cdot\frac{s(x-\mu)}{\|s(x-\mu)\|}$ — direction preserved, norm mapped through $\tanh$ so the 95th-percentile feature lands at radius $1/\sqrt2$. The revision rewrites this subsection with the geometric reading of each factor and a small diagram.

> **W.** *"The lack of a table in the main paper that reports the results discussed by the authors penalizes the proposed score. ... the authors placed many numbers and comparisons within the text, in parentheses, making the reading harder."*

**A7 — committed writing revisions.**
1. **A main-text results table** — we agree this is the paper's biggest presentation defect. The revision adds a compact main-text table (per-family means for $\hat\delta$, ORC, NC/FS advantages, plus headline cells), splits the appendix monolith into per-section tables, and moves inline parenthetical numbers into them.
2. **Introduction**: merge the four fragments into two paragraphs; add references for the capacity/hierarchy claims (or mark them as hypotheses under test); fix the missing verb at L59.
3. **Related work**: restructure §2.3 as "Background" (it is preliminaries, as the reviewer notes), merge the PRH paragraph into §2.2, avoid symbols like $\mathbb{H}$ in prose.
4. **Methodology**: add citations in §3.1; motivate the choice of the two descriptors in §3.2 ($\hat\delta$: global, coordinate-free, tied to embeddability theorems; ORC: local, edge-level, complementary — plus the calibration table above); replace "argmin" phrasing; give the few-shot protocol as an explicit algorithm box. The §4.2 ORC ranking (L224–228) will also get its missing interpretation: a new edge-type analysis (reported to Reviewer RJje) shows high mean ORC reflects tight within-superclass clusters while negative curvature concentrates on the between-superclass bridge edges — this is the conclusion the ranking supports.
5. **Experiments**: rewrite §4.1 to state its conclusion first and reference the new main-text table; enlarge Figure 2; fix the broken figure reference at L242; move "Practical protocol" next to the setup as suggested; expand A1–A4 into claim + evidence + interpretation.

We hope these answers resolve the specific confusions — particularly the absolute-vs-normalized $\delta$ scales, which we regard as the root of several concerns — and show the underlying methodology is sound. We are happy to clarify anything else during the discussion phase.
