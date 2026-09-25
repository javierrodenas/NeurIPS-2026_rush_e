Thank you for the encouraging assessment and for four questions that are exactly the right stress-tests. We ran new experiments for each (same frozen features, splits and estimators as the paper; code available to the AC on request) and report everything we found, including the negatives, so that the significance judgment can rest on the full picture.

> **Q1.** *"Can you validate the post-hoc projection on at least one additional downstream task, such as retrieval, clustering, or potentially a text-based task? If $\delta$ and ORC continue to predict when hyperbolic distances help beyond NC and FS vision classification, it would definitely make me reconsider the significance of the study."*

**A1. Five new directions; three are negative (reported as-is), two sharpen the contribution:**

- **Sample-level tasks (negative).** kNN and sample-to-sample retrieval across the full 60-cell grid: the projection *hurts* both (kNN −2.6pp mean; retrieval positive in 0/60 cells). Mechanism: the hierarchy the diagnostics detect lives at the prototype level, while the radial map distorts local sample neighborhoods. The revision restricts the domain of validity to prototype-based metric tasks.
- **Hierarchy recovery (negative for H).** Cutting CIFAR-100 centroid dendrograms at 20 clusters vs the true superclasses (four linkage methods checked): Euclidean Ward recovers them well (ARI up to 0.61); Poincaré-distance linkage does not improve it.
- **Metric-controlled study (positive).** A matched radial-rescaling control changes nothing (|RT−R| ≤ 0.25pp), so the H−R gain is the metric itself. Against cosine, the strongest zero-cost alternative, the hyperbolic advantage concentrates on contrastive VLMs: CLIP few-shot +0.9..+1.3pp over cosine even after L2 normalization (ImageNet +0.1..+0.2pp), while DINOv2 prefers cosine. $\delta$/ORC still predict H−R as in the paper; the practical protocol becomes three-way.
- **Cross-model alignment (negative; see A4).**
- **Text-based task (new).** The exact NC/FS protocol transplanted to text: DBpedia Classes (a real 3-level hierarchy, 219 leaf classes) with the BGE/E5/GTE embedders. Geometry: genuine tree structure beyond spectrum-matched nulls, and strong sibling alignment with the true level-2 hierarchy (triplet agreement 0.88 vs 0.5 chance). Tool: these embedders sit *above* the paper's $\hat\delta\lesssim0.10$ threshold ($\hat\delta$ = 0.12–0.13, ORC ≈ +0.37), and the projection behaves exactly as both diagnostics predict: no meaningful gain (NC slightly negative; FS +0.04..+0.08pp at a ~99% ceiling). The diagnostic transfers to a new modality, correctly predicting *when hyperbolic distances help little*, which the paper frames as half of its claim.

> **Q2.** *"How robust are the text-model geometry results to prompt variation? ... an additional, simple validation using variations of the prompt 'a photo of a {object}' would provide more reassurance."*

**A2.** Six templates over the 1000 ImageNet class names ("a photo of a {}", "{}" name-only, "an image of a {}", "a close-up photo of a {}", "this is a photo of a {}", "a {} in the wild"):

| model | min–max $\hat\delta$ across templates | range |
|---|---|---|
| BGE-base | .115–.126 | .011 |
| E5-base | .114–.120 | .006 |
| GPT-2 | .080–.145 | .065 |
| GPT-2-M | .091–.134 | .043 |

**Embedders are highly robust** (the paper's template reproduces our published values, e.g. BGE .124 vs .123). **Causal LMs are template-sensitive**, and under name-only prompts the LM-vs-embedder ordering attenuates or reverses: we will report causal-LM geometry as mean±range and narrow those claims. The vision results, the paper's core, involve no prompts. We thank the reviewer for prompting exactly the check that surfaced this.

> **Q3.** *"Given the sometimes very small improvements on downstream tasks (e.g., +0.80 pp on DTD NC), it would be useful to include an additional statistical measure, such as standard deviation over 3 or 5 seeds, to show that even small improvements are consistent."*

**A3.**
- **Few-shot:** per-episode *paired* CI95 over 1000 episodes, ±0.03–0.20pp across the 60 cells. Headline gains exceed their CI by large factors (e.g. DINOv2-G CIFAR-10 +1.5pp vs ±0.11).
- **NC:** deterministic given the frozen features (no seed variance); McNemar tests now run: the +2.1pp DINOv2-L/CIFAR-100 gain corresponds to 242 vs 37 discordant test decisions on n=10,000 (p ≈ 4×10⁻³⁸); full per-cell table in the revision.
- The DTD NC +0.8pp cell the reviewer cites is among the smaller effects and will be presented with its uncertainty rather than as a headline.

> **Q4.** *"I would like the authors to clarify the main reasons why, and in what sense, this work is supposedly tightly connected to the Platonic Representation Hypothesis."*

**A4.** We agree the connection as framed was too strong, and we tested the natural bridge: mutual-kNN alignment between all 66 pairs of the 12 vision models on the shared 1000 centroids, Euclidean vs Poincaré. **Hyperbolic does not increase cross-model alignment** (0.474 vs 0.427). This was the paper's stated position all along, §2.4: *"a within-model property that stands on its own"*, but that framing sits in Related Work while the title and abstract lean on PRH language; the mismatch is ours. The revision promotes §2.4 into the title/abstract, softens "universal", and discusses the critical PRH literature raised by Reviewer RJje. The PRH-adjacent statement that survives is **convergence in form, not in content**: 69/72 model×dataset combinations show genuinely tree-like structure (beyond matched nulls), yet each model builds its *own* tree, which is exactly why hyperbolic distances do not increase cross-model agreement.

We hope the text-modality test, the prompt-robustness table, the paired CIs, and the metric-controlled study constitute the "one additional set of results" the reviewer asked for, with the negatives included so the significance judgment can be made on the full picture. We are glad to run further checks during the discussion phase, and we thank the reviewer again for the constructive engagement.
