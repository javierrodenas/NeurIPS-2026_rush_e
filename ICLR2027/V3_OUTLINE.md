# V3_OUTLINE — the claim list of the brief and where each claim landed in `main_iclr2027_v3.pdf`

Line numbers are the ICLR margin numbers of the compiled v3 PDF; every number in a claim is the fill read from its result file (`rebuttal/results/phaseE_v3_fills.json`). Table numbers in parentheses are the brief's; the file numbers them in citation order (Table 1 census, then Table 2 robustness, Table 3 calibration, ...).

## 1 Introduction (verbatim, with Figure 1)

- line 67: the premise and the three confounds
- line 77: the instrument and the three-part answer

## 2 Related Work (verbatim)

- line 107: network science, representation analyses, convergence program

## 3.1 Background: Gromov δ

- line 127: verbatim paragraph, Eq. 1 (pairings)
- line 139: Definition 1 (four-point defect)

## 3.2 Estimation and normalization

- line 143: verbatim paragraph, Eq. 2
- line 159: Definition 2 (reading of record)
- line 161: Proposition 1(a) (range bound)
- line 162: Proposition 1(b) (dimension confound)
- line 165: Remark: spectrum and statistic confounds
- line 168: calibration on reference geometries (Table 3 of the brief)

## 3.3 Spectrum-matched null, excess and genuineness

- line 172: Definition 3 (Haar null)
- line 178: why Haar and not Gaussian coefficients
- line 182: Definition 4 (excess), Eq. 3
- line 188: rank and left-tail p, Eq. 4, K = 200
- line 192: Definition 5 (genuine), Eq. 5
- line 196: clustered structure vs tree-like; Figure 2 cited

## 3.4 Depth test

- line 204: Definition 6 (hub null and excess B)
- line 211: Definition 7 (matched star)
- line 235: Eq. 6, certification at z ≤ −2
- line 235: the star caveat
- line 239: validity measured in §5.3

## 3.5 Comparing trees across models

- line 240: dendrograms, three measures, degeneracy criterion, calibrated local agreement

## 3.6 Projection and tasks

- line 254: the map, the two tasks, three readouts

## 4 Experimental Setup

- line 260: models and the compact panel
- line 286: datasets, cells, two levels
- line 293: extraction
- line 300: replicates, seeds, budgets, BH, frames, star seeds

## 5.1 The premise at the sample level

- line 85: not genuine in 14 of 24; at most 39% below the null, largest for DINOv2-G on CIFAR-100; genuine cells are the DINOv2 family (Table 4 of the brief)

## 5.2 Clustered structure at the class level

- line 344: 70/72 negative, 49/72 genuine (Table 1, Figure 3)
- line 350: 18/24 on ImageNet and CIFAR-100; 42–47 under joint sensitivity
- line 355: every family; ViT-T on DTD only; flat datasets; class-count control; two ResNets
- line 360: ImageNet excess 1–36 null spreads (brief: 3–8; file: see report)
- line 235: the star caveat (synthetic star reaches -0.115)
- line 367: neural collapse: ETF is the limit, hubs show it is not the whole story
- line 374: scale: DINOv2 deepens on CIFAR-10, weakly on CIFAR-100; no general trend

## 5.3 Depth above the labelled clusters

- line 396: certified 4/12 (ViT-S/B/L, DINOv2-L); the same four under the spectrum-matched-hub star (z -2.0 to -3.7) (Figure 4, Tables 5–6 of the brief)
- line 402: 0 of 60 false alarms with randomized hubs; the four never fire at zero strength (0 of 20)
- line 406: clean implanted tree detected in almost none (power 0.05); ratios 1.3–3.9 outside the synthetic sweep
- line 410: eight 'not detected'; certified set shifts with the frame; CIFAR-100 outside the validated regime; isotropic star withdrawn; shrunk variant diagnostic only
- line 417: ViT-B augreg certified (z -3.03) and WordNet-aligned (+0.52)
- line 422: DeiT-B not certified, unaligned (+0.08), recovers superclasses (0.56); label sets are not the explanation; alignment recipe-dependent
- line 427: MERU: same clustering, no detected depth, near-flat regime (radius·√c 0.258–0.279; Lorentz/Euclidean 0.997)

## 5.4 Whose tree

- line 454: naive map isolates DINOv2 (0.03 vs 0.64): chaining artifact; DINO-B not in the island (Figure 5)
- line 461: selected configuration 0.38 vs 0.48; other admissible 0.13–0.17 vs 0.50–0.55
- line 468: cophenetic 0.48 vs 0.78; triplets 0.77 vs 0.76 under cosine-average only
- line 99: angular tree: 0.91/0.85 cosine vs 0.71/0.63 Euclidean for DINOv2-L/G
- line 478: cosine census agrees in 65/72
- line 485: WordNet alignment: contrastive +0.57–+0.59, supervised +0.49–+0.53
- line 487: DINOv2 +0.18–+0.22; supervision deepens on average, recipe-dependent; form and content anti-correlate
- line 494: not WordNet circularity: DBpedia three embedders genuine (-0.021 to -0.017), sibling agreement 0.89
- line 499: HierarCaps 0.83–0.97
- line 502: text census: 7/15 genuine; GPT-2 S genuine (p .020), M not, L/XL under every reading
- line 506: Pythia at every scale; OLMo-1B; embedders absent on class names, present on DBpedia; template and extraction move the fragile values only

## 5.5 What is shared is local and ordinal

- line 510: calibrated mKNN 0.464 and CKA 0.577 over 66 pairs, all significant (Table 10 of the brief)
- line 514: Poincaré improves neither reading (the values are in Table 13); shared neighborhoods, model-specific metric

## 6 Implications for hyperbolic representation learning

- line 518: Khrulkov rule (Eq. 7) assigns c from 0.48 (d = 192) to 2.5 (d = 1536) to Gaussian clouds
- line 527: calibrated reading predicts NC gain on four datasets and FS on three; depth predicts nothing (Table 11 of the brief)
- line 531: cosine collects the self-supervised structure; Poincaré +0.9 to +1.3 pp only for contrastive VLMs; nothing on flat labels; sample-level tasks never benefit
- line 100: what to measure before imposing curvature

## 7 Conclusion and Limitations

- line 27: the thesis sentence verbatim
- line 548: limitations (i)–(viii)

## Statements

- line 559: Reproducibility (anonymized repo, tool/)
- line 569: Ethics
- line 573: AI use (four items)
