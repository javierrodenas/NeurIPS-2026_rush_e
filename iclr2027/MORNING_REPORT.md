# Informe de la noche (20→21 ago) — todos los datos para la reescritura

Todo verificado y en `rebuttal/results/` (+ `night/`). Cada número citable tiene CSV.

## 1. exp18 — nulls para los modelos de texto (15/16; el hallazgo de la noche)

| Familia | Modelo (d) | δ̂ crudo | null | **exceso** | veredicto |
|---|---|---|---|---|---|
| GPT-2 | S (768) | 0.131 | 0.083 | **+0.048** | por ENCIMA del null |
| | M (1024) | 0.090 | 0.103 | −0.013 | marginal |
| | L (1280) | 0.095 | 0.129 | **−0.034** | genuino |
| | XL (1600) | 0.083 | 0.131 | **−0.048** | genuino |
| Pythia | 410M/1B/2.8B | .098/.096/.098 | .136/.131/.139 | **−0.037/−0.035/−0.041** | genuino, estable |
| OLMo | 1B (2048) | 0.100 | 0.101 | −0.001 | **at-null** |
| | 7B | — | — | — | no medido (disco lleno; ver §7) |
| Embedders | BGE b/l | .121/.120 | .117/.120 | +0.005/+0.001 | **at-null** |
| | GTE b/l | .124/.120 | .113/.116 | +0.011/+0.004 | **at-null** |
| | E5 b/l | .115/.114 | .113/.113 | +0.003/+0.001 | **at-null** |
| | GTE-Qwen2 1.5B | 0.107 | 0.126 | **−0.019** | genuino |

**Lecturas (por orden de importancia):**
1. **El "objective, not modality" impreso NO sobrevive entero al instrumento.** El paper decía "text embedders cluster at δ≈0.110–0.124, identical to vision contrastive models". Los δ̂ crudos sí son idénticos — pero CLIP/SigLIP tienen exceso genuino (−0.016..−0.030) y los embedders clásicos están AT NULL (+0.001..+0.011): **mismos valores crudos, veredictos opuestos**. Es el argumento definitivo a favor del instrumento (y hay que reescribir esa frase del paper).
2. **GPT-2: la forma genuina emerge monótonamente con la escala** (+0.048 → −0.048). Pythia la tiene a todas las escalas. OLMo-1B no la tiene. → dentro de texto, manda la **receta** de entrenamiento, y en GPT-2, la escala. Historia de escalado en la base correcta (exceso), que W2 exigía.
3. **El contraste con DBpedia** (exp14): los MISMOS embedders BGE/GTE/E5 que aquí están at-null con prompts de nombres de clase mostraron exceso genuino (−0.020..−0.028) con texto real de DBpedia. → la forma es una propiedad del **par (modelo, conjunto de conceptos)**, no del modelo a secas — igual que DINOv2 (at-null en ImageNet, el más fuerte en transfer). Todo converge.
4. gte_qwen2 (el único embedder con backbone LLM) es genuino — coherente con el patrón LLM.
5. ⚠️ Nota de procedencia: OLMo-1B crudo nos da 0.100 vs 0.083 impreso (checkpoint -hf vs original). Documentado; el veredicto at-null es sobre nuestra medición interna coherente.

## 2. exp20 — z-table de las 72 celdas de visión (W6 respondido)

Formulación de dos niveles para el paper:
- **69/72 con exceso negativo** (point estimates; las 3 positivas son DINOv2-S/B/G@ImageNet, todas dentro de +1.2σ = at-null, no por encima).
- **Significancia**: 51/72 con z<−2; en datasets jerárquicos (C≥47): **41/48 con z<−2, 34/48 con z<−3**; Holm (unilateral, α=.05): 36/72. Los C=10 (MNIST/FMNIST) son 24/24 sign-negativos pero solo 10/24 con z<−2: **potencia, no ausencia** (210 cuádruplas; diagnóstico de exp19).
- Marginales notables en jerárquicos: dinov1_b/IN (z=−0.6), dinov2_l/IN (−1.1), siglip/IN (−1.7), i21k_b/C100 (−1.9). El texto debe citar dinov2_l/IN como "sign-negative but marginal", no como sólido.

## 3. exp19 — C vs jerarquía (W3 respondido; 220 filas)
- A C fijo, subsets aleatorios (atraviesan la jerarquía completa) son MÁS tree-like y ganan MÁS que los WordNet-coherentes (hermanos = jerarquía efectiva plana): DINOv2-L@C=50: NC +1.58pp vs +0.53; δ̂ .049 vs .087. **La variable es la profundidad abarcada, no C.** (Redactar con cuidado: "coherente" suena jerárquico pero es lo plano.)
- δ̂ crudo SUBE con C → el δ̂ alto de MNIST (C=10) no es artefacto de C; su planitud es real y subestimada.
- DINOv2-G llega a exceso ≈0 en C=500–1000 (+0.003 vs +0.0037 de exp11): reproducción independiente de la excepción.
- C=10 con enumeración exacta (degeneración del estimador tratada).

## 4. Barrido de t (W7; `night/t_sweep.csv`)
FS positivo en los 8 valores de t (100% de celdas para t≥0.35); la ganancia FS es MAYOR a t bajos (t=0.25: +0.88 medio; t=1/√2: +0.76) → **t=1/√2 es conservador, no cherry-picked**. NC cruza cero cerca de 1/√2. Frase para el paper: "conclusions are insensitive to t; the headline t was fixed a priori by the p95→1/√2 convention".

## 5. ORC en puentes (W5; `night/orc_bridges.csv`)
fneg_across > fneg_within en todo el panel (ej. i21k_t/IN: 9.4% vs 5.2%) y orc_within > orc_across. El estadístico principal del paper pasa a ser (fneg_across, orc_across); la media global se reporta como compacidad de clúster. Excepción DINOv2-S/IN (coherente con at-null).

## 6. Triplete arch-matched (W8; `night/arch_matched_triplet.csv`)
ViT-B con tres objetivos (i21k_b / dinov1_b / clip_b; d=768/768/512): los tres con forma genuina en casi todos los datasets; el ordering fuerte por objetivo emerge sobre todo CON LA ESCALA (dentro de familia), no a tamaño fijo. Matiza W8 honestamente: "at matched architecture and dimension, all three objectives yield genuine form; the large family separations emerge with scale".

## 7. ICs bootstrap (W6; `night/correlation_cis.csv`)
- NC pooled r=−0.45, cluster-bootstrap sobre datasets **[−0.60, −0.36]** → robusto.
- FS pooled r=−0.18, **[−0.50, +0.12]** → NO robusto; se presenta solo within-dataset (C100 [−.95,−.15], C10 [−.93,−.23], DTD [−.94,−.32] excluyen 0; ImageNet no).
- NC within: IN [−.97,−.65] y C100 [−.97,−.78] sólidos; C10 y DTD anchos (decir "consistent in direction").
- OLMo-7B: excluido por disco (partición única 855GB al 100% durante la descarga de 27GB; caches de modelos medidos ya liberados). Si se quiere, se corre en cuanto haya 30GB libres.

## 8. Veredictos de hipótesis
- **H1 (forma):** confirmada y REFINADA. En visión: casi universal (69/72 signo; 41/48 jerárquicos con z<−2). En texto: depende de la receta y la escala (GPT-2 emergente, Pythia sí, OLMo-1B no, embedders dependientes del corpus de sondeo). Reformulación: *"genuine tree form is widespread but not universal, it is a property of the (model, concept set) pair, and raw δ̂ cannot tell you where it is — the calibrated instrument can."*
- **H2 (contenido):** intacta (decoupling WordNet, cross-model 0.474/0.427).
- **H3 (regla débil):** confirmada con su estadística honesta (NC pooled robusto; FS within-dataset; t-sweep insensible; RT/coseno como estaban).

## 9. Propuesta de abstract (borrador para discutir, NO insertado)
> Foundation models are often claimed to converge toward shared representations; recent calibrated analyses show that much of that convergence is an artifact of uncalibrated similarity measures. We ask the same question of geometry: what structure of a model's inter-class representation survives calibrated measurement, and is it shared? We measure tree-likeness (Gromov δ-hyperbolicity, Ollivier-Ricci curvature) for 31 foundation models across vision and text, calibrating every reading against reference geometries and nulls matched in dimension and covariance spectrum, since raw δ is confounded by dimension, class count, and spectrum. Three findings. (1) Genuine tree-like *form* is widespread but not universal: 69/72 vision model×dataset combinations sit below their matched null (41/48 of the ≥47-class cells beyond two null s.d.), while in text it depends on the training recipe and scale, emerging monotonically with size in GPT-2 and absent in classic sentence embedders on the same probe, whose raw δ equals that of vision-contrastive models yet reflects only their spectrum. (2) The *content* is model-specific: the most tree-like family is the least aligned with WordNet, and hyperbolic distances do not increase cross-model agreement: models converge in the form of their class geometry, not in its content. (3) The calibrated descriptors support a modest, training-free decision rule: they predict when leaving Euclidean distances pays on prototype tasks, and the training objective selects which zero-cost metric collects the gain. Ablations (random weights, non-hierarchical fine-tuning, layer-wise emergence, label shuffles) show the form is learned, and a class-subsampling study shows the operative variable is the depth of hierarchy spanned, not the number of classes.

## 10. Cambios obligados en main_iclr2027.tex (cuando Javi valide)
1. La frase "text embedders... identical to vision contrastive models, confirming objective-not-modality" → reescribir con el veredicto de nulls (§1.1 de este informe).
2. "Causal LMs achieve comparably low δ" → historia por familias con excesos (GPT-2 escala, Pythia, OLMo).
3. El 69/72 → formulación de dos niveles (con z y Holm), y dinov2_l/IN citado como marginal.
4. ORC: estadístico principal = puentes.
5. §4.1/4.3: correlaciones con sus ICs; FS solo within-dataset.
6. Nueva subsección C-sweep (W3) y frase de t-sweep (W7).
7. Panel: 31 modelos medidos con nulls (OLMo-7B fuera con nota), contar y listar EXACTO.
