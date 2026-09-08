# CHANGELOG — pasada de versión final (2026-09-03)

Brief: "Revision brief — Is There a Platonic Tree? — final-version pass". Todo número del paper sale de
un fichero de `rebuttal/results/`; ningún número se ha teclado a mano salvo las filas árbol/H²/esfera de
la tabla de calibración (sin CSV; ver `TODO_author.md`). Los seis protocolos vigentes de `CODE_MAP.md`
no se han tocado.

## 0. Desviación importante respecto al brief (y por qué)

El brief asumía que las cachés por imagen de ImageNet existían en esta máquina ("all CPU, all cheap").
No existían: `*_imagenet_train.npz` y `*_imagenet_fulltrain.npz` se extrajeron en la máquina remota
(4090) y nunca se copiaron. El ImageNet crudo sí está en `/media/HDD_4TB_1/javi/ILSVRC2012_img_train`,
y ambos splits son funciones deterministas del listado ordenado de ficheros, así que:

- `rebuttal/scripts/extract_imagenet_cache_local.py` regenera `{m}_imagenet_train.npz` (100 img/clase,
  `files[:100]` por clase) con el pipeline original **idéntico** (mismos `MODEL_DEFS`, transforms de
  timm/open_clip por modelo, mismas reglas fp16, mismo orden). GPU: 2× RTX 2080 Ti, ~3.5 h de pared
  (DINOv2-G a 518 px en dos shards, uno por GPU, fusionados después).
- **Evidencia de fidelidad**: la δ_norm cruda de ImageNet de los 12 modelos coincide con la de
  `exp20_null_ztable.csv` (caché original) a 4 decimales en los 12 casos
  (p. ej. ViT-T 0.1227/0.1227, DINOv2-S 0.1185/0.1185, DINOv2-G 0.0808/0.0808). Check permanente en
  `sweep_freeze.py` ("R1 fidelity").
- Para **R2** no se regeneró el submuestreo de exp5 (100/clase con `RandomState(0)` sobre el fulltrain,
  otras 100k imágenes por modelo): `exp21b` usa los centroides de la caché del censo, que es el
  protocolo que §3 declara ("the mean of each class's cached training embeddings, 100 on ImageNet") y
  trata a los 12 modelos igual. Los raws cambian ligeramente respecto a exp21 (ver §5 abajo).
  **Pendiente de veto del autor** (`TODO_author.md`).
- El centroid store (`results/centroids/imagenet_train/*.npy`) difiere de la caché en los **12** modelos
  (0.001–0.004 en DINOv2/CLIP/SigLIP, 0.017–0.025 en ViT supervisados), no solo en los marcados con †:
  por eso R1 recomputó las 12 celdas de ImageNet y no solo las 4 supervisadas.

## 1. Entorno (§1 del brief)

- `gen_appendix.py`, `gen_appendix2.py`, `sweep_freeze.py`: rutas `/home/javi/...` sustituidas por
  `PLATONIC_RESULTS` (default: `<repo>/rebuttal/results`). Scripts nuevos usan `PLATONIC_ROOT` /
  `PLATONIC_RESULTS`. `HF_HOME`/`TORCH_HOME` → `/media/HDD_4TB_2/javi/hf_cache`.
- Shim `numpy._core → numpy.core` en `gen_appendix2.py`, `make_treemap_fig.py`, `make_fig_overview.py`,
  `sweep_freeze.py`: `exp23_treemap_controls.npz` fue picklado con numpy ≥ 2 y este entorno tiene 1.x.
- Compilación: tectonic en un entorno conda de scratch (no hay TeX local); `tcolorbox` añadido al preámbulo.

## 2. Reruns

### R1 — `expR39b_census20_cache.py` → `expR39b_census20_cache.csv` (72 celdas)
Mismo estimador, misma null y mismas semillas que `expR39` (real 10 semillas; 20 réplicas espectrales,
semillas 300+rep, 5 semillas cada una). Las 12 celdas de ImageNet desde la caché regenerada; las 60 filas
de transfer copiadas **verbatim** de `expR39_census20.csv` (ya eran cache-based) — dicho en el log del
script. Ninguna fila usa el store (`store_centroids = 0` en las 72).

Recuentos recalculados (no copiados del brief): **69/72** sign-negativas · **57/72** por debajo de las 20
réplicas · **43/48** en los cuatro datasets jerárquicos · planos **24/24** en signo, **14/24** bajo todas ·
excepciones sign-positivas: DINOv2-S/B/G en ImageNet, rango 0/20 y z ≤ +1.1 · resumen z secundario para
la leyenda de B20: |z| ≥ 2 en 50/72, z ≤ −3 en 31/48 jerárquicas.
(El brief esperaba 69/72, 56/72 y 42/48 a partir de las filas de transfer de expR39; con las 12 celdas
ImageNet desde caché: 57/72 y 43/48.)

### R2 — `exp21b_local_global_K200.py` → `exp21b_local_global_K200.csv` (66 pares)
K = 200 permutaciones (`RandomState(0)`), α = 0.05, calibración escalar (sin búsqueda de capas). Por par
y métrica (mKNN k=10 y CKA lineal, en R y en H): raw, media y s.d. de la null, τ95 (eq. 9: estadístico de
orden ⌈(1−α)(K+1)⌉ de {raw} ∪ nulls), p = (1+#{null ≥ raw})/(K+1) (eq. 10), score calibrado
max((raw−τ95)/(1−τ95), 0) (eq. 12), y el antiguo raw − media (continuidad). Números en §5 abajo.

### R3 — criterio del tree map ejecutable
`gen_appendix2.py`: `SELECTED` ya no es un dict tecleado; se calcula de `exp23_config_diagnostics.json`
(entre configuraciones con `maxfrac ≤ 0.5`, mayor `cpcc`, por dataset) y se **asserta** igual a
{ImageNet: cosine-average, CIFAR-100: cosine-complete}; falla ruidosamente si cambia. Misma función en
`make_treemap_fig.py` y `make_fig_overview.py`.

### R4 — tabla de calibración generada
`gen_appendix.py` escribe `appendix_tables/tab_calibration.tex` (el .tex tecleado se sustituyó por
`\input`). Fila gaussiana desde `exp1_delta_controls.csv` (variante `gauss`, deduplicada por d):
d = 192/384/512/768/1024/1536 → δ_norm 0.104/0.081/0.074/0.061/0.055/0.046 (**antes** solo 192/768/1536:
0.104/0.061/0.046). Filas árbol/H²/esfera: sin fichero de resultados (exp6 imprimió a stdout), tecleadas
dentro del generador y re-verificadas en vivo por `sweep_freeze.py`.

### R5 — regeneración y verificación
`gen_main_table.py`, `gen_appendix.py`, `gen_appendix2.py`, `gen_review_tables.py`, `gen_provenance.py`,
figuras, y `sweep_freeze.py` con la sección 2 (censo) leyendo `expR39b` y la sección 5 (§5) leyendo
`exp21b`. Resultado del sweep al cierre: ver "Verificación" al final.

## 3. Cambios estructurales

- **Caja de página 1** (`tcolorbox`) antes del primer párrafo de §1, texto del brief.
- **Figura 1** (conceptual): `\IfFileExists{figures/fig1_concept.pdf}` → imagen; si no, placeholder del
  mismo tamaño (1.25 in de alto, ancho de texto) para que el layout sea final. Leyenda del brief.
  `fig_intro_concept.pdf` (cueva NeurIPS) borrado; ya no se referenciaba.
- **Figura 2 nueva** (`figures/make_fig_overview.py`): (a) δ_norm cruda vs d con la curva gaussiana
  (exp1) y los 12 modelos (expR39b); (b) "same raw, opposite verdict": BGE-base (expR48) frente a la celda
  de visión con raw más cercano, elegida por datos = SigLIP-B/DTD (raw 0.121 ambos; exceso +0.00x vs
  −0.062); (c) ARI DINOv2-B/L/G vs bloque, naive vs seleccionada, ImageNet 0.029→0.380 y CIFAR-100
  0.005→0.389 (exp23).
- **Tabla 1**: 10 → 7 columnas (modelo · familia · δ_norm IN · exceso IN (r/20) · exc. C100 · exc. C10 ·
  ρ_WN) desde `expR39b` + `exp3_alignment.csv`; negrita en celdas sign-positivas. Las columnas p99.9,
  DTD, FMNIST/MNIST, best−R y métrica → `appendix_tables/tab_census_extra.tex` (nueva, generada en
  `gen_main_table.py` desde `expR34`, `exp2`, `expR39b`).
- **Tree map**: la figura principal pasa de 4 paneles a 2 (solo ImageNet, naive vs seleccionada; etiquetas
  ≥ 7 pt); la pareja CIFAR-100 es una figura de apéndice (`fig_treemap_c100.pdf`, mismo script con
  `--appendix`).
- **Figura best-metric** (scatter) movida de §6 al apéndice (subsección Statistics), referenciada desde
  §6. Motivo: §6 se comprimió a media página y el corolario queda en una línea de contribuciones; la
  figura a ancho completo allí desbordaba el texto principal a la p10.
- **§4** retitulado "Findings I: Structure beyond the Null"; el párrafo "The excess certifies clustering,
  not depth" sigue inmediatamente al censo (ya lo hacía).
- **§5**: nuevo párrafo final "Form and content anti-align" (dos frases, movidas desde §7).
- **§6**: comprimido a tres párrafos ("δ_norm predicts the gain within datasets" / "The training objective
  picks the metric" / "A diagnostic, not a policy"), un número cada uno; punteros y comparación de
  políticas al apéndice.
- **Apéndice**: nueva etiqueta `app:census` en "Vision census: excess and significance per cell";
  `tab_census_extra` insertada antes de B2; figura CIFAR-100 del tree map en `app:treemap`.
- Todos los lead-ins de §3–§6 son afirmaciones (p. ej. "A low raw value is not evidence.", "The reading is
  an excess over a matched null.", "ξ reads norm structure, not curvature.", "The island is an artifact;
  sharing is graded.", "WordNet alignment follows supervision, not tree-likeness.", "The recovery is not
  WordNet circularity.", "Local agreement survives calibration.", "Negative curvature concentrates on
  bridges.").
- Leyendas: primera frase en negrita con el takeaway, en todas las figuras y en Tabla 1, B20 y calibración.

## 4. Párrafos reescritos (texto)

- **Abstract**: sustituido por el borrador del brief íntegro (los números que dependían de R1 no cambiaron:
  "69 of 72"). Antes: "(1)~Tree-like class geometry… (2)~The trees are moderately shared… (3)~…";
  ahora prosa continua con la frase de la estrella "…whether or not they were trained in hyperbolic space".
- **§1**: párrafos 1–2 intactos; "Our first contribution is therefore an instrument…" y "The answer has two
  halves…" → tres párrafos: instrumento (2 frases), "What survives is clustered form." y "Sharing is graded,
  and its naive measurement misleads." Solo dos números en toda la intro: "69 of 72" y "0.03 → 0.38"
  (exp23: `('euclid','average')` big_vs_sup 0.029 → `('cosine','average')` 0.380). Contribuciones: cuatro
  ítems ≤ 2 líneas, sin cadenas de puntos y coma; el 4.º es una línea.
- **§3 "The reading is an excess over a matched null."**: "5 null replicates for δ, 20 for ξ" y la frase
  del z-score → 20 réplicas para ambos; evidencia primaria = rango de percentil (20/20 = bajo todas);
  justificación rank-based citando Gröger §5.1; z relegado al apéndice; frase de null "aggregation-aware"
  (Gröger §5.2): mismo supremo con el mismo presupuesto (5×10⁵ cuádruplas por semilla) sobre cada
  réplica; real 10 semillas / réplica 5 afecta a la varianza, no al sesgo. Eliminado el paréntesis largo
  sobre construcciones alternativas de la null (→ puntero a A.robust) y el aparte del whitening.
- **§4 censo**: "Vision: 69/72 cells, stated at two levels." → "Vision: structure beyond the null in 69 of
  72 cells." en lenguaje de rangos (arriba). Eliminados del texto principal: "51/72 |z|≥2", "41/48",
  "34/48 z<−3" (exp20) → sustituidos en la leyenda de B20 por 50/72 y 31/48 (expR39b); "−0.04..−0.08 →
  −0.007..−0.03" (→ A.robust); "−0.039/−0.020, z −4.8/−2.4" de Barlow/BYOL (→ B24).
- **§4 texto**: frases cortas; eliminadas las magnitudes en cadena (S/M −0.015/−0.010, L/XL
  −0.041/−0.049, Pythia −0.032..−0.036, OLMo −0.039/−0.015, erank 1.3→56/98, +0.002..+0.011,
  0.114–0.124, −0.019) → viven en B23/B28/Fig. 4; se conserva una magnitud (0.02 del padding).
- **§4 profundidad-vs-C, training, ORC**: una magnitud por párrafo (se conservan −0.124 vs −0.093 y
  19–91 %; eliminados +1.58/+0.53 pp, −61 %/−46 %, p>0.12, +65–144 %, 7.6/3.2 %, 8.0/4.4 %, 0.3/0.5 % →
  B3, Fig. 3, B6).
- **§5**: "The island is an artifact; sharing is graded." sin cadena (eliminados 0.2–0.5 vs 0.5–0.6, gaps
  +0.03..+0.36, −0.02, 0.13–0.24 vs 0.46–0.57 → B7/B11); párrafo del criterio compactado ("highest CPCC
  among non-degenerate configurations"); "Under the selected configurations…" conserva 0.38 vs 0.48
  (eliminados 0.39 vs 0.63, 0.003–0.12, 0.48–0.52 vs 0.54–0.60, 0.56 vs 0.61, 0.74/0.74 → B7);
  HierarCaps conserva 0.83–0.97 (eliminados +0.32..+0.87, 10–54 % vs 4.2 % → A8); semántica angular
  conserva 0.91 vs 0.71 (eliminados 0.85/0.63); ARI 0.61 conserva; definición de ARI y CPCC en una cláusula.
- **§6**: ver estructural; conserva r = −0.83 (ImageNet NC) y +1.3 pp (CLIP); eliminados del texto
  principal −0.87, −0.67..−0.78, −0.62/−0.79, −0.31..−0.86, −0.87/−0.65, +0.23/−0.34, −0.20..−0.53,
  +2.05 pp, p≈4×10⁻³⁸, ≤0.2 pp, +0.9, −0.60..−0.83, −0.55..−0.82, +0.63/+0.64, +0.41/+0.28, p=0.011,
  −0.24 pp, +0.5..+2 pp, −2.6 pp, 0/60 → B12, A2/A3, B8, A6, B9.
- **§7**: primer párrafo abre con el espejo de la caja; conserva la frase del "corrected picture" (0.13–0.38);
  la frase de anti-alineación se movió a §5; Limitations (i)–(vii) intactas.
- **§2**: frase del nivel-muestra acortada (puntero a A.sample).

## 5. Tabla 1 — celda a celda (exp20, 5 réplicas → expR39b, 20 réplicas; fuente de cada valor en la cabecera)

| modelo | exc IN (z) exp20 → exc IN (r/20) expR39b | exc C100 exp20 → expR39b | exc C10 exp20 → expR39b |
|---|---|---|---|
| i21k_t | −0.021 (−2.7) → −0.019 (20/20) | −0.043 → −0.039 | −0.059 → −0.051 |
| i21k_s | −0.020 (−2.3) → −0.021 (20/20) | −0.033 → −0.031 | −0.096 → −0.090 |
| i21k_b | −0.028 (−3.0) → −0.027 (20/20) | −0.017 → −0.015 | −0.090 → −0.083 |
| i21k_l | −0.030 (−4.4) → −0.031 (20/20) | −0.048 → −0.043 | −0.098 → −0.096 |
| dinov1_b | −0.005 (−0.6) → −0.005 (16/20) | −0.068 → −0.063 | −0.097 → −0.094 |
| dinov2_s | +0.012 (+1.1) → +0.012 (0/20) | −0.053 → −0.051 | −0.095 → −0.095 |
| dinov2_b | +0.006 (+0.6) → +0.007 (0/20) | −0.043 → −0.042 | −0.135 → −0.134 |
| dinov2_l | −0.008 (−1.1) → −0.008 (20/20) | −0.051 → −0.051 | −0.155 → −0.154 |
| dinov2_g | +0.004 (+0.5) → +0.004 (0/20) | −0.079 → −0.079 | −0.157 → −0.155 |
| clip_b | −0.025 (−3.1) → −0.027 (20/20) | −0.047 → −0.044 | −0.077 → −0.071 |
| clip_l | −0.021 (−2.3) → −0.020 (20/20) | −0.043 → −0.038 | −0.083 → −0.080 |
| siglip_b | −0.019 (−1.7) → −0.017 (20/20) | −0.042 → −0.037 | −0.091 → −0.088 |

Cambio máximo en las filas de transfer (nulls 5 → 20 réplicas): 0.012. ρ_WN sin cambios (exp3).
Ceros a la izquierda restaurados en todas las tablas (−0.021, no −.021): eliminados los `.replace("0.", ".")`
de `gen_main_table.py` y `gen_review_tables.py`.

## 6. Figuras

- `figures/style.mplstyle` (STIX, 8/7 pt, sin spines superior/derecho, fonttype 42) y `figures/palette.py`
  (`FAMILY_COLORS`: supervisado #4C72B0, SSL #DD8452, contrastivo #55A868, LM causal #8172B2, embedder
  #937860, null #7F7F7F discontinuo; datasets solo por marcador) usados por todos los scripts.
- `fig_text_nulls.py`: **corregido** el uso de colores (GPT-2 iba en naranja SSL y Pythia en azul
  supervisado): todos los LMs causales en morado, embedders en marrón; barras huecas = at null (no bajo
  todas las réplicas), en lugar de los tres callouts; barras de error conservadas; fuente `expR48`.
- `make_figs.py`: fuente `expR39b` (antes exp20); título (a) calculado ("69/72"); leyenda única fuera
  (abajo); scatter best-metric sin r en el título → r a la leyenda (FS −0.60..−0.83; NC −0.61..−0.86;
  pooled −0.31/−0.22, salida del script).
- `make_fig3_causal.py`: barras en el color de familia (no rojo); líneas de profundidad DINOv2-B naranja,
  CLIP-B verde.
- `make_treemap_fig.py`: 2 paneles + `--appendix`; viridis; etiquetas 7 pt.
- Alturas: Fig. 1 1.25 in, overview 1.55, excesos 1.85, texto 1.75, causal 1.45, tree map 2.05,
  best-metric 1.5 (apéndice).

## 7. Verificación / estado de compilación

- Compila con tectonic: 0 warnings (undefined/overfull), 0 `??`; texto principal + Ethics Statement en
  p9; Reproducibility abre p10; 30 páginas en total.
- `sweep_freeze.py`: se completa en el bloque "Cierre" al final de este fichero.

## 8. §5 — calibración Gröger (exp21 → exp21b)

Fuente nueva: `exp21b_local_global_K200.csv` (66 pares; K = 200; α = 0.05; centroides = caché del censo,
ver §0). Tabla de apéndice nueva `tab_b31_groger.tex` (B31) con raw / null mean / τ₀.₀₅ / calibrado /
null-centered / pares p<0.05 para mKNN y CKA en R y H.

| medida | exp21 (3 perms, raw → raw−null) | exp21b raw | null mean | τ₀.₀₅ | calibrado (eq. 12) | null-centered | pares p<0.05 |
|---|---|---|---|---|---|---|---|
| mKNN R | 0.474 → 0.464 | 0.472 | 0.0100 (= 10/999) | 0.015 | **0.464** | 0.462 | 66/66 (p = 1/201 todos) |
| mKNN H | 0.427 | 0.425 | 0.0100 | 0.019 | 0.414 | 0.415 | 66/66 |
| CKA R | 0.636 → 0.529 | 0.633 | 0.108 | 0.111 | **0.577** | 0.524 | 66/66 |
| CKA H | 0.644 | 0.640 | 0.108 | 0.111 | 0.586 | 0.532 | 66/66 |

Texto de §5 "Local agreement survives calibration.": antes "mutual-kNN 0.474→0.464 calibrated…; null 0.010",
"CKA 0.636→0.529", "kNN 0.474→0.427; CKA 0.636→0.644"; ahora K/α/eq. 12 explícitos, "0.472 raw, 0.464
calibrated, every pair at p=1/201, null mean 0.010 = k/(n−1)", "CKA 0.633 raw, 0.577 calibrated, null 0.108",
Poincaré "0.472→0.425; 0.633→0.640". La frase de Poincaré se conserva.

## 9. Cierre — verificación

- `sweep_freeze.py` (sección 2 → expR39b, sección 5 → exp21b, check de fidelidad R1 añadido): **67/67 PASS**
  (`rebuttal/results/sweep_freeze_final.log`).
- Compilación tectonic: 30 páginas, 0 warnings (undefined/overfull), 0 `??`; Ethics Statement en p9,
  p10 abre con Reproducibility Statement. PDF: `ICLR2027/main_iclr2027_final.pdf`.
- Generadores ejecutados en orden: gen_main_table, gen_appendix, gen_appendix2, gen_review_tables,
  gen_provenance (`tab_z_provenance.tex` en sincronía; no se incluye en el PDF por decisión previa del autor).

## 10. Follow-up (cinco arreglos)

1. **Figura 1** `[t]` → `[H]`: queda bajo la caja de página 1 (placeholder `\IfFileExists` intacto).
2. **make_fig_overview.py**: panel (a) con leyenda de familias (sup./SSL/contr.) + curva gaussiana, colocada
   bajo los ejes; leyenda raw/exceso del panel (b) bajo los ejes (etiqueta "SigLIP-B (DTD)" en una línea).
3. **fig_text_nulls.py**: leyenda dentro de los ejes, abajo a la derecha (zona vacía), ya no cubre GPT-2 S/M.
4. **Línea ~137 (§4)**: "DINOv2-L is sign-negative but marginal there (z=−1.1)" → "DINOv2-L is below every
   replicate on ImageNet, with a small excess (−0.008)" (expR39b: −0.0084, rango 20/20). En la misma frase, los
   rangos por familia pasan de exp20 a expR39b: ViT −0.020..−0.030 → **−0.019..−0.031**; CLIP/SigLIP
   −0.019..−0.025 → **−0.017..−0.027**; S→G CIFAR-10 −0.095→−0.157 → **−0.095→−0.155**. La frase p99.9
   "40/48 hierarchical cells beyond 2σ" → "**42/48** hierarchical cells below every replicate" (expR40, rango);
   70/72 sin cambio. En el párrafo del censo, "(z ≤ +1.1)" → "(excess ≤ +0.012)" (expR39b, máx. sign-positivo
   +0.0118) y "per-cell ranks and z" → "per-cell ranks". Comentario de fuente de la línea actualizado
   (exp26/exp12/exp11 → expR39b, expR40).
   **Grep de z en líneas 1–210 tras el cambio**: no queda ningún valor de z-score. Quedan dos menciones del
   concepto en §3 (línea 115: "a z-score on a handful of replicates is not rank-based, so z is kept only as a
   secondary summary in the appendix"), que es la frase que el brief §3.4 pide explícitamente.
5. **Párrafo "Text: …"**: tres magnitudes — 0.02 (padding), S/M vs L/XL (−0.015/−0.010 vs −0.041/−0.049,
   expR48/B23) y el rango DBpedia (−0.020..−0.028, exp14), este último absorbido desde la frase "Second, …".
   Eliminadas del cuerpo las cifras de plantillas ("Two scopes": .065/.012, medias −0.012/−0.023/−0.040/−0.041,
   9/10, 10/10) → leyenda de B14 (ya llevaba medias y recuentos; añadido el rango de sensibilidad .065/.012 con
   puntero a la Tabla A7); el detalle GTE-Qwen2 (−0.019 con 3 réplicas; +0.002, rango 8/20 con 20) → leyenda
   de B23. El bloque queda en tres párrafos cortos con una frase-puntero a A.templates.
- Verificación: `sweep_freeze.py` **67/67 PASS**; compilación 30 páginas, warnings = 0, `??` = 0; Ethics en
  p9 (1); p10 abre con "R EPRODUCIBILITY S TATEMENT". PDF actualizado en `ICLR2027/main_iclr2027_final.pdf`.

## 11. R6 — censo a 200 réplicas (resolución Gröger): censo de registro

Ficheros nuevos: `expR39c_census200_cache.csv` (72 celdas; copia de expR39b con `range(200)`, semillas 300+rep, 5 semillas/réplica) y
`expR48b_text_census200_bs1.csv` (15 modelos; copia de expR48 con `range(200)`, 3 semillas/réplica como expR48). Ambos guardan `frac_null_above`,
`r_above` y `p_left = (1+#{null ≤ real})/201`. Cómputo: 3 procesos de visión (ImageNet en dos mitades + transfer) y 2 de texto (una GPU cada uno),
BLAS monohilo, ~1.5 h de pared en máquina compartida. Los CSV de 20 réplicas (`expR39b`, `expR48`) se conservan como histórico; sus tablas B20/B23
se sustituyen por las de 200 réplicas (labels `tab:b20-census200`, `tab:b23-text200`). **"Genuine" se define una sola vez en §3: p ≤ 0.05, r ≥ 191/200**;
todas las frases "below every replicate" del texto principal se han sustituido.

### Recuentos (texto principal)

| enunciado | antes (20 réplicas, expR39b) | ahora (200 réplicas, expR39c) |
|---|---|---|
| sign-negativas (abstract, intro, §4) | 69/72 | **68/72** (ViT-T FashionMNIST pasa de -0.012 (11/20) a +0.001 (95/200, p=0.527)) |
| celdas genuinas (antes: bajo las 20 réplicas) | 57/72 | **55/72** (p ≤ 0.05); bajo las 200 réplicas: 44/72 |
| jerárquicas genuinas | 43/48 | **43/48** |
| planas: signo / genuinas | 24/24 / 14/24 | **23/24 / 12/24** |
| excepciones sign-positivas | DINOv2-S/B/G IN (excess ≤ +0.012, rango 0/20) | DINOv2-S/B/G IN (excess ≤ +0.011, r = 0/0/1 de 200, p ≥ 0.99) + ViT-T FMNIST (+0.001, p 0.53) |
| leyenda B20: \|z\| ≥ 2 / z ≤ −3 jerárquicas | 50/72 / 31/48 | 49/72 / 32/48 |
| §4 l.137 rangos ImageNet: ViT / CLIP-SigLIP | −0.019..−0.031 / −0.017..−0.027 | −0.020..−0.030 / −0.016..−0.025 |
| §4 l.137 DINOv2-L ImageNet | "below every replicate, excess −0.008" | "genuine, excess −0.008, r = 200/200" |
| §4 l.137 p99.9 (expR40, 20 réplicas) | "42/48 below every replicate" | "42/48 genuine at 20 replicates" (mismo recuento, lenguaje de p) |
| Fig. 3 título (a) | 69/72 | 68/72 (calculado) · huecos = p > 0.05 |
| Fig. 2 (b) exceso SigLIP-B/DTD | −0.062 | −0.066 |

### Tabla 1, celda a celda (exceso (r) 20 réplicas → exceso (r/200, p) 200 réplicas)

| modelo | ImageNet | CIFAR-100 | CIFAR-10 |
|---|---|---|---|
| i21k_t | -0.019 (20/20) → -0.020 (199/200, p=0.010) | -0.039 (20/20) → -0.038 (200/200, p=0.005) | -0.051 (20/20) → -0.042 (191/200, p=0.050) |
| i21k_s | -0.021 (20/20) → -0.022 (200/200, p=0.005) | -0.031 (20/20) → -0.031 (199/200, p=0.010) | -0.090 (20/20) → -0.081 (200/200, p=0.005) |
| i21k_b | -0.027 (20/20) → -0.025 (200/200, p=0.005) | -0.015 (18/20) → -0.016 (187/200, p=0.070) | -0.083 (20/20) → -0.073 (200/200, p=0.005) |
| i21k_l | -0.031 (20/20) → -0.030 (200/200, p=0.005) | -0.043 (20/20) → -0.044 (200/200, p=0.005) | -0.096 (20/20) → -0.088 (200/200, p=0.005) |
| dinov1_b | -0.005 (16/20) → -0.003 (162/200, p=0.194) | -0.063 (20/20) → -0.064 (200/200, p=0.005) | -0.094 (20/20) → -0.085 (200/200, p=0.005) |
| dinov2_s | +0.012 (0/20) → +0.011 (0/200, p=1.000) | -0.051 (20/20) → -0.052 (200/200, p=0.005) | -0.095 (20/20) → -0.095 (200/200, p=0.005) |
| dinov2_b | +0.007 (0/20) → +0.007 (0/200, p=1.000) | -0.042 (20/20) → -0.042 (200/200, p=0.005) | -0.134 (20/20) → -0.135 (200/200, p=0.005) |
| dinov2_l | -0.008 (20/20) → -0.008 (200/200, p=0.005) | -0.051 (20/20) → -0.051 (200/200, p=0.005) | -0.154 (20/20) → -0.153 (200/200, p=0.005) |
| dinov2_g | +0.004 (0/20) → +0.005 (1/200, p=0.995) | -0.079 (20/20) → -0.079 (200/200, p=0.005) | -0.155 (20/20) → -0.155 (200/200, p=0.005) |
| clip_b | -0.027 (20/20) → -0.025 (200/200, p=0.005) | -0.044 (20/20) → -0.044 (200/200, p=0.005) | -0.071 (20/20) → -0.062 (200/200, p=0.005) |
| clip_l | -0.020 (20/20) → -0.018 (200/200, p=0.005) | -0.038 (20/20) → -0.039 (200/200, p=0.005) | -0.080 (20/20) → -0.072 (200/200, p=0.005) |
| siglip_b | -0.017 (20/20) → -0.016 (199/200, p=0.010) | -0.037 (20/20) → -0.038 (200/200, p=0.005) | -0.088 (20/20) → -0.080 (200/200, p=0.005) |

Cambio máximo de exceso entre 20 y 200 réplicas: 0.0131. Celdas que cambian de veredicto (bajo-todas a 20 vs p ≤ 0.05 a 200): i21k_s/mnist (-0.044, r=186, p=0.075; antes 20/20); i21k_b/mnist (-0.030, r=181, p=0.100; antes 20/20).
Marca ∘ en Tabla 1 (transfer no genuino): i21k_b/cifar100 (p=0.070).

### Censo de texto (expR48, 20 réplicas → expR48b, 200 réplicas)

| modelo | exceso 20 (rango/20) | exceso 200 (r/200, p) | veredicto 200 |
|---|---|---|---|
| gpt2 | -0.0150 (20/20) | -0.0132 (183/200, p=0.090) | no genuino |
| gpt2_m | -0.0098 (18/20) | -0.0090 (173/200, p=0.139) | no genuino |
| gpt2_l | -0.0406 (20/20) | -0.0389 (200/200, p=0.005) | genuino |
| gpt2_xl | -0.0485 (20/20) | -0.0472 (200/200, p=0.005) | genuino |
| pythia_410m | -0.0361 (20/20) | -0.0352 (200/200, p=0.005) | genuino |
| pythia_1b | -0.0319 (20/20) | -0.0310 (200/200, p=0.005) | genuino |
| pythia_2b8 | -0.0361 (20/20) | -0.0346 (200/200, p=0.005) | genuino |
| olmo_1b | -0.0393 (20/20) | -0.0381 (200/200, p=0.005) | genuino |
| bge_base | +0.0037 (5/20) | +0.0053 (23/200, p=0.886) | no genuino |
| bge_large | +0.0029 (5/20) | +0.0034 (53/200, p=0.736) | no genuino |
| gte_base | +0.0113 (0/20) | +0.0121 (2/200, p=0.990) | no genuino |
| gte_large | +0.0069 (1/20) | +0.0071 (11/200, p=0.945) | no genuino |
| gte_qwen2 | +0.0019 (8/20) | +0.0033 (63/200, p=0.687) | no genuino |
| e5_base | +0.0024 (9/20) | +0.0040 (39/200, p=0.806) | no genuino |
| e5_large | +0.0047 (3/20) | +0.0049 (28/200, p=0.861) | no genuino |

Genuinos: 6/15. GPT-2 S (p = 0.090) y M (p = 0.139) no son genuinos y L/XL sí → se mantiene "emerges with scale" (regla del brief).
Frase de §4: "S and M sit at their null (−0.015/−0.010) while L and XL lie below every replicate (−0.041/−0.049)" → "S and M are not genuine (−0.013/−0.009) while L and XL are (−0.039/−0.047)"; "OLMo at both scales tested" → "OLMo-1B" (OLMo-7B no re-extraído);
embedders "sit at their nulls" → "are not genuine"; GTE-Qwen2 "not robust across replicate counts" → "not genuine either". Leyenda B14: S/M (z −1.6/−1.2, −0.015/−0.010) → (−0.013/−0.009, p 0.09/0.14).
§3: "over 20 spectrum-null replicates, for δ and ξ alike" → "over 200 (20 for ξ)"; definición de genuine; "each replicate 5" → "5 (3 in the text census)". Intro: "ranked against 20" → "200". Leyendas Fig. 3/4: huecos = p > 0.05; Fig. 4 "20 null replicates" → "200".

### Verificación
- `sweep_freeze.py` (sección 2 → expR39c; sección 3 → expR48b): **67/67 PASS**.
- Compilación: 30 páginas, warnings = 0 (Overfull de B20 corregido con tabcolsep 1.8 pt), `??` = 0; Ethics en p9; p10 abre con Reproducibility Statement. PDF: `ICLR2027/main_iclr2027_final.pdf`.

## 12. Pasada de contribución (cuatro ediciones de texto + el instrumento como herramienta)

1. **Abstract**: "nearly universal in vision (68 of 72 cells)" → "nearly universal on datasets with a class hierarchy (43 of 48 cells;
   55 of 72 overall)" (expR39c, genuine = p ≤ 0.05). §4 (párrafo del censo): "p ≥ 1.00" → "p ≥ 0.99" (DINOv2-G ImageNet, r = 1/200 → p = 0.995).
2. **§4, nuevo párrafo "Relation to neural collapse."** (8 líneas, solo `papyan2020prevalence`), inmediatamente después de "The excess certifies
   clustering, not depth.": el ETF de neural collapse como caso extremo de la estrella (δ = 0 sin jerarquía); lo que separa nuestra lectura:
   (i) la organización certificada no es plana — hubs de superclase recuperados en §5 y curvatura negativa en los puentes; (ii) aparece en
   modelos sin etiquetas, en datasets de transfer y en LMs causales, fuera del ámbito de neural collapse, y el test de estrella emparejada
   pregunta lo que neural collapse no pregunta. §2: "…without testing tree-likeness" + "; §4 relates our excess to the collapsed simplex".
3. **Intro**: al final del párrafo del instrumento, dos frases de posicionamiento: Gröger et al. calibran similitud entre modelos; nosotros
   geometría dentro de un modelo, y comparar los árboles resultantes entre modelos necesita su propia calibración (§5).
4. **El instrumento como herramienta**: `ICLR2027/tool/calibrated_delta.py` (entrada: matriz de centroides n×d `.npy` + etiquetas de superclase
   opcionales; salida: δ_norm cruda, exceso espectral con rango r/200 y p de cola izquierda, y el test de profundidad con estrella emparejada de
   B29; funciones y semillas copiadas verbatim de `expR39c_census200_cache.py` y `expR50_depth_test.py`). `ICLR2027/tool/run_checks.py`
   lo ejecuta en ViT-L/CIFAR-100 y DINOv2-L/ImageNet y escribe `rebuttal/results/tool_check.json`; `sweep_freeze.py` (sección TOOL) compara con
   la Tabla 1 (exceso 3 dp, r, p) y con B29 (profundidad 3 dp, z 1 dp; en ImageNet con los centroides del store, como expR50).
   **Resultado: reproducción exacta** — ViT-L/C100 exceso −0.0441, r 200, p 0.005; DINOv2-L/IN −0.0079, r 200, p 0.005; profundidad
   +0.0396 (z +4.06) y −0.0289 (z −2.46), idénticos a las tablas. Frase nueva en §3 ("The instrument is a single script: …") y en el
   Reproducibility Statement con el placeholder `\url{ANONYMIZED-REPO}` (ver `TODO_author.md`).
5. **Ajuste de páginas**: el párrafo nuevo y las frases añadidas empujaban ~9 líneas a p10 → §6 comprimido de nuevo (sin la pregunta
   introductoria; tres párrafos de 3–4 líneas; se conservan r = −0.83 y +1.3 pp), cláusula de la null PC-permutación eliminada en §4
   ("a PC-permutation null confirms every verdict": los veredictos son ahora por rango), cláusula de Poincaré acortada en §5, frase de las
   ResNet-50 acortada en el censo, Figura 1 a 0.95 in, overview 1.65 in, excesos 1.7 in, tree map 1.95 in.
- Verificación: `sweep_freeze.py` **71/71 PASS** (67 + 4 checks del tool); compila con 30 páginas, 0 warnings, 0 `??`; el texto principal
  (Limitations incluidas) termina en p9 y el Ethics Statement abre p10. PDF: `ICLR2027/main_iclr2027_final.pdf`.

## 13. Pasada final de consistencia

1. **Vocabulario signo/genuino alineado con §3**: intro "What survives is clustered form." → "Structure beyond the matched null is nearly
   universal on datasets with a class hierarchy (43 of 48 cells; 55 of 72 overall; negative excess in 68 of 72)"; lead-in de §4 →
   "Vision: negative excess in 68 of 72 cells, genuine in 55."; contribución 2 → "Genuine tree-like form in nearly every vision cell with
   a class hierarchy".
2. **§3**: "(exact enumeration when C ≤ 30)" → "(for C = 10 the 5×10⁵ draws cover every quadruple many times over, so the supremum is exact)".
   Verificado: `delta_norm` de `expR39c_census200_cache.py` (y de todos los censos) muestrea 5×10⁵ cuádruplas por semilla en todos los casos;
   con C = 10 hay 210 cuádruplas distintas, cubiertas miles de veces.
3. **MERU**: "whether or not they were trained in hyperbolic space" (abstract) y "whether or not the model was trained in hyperbolic space"
   (intro) → "including a backbone trained in hyperbolic space" (un solo modelo, MERU). La caja de p1 no contenía la frase.
4. **ORC**: el párrafo "Negative curvature concentrates on bridges" pasa íntegro al apéndice A.14 ("ORC edge types", nuevo label `app:orc`);
   en §4 queda un puntero de una línea; en el párrafo de neural collapse "(below)" → "(Appendix A.14)". Hueco de la Figura 1 de vuelta a 1.2 in.
5. **p99.9 a 200 réplicas** (`expR40b_p999census200.py` → `expR40b_p999census200.csv`; copia de expR40 con `range(200)` y la caché del censo
   como fuente de centroides en todos los datasets — expR40 leía ImageNet del store —, mismo estadístico p99.9 y mismas semillas; 3 procesos,
   ~70 min). Recuentos: sign-negativas **70/72 (sin cambio)**; jerárquicas genuinas **44/48** (antes "42/48 below every replicate" a 20 réplicas
   sobre el store); genuinas 56/72; ImageNet: DINO/DINOv2 genuinos (p 0.005), ViT-T no (p 0.114), CLIP-B p 0.0498. Frase de §4 reescrita
   ("…44/48 hierarchical cells genuine under the same protocol (200 replicates)…"); B21 regenerada desde expR40b (exc/r/p por celda, recuentos
   en la leyenda); la columna p99.9 de `tab_census_extra` pasa de expR34 (20 réplicas) a expR40b (200). Check nuevo en el sweep.
   Sign-positivas bajo p99.9: ViT-B/CIFAR-100 (+0.001, p 0.56) y ViT-T/FMNIST (+0.001, p 0.53).
6. **`ICLR2027/tool/demo.py`** (n = 60, d = 32, 50 réplicas, semillas fijas): nube aleatoria → exceso −0.014, r 47/50, p 0.078 → no genuino;
   estrella de 6 clusters → exceso −0.054, p 0.020 → genuino, profundidad +0.039 (z +0.9) → sin profundidad; jerarquía 6×5 → exceso −0.115,
   p 0.020 → genuino, profundidad −0.025 (z −0.6) → más arbóreo que su estrella. A n = 60 con K = 6 hubs el test de profundidad está
   infra-potenciado (probé cuatro parametrizaciones; |z| ≤ 0.6 en todas), así que el demo y el README leen el signo de la profundidad y remiten
   a B25/B29 (n = 100–1000) para la calibración. `tool/README.md`: uso, check de reproducción de dos celdas, demo.
- Verificación: `sweep_freeze.py` **73/73 PASS**; compila con 30 páginas, 0 warnings, 0 `??`; el texto principal (Limitations incluidas)
  termina en p9 y el Ethics Statement abre p10. PDF: `ICLR2027/main_iclr2027_final.pdf`.

## 14. Respuesta a la revisión — FASE A (experimentos y memo; sin ediciones de prosa) — CHECKPOINT

Todo lo que sigue está fijado antes de las ejecuciones; ningún ajuste se eligió mirando los números. Memo con solo números:
`ICLR2027/MEMO_phaseA.md` (generado por `rebuttal/scripts/make_memo_phaseA.py`).

**A1 — 2×2 {null} × {estadístico}, 200 réplicas, caché del censo, BH (FDR 0.05) por 72 celdas / 15 modelos.**
Scripts: `expR52_census_haar_p999_200.py` (motor de visión; `--null gauss|haar --stat sup|p999`), `expR54_census_haar_sup_200.py`
(wrapper Haar × sup), `expR53_text_haar_p999_200.py` (motor de texto; embeddings extraídos una vez y cacheados en
`Platonic/results/text_cache`). Ficheros: `expR52_census_haar_p999_200.csv` (registro propuesto), `expR54_census_haar_sup_200.csv`,
`expR53_text_{haar_p999,haar_sup,gauss_p999}_200.csv`; los cuadrantes Gauss × sup/p99.9 de visión son expR39c/expR40b.
Semillas: null `300+rep` en ambas construcciones (expR46 usaba `700+rep` para Haar); real 10 semillas de cuádruplas; réplica 5 (visión) /
3 (texto). Bootstrap ImageNet bajo el registro: `expR59_imagenet_bootstrap.py` (30 remuestreos × 12 modelos; **20** réplicas Haar por
remuestreo en vez de 200 — basta para la s.d. del exceso; documentado).

**A2 — test de profundidad.** `expR55_depth_power.py` (barrido sintético: {estrella, 2 niveles, 3 niveles} × K {6,12,20,30} ×
ratio {0.1,0.3,0.6} × {iso, aniso} × n {100,1000}, d = 768, 5 semillas; protocolo exacto de B29 con estrella **isotrópica**; 720 runs)
→ `expR55_depth_power.csv` y `figures/fig_depth_power.pdf`. `expR56_depth_variants.py` (backbones reales: estrella isotrópica (a) vs
anisotrópica (b) = muestra Haar con el espectro real de cada superclase; **10** semillas de estrella; K: CIFAR-100 {20,10,5} — los marcos
10/5 son un clustering aglomerativo consenso de los 20 hubs, `expR56_frames_cifar100.csv` —, ImageNet cortes WordNet {30,10,60})
→ `expR56_depth_variants.csv`. Control con checkpoints fine-tuned: **no disponible** (sin pesos ni features guardados; coste en
`TODO_author.md`). Hueco anotado: la potencia sintética se barrió solo con la estrella isotrópica (la de B29), no con la anisotrópica.

**A3 — censo coseno.** `expR57_census_cosine.py`: centroides L2-normalizados, distancia geodésica esférica (arccos), null Haar
construida sobre la nube normalizada y renormalizada (como expR38), p99.9, 200 réplicas, BH → `expR57_census_cosine_haar_p999_200.csv`,
`expR57_text_cosine_haar_p999_200.csv`.

**A4 — tree map sin corte.** `expR58_treemap_cutfree.py`: dendrogramas reconstruidos desde la caché del censo (exp22/23 solo guardan
resúmenes), 6 configuraciones, correlación cofenética entre modelos y acuerdo en 10⁴ tripletes (`RandomState(0)`), más ARI en el corte
→ `expR58_treemap_cutfree{,_summary}.csv`.

**A5.** Semillas de estrella 3 → 10 en expR56. Bib: `he2026helm` y `yang2025hyperbolic` son NeurIPS **2025** (39.ª edición; verificado
en dblp) → `year={2025}`. Añadidos `narayan2011curvature` (Phys. Rev. E 84, 066108, 2011; arXiv 0907.1478), `kennedy2013hyperbolicity`
(arXiv:1307.0031, 2013) y `adcock2013treelike` (ICDM 2013, pp. 1–10; Crossref DOI 10.1109/ICDM.2013.77). Sin usar aún en el texto.

**Números clave (detalle en el memo).** Registro (Haar × p99.9, BH): visión sign-negativas 70/72, genuinas 49/72 (51 sin BH),
ImageNet+CIFAR-100 18/24, otros cuatro 31/48; texto 7/15 (GPT-2 S, L, XL, Pythia ×3, OLMo-1B; GPT-2 M p = 0.15). Otras construcciones:
Haar × sup 66/72, 46/72, 15/24, 31/48; Gauss × p99.9 70, 52, 19, 33; Gauss × sup 68, 52, 19, 33. DINOv2-S/B/L/G en ImageNet: genuinos bajo
las dos construcciones p99.9, por encima de todas las réplicas (r = 0) bajo las dos supremo. Bootstrap ImageNet: s.d. del exceso
≤ 0.0011, 30/30 remuestreos negativos en los 12 modelos. Potencia del test de profundidad (estrella isotrópica, sintético): **0.00**
en las 360 jerarquías (2 y 3 niveles, todo K, ratio, anisotropía, n); falsas alarmas en estrellas puras: z ≥ +2 en 16 %, z ≤ −2 nunca.
Backbones reales: la estrella isotrópica da z ≥ +2 en 8/12 (CIFAR-100, K = 20; máx. +6.8); la anisotrópica lo elimina (0/12) y da
z ≤ −2 en 3/12 (CIFAR-100) y 4/12 (ImageNet, K = 30). Coseno: 70/72, BH 56/72, IN+C100 22/24; acuerdo de veredictos con el registro
euclídeo 65/72; DINOv2 ImageNet genuino; texto 8/15 (incl. GPT-2 M y GTE-Qwen2; no GPT-2 S). Tree map: la isla naive aparece también sin
corte (ImageNet euclid-average: cofenética 0.36 vs 0.80 dentro del bloque, tripletes 0.47 vs 0.74); bajo cosine-average tripletes 0.77 vs
0.76 y cofenética 0.48 vs 0.78.

**Verificación.** `sweep_freeze.py` con la sección de Fase A (existencia, consistencia r↔p, BH monótono, flags): **84/84 PASS**.
No se ha tocado el .tex ni el PDF en esta fase.

## 15. Respuesta a la revisión — FASE B (decisiones del autor aplicadas al texto)

Decisiones fijadas por el autor antes de tocar el .tex: censo de registro = Haar × p99.9 × BH euclídeo (el coseno solo como
robustez); titular 49/72, 18/24, 70/72; excepción DINOv2–ImageNet resuelta como artefacto del supremo; escala solo DINOv2 en
CIFAR-10/CIFAR-100; texto "robust at the larger scales, fragile at the smaller"; test de profundidad reejecutado con el marco en las
hojas y estrella anisotrópica bajo una regla de decisión fijada de antemano; B29 retirada del texto principal. Commits: WIP `552fbf5`,
cierre en el commit de esta sección.

**B1 — censo de registro en Tabla 1, Figs. 2(a,b), 3 y 4.** `gen_main_table.py` lee `expR52_census_haar_p999_200.csv` (columnas: modelo,
familia, δ̂₉₉.₉ IN, exceso IN con ∘ si no es genuino BH, r/200 (p), exceso C100, exceso C10, ρ_WN; `tab_census_extra` añade la columna
"sup. IN (r)" de expR54). `make_figs.py` (Fig. 2a: hueco = no genuino BH; título "70/72 sign-negative, 49 genuine"),
`make_fig_overview.py` (medias del null como marcadores huecos; celda más cercana a BGE-base = ViT-L/DTD), `fig_text_nulls.py` (expR53),
`make_fig3_causal.py` (1.3 in). Apéndice: B20 desde expR52; B21 sustituida por la tabla de veredictos 2×2 (`tab_b21_2x2.tex`, códigos
Hp/Hs/Gp/Gs, 49/46/52/52); B23 desde expR53 con columna 2×2; nuevas B32 (coseno), B33 (tree map sin corte), B34 (estrella iso/aniso ×
K), B35 (potencia, marcos superior y de hojas), B36 (bootstrap ImageNet). Eliminadas del apéndice: B2 (tabla z), B18, B21 antigua,
B29. §3: dos frases de disclosure ("Two choices changed after pre-registration…": estadístico p99.9 y null Haar, con el motivo de cada una
y el puntero a la 2×2) — en el cuerpo, no en nota. §4: explicación mecanicista del artefacto del supremo en DINOv2–ImageNet (el supremo
sobre 5·10⁵ cuádruplas de 1000 centroides es un extremo de una sola cuádrupla; en DINOv2 unas pocas clases muy separadas lo elevan por
encima de todas las réplicas mientras el p99.9 queda por debajo).

**B2 — nombres y recuentos.** "tree-like" → "beyond-null (clustered) structure" en abstract, intro, contribuciones, §4, §7 y pies;
"genuine" := p BH ≤ 0.05 (frase de definición en §3). Recuentos, viejo → nuevo: signo negativo 68/72 → **70/72**; genuinas 55/72
(Gauss × sup, sin BH) → **49/72** (Haar × p99.9, BH); "43 of 48 …" → **18/24 en ImageNet + CIFAR-100** (y 31/48 en los otros cuatro
solo en el apéndice); "every family" se mantiene con la salvedad explícita de que ViT-T es genuino solo en DTD; celdas no genuinas en
ImageNet/CIFAR-100 listadas (ViT-T ambas, CLIP-B y SigLIP-B en ImageNet, ViT-S/B en CIFAR-100; el resto en B20). DTD y CIFAR-10 nunca "hierarchical".
Texto: 7/15 → **7/15** (mismo número, distinto conjunto: entra GPT-2 S p = 0.02, sale GPT-2 M p = 0.15); GPT-2 L/XL genuinos bajo las
cuatro construcciones y bajo coseno; Pythia ×3 y OLMo-1B genuinos en todo; embedders +0.002..+0.004 en nombres de clase, DBpedia
−0.020..−0.028 (bajo el supremo original, dicho en el texto).

**B3 — test de profundidad.** El barrido con el marco en las hojas (`expR55b_depth_power_leafframe.py`: {estrella, 2 niveles, 3 niveles} × K {6,12,20,30} × ratio {0.1,0.3,0.6} × {iso, aniso} × n {100,1000}, d = 768, 5 semillas; estrella anisotrópica con 10 semillas; configuraciones con más hojas que puntos omitidas; 600 runs, 2 h 08 min en 4 procesos) → `expR55b_depth_power_leafframe.csv`, `fig_depth_power.pdf` (el de marco superior queda como `fig_depth_power_topframe.pdf`). Regla fijada de antemano: potencia ≥ 0.8 para 2 y 3 niveles a ratio ≤ 0.3 para ambos n, y falsas alarmas ≤ 5 % en cada dirección. Resultado: potencia **1.00** en todas esas configuraciones (≥ 0.85 a ratio 0.6 con n = 1000; 0.70 en K = 30, ratio 0.6); falsas alarmas z ≥ +2: **0 %**; z ≤ −2: **6.2 %** agregadas — 0 % con n = 1000 y con K = 6, **17 %** con n = 100 y K ≥ 12 (3–8 puntos por cluster). La regla **falla** por la dirección z ≤ −2 → rama de apéndice: el texto principal dice que la profundidad por encima de los clusters etiquetados **no está certificada**, con la regla, las cifras y el motivo (párrafo nuevo tras el censo en §4); `fig_depth_test.pdf` ((a) CIFAR-100 K = 20, (b) ImageNet K = 30, real vs estrella iso/aniso con z; (c) potencia n = 1000 y falsas alarmas para n = 100 y 1000) va al apéndice A.5 con el párrafo "Real-data readings of the anisotropic test, not validated" (aniso: 4/12 ImageNet, 3/12 CIFAR-100 con z ≤ −2, 0/24 con z ≥ +2; las lecturas de CIFAR-100 caen en el régimen n = 100, K ≥ 12 con 17 % de falsas alarmas; las de ImageNet en el régimen n = 1000 con 0 % y potencia ≥ 0.85). B29 (estrella isotrópica) retirada del texto principal: el párrafo dice que dio el veredicto contrario en CIFAR-100 (z hasta +6.8, 8/12) y 16 % de falsas alarmas en estrellas sintéticas. Control positivo fine-tuned: no se pudo correr (sin checkpoints), dicho en el texto. Abstract e intro (ii): una frase "does not certify hierarchy above the labelled clusters". Frases MERU/NC rebajadas (B30, §4). Decisión: `rebuttal/results/phaseB_depth_decision.json`; script de aplicación con las dos ramas: `rebuttal/scripts/phaseB_t3_apply.py`. Limpieza: los pies de B30/B34/B35 ya no nombran scripts internos; B34 dice que sus lecturas no están certificadas.

**B4 — afirmaciones recortadas.** Escala: solo "DINOv2 deepens with scale on CIFAR-10 (−0.071 → −0.130) and, weakly, on CIFAR-100
(−0.031 → −0.040)"; sin afirmación general. Anti-alineación: una frase sin lead-in dentro del párrafo WordNet de §5. "Most tree-like
family" eliminado. Tree map con tres medidas (ARI en el corte, cofenética entre modelos, acuerdo en 10⁴ tripletes): la isla naive es
propiedad del corte y de la configuración euclídea (cofenética 0.36 vs 0.80, tripletes 0.47 vs 0.74; bajo cosine-average tripletes
0.77 vs 0.76). DBpedia abre el párrafo de texto. Párrafo coseno en §4 ("The angular reading agrees": 56/72, 22/24, acuerdo 65/72; DINOv2
ImageNet genuino en ambas lecturas).

**B5 — estructura.** §6 (corolario ORC) íntegro al apéndice (`app:corollary`) con un puntero de tres líneas en §7; ξ a una frase en §3;
párrafo de proyección/tareas de §3 al apéndice; related work: párrafo "δ against a random null, in network science" (Narayan & Saniee
2011; Kennedy et al. 2013; Adcock et al. 2013) que dice qué es nuevo (null con espectro exacto, corrección BH en censo, test de
profundidad con estrella emparejada, backbone hiperbólico como control); intro "What is new, in order": test de profundidad, MERU,
receta/escala. Limitación (i): la frase "against matched stars the residual depth is marginal" afirmaba un resultado retirado (B29)
y se sustituye por el alcance del test (certifica jerarquía por encima de su marco; dentro de los clusters no se testea) — única
edición en Limitations, forzada por la retirada.

**B6 — verificación.** `sweep_freeze.py` con sección de Fase B (recuentos de registro, ViT-T solo DTD, conjunto no genuino, rangos,
tendencias DINOv2, 2×2, bootstrap, coseno, texto, tree map, tool bajo el protocolo de registro, potencia/falsas alarmas del test de profundidad contra la regla, lecturas reales aniso/iso, marco superior): **100/100 PASS**.
Tool: `calibrated_delta.py` reproduce Tabla 1 (exceso, r, p exactos) y B34 (profundidad 3 dp, z 1 dp) en ViT-L/CIFAR-100 y
DINOv2-L/ImageNet. Compilación: 33 páginas, 0 warnings, texto principal termina en la p. 9. `main_iclr2027_final.pdf` exportado.
Checklist de preguntas de revisor: `ICLR2027/REVIEWER_CHECKLIST_phaseB.md`.

## 16. Pasada final tras la lectura completa (seis puntos del autor)

**(1) Test de profundidad acotado por régimen.** Se mantiene la regla prefijada y su fallo en n = 100 (K ≥ 12: 17 % de falsas alarmas
z ≤ −2; K = 6: 0 %) y se añade la lectura por régimen: en n = 1000 el test cumple la barra (potencia 1.00 a ratio ≤ 0.3,
0.90 a 0.6; 0 % de falsas alarmas en ambas direcciones), así que las lecturas de ImageNet (K = 30) quedan **certificadas dentro
del régimen validado**: profundidad por encima de las superclases WordNet en 4/12 backbones (ViT-S, ViT-B, ViT-L, DINOv2-L;
z de -2.4 a -4.0), no en los otros ocho, ninguno menos jerárquico que su estrella; CIFAR-100 (n = 100, K = 20: 3/12)
sigue sin validar. El texto dice que el acotado por régimen se adoptó **después** del barrido (§4 y A.5). Dominio de validez declarado:
n ≈ 1000, no validado por debajo de ~10 puntos por cluster (Limitación (i)). Actualizados: abstract, intro (ii), contribución 2, frase MERU
("its depth reading used the withdrawn isotropic star and is not part of the certified result"), frase NC ("answers yes for four ImageNet
backbones, three of them supervised ViTs"), Limitación (i), §7 (una cláusula), A.5 (párrafo "Real-data readings by regime"). Figuras: la
nueva figura del cuerpo es `fig_depth_main.pdf` ((a) ImageNet K = 30 real vs estrellas con z; (b) potencia n = 1000 y falsas alarmas
n = 100/1000) — queda numerada **Figura 4** por orden de aparición (la de texto pasa a ser la 5); `fig_depth_cifar100.pdf` (solo CIFAR-100)
en A.5; la figura de intervenciones (`fig3_causal.pdf`) pasa al apéndice, nueva subsección "Training interventions" (`app:interventions`),
con puntero de una línea en "The form tracks training". `phaseB_t4_apply.py` aplica todo esto desde expR56/expR55b
(`phaseB_depth_regime.json`); la rama estricta anterior sigue en `phaseB_t3_apply.py`.

**(2) §3, párrafo del null.** La construcción Haar descrita como registro ("random orthogonal coefficients recombined with the real
singular values, so the sample spectrum is exact") y la gaussiana entre paréntesis como alternativa reportada en A.14 (`app:robust`).
**(3)** Frase de agregación: "the same supremum over the same quadruple budget" → "the same statistic, the 99.9th percentile of the same
5×10⁵ sampled defects per seed". **(4)** Frase duplicada "A low raw value is not evidence." eliminada al final del párrafo homónimo;
también la cláusula obsoleta "a 99.9th-percentile variant preserves every sign" de los ajustes del estimador.

**(5) Barrido de clases bajo el registro (`expR60_c_sweep_record.py`, Haar × p99.9 × 200; mismos subconjuntos y semillas que exp19;
3 modelos de la tabla; 165 celdas, 67 min con 6 workers) → `expR60_c_sweep_record.csv`; NC adv copiado de exp19.** Viejo → nuevo:
"DINOv2-L at C = 50: excess −0.124 vs −0.093" (supremo) → -0.046 vs -0.042 bajo el registro (solo en la tabla; el
texto da los umbrales: los subconjuntos aleatorios llevan más exceso que los coherentes en CLIP-L a todo C, DINOv2-L hasta C = 100,
DINOv2-G hasta C = 20; más allá convergen). "Raw δ rises with C" → falso bajo el registro (δ̂₉₉.₉ plano en C: 0.030–0.034 para
DINOv2-L aleatorio); sustituido por "random subsets still score -0.08 to -0.14 at C = 10". "DINOv2-G reaches its null
at C ≥ 500 … reproducing the census exception" → "stays below its null at every C (excess -0.008 at C = 1000, r = 200)"
(bajo el supremo, +0.0035). §5: "The census exception is consistent with this…" → "This is consistent with the census: DINOv2's ImageNet
structure is genuine under the record and the least aligned with WordNet's top level". Tabla B3 regenerada (`gen_appendix2.py`, columna
"below" = semillas con p ≤ 0.05). La ventaja de explotabilidad (NC adv) de los aleatorios se mantiene en todo C.

**(6) DBpedia bajo el registro (`expR61_dbpedia_record.py`; embeddings re-extraídos como exp14, centroides cacheados en
`Platonic/results/text_cache/dbpedia_{m}.npz`) → `expR61_dbpedia_record.csv`.** −0.028 / −0.024 / −0.020 (supremo, null gaussiana,
3 réplicas) → -0.021 .. -0.017 (BGE / E5 / GTE), r = 200/200, p = 0.005, genuinos BH los tres; también Haar × sup y
Gauss × p99.9 en el CSV. Texto: "genuine excess (−0.020 to −0.028 under the original supremum reading)" → "beyond-null excess under the
record (−0.017 to −0.021, every replicate above the real value; Table A9)". Tabla A9 con las dos lecturas (CI95 fundido en la columna FS).

**Ajuste a 9 páginas.** Recortes: párrafo de profundidad compactado; "Negative curvature concentrates on bridges" fundido en el párrafo NC;
frases acortadas en §1 (calibración de similitud), §3 (panel, ORC k, resolución), §4 (extracción, anisotropía, coseno), §5 (WordNet, DBpedia,
Gröger), §7 (disociación, cláusula de profundidad); alturas: Fig. 1 placeholder 0.3 in, overview 1.5 in, panel de exceso 1.45 in, profundidad
1.5 in, text-nulls 0.75\linewidth, tree map 0.92\linewidth. Limitaciones: solo (i) editada (dominio de validez).

**Verificación.** `sweep_freeze.py`: checks nuevos de régimen (potencia/falsas alarmas por n, hits de ImageNet por nombre y z), de expR60
(umbrales por modelo, C = 10, DINOv2-G bajo el null a todo C, NC adv) y de expR61 (rango y r): **103/103 PASS**. Compilación: 33 páginas,
0 warnings, 0 overfull, texto principal termina en la p. 9; ningún nombre de script en el PDF. `main_iclr2027_final.pdf` exportado.

## 17. Hueco para la Figura 1 real (1.2 in)

Placeholder de la Fig. 1 a **1.2 in** (antes 0.3 in tras los recortes de la pasada final). Para que el texto principal vuelva a terminar
en la p. 9 con ese hueco: (a) el párrafo de §3 "Class count is a measured variable, not a nuisance" se funde en el de §4 "Hierarchy depth,
not class count" (mismo control; el párrafo de §4 abre ahora con la descripción del barrido: C ∈ {10, …, 1000}, composiciones aleatoria y
WordNet-coherente); (b) "Local agreement survives calibration" recortado en dos líneas (kNN, CKA y lectura Poincaré en una frase; ningún
número cambia). (c) No hizo falta reducir la Fig. 3 (sigue a 1.45 in); Figs. 1 y 4 intactas. Comprobado: texto principal termina en la p. 9
con el placeholder de 1.2 in; 33 páginas, 0 warnings; sweep 103/103 (sin números nuevos). PDF exportado.

## 18a. Reestructuración — memo de los reruns R7 y R8 (antes de tocar el texto)

**R7 — lectura a nivel de muestra bajo el registro** (`expR62_samplelevel_record.py` → `expR62_samplelevel_record.csv`; mismos subconjuntos que
expR37: CIFAR-100 10 img/clase, DTD 22 img/clase, semilla 0; Haar × p99.9 × 200, BH sobre 24 celdas; supremo crudo de la misma nube al lado).
Supremo crudo 0.087–0.145 (expR37: 0.087–0.145); δ̂₉₉.₉ 0.036–0.093;
exceso -0.0232..+0.0058, signo negativo en 18/24; |z| < 2 en 12/24; **genuinas BH 10/24**
(DINOv2-S/CIFAR-100, DINOv2-B/CIFAR-100, DINOv2-L/CIFAR-100, DINOv2-G/CIFAR-100, SigLIP-B/CIFAR-100, ViT-S/DTD, ViT-B/DTD, ViT-L/DTD, DINOv2-L/DTD, DINOv2-G/DTD); dentro del ruido (no genuinas) 14/24.

**R8 — control MERU bajo el registro** (`expR63_meru_record.py` → `expR63_meru_record.csv`; 6 modelos × 2 datasets × 2 modalidades; censo Haar ×
p99.9 × 200 con BH sobre 24 celdas; δ̂₉₉.₉ nativo (Lorentz/angular) vs euclídeo; test de profundidad anisotrópico, 10 semillas: ImageNet K = 30
validado, CIFAR-100 K = 20 no validado). Imágenes de ImageNet: exceso MERU -0.0101..-0.0075 vs CLIP
-0.0100..-0.0064 (r = 200 en todas: False); |nativo − euclídeo| ≤ 0.0002;
profundidad z MERU -1.54..-0.52 vs CLIP -1.85..-0.99;
celdas con z ≤ −2 en ImageNet: ninguna; MERU con profundidad más allá de su gemelo: ninguno.
Genuinas BH en total 12/24.

**Condiciones de parada del brief.** (a) exceso a nivel de muestra ya no dentro del ruido: no; (b) MERU con profundidad
más allá del gemelo: no. → GO: la lectura cualitativa se mantiene; sigue la fase de texto.
Checks en `sweep_freeze.py` (sección R7/R8) contra `phaseC_memo.json`.

## 18. Reestructuración: una tesis, tres actos (fase de texto)

Sin cambios en censo, test de profundidad, tree map ni texto; cambia el orden y la historia. Dos resultados suben del apéndice al cuerpo y por eso se
rehicieron bajo el registro (memo en §18a). Decisiones del autor aplicadas: subtítulo sin cambiar (pendiente); nivel de muestra al cuerpo (R7);
MERU al abstract (R8); "tool" eliminado del texto (el script queda en el repositorio y en una frase del Reproducibility Statement); corolario en el
apéndice, §6 lleva solo su consecuencia. Scripts: `rebuttal/scripts/phaseC_main_body.tex.tmpl` (cuerpo nuevo con marcadores) y
`phaseC_restructure.py` (rellena desde los ficheros, empalma, mueve los párrafos al apéndice, decide la Fig. 2(b), comprueba etiquetas y palabras
prohibidas); el cuerpo anterior queda en `rebuttal/results/phaseC_old_main_body.tex`.

**Mapa de secciones (párrafo antiguo → destino).**
- Abstract, caja, §1 (4 párrafos + 4 contribuciones): reescritos según el brief; "What is new, in order" → §2(b).
- §2 antiguo (5 párrafos) → 3: (a) "Latent hyperbolicity, the target claim" (hyperbolic DL + lecturas de latent hyperbolicity); (b) "Calibrating δ and
  geometry against nulls" (network science + Kornblith/Ansuini/Pope/NC + PRH/Koepke/Gröger/Park + qué es nuevo); (c) "Background" (cuatro puntos, ln 2,
  δ_norm, orden, caveats de Fournier, ORC en una frase con puntero a A.16).
- §3: "Classes are centroids…" → "Where the premise is read, and where hierarchy is measured" (misma prosa + frase nivel de muestra/nivel de clase);
  "A low raw value…", "The reading is an excess…" (sin la frase "The instrument is a single script"); nuevo "Projection and tasks" (dos frases, puntero A.10);
  párrafo ξ → A.21; "Class count…" ya estaba fundido en §4 → ahora A.7.
- §4 (7 párrafos): 1 "Sample-level readings sit within null noise in most cells" (nuevo, R7); 2 censo (antiguo "Vision: beyond-null…", + puntero al
  control de clases en A.7); 3 "The excess certifies clustering, not depth" (sin la frase MERU); 4 profundidad (sin cambios); 5 "Imposing the geometry
  does not create depth" (nuevo, R8; sustituye la frase MERU y la mención B30); 6 NC (−2 líneas: cláusula ORC → puntero); 7 "Scale and training"
  (afirmación de escala DINOv2 + una frase de intervenciones con puntero A.6). Salen de §4: "The angular reading agrees" → §5.3; "Text: recipe and
  scale…" → §5.6 (8 líneas); "Hierarchy depth, not class count" → A.7 (`app:csweep`); "The form tracks training" → A.6 (`app:interventions`);
  figura text-nulls → apéndice "Text census" (`app:textcensus`).
- §5 (7 párrafos): 1 isla naive (fundido: extracción, mapa, cortes degenerados, criterio); 2 "The island is an artifact; sharing is graded" (tres medidas,
  una frase cada una; cierre total bajo tripletes angulares); 3 "The self-supervised tree lives in angles" (nuevo párrafo unificado: tripletes cos/eucl
  DINOv2-L + censo coseno 56/72, 22/24, 65/72 + mecanismo de la isla; puente a §6); 4 WordNet (+ ARI 0.61; anti-alineación como última frase); 5 DBpedia
  (+ los tres controles de circularidad); 6 texto (movido de §4); 7 acuerdo local (5 líneas). HierarCaps → A.19 (`app:hierarcaps`).
- §6 "Consequences for imposing curvature" (nuevo, 3 párrafos): regla de Khrulkov, "measure before imposing", métrica de coste cero. Sustituye a
  "A corollary, in the appendix".
- §7: "What the paper establishes" (caja + tres actos), "Open questions" (nuevo), Limitaciones (i)–(vii) literales + (viii).
- Apéndice recibe, con puntero de una línea desde el cuerpo: ξ (A.21), intervenciones (A.6), ORC (A.16), censo coseno (B32), medidas sin corte (B33),
  corolario (A.10), barrido de clases (A.7, párrafo + B3), HierarCaps (A.19), 2×2 (B21), potencia (B35), A.18 (B17 reescrita desde R7), texto (frases
  de extracción/anisotropía/GTE-Qwen2/plantillas + figura text-nulls).

**Números que suben del apéndice al cuerpo (fuente).** Nivel de muestra (`expR62_samplelevel_record.csv`): supremo crudo 0.087–0.145;
no genuinas 14/24, genuinas 10/24 (DINOv2-S/CIFAR-100, DINOv2-B/CIFAR-100, DINOv2-L/CIFAR-100, DINOv2-G/CIFAR-100, SigLIP-B/CIFAR-100, ViT-S/DTD, ViT-B/DTD, ViT-L/DTD, DINOv2-L/DTD, DINOv2-G/DTD); exceso genuino hasta -0.023; columnas nuevas de la Tabla 1
(exc. C100 / exc. DTD a nivel de muestra, BH sobre 24). MERU (`expR63_meru_record.csv`, imágenes de ImageNet): exceso MERU -0.007..-0.010 vs
CLIP -0.006..-0.010; |nativo − euclídeo| ≤ 0.0002; z de profundidad MERU -1.5..-0.5, CLIP -1.8..-1.0.
Censo coseno (`expR57_census_cosine_haar_p999_200.csv`): 56/72, 22/24, acuerdo 65/72 (ya estaba en §4; ahora en §5.3). Ganancias
(`exp2b_normalized_stack.csv`, FS, Poincaré − coseno tras L2, conjuntos jerárquicos): contrastivos +0.1..+1.3 pp; resto −1.7..+1.1 pp. Regla de Khrulkov
(verificada en el PDF, ec. 5: c = (0.144/δ_rel)², δ_rel = 2δ/diam, c ≈ 0.33 en sus datos): sobre la banda gaussiana de `exp1_delta_controls.csv`
(δ_norm 0.104 en d = 192, 0.046 en d = 1536) da c de 0.48 a 2.5. Rango de exceso ImageNet a nivel de clase (`expR52`): −0.005..−0.020.

**Frases eliminadas del cuerpo y destino.** "The instrument is a single script…" (§3) → eliminada (queda la frase del Reproducibility Statement).
"Comparing the resulting trees…" (§1) → §1 párrafo 3, reformulada. Párrafo "Geometric analyses of representations" → fundido en §2(b). Párrafo
"Representational convergence" → fundido en §2(b). Frase "Those readings are sample-level and uncalibrated; transplanted…" (§2) → §2(a), reformulada.
Frase MERU de "The excess certifies…" → §4.5. Cláusula ORC del párrafo NC → puntero. Frase "Whether there is hierarchy above…" → §4.3 reformulada.
"Hierarchy depth, not class count" → A.7. "The form tracks training" → A.6 (una frase resumida queda en §4.7). Frases de extracción, anisotropía,
GTE-Qwen2 y plantillas del párrafo de texto → A "Text census". Frase HierarCaps → A.19. Controles de circularidad → §5.5. "A corollary, in the
appendix" → §6.3. "What the census establishes" → "What the paper establishes" (reescrito). Figura text-nulls → apéndice.

**Figura 2(b).** Decidido por los números: par a nivel de muestra ViT-T/DTD (imágenes; crudo 0.0873, exceso -0.0004) vs
SigLIP-B/CIFAR-10 (centroides; crudo 0.0874, exceso -0.050), diferencia de crudo 6.4e-05 frente a 2.4e-04 del par BGE/ViT-L;
`phaseC_fig2b.json` lo registra y `make_fig_overview.py` lo lee. (a) y (c) sin cambios.

**Lo que no se hizo y por qué.** Survey de Mettes et al. (IJCV 2024): el texto accesible (arXiv 2305.06611) define la hiperbolicidad de Gromov como
propiedad de espacios pero no enuncia la premisa de que las redes estándar sean latentemente hiperbólicas → no se cita, como pedía el brief.
Subtítulo: sin cambiar hasta decisión del autor.

**Matices respecto al borrador del brief.** (1) A nivel de muestra no son "dos o tres" celdas genuinas sino 10/24 (la familia DINOv2 en
CIFAR-100 y DTD, SigLIP-B en CIFAR-100 y los ViT S/B/L en DTD), con excesos pequeños (≥ -0.023); el abstract, §1, §4.1, §6.1 y §7 dicen
"dentro del ruido en la mayoría de las celdas; pequeño donde sobrevive" en lugar de "dentro del ruido". La condición de parada (a) no se disparó
(umbral fijado en > 12/24). (2) Las ganancias Poincaré − coseno no son "nada para el resto": DINOv2-S en CIFAR-10 da +1.1 pp y ViT-L en DTD +1.0 pp
(y DINOv2-G −1.7); §6.3 dice "inconsistente para el resto (−1.7 a +1.1 pp)". (3) Las celdas MERU/CLIP de imágenes de ImageNet no son todas genuinas
BH (r 171–197); el párrafo habla de "the same clustering excess" sin llamarlas genuinas.

**Verificación.** `sweep_freeze.py`: sección R7/R8 (4 checks) y sección de texto reestructurado (10 checks: frases con recuentos/rangos/ganancias/
curvatura, decisión Fig. 2(b), sin "tool"/"we believe"/"interestingly"): **116/116 PASS**. Compilación: 34 páginas, 0 warnings, 0 overfull, texto
principal termina en la p. 9 con el hueco de 1.2 in de la Fig. 1 (Ethics abre en la p. 9); sin nombres de scripts en el PDF. PDF exportado.
Entregables: `ICLR2027/REVIEWER_CHECKLIST_restructure.md`, `ICLR2027/SUMMARY_plain.md`.

## 19. Título

Decisión del autor: `\title{Is There a Platonic Tree? \\ Calibrating Latent Hyperbolicity in Foundation Models}` (antes "Calibrated Class Geometry in
Foundation Models"). Recompilado: 34 páginas, texto principal en la p. 9, 0 warnings; sweep sin cambios. PDF exportado.

## 20. Pasada de prosa (estilo Gröger et al.) — la §19 del brief; la 19 de este fichero ya era el título

Cambia cómo se lee, no lo que dice: estructura, secciones, figuras, tablas, números y afirmaciones congelados. Se reescribe la prosa del texto
principal (abstract, caja, §1–§7, pies de figura) con el vocabulario fijo del brief y las reglas de números, ritmo y frase. Scripts:
`rebuttal/scripts/phaseD_main_body.tex.tmpl` (cuerpo nuevo con marcadores para los números que quedan en la prosa) y `phaseD_prose.py`
(rellena desde los ficheros de resultados, empalma, aplica los renombrados del apéndice y da casa en el apéndice a los números que no la
tenían); el cuerpo anterior queda en `rebuttal/results/phaseD_old_main_body.tex`; `phaseD_changelog.py` lista los números que salieron de la
prosa y dónde están.

**Vocabulario fijo aplicado (renombrado en cuerpo, pies, títulos de figura y apéndice).** the premise (latent hyperbolicity) · the raw reading
(δ̂₉₉.₉ sin calibrar; sustituye a "the reading of record") · the instrument (nunca tool/method/approach) · the excess · clustered structure
(sustituye a "beyond-null (clustered) structure", "tree-like structure", "form": Tabla 1, Fig. 3 y su título de panel `make_figs.py`, A.6
"Clustered structure tracks training", pie de la figura text-nulls, párrafo A.7) · genuine (definido una vez en §3) · the star caveat · the
depth test ("certified" solo para sus veredictos positivos) · the island · the angular tree · la tesis, literal cinco veces (abstract, caja,
final de §1, final de §5, §7). Ideas bautizadas una vez y reutilizadas: dimension / spectrum / statistic confound (§1 → §3, §6).

**Mapa párrafo a párrafo (viejo → nuevo).** Abstract: mismo contenido, una cifra por frase, cierra con la tesis. Caja: la tesis literal.
§1: P1 premisa y quién la cita (citas al final de frase, no encadenadas) → pregunta; P2 "The raw reading is confounded" (bautiza los tres
confounds); P3 el instrumento y la pregunta del título; P4 "The answer has three parts" (49/72, 4/12; cierra con la tesis); contribuciones
sin números; puente a §2. §2: "The premise" / "Calibrating geometry against nulls" / "Background", mismas citas, puente a §3.
§3: apertura en tensión (dos frases) + "The premise lives at the sample level; hierarchy lives at the class level" (antes "Where the premise
is read…"; sin los tamaños 80–6,000, ahora en la nota del panel de modelos) / "A low raw reading is not evidence" (esferas 0.14–0.18 →
Tabla 5; queda el par de extremos gaussianos) / "The reading is an excess over a matched null" (define excess, rank, genuine, clustered
structure; sin la frase del script) / "Two choices changed after pre-registration" (nuevo párrafo, separado del anterior) / "Projection and
tasks"; puente a §4. §4: apertura en tensión; 4.1 "The premise does not survive calibration" (antes "Sample-level readings sit within null
noise in most cells"; 14/24 queda, banda/nombres/máximo → pie B17); 4.2 "Clustered structure is the rule" (el ejemplo del brief, + 18/24;
70/72 → título del panel Fig. 3a, 0.001 → pie B36, lista de celdas no genuinas → marcas ° de la Tabla 1); 4.3 "A star passes the census"
(−0.11 → B25); 4.4 "Depth is certified on ImageNet and not on CIFAR-100" (17 % y 4/12 quedan; barra, potencias, z, +6.8, 16 % → pies
B35/B34); 4.5 "Imposing the geometry does not create depth" (rangos → pie B30); 4.6 "Neural collapse is the flat limit, not the reading"
(−2 líneas); 4.7 "Scale deepens the excess for one family only" (−0.071→−0.130 etc. → Tabla 1 columnas C10/C100; 19–91 % → A.6); cierre
puente a §5. §5: 5.1 "The naive map manufactures an island" (0.82, ≤0.06, 97 % → pie B7; 0.13–0.38 → prosa A.9); 5.2 "The island is an
artifact, and sharing is graded" (0.38 vs 0.48 y 11/12 quedan; cofenética/tripletes → B33; 0.2–0.5 → pie B11); 5.3 "The self-supervised
tree lives in angles" (65/72 queda; 0.91/0.71 → pie B32; 56/72, 22/24 → pie B32); 5.4 "WordNet alignment follows supervision, not
tree-likeness" (0.61 queda; ρ por familia → columna ρ_WN de la Tabla 1); 5.5 "The recovery is not WordNet circularity" (11/12 y +0.26
quedan; 219, 70-way, 0.21–0.29, 0.65–0.82 → pie A9); 5.6 "In text, clustered structure depends on recipe, scale and probe" (rangos y p →
A9/B23); 5.7 "Local agreement survives calibration" (66 queda; 0.472/0.464/0.633/0.577/0.108/0.425/0.640/1/201/10/999 → B31); cierra con
la tesis. §6: apertura en tensión; 6.1 "The raw reading cannot select a curvature" (regla de Khrulkov, 0.48→2.5 y −0.005..−0.020 quedan;
c≈0.33 → A.10 "A diagnostic, not a policy"; −0.023 → B17); 6.2 "Measure before imposing"; 6.3 "A zero-cost readout collects what is
there" (+0.1..+1.3 y −1.7..+1.1 quedan). §7: tesis + tres actos (49/72, 18/24, 4/12), "Open questions", Limitaciones (i)–(viii) sin cifras
(n≈1000 y "ten points" → "class sets of ImageNet's size"; 0.018 → "of the order of the fragile excesses", vive en B28; "six datasets" →
"the datasets of Table 1"). Pies de figura: primera frase en negrita = conclusión, segunda = qué se dibuja (Figs. 1–5).

**Números que salieron de la prosa y dónde viven ahora** (`phaseD_changelog.py` los contrasta; el check del sweep exige que ninguno
desaparezca). Tabla 1: ρ_WN por familia (+0.57..+0.59, +0.49..+0.53, +0.36, +0.18..+0.22), C10/C100 de DINOv2 (−0.071→−0.130,
−0.031→−0.040), exceso IN (−0.005..−0.020), 10/24 genuinas a nivel de muestra. Fig. 3a (título): 70/72. B17: banda 0.087–0.145, 14/24,
nombres de las 10 genuinas, −0.023. B25: estrella de 30 clusters −0.103/−0.104 (la prosa decía ≈−0.11). B30: MERU −0.007..−0.010, CLIP
−0.006..−0.010, 0.0002, z −1.5..−0.5 / −1.8..−1.0. B34: z −2.4..−4.0 de los cuatro certificados, +6.8 iso. B35: barra (0.8, 0.3, 5 %),
1.00/0.90, 0 %, 17 %, marco superior 0.00 y 16 %. B36: 0.0011. B7: 0.82 (ViT-T vs CLIP-B), ≤0.06, 97 %. B11: 11/12 emparejados, 0.19–0.52.
B32: 56/72, 22/24, 65/72, tripletes 0.91/0.71. B33: 0.48/0.78, 0.77/0.76, 0.47/0.74, 0.36/0.80. A9: 219, 70-way, ≤10 %, 0.21–0.29,
0.65–0.82, −0.017..−0.021. B23: +0.002..+0.004, p 0.02/0.15. B31: todos los de acuerdo local. A.6: 19–91 %. A.9 (prosa): 0.13–0.38, 50 %.
A.10 (prosa): c≈0.33. A.3 (nota del panel): 100 imágenes/clase, 80–6,000. Tabla 5: esferas 0.14–0.18. B28: 0.018.

**Verificación.** `sweep_freeze.py`: los checks numéricos no cambian; los checks de texto de la §18 se reanclan a los pies B17/B30
(donde viven ahora los rangos) y se añaden tres checks de prosa: tesis literal ×5; métricas por párrafo (≤2 grupos numéricos por párrafo y
≤1 por frase en §4–§6, "X of Y" / "from A at d=B to C at d=D" cuentan como un grupo, dígitos pegados a nombres (CIFAR-100, GPT-2, DINOv2,
L2) y fórmulas con "=" no cuentan; solo 49/72, 18/24, 4/12 en §1 y §7; ≤1 paréntesis por párrafo, (i)–(viii) exentos; sin punto y coma en
§4–§6 salvo la tesis; sin tool/beyond-null/tree-like structure/form/reading of record/our approach/the method/essentially/largely/
substantially/somewhat/interestingly/notably/importantly/we believe/we note), imprime la frase infractora; conservación de números (todo
literal decimal del cuerpo anterior sigue en cuerpo, apéndice o tablas, con tolerancia de redondeo; única excepción declarada: ≈−0.11 →
−0.103 en B25). **119/119 PASS.** Compilación: 35 páginas, 0 warnings, texto principal en la p. 9 con el hueco de 1.2 in de la Fig. 1;
alturas: Fig. 2 1.35 in, Fig. 3 1.28 in, Fig. 5 0.86\linewidth (Figs. 1 y 4 intactas). `SUMMARY_plain.md` regenerado del texto nuevo.

## 21. Carpeta renombrada a `ICLR2027/`

Decisión del autor (2026-09-08) tras dos syncs de Overleaf que renombraban la carpeta y provocaban conflictos: la carpeta exterior pasa de
`iclr2027/` a `ICLR2027/` en el repo (`git mv`); la carpeta interior del paper sigue siendo `ICLR2027/iclr2027/`. Actualizadas todas las
referencias a la ruta exterior en `rebuttal/scripts/*.py` (sweep, phase*, expR55/55b/60/61/62/63, memos), `ICLR2027/iclr2027/fig_text_nulls.py`,
`CODE_MAP.md` y los .md de la carpeta; las referencias interiores (`HERE.parent/"iclr2027"/"figures"` en los scripts de figuras) no cambian.
Verificado tras el renombrado: generadores de tablas y figuras, sweep 119/119, compilación (35 páginas, texto principal en la p. 9, 0 warnings),
PDF exportado a `ICLR2027/main_iclr2027_final.pdf`. Ninguna línea de contenido cambia.

## 22. Cuatro arreglos de prosa (sin cambio de contenido)

(1) §1, párrafo 3: la frase encadenada se divide en "…what survives calibration, and is it shared? In the words of the title: is there a
Platonic tree?", con las dos citas tras "convergence work". (2) §1, párrafo 2: "(Gromov, 1987)" pasa a seguir a "Gromov δ assigns zero to
trees". (3) §4.1: tras "concentrated in the self-supervised family" se añade "and that family's structure is the angular tree of Section 5,
which cosine collects". (4) §2 "Calibrating geometry against nulls": la frase de network science se reescribe según lo que afirman los tres
trabajos, comprobados en sus textos: Narayan & Saniee (2011) y Kennedy et al. (2013) comparan la curvatura (δ) de redes de comunicación y
sociales con retículos, rejillas hiperbólicas y grafos aleatorios de Erdős–Rényi de tamaño comparable, leyendo δ respecto al diámetro del
grafo (y las redes de carreteras como control negativo); Adcock et al. (2013) miden δ junto con descomposiciones en árbol y concluyen que el
valor de δ a secas, "the simplest and most popular metrics", no basta para caracterizar la estructura arbórea. Se retira la atribución
"found long ago that the raw reading does not certify tree-likeness"; queda "Reading δ against a reference is therefore not new". Recompilado:
texto principal en la p. 9, 35 páginas, 0 warnings; sweep sin cambios, 119/119. PDF exportado.

## 23a. Positive-control pass — memo de la Fase A (sin prosa; `ICLR2027/MEMO_positive_control.md`)

**R9 — profundidad implantada en centroides reales de ImageNet** (`expR64_implanted_depth.py` → `expR64_implanted_depth.csv`, `_summary.csv`,
`fig_implanted_depth.pdf`; 12 backbones × s ∈ {0, 0.25, 0.5, 0.75, 1} × 5 semillas de implante = 300 tests de profundidad con el
protocolo exacto de B34, más censo Haar × p99.9 × 200 en la semilla 0; partición rand6 = 6 super-hubs de 5 (RandomState(0)); el 6-corte
WordNet existe y está anidado pero es 10/5/11/2/1/1, corrido como `wn6` para la semilla 0). Falsas alarmas a s = 0: **0** backbones
(z máx. +0.50). Potencia por s: s=0.0: 0.00, s=0.25: 0.00, s=0.5: 0.00, s=0.75: 0.10, s=1.0: 0.32. s* (primer s con z ≤ −2 en ≥ 4/5 semillas): i21k_t None, i21k_s None, i21k_b None, i21k_l 1.0, dinov1_b None, dinov2_s None, dinov2_b None, dinov2_l None, dinov2_g None, clip_b None, clip_l 0.75, siglip_b 1.0.
Censo sobre las mismas nubes: dispersión máxima entre s de 0.0076 (≤ 1.85 s.d. del null), r mínimo 172/200. Nube real por el
mismo código = B34 (|Δz| máx. 0.00). Cociente intra/entre de las nubes reales a K = 30 (el barrido sintético cubrió 0.1–0.6):
i21k_t 1.55, i21k_s 1.93, i21k_b 1.96, i21k_l 1.88, dinov1_b 2.12, dinov2_s 3.0, dinov2_b 3.71, dinov2_l 3.88, dinov2_g 3.92, clip_b 1.33, clip_l 1.49, siglip_b 1.54. Tamaños de cluster del marco WordNet-30: [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 10, 11, 12, 13, 16, 17, 18, 22, 23, 31, 67, 71, 158, 175, 316].
**Variante tight** (offsets reales encogidos a cociente 0.6, mismo implante, s ∈ {0, 0.5, 1}, 2 semillas; `--tight`, chain r14): i21k_t z(s=0/0.5/1) -0.6/-4.7/-10.2, s=0 hits 0/2; i21k_s z(s=0/0.5/1) -0.2/-7.0/-10.5, s=0 hits 0/2; i21k_b z(s=0/0.5/1) -1.0/-7.7/-11.1, s=0 hits 0/2; i21k_l z(s=0/0.5/1) -1.6/-8.7/-12.3, s=0 hits 0/2; dinov1_b z(s=0/0.5/1) -1.6/-9.4/-11.4, s=0 hits 0/2; dinov2_s z(s=0/0.5/1) -0.2/-6.6/-11.9, s=0 hits 0/2; dinov2_b z(s=0/0.5/1) -0.0/-8.5/-11.0, s=0 hits 0/2; dinov2_l z(s=0/0.5/1) +0.2/-7.4/-10.9, s=0 hits 0/2; dinov2_g z(s=0/0.5/1) +0.2/-9.1/-10.7, s=0 hits 0/2; clip_b z(s=0/0.5/1) -1.7/-6.5/-12.5, s=0 hits 0/2; clip_l z(s=0/0.5/1) -2.4/-9.8/-13.4, s=0 hits 2/2; siglip_b z(s=0/0.5/1) -2.1/-8.9/-11.8, s=0 hits 2/2.

**R11 — sensibilidad conjunta** (`expR66_joint_sensitivity.py` → `expR66_joint_sensitivity{,_summary}.csv`; 72 celdas × 30 remuestreos × 50 réplicas
Haar). Registro 49/72, 18/24. z_joint ≤ −2: **42/72**, 18/24 (caen: CLIP-B/cifar10, CLIP-L/cifar10, ViT-T/dtd, DINOv2-L/fashionmnist, ViT-S/fashionmnist, SigLIP-B/fashionmnist, SigLIP-B/mnist). BH por remuestreo, genuina en ≥ 27/30:
**47/72**, 17/24 (caen: CLIP-B/cifar10, ViT-T/dtd, CLIP-L/imagenet; entra: DINOv2-S/fashionmnist). s.d. bootstrap máx. 0.0026 (0.0011 en ImageNet).

**R10 — ViT-B/16 fine-tuned** (`expR65_hier_finetune.py` → `expR65_hier_finetune.csv`, `expR65_train_log.csv`; subconjunto del censo de 100 img/clase,
2 épocas, mismos batches para (a) CE y (b) CE + CE jerárquica sobre el 30-corte WordNet; el train completo no es legible a velocidad de
entrenamiento desde el disco): frozen exceso -0.0162 z -3.62; CE -0.0153 z -2.33; jerárquico -0.0162 z -2.77.
Criterio del brief (b certificado y a no, o z de b claramente por debajo de a, umbral 1): **False**.

Checks en `sweep_freeze.py` (sección positive-control) contra `positive_control_memo.json`. Sin ediciones de prosa.
