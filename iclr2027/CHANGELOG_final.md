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
  p10 abre con Reproducibility Statement. PDF: `iclr2027/main_iclr2027_final.pdf`.
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
  p9 (1); p10 abre con "R EPRODUCIBILITY S TATEMENT". PDF actualizado en `iclr2027/main_iclr2027_final.pdf`.

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
- Compilación: 30 páginas, warnings = 0 (Overfull de B20 corregido con tabcolsep 1.8 pt), `??` = 0; Ethics en p9; p10 abre con Reproducibility Statement. PDF: `iclr2027/main_iclr2027_final.pdf`.

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
4. **El instrumento como herramienta**: `iclr2027/tool/calibrated_delta.py` (entrada: matriz de centroides n×d `.npy` + etiquetas de superclase
   opcionales; salida: δ_norm cruda, exceso espectral con rango r/200 y p de cola izquierda, y el test de profundidad con estrella emparejada de
   B29; funciones y semillas copiadas verbatim de `expR39c_census200_cache.py` y `expR50_depth_test.py`). `iclr2027/tool/run_checks.py`
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
  (Limitations incluidas) termina en p9 y el Ethics Statement abre p10. PDF: `iclr2027/main_iclr2027_final.pdf`.

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
6. **`iclr2027/tool/demo.py`** (n = 60, d = 32, 50 réplicas, semillas fijas): nube aleatoria → exceso −0.014, r 47/50, p 0.078 → no genuino;
   estrella de 6 clusters → exceso −0.054, p 0.020 → genuino, profundidad +0.039 (z +0.9) → sin profundidad; jerarquía 6×5 → exceso −0.115,
   p 0.020 → genuino, profundidad −0.025 (z −0.6) → más arbóreo que su estrella. A n = 60 con K = 6 hubs el test de profundidad está
   infra-potenciado (probé cuatro parametrizaciones; |z| ≤ 0.6 en todas), así que el demo y el README leen el signo de la profundidad y remiten
   a B25/B29 (n = 100–1000) para la calibración. `tool/README.md`: uso, check de reproducción de dos celdas, demo.
- Verificación: `sweep_freeze.py` **73/73 PASS**; compila con 30 páginas, 0 warnings, 0 `??`; el texto principal (Limitations incluidas)
  termina en p9 y el Ethics Statement abre p10. PDF: `iclr2027/main_iclr2027_final.pdf`.
