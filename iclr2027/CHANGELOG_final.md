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
