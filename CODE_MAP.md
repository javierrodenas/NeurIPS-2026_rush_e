# CODE_MAP — qué hay en este repo y de dónde sale cada número

Actualizado: 2026-09-03. Dos "papers" conviven aquí: la sumisión **NeurIPS 2026** (raíz, congelada,
decisión 2026-09-24) y la resumisión **ICLR 2027** (`ICLR2027/`, activa). Todo número del paper ICLR
sale de un CSV en `rebuttal/results/` vía un generador; nada se teclea a mano.

## Raíz (versión NeurIPS 2026 — congelada)

- `main.tex`, `neurips_2026.tex`, `neurips_2026.sty`, `references.bib`, `checklist.tex` — la sumisión 18165 tal como se envió. No tocar.
- `tab_breadth.tex`, `tab_full_results.tex`, `tab_orc_k.tex` — tablas de esa versión.
- `figures/`, `figures_archive/` — figuras de esa versión y archivo histórico.
- `codigo/` — backup del código pre-rebuttal (ver su `README.md`): extracción de features, ORC, tareas
  downstream, plots antiguos; `codigo/experiments/training/` = cabezas Hyperbolic/Cosine/Linear (no en el paper).
- `rebuttal/` (fuera de `scripts/` y `results/`) — respuestas OpenReview a los 4 revisores de NeurIPS,
  estrategia y checklist de cobertura.

## El paper ICLR 2027 (`ICLR2027/`)

- **`ICLR2027/iclr2027/main_iclr2027.tex`** — el draft. Título actual: *Is There a Platonic Tree?
  Calibrated Class Geometry in Foundation Models*. 9 págs. de texto principal + apéndice.
- Estilos ICLR: `iclr2027_conference.{sty,bst}`, `math_commands.tex`, `fancyhdr.sty`, `natbib.sty`.
- `references.bib` — bibliografía de esta versión (no confundir con el de la raíz).
- `archive_draft_v0.tex` — borrador inicial, solo histórico.
- **Compilar**: Overleaf (vía normal). En local no hay TeX: instalar `tectonic` con conda en un
  directorio scratch, copiar `ICLR2027/iclr2027/` allí y compilar (los artefactos de build no van al repo).
- Documentos de trabajo en `ICLR2027/`: **`REVIEW_31ago.md`** (log completo de las 10 rondas de
  revisión simulada + experimentos expR29–51 — la fuente de verdad de qué se hizo y por qué),
  `PLAN.md` (fechas/reglas ICLR), `COAUTHOR_TASKS.md`, `ATTACK_PLAN.md`, `MORNING_REPORT.md`,
  `HEADLINE_CHANGES.md`, `fig3_recaption.md`.

### Generadores de tablas (en `ICLR2027/iclr2027/`)

| script | produce | fuentes |
|---|---|---|
| `gen_main_table.py` | `tab_census.tex` (Tabla 1, censo visión 12×6) | `exp20_null_ztable.csv`, `expR39_census20.csv`, `expR40`, `exp2`, `exp22` |
| `gen_appendix.py` | `appendix_tables/tab_a*.tex` (A0–A10: modelos, excess, NC/FS, cos, retrieval, McNemar, prompts, HierarCaps, DBpedia, curvatura) | CSVs `exp1–exp27` |
| `gen_appendix2.py` | tablas B tempranas (B1–B9: text nulls, z-table, C-sweep, CIs, bridges, treemap, val-pick, políticas NC) | CSVs de la reestructuración |
| `gen_review_tables.py` | **B11–B30**: recovery, familycorr, flatnull, templates (B14, prefiere expR49 sin padding), robust, kmhubs, sample-level, p999, ξ-geo, census20, p999census, flatnull-p999, **B23 censo texto canónico** (prefiere expR48), extra backbones, **B25 calibración estrella**, ξ-norm, variantes de null, **B28 extracción/padding**, **B29 test de profundidad**, **B30 control MERU** | CSVs `expR29–expR51` |
| `gen_provenance.py` | post-proceso: quita `Source(s): ...` de las leyendas generadas | — |
| `fig_text_nulls.py` | `figures/fig_text_nulls.pdf` (censo texto con barras de error) | `expR48_text_census_bs1.csv` |
| `sweep_freeze.py` (en `rebuttal/scripts/`) | verificador: coteja los números de la prosa del .tex contra los CSVs | todos |

### Figuras (en `ICLR2027/figures/`, se copian a `ICLR2027/iclr2027/figures/`)

- `make_figs.py` — Fig. panel de excesos (a/b) y scatter best-metric (`exp20`, `exp2`).
- `make_treemap_fig.py` — Fig. tree map naive vs criterio (`exp23`, `exp22`).
- `make_fig3_causal.py` — Fig. 3, 2 paneles: fine-tuning no jerárquico y profundidad por capas.
- `fig_expmap.py` — esquema de la proyección de Poincaré (diagrama, sin datos).
- La Fig. 6 (proyección 2-D CIFAR-100) **no tiene script en el repo**; el nodo "Reality" está tapado
  con cajas overpic ("root") en el .tex — regenerar la figura de origen (tarea de coautores).

## Experimentos (`rebuttal/scripts/` → `rebuttal/results/`)

Convención: `<script>.py` escribe `<mismo nombre>.csv` (+ `.log`) en `rebuttal/results/` (117 ficheros).
`exp1–exp27` = rebuttal NeurIPS y primeras rondas ICLR; `expR29–expR51` = rondas de revisión simulada
(no existen exp25, exp28 ni expR35). Una línea por script:

**Fundamentos del instrumento**
- `exp1_delta_controls` δ con controles esférico/euclídeo/whitening; `exp1b_dim_matched` δ a dimensión igualada (PCA); `exp1c_spectrum_null` la null espectral original.
- `exp6_h2_sphere_check` δ teórica de H² y esferas; `exp19_c_sweep` deconfound nº clases vs jerarquía; `exp20_null_ztable` formalización "beyond null noise" (z); `exp26_xi_nulls` nulls para ξ.
- `expR31_quad_sweep` estabilidad al presupuesto de cuádruplas; `expR32_centroid_bootstrap` bootstrap de centroides; `expR33_budget_flatnull` flat-null a 4× presupuesto; `expR46_null_variants` sensibilidad a la construcción de la null.

**Censo y estadísticos**
- `exp11_null_per_dataset` exceso por dataset; `expR34_p999_imagenet` p99.9 en ImageNet; `expR39_census20` censo 72 celdas con 20 réplicas; `expR40_p999census` censo bajo p99.9; `expR45_convnet_rows` filas ConvNet.

**Jerarquía vs clustering (la línea que acabó en B25/B29/B30)**
- `expR29_flatnull` null B (hub-randomizing) — **retractada como test por sí sola**; `expR36_kmeans_hubs` null B con hubs k-means; `expR41_flatnull_p999` null B bajo p99.9; `expR42_star_calibration` una estrella pura puntúa "jerárquica" → retractación (B25); `expR50_depth_test` **test de profundidad calibrado por estrella** (B29); `expR51_meru_control` **control MERU vs gemelos CLIP** (B30).

**Curvatura / ξ**
- `exp12_curvature_sign` estimador de signo ξ; `expR38_xi_angular` ξ angular; `expR43_xi_norm_references` ξ confundido por normas → **ξ demovido a descriptor de normas** (B26).

**Texto**
- `exp4_prompt_variation` robustez de prompts; `exp14_dbpedia_text` jerarquía DBpedia 3 niveles; `exp16_hierarcaps` HierarCaps; `exp17_wordnet_prompts` prompts de definición; `exp18_text_nulls`/`exp18b_text_anisotropy` nulls y anisotropía de texto; `expR30_template_nulls` nulls por plantilla (batched); `expR44_text_census20` censo texto 20 réplicas (batched); `expR47_extraction_variance` variabilidad de extracción → **artefacto de left-padding**; `expR48_text_census_bs1` **censo canónico sin padding** (B23); `expR49_template_nulls_bs1` plantillas sin padding (B14).

**Contenido (¿de quién es el árbol?)**
- `exp3_semantic_alignment` alineamiento semántico + ORC; `exp7_hierarchy_recovery` recuperación de jerarquía; `exp10_local_vs_global` local vs global; `exp21_local_global_calibrated` arco Platonic→Aristotelian; `exp22_tree_similarity` mapa árbol-vs-árbol; `exp23_treemap_controls` **controles del tree map (crítico)**; `exp27_dbpedia_treemap` tree map no-WordNet; `exp24_val_metric_selection` selección de métrica en validación.

**Tareas downstream**
- `exp2_metric_controls` controles de métrica NC/FS; `exp2b_normalized_stack` ¿apila sobre L2?; `exp13_mcnemar` McNemar; `exp15_verify_345` re-run kNN/retrieval; `exp9_dino_verify` verificación cosine>hyp; `exp5_crossmodel_alignment` alineamiento entre modelos; `exp8_audit` auditoría global; `expR37_sample_level` censo a nivel de muestra.

## Datos y modelos externos (NO están en git)

- `/media/HDD_4TB_2/javi/Platonic/` — la caché del proyecto:
  - `data/` — CIFAR-10/100, DTD, MNIST/FashionMNIST crudos; `imagenet_train_50/` (50 imgs/clase, 1000 wnids en orden estándar).
  - `results/practical_tasks_cache/` — features por imagen de los 12 backbones en los 5 datasets de transfer.
  - `results/centroids/imagenet_train/*.npy` — centroides ImageNet por modelo (¡los de ViT supervisado difieren de la caché del censo — flag † en el paper!).
  - `results/features/imagenet/imagenet1k_wordnet_dist.npy` — distancias WordNet (frame WN-30 = AgglomerativeClustering(30, average, precomputed)).
  - `results/meru_cache/` — embeddings MERU/CLIP de expR51.
- `/media/HDD_4TB_2/javi/hf_cache/` — HF_HOME (el disco raíz está lleno: **nunca** descargar a `~/.cache`); `hf_cache/meru/*.pth` = checkpoints MERU/CLIP S/B/L.
- `/media/HDD_4TB_2/javi/meru/` — clon del repo de MERU (Desai et al. 2023), en `sys.path` de expR51.
- GPUs: 2× RTX 2080 Ti (11 GB).

## Protocolos vigentes (decisiones que NO deben revertirse sin releer REVIEW_31ago.md)

1. **Estadístico primario = supremo** del four-point (pre-registrado); p99.9 solo como robustez. No se adjudica la excepción DINOv2-ImageNet.
2. **Censo de texto canónico = extracción sin padding, batch 1** (expR48 → B23). El left-padding mueve δ̂ de GPT-2 hasta 0.018 (B28). B1 se conserva solo como "extracción original".
3. **Null B (hub-randomizing) no es un test de jerarquía por sí sola** (una estrella pura la pasa, expR42). Solo se usa **calibrada por estrella emparejada** (expR50 → B29).
4. **ξ no mide signo de curvatura**; es un descriptor de estructura de normas (expR43 → B26).
5. Framing: árboles "**moderately shared, configuration-dependent**"; el exceso certifica clustering, la profundidad residual es marginal (B29) y entrenar en hiperbólico no la crea (B30).
6. El criterio del tree map se aplica **por dataset** (el elegido en DBpedia es el degenerado en ImageNet).

## Añadido en la respuesta a la revisión (Fase B)

- `rebuttal/scripts/expR55b_depth_power_leafframe.py` → `rebuttal/results/expR55b_depth_power_leafframe.csv` (barrido de potencia del test
  de profundidad con el marco en las hojas y estrella anisotrópica; alimenta la Tabla B35 y `ICLR2027/figures/fig_depth_test.pdf` vía
  `ICLR2027/figures/make_fig_depth.py`, junto con `expR56_depth_variants.csv`). Chain: `rebuttal/scripts/r10_chain.sh`.
- `rebuttal/scripts/phaseB_t3_apply.py`: aplica la regla de decisión fijada (potencia ≥ 0.8 a ratio ≤ 0.3 para ambos n; falsas alarmas
  ≤ 5 % por dirección) y escribe la rama correspondiente en el .tex; deja la decisión en `rebuttal/results/phaseB_depth_decision.json`
  (rama aplicada: apéndice, "not certified").
- `ICLR2027/REVIEWER_CHECKLIST_phaseB.md`: preguntas de revisor con la respuesta tal como está en el texto principal.
- Pasada final: `rebuttal/scripts/expR60_c_sweep_record.py` (+ `r11_chain.sh`) → `expR60_c_sweep_record.csv` (barrido de clases bajo el
  registro; alimenta la Tabla B3 vía `gen_appendix2.py`); `rebuttal/scripts/expR61_dbpedia_record.py` → `expR61_dbpedia_record.csv`
  (DBpedia bajo el registro; Tabla A9 vía `gen_appendix.py`; centroides en `Platonic/results/text_cache/dbpedia_{m}.npz`);
  `phaseB_t4_apply.py` (test de profundidad acotado por régimen, figuras, §3) y `phaseB_t5_apply.py` (frases de §4 desde expR60/61,
  `phaseB_final_numbers.json`, que `sweep_freeze.py` contrasta con los CSV).
- Reestructuración: `rebuttal/scripts/expR62_samplelevel_record.py` → `expR62_samplelevel_record.csv` (nivel de muestra bajo el registro; Tabla 1
  columnas nuevas vía `gen_main_table.py`, B17 vía `gen_review_tables.py`, Fig. 2(b) vía `make_fig_overview.py` + `phaseC_fig2b.json`);
  `expR63_meru_record.py` → `expR63_meru_record.csv` (MERU bajo el registro; B30); chain `r12_chain.sh`; `make_memo_R7R8.py` → `phaseC_memo.json` +
  memo en CHANGELOG §18a; `phaseC_main_body.tex.tmpl` + `phaseC_restructure.py` (cuerpo nuevo, apéndice, decisión de Fig. 2(b)); el cuerpo anterior en
  `rebuttal/results/phaseC_old_main_body.tex`. Checks: secciones R7/R8 y "texto reestructurado" de `sweep_freeze.py`.
- Pasada de prosa: `rebuttal/scripts/phaseD_main_body.tex.tmpl` + `phaseD_prose.py` (cuerpo del texto principal, rellenado desde los CSV;
  cuerpo anterior en `rebuttal/results/phaseD_old_main_body.tex`), `phaseD_changelog.py` (números que salieron de la prosa y su destino);
  checks "prose" en `sweep_freeze.py` (métricas por párrafo, tesis ×5, conservación de números). Los pies de B7/B11/B17/B30/B32/B34/B35/A9
  y de la Tabla 1 llevan ahora los números que antes estaban en la prosa (todos calculados en los generadores).
- **2026-09-08: la carpeta exterior es `ICLR2027/`** (antes `iclr2027/`; la interior del paper sigue siendo `ICLR2027/iclr2027/`). Toda mención
  anterior a `iclr2027/...` en este fichero y en los CHANGELOG/TODO se lee como `ICLR2027/...`. Compilar: copiar `ICLR2027/iclr2027/` al build y
  `tectonic main_iclr2027.tex`; exportar a `ICLR2027/main_iclr2027_final.pdf`.
