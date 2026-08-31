# Revisión del borrador ICLR 2027 (`iclr2027/iclr2027/main_iclr2027.tex`) — 31 ago 2026

Compilado con tectonic sobre una copia (la máquina no tiene TeX): **compila sin errores, 21 páginas**.
Cifras del texto principal contrastadas una a una con `rebuttal/results/*.csv` (lista al final): **todas cuadran**.
Lo que sigue está ordenado por gravedad.

## A. Bloqueantes (no se puede enviar así)

1. **Se pasa del límite de 9 páginas.** La Discusión ("Excess measures existence…"), las Limitations y la
   Figura 7 caen en la página 10 (~0.6 página de exceso). Ethics/Reproducibility/LLM no cuentan, pero la
   Discusión sí.
2. **Abstract de 409 palabras** (`main_iclr2027.tex:47`): ocupa el 60 % de la página 1 y es exactamente lo
   que 1Eqj penalizó ("many numbers within the text, in parentheses, fragmented"). Objetivo: ≤ 220 palabras,
   tres hallazgos, como mucho 3-4 cifras.
3. **No hay ninguna tabla de resultados en el texto principal** (solo la de calibración, Tabla 1). El rebuttal
   a 1Eqj (A7) prometía "a main-text results table"; el borrador tiene 7 figuras y todos los números en
   paréntesis dentro de párrafos de 200-400 palabras. Propuesta: una tabla compacta por familia con
   exceso en ImageNet (z), exceso en transfer, ρ WordNet, ganancia NC/FS de la mejor métrica y métrica
   ganadora. Cabe en 8 filas y sustituye a la mitad de los paréntesis de §4-§6.
4. **Figura 1 sigue con el artwork viejo**: dentro del PDF pone "How is the reality?", "Hyperbolic" /
   "Euclidean" y nodo raíz "Reality"; el caption (`:73-76`) habla de "tree-like (left) … or flat (right)" y
   de "Is there a Platonic tree?". Contradice título y reescritura (COAUTHOR_TASKS #1). O se re-etiqueta o
   se quita; quitarla resuelve además el problema de páginas.
5. **Referencias `??` en el índice de procedencia** (p. 20): `tab_z_provenance.tex:10,17` referencian
   `tab:a1` y `tab:a8`, pero `tab_a1_excess` y `tab_a8_hierarcaps` no se incluyen con `\input`.
6. **HierarCaps no aparece en el paper.** La tabla está generada (`appendix_tables/tab_a8_hierarcaps.tex`,
   exp16) pero no se incluye ni se cita en el texto. Fue compromiso público con pux6 ("we commit to such an
   evaluation in the revision") y pux6 dijo que con ello el paper "reads as substantially more complete". Si
   los mismos reviewers reaparecen (habitual entre NeurIPS e ICLR), lo notarán. El resultado es además
   bueno: orden radial ρ(level, radius) +0.47..+0.87, monotonía estricta 19-54 % frente a 4.2 % de azar,
   tripletas 0.83-0.90. Añadir 2-3 frases en "Whose tree?" (`:190`) y el `\input` en el apéndice.
7. **Dos tablas del apéndice se salen del margen**: `tab_a0_models` (+10.7 pt) y `tab_b7_treemap`
   (+44 pt, claramente visible). `\scriptsize` + `\setlength{\tabcolsep}{3pt}` o `\resizebox`.

## B. Contenido y framing (decisiones de autor)

8. **§6 (Corollary) se desmonta a sí misma.** La contribución 4 (`:66`) promete "when to leave Euclidean
   distances (predicted by the descriptors) … and which zero-cost metric collects the gain", pero el propio
   texto dice que "the δ-gated rule … is the weakest policy in the table" en NC y FS, y que "at the headline
   t the Poincaré projection does not improve NC on average". Es honesto (y correcto: tab_b9 lo confirma),
   pero tal como está redactado un reviewer citará esa frase contra la contribución (pux6 ya lo hizo:
   "the structure exists yet is not useful downstream"). Recomendación: (a) redefinir la contribución 4
   como *diagnóstico* ("δ predicts where the sign of the non-Euclidean gain falls; the operational policy is
   objective + label flatness, with cosine as the safe default"); (b) reordenar "Which metric collects the
   gain" para abrir con lo que funciona (validation pick +0.78 pp, objective rule +0.41 vs always-cosine
   +0.28, p = 0.011) y cerrar con el negativo del δ-gated; (c) quitar "Metric-Selection Rule" del título de
   la sección si se opta por (a).
9. **La figura principal de §5 es el mapa ingenuo** (Fig. 5), que el propio texto declara artefacto; la
   corrección solo se ve en una tabla del apéndice. Mejor un panel (a) naive / (b) configuración
   seleccionada: el lector ve la isla desaparecer en vez de leerlo. El párrafo "A naive map of trees, and
   its artifact" (~420 palabras) lee como log de auditoría (criterio, umbral 0.5, sensibilidad
   [0.40-0.50], seis configuraciones); dejar el criterio en dos frases y mover la sensibilidad al apéndice.
10. **La respuesta a la pregunta del título aparece tres veces casi literal** ("to a substantial degree,
    yes; the shared tree is the human taxonomy; supervision deepens and stabilizes…"): abstract (`:47`),
    final de §5 (`:180`) y Discussion (`:221`). Una vez en Discussion basta; ahorra ~0.1 página.
11. **Panel de modelos incoherente.** §3 (`:95`) dice "4 additional contrastive vision backbones are
    measured on ImageNet" (OpenCLIP/MetaCLIP, Tabla A.1), pero no aparecen en ningún resultado del
    borrador (0 menciones; exp20 tiene 12 modelos). O entran en el censo o se quitan de A.1 y del
    recuento. El `% TODO-coauthors` de esa línea sigue abierto.
12. **Figura 7**: las etiquetas δ = 0.107 / 0.076 dentro del gráfico son del protocolo viejo (exp20
    CIFAR-100: DINOv2-S 0.113, DINOv2-G 0.070) y el nodo "Reality" sigue (COAUTHOR_TASKS #2). Quitar los δ
    del gráfico o regenerar. Es "ilustración" por caption y estamos pasados de página: candidata al apéndice.
13. **Intro, párrafo "The answer has two halves"** (`:58-59`): 230 palabras con 12 cifras. Es el patrón que
    1Eqj criticó. Con la tabla del punto 3, este párrafo puede quedarse en la mitad y sin números.

## C. Figuras (legibilidad)

El ancho de texto de ICLR es 5.5 in (396 pt). Factores de escala reales al incluirlas:
Fig. 2 y 6 (662 pt → ×0.60), Fig. 3 (633 pt a 0.9\linewidth → ×0.56), Fig. 4 (1320 pt → ×0.30),
Fig. 5 (627 pt → ×0.63). Con tipografía de ~7 pt en origen, Fig. 2, 4 y 6 quedan a 4-5 pt en el PDF:
ilegibles impresas (1Eqj: "Fig. 2 is too small … or it is useless"). Regenerar con
`figsize=(5.5, h)` y fuentes ≥ 7 pt (Fig. 4 en dos filas de 2 paneles).

14. Fig. 3: la anotación "genuine form emerges with scale (GPT-2 and OLMo)" tapa las barras de Pythia.
15. Fig. 2b muestra la media sobre datasets de transfer, pero el texto (`:133`) cita CIFAR-10
    (−0.095 → −0.157) "Figure 2b": el lector no encontrará esas cifras. Decir "mean over transfer
    datasets" en el caption o mostrar CIFAR-10.

## D. Menores

- Títulos de §4 y §5 ocupan dos líneas cada uno; acortar.
- `:118`: "5 null replicates" para δ; para ξ el último commit usa 20 réplicas + ruido del estimador. Decir
  ambos en el párrafo "Curvature sign".
- `:127`: el sweep exp19 tiene 5 semillas con dispersión grande (NC adv random 0.5-2.6 pp, coherente
  0.04-1.0 pp); el texto da solo medias (+1.58 / +0.53). Añadir "mean of 5 seeds" o ±.
- Reproducibility: "will be released" → enlace anónimo ya en submission (gratis y los reviewers lo valoran).
- Related Work: "None of these readings is calibrated against matched nulls; … without calibration they
  are not interpretable" es una afirmación fuerte sobre trabajo ajeno; suavizar a "are not calibrated…;
  our results suggest…".
- `\author{Anonymous authors}` es inocuo (la plantilla ya lo anonimiza).

## E. Plan de recorte para entrar en 9 páginas (≈ −0.9 pág, deja sitio para la tabla)

| Acción | Ahorro |
|---|---|
| Abstract 409 → ≤ 220 palabras | −0.30 |
| Fig. 7 al apéndice (o quitar Fig. 1) | −0.30 |
| Respuesta al título solo una vez | −0.10 |
| Sensibilidad del umbral (§5) al apéndice | −0.10 |
| Intro "two halves" a la mitad | −0.10 |
| Tabla resumen nueva en §4 | **+0.25** |

## F. Cifras verificadas contra CSV (todas ✓)

69/72, 51/72, 41/48, 34/48, celdas frontera −3.03/−2.97 (exp20); rangos ImageNet ViT −0.020..−0.030 y
CLIP/SigLIP −0.019..−0.025; DINOv2-L z = −1.1; excepciones ≤ +1.2 s.d.; S→G CIFAR-10 −0.095 → −0.157;
GPT-2 +0.048/−0.013/−0.034/−0.048, OLMo −0.001/−0.015, Pythia −0.035..−0.041, embedders +0.001..+0.011,
GTE-Qwen2 −0.019 (exp18); ξ DINOv2 −0.081 → −0.152, SigLIP +0.005 (z = 2.5), 9/12 (exp26); ρ WordNet
por familia (exp3); pooling 1 img ρ ≤ +0.26 (exp8_p6; nota: al rebuttal se dijo +0.34, de otro run);
DBpedia exceso/FS/NC/tripletas (exp14); exp19 C = 50 −0.124/−0.093 y +1.58/+0.53; McNemar 242/37,
p ≈ 4×10⁻³⁸ (exp13); NC 22/60 positivas, media −0.24, hier −0.00, flat −0.70; FS 56/60
(table1_regenerated); CLIP HN−COS +0.9..+1.3 transfer, +0.1..+0.2 ImageNet (exp2b); exp24 +0.63/+0.64,
+0.78/+0.78, +0.41 vs +0.28; políticas NC (tab_b9); exp21 kNN 0.474/0.010/0.464, CKA 0.636/0.107/0.529,
H 0.427/0.644.

Única discrepancia encontrada: las δ impresas dentro de la Figura 7 (punto 12).

## Cómo compilar en local

No hay TeX en la máquina. Funciona `conda create -p <dir> -c conda-forge tectonic` y luego
`tectonic main_iclr2027.tex` dentro de `iclr2027/iclr2027/` (descarga paquetes la primera vez).

---

## Estado tras la pasada mecánica (31 ago, tarde)

Hecho en `main_iclr2027.tex` y `appendix_tables/` (compila con tectonic: **20 páginas, texto principal termina en la 9, cero warnings**):

- **A1 páginas**: ✅ Ethics Statement abre la página 10.
- **A2 abstract**: ✅ 409 → 225 palabras; mismos tres hallazgos, cuatro cifras.
- **A5 `??`**: ✅ `tab:a1` eliminado del índice de procedencia; `gen_provenance.py` ahora solo indexa tablas que el `.tex` incluye.
- **A6 HierarCaps**: ✅ frase con cifras (exp16) en "Whose tree?" + cita `alper2024hierarcaps` + nueva subsección A.13 con `tab_a8`.
- **A7 tablas anchas**: ✅ `tab_b7_treemap` (tabcolsep 3 pt, cabeceras cortas, "(deg.)"/"(selected)") y `tab_b2_ztable` (tabcolsep 4 pt); mismos cambios en `gen_appendix2.py` para que una regeneración no los deshaga.
- **B10 respuesta al título**: ✅ eliminada del final de §5; queda en abstract (corta) y Discussion.
- **B12 Fig. 7**: ✅ movida a un Apéndice B "Illustrative visualization" (`app:viz`); Limitations (vi) apunta allí. Las δ viejas dentro del gráfico y el nodo "Reality" siguen pendientes (requieren regenerar la figura).
- **B9 (parcial)**: la sensibilidad del umbral 0.5 pasa al apéndice A.5 (`app:treemap`); el criterio queda en dos frases en §5.
- **D títulos de §4/§5**: ✅ una línea cada uno ("Findings I: Genuine, Learned, Scale-Deepening Form" / "Findings II: Shared Content, Fragile Measurement").
- **B13 intro**: cifras repetidas de GPT-2/OLMo y el 41/48 quitadas del párrafo "two halves" (siguen en §4).
- Limitations (ii, iv, v, vi, vii, viii) y "What the census establishes" apretadas sin cambiar contenido.

Copia de seguridad del `.tex` original: scratchpad de la sesión (`main_iclr2027.tex.bak`); el diff completo está en `git diff`.

**Pendiente (decisiones de autor, no tocado):** A3 tabla resumen en el texto principal (hay ~0 líneas de margen: habrá que pagarla con más recortes en §5 o quitando Fig. 1); A4 Fig. 1 artwork viejo; B8 reformular §6/contribución 4; B9 Fig. 5 naive vs corregida; B11 OpenCLIP/MetaCLIP en el panel; C legibilidad de Figs. 2/4/6; 14-15 Fig. 3 anotación y Fig. 2b caption.

---

## Estado tras la pasada de pendientes (31 ago, noche)

Todo compilado con tectonic: **20 páginas, texto principal termina en la 9, cero warnings**. Commit anterior: `79da833`.

- **A3 tabla resumen en el texto principal**: ✅ nueva Tabla 2 (`tab_census.tex`, generada por `gen_main_table.py` desde exp20/exp26/exp3/exp28/exp2): 12 backbones × {δ̂ IN, exceso IN (z), exceso transfer, ξ-exceso, ρ_WN, ARI₂₀, mejor métrica − R (FS), métrica}. Referenciada desde §4, §5 y §6.
- **A4 Fig. 1 (artwork viejo)**: ✅ retirada del paper (opción B de COAUTHOR_TASKS); el fichero sigue en `figures/` por si se re-etiqueta. La frase de la intro que la citaba se reescribió.
- **B8 §6 / contribución 4**: ✅ reformulado como *diagnóstico + política*: contribución 4, abstract (3), frase de la intro, título de §6 ("A Diagnostic of Exploitability, and a Modest Policy"), párrafo de apertura, y "Which metric collects the gain" reordenado (abre con validation pick / objective rule, cierra con el negativo δ-gated en un párrafo propio "What the diagnostic is, and is not"). Ninguna cifra cambia.
- **B9 Fig. 5 naive vs. corregida**: ✅ nueva `fig_treemap_controls.pdf` (`iclr2027/figures/make_treemap_fig.py`, desde las matrices ARI por configuración de exp23): (a) ImageNet naive, (b) ImageNet cosine-average, (c) CIFAR-100 naive, (d) CIFAR-100 cosine-complete, con el ARI medio DINOv2-B/L/G vs bloque sobre cada panel (0.03 → 0.38; 0.00 → 0.39). Sustituye a la figura naive; caption nuevo. Además la sensibilidad del umbral ya estaba en A.5.
- **B11 OpenCLIP/MetaCLIP**: ✅ fuera de la Tabla A.1 y de la frase de §3 (no hay resultados calibrados); el TODO-coauthors queda resuelto: censo 12 visión × 6 + 16 texto; malla de tareas 10 visión (sin DINO-B ni SigLIP-B).
- **C legibilidad**: ✅ Figs. 2, 3 y 6 regeneradas a 5.5 in de ancho con fuentes 6.5-8 pt (`make_figs.py`, `fig_text_nulls.py`; rutas ahora relativas al repo, con `PLATONIC_RESULTS` como override). ⚠️ **Fig. 4 (ablaciones) no se puede regenerar aquí**: el script de 4 paneles y los datos de la curva por capas no están en el repo (`figures_archive/gen_fig3_causal.py` es una versión antigua de 3 paneles). Queda para coautores: regenerar a `figsize=(5.5, 1.7)` con fuentes ≥ 7 pt.
- **14 Fig. 3 anotación**: ✅ reubicada (ya no tapa Pythia). **15 Fig. 2b**: ✅ caption dice "mean over the three hierarchical transfer datasets"; el texto cita también la media de transfer (−0.063 → −0.103).
- **12 Fig. 7 δ viejas**: ✅ tapadas con `overpic` (cajas blancas sobre "δ = 0.107 / 0.076"); el nodo "Reality" sigue (necesita el script fuente).
- **Menores**: ✅ "5 null replicates for δ, 20 for ξ"; exp19 "means over 5 draws"; frase de Related Work suavizada.
- Recortes adicionales para volver a 9 páginas tras añadir la tabla: intro "two halves", contribuciones 3-4, related work (convergencia), Discussion, Limitations (i)-(ii), §5 (criterio), §6 (frases sueltas). Sin cambios de contenido ni de cifras.

**Sigue pendiente (requiere a coautores):** Fig. 4 regenerada; Fig. 7 regenerada sin "Reality" (script no localizado); enlace anónimo al código en Reproducibility; decidir si Fig. 1 vuelve re-etiquetada.

---

## Nota de numeración (los números de figura de las secciones anteriores son los del borrador original)

Tras retirar la figura de la caverna y mover la visualización al Apéndice B, la numeración compilada es:
Figure 1 = `fig_excess_panel` · Figure 2 = `fig_text_nulls` · **Figure 3 = `fig3_causal` (ablaciones; la única aún ilegible ×0.30)** · Figure 4 = `fig_treemap_controls` (naive vs. seleccionada) · Figure 5 = `fig_bestmetric_scatter` · **Figure 6 = `fig_tree_visualization` (Apéndice B; nodo "Reality" pendiente, δ viejas ya tapadas)**. La tabla de calibración está ahora en el Apéndice A.1 y la Tabla 1 del texto principal es el censo.

---

## Respuesta a la revisión simulada (31 ago, madrugada) — TODOS los puntos atacados

Cuatro experimentos nuevos (scripts en `rebuttal/scripts/expR29-32*.py`, resultados en `rebuttal/results/expR*.csv`), corridos en esta máquina (los features estaban en `/media/HDD_4TB_2/javi/Platonic`, no en `/home/javi/Platonic` como asumían los scripts viejos; GPU 2080 Ti para GPT-2).

| Punto | Resultado | En el paper |
|---|---|---|
| **W1** null que aísle jerarquía de clustering | **expR29**: null que conserva la geometría de cada cluster y sustituye la configuración de hubs por una muestra gaussiana espectro-igualada. 24/24 celdas (IN wn30 + C100) siguen negativas; 14/24 con z<−2 (ViT-L IN −0.032, z=−5.1). En C100 el componente entre-hubs lleva el 56-94 % del exceso para sup./contrastivos y menos para DINOv2 (semántica angular). DINOv2-L en IN: marginal bajo null espectral → significativo bajo el plano (−0.034, z=−3.0) | Párrafo "Hierarchy, not just clusters" en §4; Tabla B13; abstract y contribución 1 mencionan el null; Fig. — |
| **W2** pseudo-replicación por familia | Demeaned por familia: within-dataset sobrevive (NC −0.31..−0.73; FS −0.63..−0.86; best−R −0.55..−0.82, sostenido por DINOv2); el pooled del mejor-métrica NO (−0.31→−0.18, −0.33→−0.04) | Frases en §6 (dos sitios); Tabla B12 |
| **W3** ¿sobrevive la emergencia GPT-2 a la plantilla? | **expR30**: 10 plantillas × 4 tamaños con nulls. S: 0/10 genuino (7/10 POR ENCIMA del null); M: 3/10; L: 9/10; XL: 9/10 (incl. gloss_only sin nombres de clase). El orden con la escala es robusto a plantilla; los valores no | Frase "Two scopes" reescrita en §4; Limitation (iii) matizada; Tabla B14 |
| **W4** desconexión instrumento↔corolario | z pooled predice ≥ δ (NC −0.50 vs −0.45; FS −0.36 vs −0.18); exceso within-transfer −0.55..−0.84; inversión SOLO en ImageNet (la celda anómala DINOv2) | Frase en §6 y en Discussion ("dissociation localized") |
| **W5** media de excesos incomparables | Tabla 1 con columnas exc. C100/C10/DTD; Fig. 1b con líneas por dataset | hecho |
| **W6** cuádruplas / percentil | **expR31**: exceso estable desde 10⁵ cuádruplas (sd ≤0.004 salvo la celda at-null); p99.9 conserva todos los signos | Frase en §3; Tabla B15 |
| **Q4** bootstrap de centroides | **expR32**: sd del exceso 0.002-0.005 con 30 remuestreos (C100 500 img/clase, DTD 80) — un orden por debajo del exceso | Frase en §3; Tabla B15 |
| Menores | CLIP-L "marginal (z=−1.9)"; GTE-Qwen2 anotado en Fig. 2 + texto ("patterning with the causal-LM scale story"); DINO-B unificado; **§3 corregido**: los centroides de transfer usan el split cacheado completo (500 C100 / 80 DTD / ~6000 MNIST), no "100 por clase" — verificar con coautores que ImageNet sí es 100 | hecho |
| Recomendación "mover §6 al apéndice" | No seguida (Xbn5/pux6 pidieron utilidad downstream); en su lugar: §6 reformulada como diagnóstico + política (commit anterior) + los resultados W2/W4 arriba | — |
| Densidad §5/§6 | Sin resolver del todo (falta de espacio); las tablas B12-B15 absorben parte | parcial |

Para caber en 9 páginas con el material nuevo: la tabla de calibración pasó al Apéndice A.1 (sus números clave quedan inline en §3) y hubo recortes de redacción distribuidos sin tocar cifras. Compila: 22 páginas, texto principal hasta la 9, 0 warnings.

---

## Ronda 2 del revisor simulado (nota 8) — los 6 puntos restantes, resueltos

1. **Abstract sobrevendía el null B** → ahora dice "significant in 14/24 cells, concentrated in CIFAR-100 and the supervised models". (El "5 de 12" del revisor era el −1.95 de ViT-T redondeado en la tabla; el recuento estricto es 4/12 IN + 10/12 C100 = 14/24.)
2. **Null B usa la taxonomía humana** → frase de alcance (i) en la caption de B13: certifica jerarquía *alineada con WordNet/superclases*; no detectaría jerarquía propia del modelo desalineada.
3. **"Estable desde 10⁵" no valía en ImageNet** → cierto (DINOv2-L IN deriva −0.024→−0.005; ViT-L y CLIP-B derivan en dirección contraria). Frase de §3 y caption de B15 reescopadas a transfer sets con el caveat de ImageNet explícito. Y **expR33** (nuevo): a presupuesto 4× (2×10⁶) los veredictos del null B en ImageNet persisten — ViT-L zB=−5.1, CLIP-B −2.4, **DINOv2-L −2.0** (la celda rescatada no es artefacto del presupuesto); citado en la caption de B13. El bootstrap de ImageNet requiere el caché por imagen (no está en esta máquina) → nota en B15 y pendiente coautores.
4. **Discrepancia B13 vs Tabla 3 en ImageNet** → diagnosticada: los `.npy` de `results/centroids/imagenet_train/` difieren del caché del censo **solo para los 4 ViT supervisados** (δ̂ +0.016..+0.025; el resto ±0.003). Caption (ii) de B13 lo declara; dentro de cada fila todo usa los mismos centroides. Pendiente coautores: regenerar la mitad ImageNet de B13 desde el caché canónico.
5. **Fig. 5 desincronizada** → títulos ahora "within-ds r −0.60..−0.83 (pooled −0.31)" (calculado en `make_figs.py`) y caption remite a B12.
6. **Limitación (iv)** → añade "the within-dataset δ–gain prediction rests chiefly on DINOv2's scale range".

Compila: 22 páginas, texto principal hasta la 9, 0 warnings, sin `??`.

---

## Ronda 3 (nota 8, aceptar — sostenida): tres menores de redacción, hechos

1. Caption B13: "Two scopes" → "Three scope notes" (eran tres).
2. Caption B15: fuera la frase de bitácora ("not available on this machine") → "reported on CIFAR-100 and DTD".
3. Trazabilidad de A.4: comentario de fuente en el .tex (expR29/expR33 + generador + almacén de centroides), ya que el índice de procedencia renderizado se retiró a petición.

Oferta del revisor para la siguiente pasada: solo legibilidad de §5-§6 (siguen densas). Decisión de Javi.

---

## Revisión "desde cero" (segundo agente; nota 6, subiría a 8 con W1-W2-W5) — atacada con 5 experimentos nuevos

- **W1/Q3** → **expR34**: ImageNet re-puntuado con p99.9 + 20 réplicas + rangos percentiles (Tabla B18); reordena honestamente (DINO/DINOv2 por debajo de las 20 réplicas; el exceso de ViT-T era del supremo) y coincide con el orden de ξ. Frase en §4.
- **W2 (z con 5 réplicas)** → §3: |z| grande se lee como "por debajo de todas las réplicas", no cola gaussiana; B18 aporta la versión de 20 réplicas. Pendiente coautores: censo completo a ≥30 réplicas desde el caché canónico.
- **W3** → "genuine" definido como relativo al null en §3 + tabla A.2 nueva (qué conserva/destruye cada null).
- **W4** → abstract añade "41/48 beyond 2σ on hierarchical datasets" junto al 14/24.
- **W5** → **expR37**: instrumento sobre features a nivel muestra (el objeto de Khrulkov/Yang): δ crudos en la misma banda baja, excesos casi todos en ruido (6/24 con |z|≥2; DINOv2/DTD positivo). Tabla B17 + related work reescrito ("our census is about inter-class geometry") + enlace desde §6.
- **Q4** → **expR36**: null plano con hubs k-means de los propios centroides (sin taxonomía humana): 24/24 negativos, 16/24 |z|≥2, DINOv2-L/G en ImageNet a −4.4/−4.6. Tabla B16 + §4.
- **Q2** → **expR38** (ξ vectorizado): en geometría angular el ξ negativo de DINOv2-S/B/L NO sobrevive (pasan a positivo); G sigue negativo (−12.5) y CLIP también. Tabla B19 + limitación (viii): la celda "no resuelta" se estrecha a efecto de normas.
- **Q1** → pooling detallado en el apéndice del panel (timm `num_classes=0`; last-token causal; pooling por embedder).
- **W7** → definición de *cell*, fórmula de ξ en el apéndice, nota de que el no-alineamiento cross-model bajo ϕ es esperable. Densidad §5-§6: parcial (limitada por espacio).
- **No seguido**: comprimir §6 a media página (conflicto entre revisores; decisión de coautores). **Pendiente coautores**: censo a ≥30 réplicas, iNaturalist (Q5), bootstrap ImageNet.

Compila: 25 páginas, texto principal hasta la 9, 0 warnings, sin `??`.

---

## Respuesta post-revisión (condición 6→8): reconciliación de ImageNet + censo a 20 réplicas

**Reconciliación del relato de ImageNet** (la condición del revisor, ~12 ediciones): la excepción del censo ya no se cuenta como "DINOv2 at null" sino como *supremum-specific*: §4 lo resuelve explícitamente ("DINOv2's ImageNet tree is real, in the bulk of its four-point distribution and among its own clusters; only the supremum read in the WordNet frame sits at null"), §5 lo pliega en forma-vs-contenido ("a real tree, least aligned with WordNet's top level"), y todas las frases "at null" (abstract-fig.1-ORC-§6-§7) quedan escopadas al supremo. La Tabla 1 muestra ahora el exceso supremo (z) **y la columna p99.9** lado a lado. La tensión δ/ξ se cierra igual (norm structure, expR38); caption de B19 con el resultado explícito.

**Censo a ≥20 réplicas** (segunda condición): **expR39** — las 72 celdas con 20 réplicas y rangos percentiles (Tabla B20): reproduce el censo de 5 réplicas con **signo 68/68** y significancia 62/68 en las celdas comparables; 56/72 por debajo de las 20 réplicas; mismas tres excepciones (DINOv2-S/B/G en IN). Los 4 ViT-ImageNet van marcados († almacén de centroides; no comparables directos con la Tabla 3). Frase en §3.

Ajustes de página por el material nuevo: título de §6 acortado ("Corollary: Diagnosing Exploitability"), limitaciones fusionadas (v-vi), recortes menores distribuidos. Compila: texto principal hasta la p. 9, 0 warnings, sin `??`.

**Para coautores**: regenerar las filas † de B13/B18/B20 desde el caché canónico de ImageNet; el resto de pendientes sin cambios.
