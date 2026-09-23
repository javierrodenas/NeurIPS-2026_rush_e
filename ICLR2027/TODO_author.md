# TODO(author) — pasada de versión final (2026-09-03)

Lo que no pude verificar, decidí no tocar, o necesita una decisión del autor.

## Decisiones que necesitan tu veto/aprobación

1. **R2 con centroides de la caché del censo (no el submuestreo de exp5).** `exp21b` calcula mKNN/CKA
   sobre los centroides de los 100 embeddings de entrenamiento cacheados por clase (el protocolo de §3),
   no sobre el submuestreo `RandomState(0)` de 100/clase del fulltrain que usó exp5/exp21. Motivo: el
   submuestreo exige otra extracción de 100k imágenes por modelo (~3 h más) y el paper declara la caché
   como representación de clase. Consecuencia: los raws de §5 cambian ligeramente (exp21: mKNN 0.474, CKA
   0.636; valores nuevos en §5 y en `CHANGELOG_final.md`). Si prefieres reproducir exp5 exactamente,
   `extract_imagenet_cache_local.py --sets exp5sub` genera el submuestreo exacto (la reproducción del
   muestreo está verificada: `choice(replace=False)` solo depende del tamaño de bloque) y basta cambiar
   una línea en `exp21b` (`load_centroids`).
2. **Figura best-metric al apéndice.** Movida de §6 a la subsección Statistics del apéndice para que el
   texto principal termine en p9 tras añadir las Figuras 1 y 2. Si la quieres en el cuerpo, la palanca
   alternativa es recortar ~11 líneas en §2/§3 o reducir la Figura 1 a 1.0 in.
3. **Tamaño de la Figura 1.** El placeholder mide `\textwidth × 1.25 in`. Si la figura de Canva tiene
   otra proporción, cambia el `1.25in` del `\parbox` para que el layout siga siendo final; el `\IfFileExists`
   la incluye automáticamente al soltar `figures/fig1_concept.pdf`.

## Figura 1 (Canva) — especificación

Tres viñetas horizontales, mismo estilo: (a) nube aleatoria en alta dimensión, (b) clusters alrededor de un
hub (estrella), (c) jerarquía anidada. Bajo cada una, la lectura del instrumento: δ cruda baja en las tres;
exceso sobre la null espectral: nulo en (a), genuino en (b) y (c); test de estrella emparejada: sin
profundidad en (b), profundidad en (c). Paleta de `figures/palette.py`; null en gris discontinuo. Guardar
como `ICLR2027/iclr2027/figures/fig1_concept.pdf` (vectorial), ancho de texto, ≤ 1.3 in de alto.

## Números sin fichero de resultados (tecleados, verificados en vivo)

- Tabla de calibración (`appendix_tables/tab_calibration.tex`, generada por `gen_appendix.py`): las filas
  **árbol binario (0.000/0.000), H² R=2/4/8/16 (0.65/0.69/0.69/0.69; 0.162/0.087/0.043/0.022) y S⁹⁹
  cuerda/geodésica (0.25/0.37; 0.143/0.179)** no tienen CSV (`exp6_h2_sphere_check.py` imprimió a stdout).
  Están tecleadas dentro del generador y `sweep_freeze.py` las re-verifica sintéticamente en cada freeze
  (tolerancias en el script). Si quieres cerrarlo del todo: hacer que exp6 escriba `exp6_calibration.csv`
  y leerlo en `gen_appendix.py`.
- Claim "signs agree in 35/36 cells" (§4, construcción exact-sample-spectrum): viene de `expR46`
  (sin cambios en esta pasada); no lo re-verifiqué.

## Cosas que no toqué (y por qué)

- **Fig. 6 (visualización 2-D CIFAR-100, apéndice)**: no existe script (ver `CODE_MAP.md`). El nodo
  "Reality" sigue tapado con cajas `overpic` que leen "root"; la cadena "Reality" sigue en la capa de texto
  del PDF. Regenerar la figura desde su fuente.
- **Tabla B1** (`tab_b1_textnulls.tex`): su generador vive en otra máquina; la leyenda se editó a mano en
  la ronda 10 ("original extraction"). Si se regenera allí, la leyenda se perderá.
- **OLMo-7B** no re-extraído sin padding (omitido en la Fig. 4; valor original en B1).
- **Limitations (i)–(vii)**: intactas por mandato del brief.
- **§2 Related Work**: solo se acortó una frase (nivel-muestra); el resto no estaba en el alcance.
- **Tabla B2** (`tab_b2_ztable`, censo de 5 réplicas de exp20) se conserva como histórica; B20 (expR39b) es
  la canónica y su leyenda dice de dónde sale cada celda.

## Comprobaciones pendientes fuera de esta máquina

- Compilar en Overleaf con `tcolorbox` (aquí compila con tectonic sin warnings); confirmar que el estilo ICLR
  admite el paquete.
- Recuento de páginas con la Figura 1 real (ahora: texto + Ethics en p9, Reproducibility abre p10).
- `references.bib`: `groger2026aristotelian` es arXiv:2602.14486 (ICML 2026) — comprobar la entrada final
  (título/venue) cuando salga la versión de actas.

## Dónde están las cachés regeneradas (no están en git)

`/media/HDD_4TB_2/javi/Platonic/results/practical_tasks_cache/{m}_imagenet_train.npz` (12 modelos, 100
img/clase, extraídas de `/media/HDD_4TB_1/javi/ILSVRC2012_img_train` con
`rebuttal/scripts/extract_imagenet_cache_local.py`); shards de DINOv2-G en `*_train.shard{0,1}of2.npz`
(se pueden borrar). `i21k_t`/`dinov2_s` tienen además `*_imagenet_exp5sub100.npz` (de la primera pasada).

## Añadido en R6 (censo a 200 réplicas)

- El recuento de signo baja de 69/72 a **68/72**: ViT-T en FashionMNIST queda en exceso +0.001 (p = 0.53), es decir, exactamente en el null;
  el abstract y la intro dicen ahora "68 of 72". Si prefieres redondear el mensaje a "nearly universal" sin número, es un cambio de dos frases.
- El censo de texto a 200 réplicas usa 3 semillas de cuádruplas por réplica (como expR48); el de visión, 5 (como expR39). Es lo que había; si quieres
  homogeneizar a 5 en texto, `expR48b` tarda ~1.5 h más de CPU.
- OLMo-7B sigue sin re-extraer (solo extracción original, Tabla B1); el texto de §4 dice ahora "OLMo-1B".

## Añadido en la pasada de contribución

- **`\url{ANONYMIZED-REPO}`** en el Reproducibility Statement: sustituir por la URL del repositorio anónimo. El tool está en
  `ICLR2027/tool/calibrated_delta.py` (uso: `python calibrated_delta.py centroids.npy [--labels sup.npy] [--reps 200] [--json out.json]`);
  `ICLR2027/tool/run_checks.py` reproduce dos celdas de la Tabla 1 y de B29 y escribe `rebuttal/results/tool_check.json`, que
  `sweep_freeze.py` compara en cada freeze (re-ejecuta el check solo si falta el JSON; ~15 min de CPU).
- Para el check de B29 en DINOv2-L/ImageNet el tool recibe los centroides del **store** (como hizo expR50); con los de la caché del censo
  el test de profundidad da un valor ligeramente distinto (store ≠ caché, ver CHANGELOG §0). Si en algún momento se rehace B29 sobre la caché,
  cambiar la línea correspondiente de `run_checks.py`.

## Añadido en la respuesta a la revisión (Fase A)

- **Control positivo con los checkpoints fine-tuned jerárquicamente (Fig. 5a)**: no hay pesos ni features guardados en
  `Platonic/` (solo `analysis4_finetuning.csv`, `hierarchical_finetuning.csv` y logs; `run_finetune_ablation.py` y
  `exp_hierarchical_finetuning.py` no guardan nada). Coste de rehacerlo: re-entrenar ViT-B, DINOv2-S y CLIP-B con el objetivo
  jerárquico de CIFAR-100 (el log original da ~1.8 h de GPU para ViT-B; ~4–5 h para los tres en una 2080 Ti), extraer
  centroides de CIFAR-100 (minutos) y pasar `ICLR2027/tool/calibrated_delta.py --labels` (minutos). Guardar esta vez los
  centroides (`results/centroids/cifar100_ft/{m}.npy`). Es el único control positivo en datos reales disponible para el test
  de profundidad.
- **Semillas de la null Haar**: el brief de la respuesta fija `300+rep` para los censos nuevos (expR52–54, 57, 59); `expR46`
  (la fuente de la construcción Haar) usaba `700+rep`. Anotado en los docstrings.
- **Marcos K de CIFAR-100 (K = 10, 5)** para el barrido del test de profundidad: clustering aglomerativo (enlace promedio) de
  los 20 hubs de superclase sobre la matriz de distancias consenso de los 12 backbones (`expR56_frames_cifar100.csv`);
  es determinista y compartido por todos los modelos, pero no proviene de etiquetas humanas. Los cortes de ImageNet (10/30/60)
  sí son de WordNet.

## Añadido en la respuesta a la revisión (Fase B)

- **DBpedia bajo el protocolo de registro**: el rango −0.020..−0.028 de §4 sigue siendo el de `exp14` (supremo, null gaussiana,
  3 réplicas); el texto lo dice explícitamente. Para cerrarlo: re-extraer los embeddings de DBpedia (exp14, minutos) y pasar
  `expR53` en modo Haar × p99.9 (≈10 min de CPU). Lo mismo para HierarCaps (A8) si se quiere homogeneizar del todo.
- **Control positivo fine-tuned (Fig. 5a)**: no pudo correrse (sin checkpoints ni features guardados); coste en la sección anterior. El
  párrafo del test de profundidad lo dice.
- **B29 (estrella isotrópica, 3 semillas)**: retirada del apéndice y del texto principal (el fichero `tab_b29_depth.tex` y `expR50_depth_test.csv`
  se conservan como histórico); B34 (estrella iso/aniso, 10 semillas, barrido K) y B35 (potencia) la sustituyen.
- **El tool** (`ICLR2027/tool/calibrated_delta.py`) sigue ahora el protocolo de registro (Haar × p99.9; estrella anisotrópica con 10
  semillas); `--null gauss --stat sup --star iso` reproducen las lecturas antiguas. Reporta la p sin corregir (BH requiere un censo).
- **Test de profundidad: la regla falla por una sola dirección y un solo régimen.** Falsas alarmas z ≤ −2 = 6.2 % agregadas, concentradas
  en n = 100 con K ≥ 12 (17 %; 3–8 puntos por cluster, covarianza de rango bajo); con n = 1000 es 0 % y la potencia ≥ 0.85 en todo ratio;
  z ≥ +2 nunca. Se aplicó la rama de apéndice tal cual se fijó. Si decides validar **por régimen** (ImageNet, n = 1000: 4/12 con z ≤ −2), la rama
  "texto principal" (párrafo, figura en el cuerpo, frases de abstract/intro) está en `rebuttal/scripts/phaseB_t3_apply.py`; cambiar la regla
  después del barrido debería declararse en §3. Alternativa limpia: repetir `expR55b` con covarianza regularizada (shrinkage) en clusters
  pequeños o con n = 100 solo hasta K = 6 (≈ 2 h de CPU), y re-decidir.
- `expR55b` tardó 2 h 08 min (4 procesos, 600 runs); logs en `rebuttal/results/expR55b_k{6,12,20,30}.log`.
- `fig_depth_test.pdf` está en el apéndice A.5; `fig_depth_power.pdf` (solo potencia, generado por expR55b) no se incluye en el .tex.
- Limitación (i) editada (única edición en Limitations): la frase "against matched stars the residual depth is marginal" afirmaba un
  resultado retirado (B29).

## Añadido en la pasada final (seis puntos)

- **Acotado por régimen del test de profundidad**: es una decisión posterior al barrido y el texto lo dice ("a scoping adopted after the
  sweep"; A.5 "The bar was fixed before the sweep; the restriction to the validated regime was adopted after it"). Si prefieres volver a la
  regla estricta, `rebuttal/scripts/phaseB_t3_apply.py` contiene la rama "not certified" (párrafo, abstract, intro).
- **Contraste aleatorio vs coherente bajo el registro**: más débil que bajo el supremo. Los aleatorios llevan más exceso solo a C pequeño
  (DINOv2-L hasta 100, DINOv2-G hasta 20; CLIP-L a todo C); la ventaja de explotabilidad (NC adv) sí se mantiene a todo C. El título
  "Hierarchy depth, not class count" se conserva; valorar si el argumento del "alcance" de la jerarquía (antes en Limitación (i)) debe
  reformularse más.
- **Lecturas que siguen bajo el protocolo original**: A1 (exp11: rejilla supremo × gaussiana) y A8 HierarCaps (exp16); B20 es la rejilla
  de registro. A9 muestra ahora las dos lecturas. Rehacer A8 bajo el registro: re-extraer HierarCaps (minutos de GPU) + censo (~10 min).
- Coste: expR60 67 min (6 workers); expR61 ~25 min (GPU para los embeddings + 3 censos × 3 construcciones).

## Añadido en la reestructuración

- **Subtítulo**: decidido, "Calibrating Latent Hyperbolicity in Foundation Models" (CHANGELOG §19).
- **Nivel de muestra**: 10/24 celdas genuinas bajo el registro (no 2–3): DINOv2 en CIFAR-100 y DTD, SigLIP-B en CIFAR-100, ViT-S/B/L en DTD, con excesos
  ≤ 0.023. El texto dice "dentro del ruido en la mayoría; pequeño donde sobrevive". Si prefieres la frase fuerte del brief, no la sostienen los datos.
- **Ganancias Poincaré − coseno**: no son "nada para el resto" (DINOv2-S CIFAR-10 +1.1, ViT-L DTD +1.0, DINOv2-G −1.7 pp); §6.3 lo dice como "inconsistente".
- **MERU/CLIP en ImageNet (imágenes)**: r 171–197, no todas genuinas BH sobre las 24 celdas; el párrafo no las llama genuinas. Con prompts de texto en
  ImageNet los CLIP dan exceso positivo (r 2–15) y MERU ~0: dato colateral, solo en B30.
- **Lecturas aún bajo el protocolo original**: A1 (exp11) y A8 HierarCaps (exp16). Coste de A8: re-extraer HierarCaps + censo (~15 min).
- **Mettes et al. (IJCV 2024)**: no citado; el texto de arXiv no enuncia la premisa.
- Costes: R7 ≈ 1 h 10 min (4 workers, ~7 min/celda); R8 ≈ 40 min (3 workers).
- Página 9 tiene ~8 líneas de holgura con el hueco de 1.2 in; si la Fig. 1 real es más alta, hay margen.

## Añadido en la pasada de prosa

- Lee `SUMMARY_plain.md` (diez líneas) y comprueba que la historia tiene una sola voz; la tesis aparece literal cinco veces en el texto.
- Los números que salieron de la prosa están en los pies de tabla generados (lista completa en CHANGELOG §20); si echas de menos alguno en el
  cuerpo, el check "prose" del sweep te dirá si al reponerlo se rompe la regla de dos por párrafo.
- Limitación (i) se reformuló sin cifras ("class sets of ImageNet's size"); las cifras (n ≈ 1000, diez puntos por cluster) siguen en A.5 y B35.

## Añadido en el positive-control pass

- **El control de profundidad en nubes reales es negativo en potencia**: el árbol implantado (hermanos separados) no se detecta en ningún backbone al
  nivel de ruido real de ImageNet (cociente intra/entre 1.3–3.9), y sí en los 12 cuando los offsets se encogen a 0.6. El texto lo dice como "conservative
  and weak" y convierte los ocho no certificados en "not detected". Si prefieres otra lectura, el párrafo es "On real clouds the test is conservative and weak".
- **R12 (marco equilibrado, complete linkage)** no supera la regla pre-registrada (potencia 0.02 a s = 1) y queda en B37; en ese marco los certificados
  son ViT-T/B/L (ViT-S y DINOv2-L en z −1.7): el conjunto de cuatro certificados es sensible al marco, no lo dice el cuerpo, sí la tabla.
- **R10** es inconcluso (frozen -3.62, CE -2.33, CE+jerárquica -2.77) y está en A.5; la limitación del control entrenado se mantiene.
- Coste: R9b + R12 ≈ 4,5 h con 7 workers; R9 (v1) ≈ 8 h; R10 ≈ 1 h 15 min en dos GPUs.

## Añadido en el final pass (Fase B, congelado el 17 de septiembre de 2026)

- **AI Use Statement**: escrito en la forma de la plantilla ICLR 2027 (usos declarados: redacción, código, análisis; no usados: ideación,
  resultados). Cotejar las categorías con la política oficial antes de enviar.
- **Figura 1**: hueco de 1.4 in (`figures/fig1_concept.pdf`); **Figura 4**: hueco opcional para `figures/fig4_schematic.pdf` (5.5 × 0.9 in)
  encima del panel (a); si el fichero no existe no aparece nada. **Enlace anonimizado**: `TODO(author)` en el Reproducibility Statement.
- Tabla 15 (panel de modelos): el bloque "Controls" lleva parámetros tecleados (DeiT-B 86M, ViT-B augreg 86M, MERU/CLIP 22M/86M/307M,
  ResNet-50 24M), como el resto del panel heredado; verificar si se cita.
- Abstract: "a gap of about a third remains" se apoya en la mediana (0.34) de la fracción de acuerdo intra-bloque ausente sobre
  las configuraciones admisibles de ImageNet × tres medidas (pie de la Tabla 9); bajo la configuración seleccionada y el ARI es un quinto
  (0.38/0.48) y bajo la cofenética 0.38. Si prefieres otra base, el sweep acepta [0.25, 0.42].
- "+0.9 to +1.3 pp" (§6.4) es la ganancia Poincaré−coseno de los VLM contrastivos en los conjuntos de transferencia; en ImageNet es +0.1
  (Tabla 13). Lead-in de §5.2 con coma en vez de punto y coma (regla de puntuación). Tres URLs de NeurIPS retiradas de la bibliografía.
- Solo cambios tipográficos a partir de ahora; el sweep debe seguir en PASS completo tras cualquiera.

## Versión paralela v2 (2026-09-18)

- `ICLR2027/main_iclr2027_v2.pdf` (41 páginas; texto principal hasta la p. 12) frente a `main_iclr2027_final.pdf` (p. 9). Comparar y
  decidir; `V2_DIFF.md` lista qué se movió, qué pasó a definición/ecuación y qué se borró como redundante, y cinco recortes candidatos
  (nada recortado). Si se elige v2, hay que decidir los recortes hasta 9–10 páginas y renombrar el fichero de envío.
- Cualquier cambio de texto en v2 se hace en `rebuttal/scripts/phaseE_paper_v2.tex.tmpl` y se re-aplica con `phaseE_v2.py`.

## Versión v3 (2026-09-18)

- `ICLR2027/main_iclr2027_v3.pdf`: texto principal hasta la página 11 tras los tres recortes prescritos. Para llegar a la página 9 hacen
  falta ~65 líneas más; candidatos (nada aplicado): enunciados del Lema 1, Corolario 1 y Remark al apéndice con un puntero (~8 líneas);
  §4 a media página quitando "Extraction" (ya en A.14, ~6 líneas); Definiciones 6–7 fundidas en un párrafo de prosa (~6 líneas);
  §5.4 de once párrafos a siete uniendo los de un solo número (~10 líneas); §3.5 a la mitad (~4 líneas). Con todo ello queda en ~10 páginas;
  la página 9 exige además recortar afirmaciones de §5.
- Dos cifras del brief no coinciden con los ficheros y se han escrito con el valor del fichero: "3–8 null spreads" → 1 a 36; "0.88" → 0.89.
- Cambios de texto en v3: en `rebuttal/scripts/phaseE_paper_v3.tex.tmpl`, re-aplicar con `phaseE_v3.py` (después de `phaseE_final.py`).
- (2026-09-18, segunda y tercera ronda) Con los tres recortes prescritos aplicados, las referencias de v3 empiezan en la página 11.
  Para que empiecen en la 10 hay que liberar las líneas que ocupa la página 11 antes de las referencias (ver el informe de la sesión);
  candidatos sin tocar afirmaciones: §3.5 a la mitad (~4 líneas), §4 'Extraction' en una frase (~4), Proposición y remark al apéndice
  con un puntero (~6), fusión de párrafos de un solo número en §5.4 pasando la cifra al puntero de tabla (~6). Decidir.

## Versión final (2026-09-18, `main_iclr2027_final.pdf`)

- La página 9 se alcanzó con recortes de prosa más allá del orden de corte del brief y con espaciado tipográfico (CHANGELOG §27,
  `FINAL_CHECK.md`). Revisar ambos: las frases eliminadas no quitan afirmaciones, pero son decisión del autor; el espaciado se revierte
  borrando las líneas marcadas "typographic only" que `phaseE_submission.py` escribe en el preámbulo.
- Apéndice: 16 páginas frente a las 14 del brief. Candidatos (no citados desde el texto principal): paneles (f) y (g) de la tabla del
  corolario, panel (c) de extracción de la tabla de texto, notas de extracción de la tabla del panel de modelos.
- Dos paneles se eliminaron por extensión de la lista del brief ("null-variant panel", "uncited supremum tables"): el (b) del censo
  (veredicto bajo las cuatro construcciones) y el (b) del corolario (correlaciones del supremo). Restaurar = quitar la entrada de `DROP`
  en `phaseE_submission.py`.
- La limitación (viii) sobre ξ se eliminó al irse ξ del paper; el resto de las limitaciones va en un solo párrafo de seis puntos.
- `figures/fig1_concept.pdf` no está en el repo: la Figura 1 compila como la caja de 1.4 in del `\IfFileExists` del autor.
- (Cuarta revisión) Tabla 11(c): la cabecera "supremum, Gaussian" describía correctamente `exp14_dbpedia.csv` (coeficientes gaussianos).
  Para que diga "Haar" sin mentir, la copia final usa el exceso del supremo bajo el nulo Haar de `expR61_dbpedia_record.csv`
  (`excess_haar_sup`); los tres valores cambian en ~0.004. Confirmar que era eso lo que se quería.
- (Cuarta revisión) §6 "the best simple policy is cosine everywhere": en la Tabla 13e la regla por objetivo (sin validación) da +0.28 pp
  frente a +0.23 de "always cosine" (60 celdas; +0.41 frente a +0.28 en las 40 jerárquicas). El texto lo dice como "the objective rule
  adds little"; decidir si se prefiere otra formulación.
- (Cuarta revisión) Tabla 2(d) vuelve a listar el barrido completo de C para DINOv2-L (la leyenda del brief necesita varios C).
- (Quinta revisión) Regla de Khrulkov: el rango 0.48–2.5 ya venía del supremo (`exp1_delta_controls.csv` sólo tiene `delta_max`, la
  misma columna que la fila gaussiana de la Tabla 3); recalcularlo con esa banda no lo cambia. Si se quería otra banda, decir cuál.
- (Quinta revisión) Entrada bib `gu2019learning` (Gu, Sala, Gunel, Ré, ICLR 2019): DBLP, OpenReview y Semantic Scholar bloquean el
  acceso desde esta máquina; escrita desde las actas. `sala2018representation` verificada contra proceedings.mlr.press (v80, 4460–4469).
- (Quinta revisión) "up to fourteen levels": h = profundidad máxima en WordNet del hipónimo común de cada clúster del corte K=30
  (mediana 9); el máximo lo da un clúster de una sola clase (assault_rifle, profundidad 14). Si se prefiere la mediana, cambiar `h_word`.
- (Decisiones del 2026-09-20, aplicadas; CHANGELOG §30) El registro es el nulo Haar centrado (expR75) y la afirmación de profundidad
  lleva la redacción del autor en resumen, §1, contribución 2, tesis, §5.3, MERU y Figura 4. Queda sin regenerar bajo el nulo centrado
  lo que no pedía el brief: bootstrap (expR59/expR73), presupuesto (expR72), nivel de muestra (expR62, n=1000), censos de texto
  (expR53/expR61), corolario (expR68). El término de centrado mueve el exceso como máximo 0.0013 en DTD (n=47), 0.0008 en CIFAR-100
  (n=100) y nada en ImageNet (n=1000), frente a 0.053 en los conjuntos de diez clases; decidir si se rehacen por coherencia.
- (Quinta revisión) La Figura 1 del autor pasa a `[t]` (flota a la cabeza de la página 2) para no perder seis líneas al pie de la página 1;
  y `parskip` baja de 6 pt a 4 pt. Ambas se revierten en `phaseE_submission.py` (EDITS y preámbulo).
- Figura 5(c): la leyenda literal dice "DINOv2 loses most of its structure under Euclidean distance". Con azar 0.5, la pérdida de la
  estructura por encima del azar es 63% (DINOv2-G), 50% (L), 30% (B) y 8% (S): "most" vale para L y G; S y B pierden menos. Decidir si
  se matiza ("the larger DINOv2 models").
- `TODO(author)`: enlace del repositorio anónimo en el Reproducibility Statement.
- Cambios de texto de la versión final: en `rebuttal/scripts/phaseE_paper_final.tex.tmpl`; regenerar con
  `FINAL_CUTS=s55,s6,table,s2 python rebuttal/scripts/phaseE_submission.py` y después `gen_provenance.py main_iclr2027_final.tex
  tab_z_provenance_final.tex final`.
- (Prioridad 1c, si la regla se cumple) El resumen recibe la frase del autor y, para no pasar de 250 palabras, la misma oración pierde
  cinco palabras (CHANGELOG §33). Confirmar o proponer otro recorte. Con esa integración los tres statements pasan de la página 9 a la 10
  (el texto principal sigue acabando en la 9).
- Proceso ajeno a esta sesión: `conda install -y -c conda-forge tectonic` (PID 2767839) lleva 157 días al 99 % de una CPU con 8,4 GB de
  memoria; es un solver colgado de una sesión de abril. No se ha tocado; matarlo libera un núcleo para los experimentos.
- (Brief del 2026-09-21, CHANGELOG §37) Tres decisiones: (a) el título de la Tabla 9 fijado en R2 dice "none at the top-level frame
  used on real backbones"; expR79 muestra potencia 0.8 en ese marco con el espectro real y tres niveles: matizarlo o dejarlo (describe
  el barrido de expR55). (b) Se actualizó la limitación (vi), no la (viii) que decía el brief, porque es la que habla de la lectura a
  nivel de muestra. (c) Siete recortes de una o dos palabras en el resumen para quedar en 249 con tus dos frases nuevas; la frase
  "including a published reading reproduced and calibrated" no cabía.
- (Retoque del resumen, 2026-09-21) Tu frase nueva sobre el backbone hiperbólico son ocho palabras más; para quedar en 250 se
  recortaron cinco sitios más (CHANGELOG §37, último punto). Si prefieres el texto sin esos recortes, queda en 257 palabras.
- (Resumen, 2026-09-21, pendiente) Con tus tres reversiones el resumen tiene 253 palabras, tres sobre el tope de 250 de la sexta
  revisión; el check del sweep queda en FAIL hasta que decidas: aceptar 253 (subo el tope del check), o recortar tres palabras de tu
  elección (los dos recortes sin contenido que quedan, "The trees are" → "Trees are" y "the naive comparison" → "naive comparison",
  sólo llegan a 251).
- (Séptima revisión, 2026-09-21) La acotación del titular añade once palabras al resumen: 264 con tus tres reversiones anteriores.
  El sweep mantiene el tope de 250 (check en FAIL) hasta que decidas qué hacer con el resumen. La limitación (iii) da el rango DINOv2
  como 3.0–3.9 (los cuatro DINOv2, desde expR64b); tu brief decía 3.7–3.9, que es el rango de B/L/G sin DINOv2-S (3.0).
- (Séptima revisión, continuación) Tres notas: (a) el resumen sube a 276 palabras con la forma corta en las dos frases; el sweep
  sigue en FAIL por el tope de 250. (b) El control plano desacoplado va de −1.6 a −1.8 según expR79 (la media más baja es −1.76); el
  brief decía −1.7; el texto lleva el valor del fichero. (c) DINO-B (razón 2.1) no está ni en "supervised and contrastive (1.3–2.0)"
  ni en "the DINOv2 family (3.0–3.9)"; expR81 lo mide igualmente.
- (Octava revisión) (a) La declaración de uso de IA nombra los tres puntos del formulario y conserva la aclaración de la quinta
  revisión ("namely the implementation and execution of experiments… and LLM-simulated reviews…"); si las casillas de OpenReview dicen
  otra cosa, pega su texto y lo igualo. (b) Las z de §5.3 van con dos decimales (−4.19 to −4.82; −1.61 to −1.76; ViT-L −1.61), no con
  los redondeos mixtos del brief. (c) "DINO-B (2.1) and ViT-B (2.0) lie at the edge of the covered range" está en la limitación (iii).
- (expR82, 2026-09-21) Tres ajustes de forma en tu texto: el paréntesis "(z from −2.26 to −3.99 under both stars)" va en aposición
  (regla de paréntesis); el punto y coma se parte en dos frases para que el párrafo nuevo tenga tres con el puntero; en la limitación
  (iii) sustituí sólo la frase de "in progress" y dejé "It certifies alignment above its frame only, and its one trained positive
  control is inconclusive." (leí "the sentence before it" como la primera mitad de esa misma frase). El rango −2.26 a −3.99 es el de la
  estrella anisótropa; bajo la estrella Haar-hub el rango es −2.08 a −3.57, también todo ≤ −2.
- (Pase de cierre, 2026-09-22) Decisiones de forma mías, por si las quieres cambiar: el párrafo "The two-level implant is missed at
  ImageNet's noise level" sale de §5.3 (su contenido está en tu frase, la Figura 4b y la Tabla 9; el puntero a la Figura 4b pasa al
  párrafo del desacoplamiento); "Text depends on recipe, scale and probe" pierde las dos frases de la sonda y las plantillas y su
  entradilla pasa a "Text depends on recipe and scale"; en el control positivo "the WordNet 30, 6 and 2 cuts" va con palabras ("cuts
  of thirty, six and two superclasses") por la regla de números; la frase de los recuentos se parte tras "on the frame of record".
- (Lista del revisor) `make_reviewer_checklist_third.py` busca en el PDF frases ancla de la tercera revisión para citar líneas; unas
  doce ya no existen en la prosa actual (p. ej. "Supervision on leaf labels alone can produce this depth", "The calibrated reading
  predicts the zero-cost gain") y el script las señala como "missing anchors" sin dejar de escribir la lista. Cosmético; se pueden
  actualizar a las frases actuales si la lista se va a entregar.
- (Novena revisión, 2026-09-22) Dos decisiones tuyas: (a) RESUELTO, se queda "as a whole". "about a third" en §5.4: la mediana de agreement que falta (0.34) es
  sobre las doce medidas de las cuatro configuraciones admisibles de ImageNet, la seleccionada incluida; sobre las otras nueve medidas
  sale 0.50. El texto dice ahora "About a third of the within-block agreement is missing under the admissible configurations as a
  whole."; si quieres que se refiera a las otras configuraciones, lo honesto es "about half". (b) RESUELTO por tu brief de la tarde del 22 (tu frase, con los tres recuentos como rellenos). Frame balanceado: la construcción
  emparejada (hubs por el propio frame) da 0 de 50 desacoplado, y por tu regla entró "ViT-B shows hub structure under that grouping and
  the frame of record does not". Pero la literal (clouds planos agrupados por WordNet-30 y leídos en el balanceado) dispara 5 de 5
  intacta y 38 de 50 desacoplada: un cloud organizado por un frame dispara en el otro aunque se desacople. El 7 de 10 del ViT-B
  congelado en el balanceado admite esa lectura de desajuste de frames; si prefieres decirlo así, la frase sería "That frame's flat
  control fires once decoupled in 0 of 50 runs, but a flat cloud clustered by the frame of record fires on it in 38 of 50, so the
  balanced-frame reading of ViT-B may be frame mismatch". Las dos filas están en la Tabla 9(f).
- (Limpieza, 2026-09-22 tarde) Tres entradillas nuevas son mías, porque la regla exige negrita al frente de cada párrafo y tus
  reemplazos quitaban la que había: "Size and evidence." (§3.3), "What a genuine excess means." (§3.3) y "Calibration budget and
  frames." (§4). Cámbialas si prefieres otras. Y la frase acordada del frame balanceado va en §5.3(e) sin los dos recuentos entre
  paréntesis (están una sola vez en (c), como pide "each number once"): "That frame's matched flat control does not fire once
  decoupled, so this is not a false alarm: … even once decoupled, so frames are not interchangeable."
- (Décima revisión, 2026-09-22 noche) Dos avisos: (a) "not that the hubs of the frame form a hierarchy" sube el resumen a 303
  palabras sin fórmulas, tres sobre tu tope de 300; el check del sweep falla hasta que recortes tres palabras o subas el tope (no he
  tocado tu texto). (b) El techo intra-modelo de tripletas es 0.90 de media (0.85 en DINOv2-L), justo en el umbral de tu regla; seis
  de los doce backbones quedan por debajo de 0.9. He aplicado la rama "keep" y la tesis no cambia; si prefieres leerlo como cercano
  al 0.76, dímelo y cambio la tesis a "share their trees up to the noise of the comparison".
- (Tesis tras expR84, 2026-09-22 noche) Dos cifras de tu brief no coinciden con el fichero y el texto sigue al fichero: el techo de
  tripletas por backbone va de 0.85 a 0.96 (DINO-B 0.965, ViT-L 0.955), no a 0.95; y "ARI 0.64–0.79" / "cophenetic 0.91–0.94" eran el
  mínimo y la media, no el rango (rangos reales 0.64–0.94 y 0.91–0.97, medias 0.79 y 0.94). Si preferías rango mínimo–media, dímelo.
- (Tesis tras expR84) El resumen queda en 318 palabras sin fórmulas con tu frase nueva del compartir, ocho sobre el tope de 310 que
  fijaste; el check del sweep falla hasta que recortes o subas el tope.
- (Pasada consolidada, 2026-09-22 noche) Tu resumen nuevo tiene 361 palabras sin fórmulas y tu tope es 310: el check del sweep
  falla por eso y por nada más. O recortas unas 50 palabras o subes el tope (dime a cuánto).
- (Pasada consolidada) La página 9 depende del mismo recorte: con el resumen de 361 palabras el texto principal termina seis
  líneas dentro de la página 10 (limitaciones vi–ix). Con el resumen en tu tope de 310 cierra en la 9 sin tocar más texto.
- (Resumen de las 17:25) 337 palabras sin fórmulas: 27 sobre el tope de 310. La página 9 tampoco cierra con él (seis líneas de
  las limitaciones vi–ix en la página 10): o recortas ~6 líneas de contenido tuyo (limitaciones o §5.4), o aceptas la página 10.
- (Figura 5, disposición, 17:30) Página 9: tras vaciar §5.5 y §6 hasta párrafos de dos frases, la página 9 acaba en la limitación
  (vi) con una línea libre y el párrafo de alcance (vii)–(ix), de cuatro líneas, pasa entero a la 10 (la clase impide partirlo con
  una línea libre). Faltan tres líneas de contenido tuyo: por ejemplo fundir (vii)–(ix) en dos frases, o recortar (iii).
- (Legibilidad §5.3–§7, 2026-09-23 12:45) §5.4 "Controlled, the trees share their topology, not their metric.": tu frase decía "0.5 by
  chance", pero el acuerdo de tripletas de expR58/expR84 es cuál de los tres pares de cada tripleta se fusiona primero, y dos árboles
  independientes coinciden en 1 de 3; el 0.5 es el azar de la tarea binaria de tripletas de hermanos de CIFAR-100 (Figura 5c). El
  texto dice "Chance gives 0.33." (frase aparte por el tope de dos números). Si quieres el 0.5, la frase tendría que referirse a la
  Figura 5c y no a la Tabla 11. Fuera también "about a third" (resuelto por tu brief). La frase de §6 "The raw and calibrated
  readings…" tenía 32 palabras y va partida en los dos puntos. El pie de la Figura 5 sigue diciendo "shared almost up to noise".
- (Bibliografía, 2026-09-23 13:40) El fichero que compila es `references.bib`, no `iclr2027_conference.bib`: tus 52 entradas están ya
  en `references.bib` (más las 3 citadas que faltaban en el tuyo). Si vuelves a tocar referencias, hazlo en `references.bib` o dime
  que cambie la plantilla a `\bibliography{iclr2027_conference}`. Decisiones a confirmar: Gröger et al. como ICML 2026 (arXiv dice
  "ICML 2026 camera-ready"; tu entrada era el CoRR) y "Shuo Wen" (la anterior decía "Song Wen"); Aggarwal et al. con "High
  Dimensional Space" en singular (título del PDF de los autores; DBLP pone "Spaces").
- (§1 reescrita, 2026-09-23 16:50) "depth test" → "hierarchy test" hecho en todo el envío. Quedan con "depth" en sentido de concepto,
  no de estadístico, y no los toqué: título de §5.3 "Depth above the labelled clusters", pie de la Figura 1 ("clusters without
  depth… with depth"), §4 "creates depth", §3.3 "not depth", limitación (i) "depth needs the separate test", pie del apéndice
  "misses implanted depth". Di si los cambias (p. ej. "Hierarchy above the labelled clusters"). Tu contribución 2 dice
  "hyperbolic-trained backbone" y sustituye a "nominally hyperbolic backbone" (vocabulario de la novena revisión). Contribución 3
  ya no menciona la isla; la isla queda en §5.4.
- (Tabla 1, 2026-09-23 17:20) Pediste en el pie "CIFAR-100 is not below the null at 0.05 under either"; con el estadístico de registro
  CIFAR-100 queda por debajo (198/200, p mayor 0.030; §5.1 lo dice). El pie dice "Under the supremum, CIFAR-100 is not below the
  null at 0.05." Confirma o da otra redacción.
- (Tabla 1, 2026-09-23 17:40) Resuelto con tu criterio de la mediana (§75). La frase nueva de §5.1 tenía 35 palabras y va partida
  en dos en "…stays below it. The verdict for CIFAR-100…".
- (Bibliografía, 2026-09-23 18:15) CUB-200-2011: el registro de Caltech Authors (cvm3y-5hh21) numera el informe CNS-TR-2010-001;
  el informe se cita habitualmente como CNS-TR-2011-001. La entrada lleva el número del registro. Cambia si prefieres el otro.
- (Apéndice, 2026-09-23 20:00) Tres decisiones tuyas: (i) el apéndice queda en 16 páginas, una por encima de tu tope, y bajar a 15
  exige quitar material citado (Tabla 5(b) o 5(c) son las candidatas); (ii) "of record" sigue en el texto principal y ya no en el
  apéndice; (iii) cuatro números que cita el texto principal ya no están escritos en el apéndice (72, 47, 34, +0.9).

