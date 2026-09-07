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
como `iclr2027/iclr2027/figures/fig1_concept.pdf` (vectorial), ancho de texto, ≤ 1.3 in de alto.

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
  `iclr2027/tool/calibrated_delta.py` (uso: `python calibrated_delta.py centroids.npy [--labels sup.npy] [--reps 200] [--json out.json]`);
  `iclr2027/tool/run_checks.py` reproduce dos celdas de la Tabla 1 y de B29 y escribe `rebuttal/results/tool_check.json`, que
  `sweep_freeze.py` compara en cada freeze (re-ejecuta el check solo si falta el JSON; ~15 min de CPU).
- Para el check de B29 en DINOv2-L/ImageNet el tool recibe los centroides del **store** (como hizo expR50); con los de la caché del censo
  el test de profundidad da un valor ligeramente distinto (store ≠ caché, ver CHANGELOG §0). Si en algún momento se rehace B29 sobre la caché,
  cambiar la línea correspondiente de `run_checks.py`.

## Añadido en la respuesta a la revisión (Fase A)

- **Control positivo con los checkpoints fine-tuned jerárquicamente (Fig. 5a)**: no hay pesos ni features guardados en
  `Platonic/` (solo `analysis4_finetuning.csv`, `hierarchical_finetuning.csv` y logs; `run_finetune_ablation.py` y
  `exp_hierarchical_finetuning.py` no guardan nada). Coste de rehacerlo: re-entrenar ViT-B, DINOv2-S y CLIP-B con el objetivo
  jerárquico de CIFAR-100 (el log original da ~1.8 h de GPU para ViT-B; ~4–5 h para los tres en una 2080 Ti), extraer
  centroides de CIFAR-100 (minutos) y pasar `iclr2027/tool/calibrated_delta.py --labels` (minutos). Guardar esta vez los
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
- **El tool** (`iclr2027/tool/calibrated_delta.py`) sigue ahora el protocolo de registro (Haar × p99.9; estrella anisotrópica con 10
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
