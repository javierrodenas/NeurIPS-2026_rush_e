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
