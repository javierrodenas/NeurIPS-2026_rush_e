# Checklist de cobertura — cada pregunta/objeción de cada reviewer → dónde se responde

Estado: ✅ respondido con datos · 📋 respondido con compromiso de revisión · Referencias A# = sección de la respuesta correspondiente.

## Reviewer 1Eqj (reject — claridad)

| # | Punto del review | Dónde | Estado |
|---|---|---|---|
| 1 | Intro fragmentada (4 párrafos) | A7.2 | 📋 |
| 2 | Afirmaciones fuertes sin referencias en intro | A7.2 | 📋 |
| 3 | Verbo que falta en L59 | A7.2 | 📋 |
| 4 | §2.3 es preliminares, no related work | A7.3 | 📋 |
| 5 | §2.4 solo discute Huh et al.; mover a 2.2 | A7.3 | 📋 |
| 6 | Símbolo $\mathbb{H}$ en prosa (L84) | A7.3 | 📋 |
| 7 | §3.1 sin citas | A7.4 | 📋 |
| 8 | §3.2 sin intuición de por qué δ/ORC | A7.4 | 📋 |
| 9 | "low δ" — ¿qué es bajo? | A3 | ✅ calibración + nulls |
| 10 | Paradoja H² (log(1+√2)) vs S^99 (0.127) | A2 | ✅ tabla absoluto-vs-normalizado |
| 11 | ¿Es correcto normalizar δ y luego decir "bajo = arbóreo"? | A4 | ✅ |
| 12 | Fórmula L170 incomprensible | A6 | ✅ + 📋 diagrama |
| 13 | "argmin no es un verbo"; FS informal | A7.4 | 📋 |
| 14 | Sin tablas en el main; apéndice sin referenciar | A7.1 | 📋 tabla main-text |
| 15 | §4.1 "¿dónde miro?" | A7.5 | 📋 |
| 16 | L214 "collapse toward zero" incorrecto (son negativos) | A5 | ✅ concedido con corrección |
| 17 | Fig. 2 demasiado pequeña | A7.5 | 📋 |
| 18 | §4.2 L224-228: ¿qué se concluye del ranking ORC? | A7.4 | ✅ análisis de puentes (vía RJje-A7) |
| 19 | Referencia rota L242 | A7.5 | 📋 |
| 20 | Practical protocol mal ubicado | A7.5 | 📋 |
| 21 | §4.4 A1–A4 telegráficos | A7.5 | 📋 |
| Q1 | ¿Qué es el plano hiperbólico unitario? | A1 | ✅ |
| Q2 | ¿De dónde sale el 0.127 de S^99? | A2 | ✅ (calibración propia, protocolo declarado) |
| Q3 | ¿Cuánto es δ "suficientemente baja"? | A3 | ✅ |

## Reviewer RJje (reject — técnico)

| # | Punto | Dónde | Estado |
|---|---|---|---|
| W1 | Interpretación excesiva: centroides tree-like ≠ espacio genuinamente hiperbólico | A2 | ✅ concesión + renombrado a "tree-like inter-class metric structure" |
| W2a | No se estima signo/valor de curvatura; −1 elegido | A2, A5 | ✅ concedido; sweeps t/c muestran insensibilidad |
| W2b | Falta comparación con curvatura positiva / baseline esférico | A1, A2 | ✅ esfera sube δ en 12/12; calibración |
| W2c | Gaussian / normalized / whitened / dimension-matched controls | A2 | ✅ todos corridos (+ nulls espectral y permutación, extra) |
| W3 | Tensión interna del ORC | A7 | ✅ análisis de aristas puente |
| W4 | Confound de etiquetas/prompts (centroides heredan taxonomía) | A6 i-ii | ✅ groupings + decoupling WordNet |
| W5 | "Objetivo > modalidad" confundido por template compartido | A6 iii | ✅ prompts + claim acotado |
| W6 | Ganancia = ¿rescalado radial? Comparar con mismo centrado/escala/normalización | A3, A4 | ✅ RT + COS/COSC + stack normalizado |
| Q1 | ¿Negativa mejor explicación que positiva? | A1 (+A2) | ✅ |
| Q2 | Baseline esférico en comparaciones principales | A2 | ✅ + 📋 tablas en revisión |
| Q3 | ¿L2 / cosine / esférica en downstream? | A3 | ✅ split por familia |
| Q4 | ¿Misma transformación radial con distancia euclídea **o esférica**? | A4 | ✅ RT (euclídea) + COSC (esférica ≡ cosine centrado, por preservación de dirección) |
| Q5 | ¿Estimar curvatura en vez de fijar −1? | A5 | ✅ sweeps + concesión explícita de scope |
| Q6a | ¿Random class groupings? | A6 i | ✅ |
| Q6b | ¿Prompts distintos / solo nombre? | A6 iii | ✅ |
| Q6c | ¿Definiciones WordNet / conceptos no visuales? | A6 iii | 📋 comprometido para revisión (no corrido) |
| M1 | Conexión PRH no establecida; probar alignment cross-model | A8 | ✅ corrido (negativo) + reposicionamiento |
| M2 | Citar [1] Koepke, [2] Gröger | A8 | 📋 se citarán y discutirán |
| Lim | Discutir problemas de class templates en limitaciones | A6 iii | 📋 explícito |

## Reviewer Xbn5 (accept)

| # | Punto | Dónde | Estado |
|---|---|---|---|
| W1 | Solo 2 tareas prototype, solo visión; "universal exploitability" excesivo | A1 | ✅ negativos honestos + estudio de métricas; texto: 📋 future work declarado |
| W2 | Conexión PRH débil | A4 | ✅ |
| Q1 | ¿Tarea adicional (retrieval/clustering/texto)? | A1 | ✅ kNN/retrieval/recovery corridos (negativos, reportados); texto: 📋 |
| Q2 | ¿Robustez a prompts? | A2 | ✅ tabla 6 templates |
| Q3 | ¿Medida estadística (std/seeds)? | A3 | ✅ CIs pareados por episodio + 📋 McNemar |
| Q4 | ¿Por qué "tightly connected" al PRH? | A4 | ✅ experimento + reformulación |

## Reviewer pux6 (borderline accept)

| # | Punto | Dónde | Estado |
|---|---|---|---|
| W1/Q1 | Corrección semántica cuantitativa; jerarquías ground-truth (HierarCaps) | A1 | ✅ ρ WordNet + ARI superclases + decoupling; HierarCaps: 📋 |
| W2/Q2 | Circularidad del centroide / sesgo lingüístico | A2 | ✅ 3 medidas (groupings, decoupling, sin pooling) |
| W3/Q3 | Claims excesivos; deployment real (retrieval complejo, dense) | A3 | ✅ narrowing + negativos + hallazgo CLIP; dense: 📋 apuntado a MERU/HypLoRA |
| Lim | Limitaciones deben cubrir sesgo lingüístico y validación no-toy | A2 (final), A3 | 📋 explícito |

## Meta-review (5 puntos de discusión) → comment_AC_global.md

| Punto AC | Dónde |
|---|---|
| 1. Claridad | §1 (→ respuesta 1Eqj) |
| 2. Baselines esféricos/matched vs curvatura negativa | §2 (→ RJje A1-A2) |
| 3. Robustez: prompts, groupings, jerarquías explícitas | §3 (→ RJje A6, Xbn5 A2, pux6 A1) |
| 4. Downstream más allá de prototipos | §4 (→ Xbn5 A1, pux6 A3) |
| 5. Narrowing de universalidad/geometría intrínseca/PRH | §5 (→ RJje A2/A8, Xbn5 A4) |

**Únicos puntos sin experimento (los tres, reconocidos y comprometidos en el texto):** definiciones WordNet + conceptos no-visuales (RJje Q6c), tarea downstream de texto (Xbn5), HierarCaps (pux6), evaluación dense (pux6). Scripts preparados y no lanzados para fase 2: `exp12_curvature_sign.py` (signo de curvatura), `exp13_mcnemar.py` (McNemar NC).
