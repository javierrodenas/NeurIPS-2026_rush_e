# PROPUESTA (pendiente de aprobación de Javi) — inserción del resultado DBpedia

## Resultados crudos (exp14, `results/exp14_dbpedia.csv`)

| Embedder | δ̂ | null | exceso | tripletas L2 | NC R→H | FS R→H | FS H−R (CI95) | FS H−COS |
|---|---|---|---|---|---|---|---|---|
| BGE-base | 0.122 | 0.151 | −0.028 | 0.875 | 90.3→89.7 | 98.83→98.88 | +0.05 ±0.05 | +0.08 |
| E5-base | 0.127 | 0.151 | −0.024 | 0.879 | 91.6→91.2 | 99.10→99.13 | +0.04 ±0.03 ✓ | +0.01 |
| GTE-base | 0.132 | 0.152 | −0.020 | 0.890 | 90.0→89.2 | 98.67→98.75 | +0.08 ±0.06 ✓ | +0.07 |

Criterio prefijado (FS H−R>0 con CI fuera de 0 en ≥2/3): **cumplido** (E5, GTE). Magnitudes: minúsculas (+0.04..+0.08pp sobre techo ~99%). NC: ligeramente negativo.

## Texto propuesto para Xbn5 A1 (sustituye la línea "not run in this period")

- **Text-based task (new).** We transplanted the exact NC/FS protocol to text: DBpedia Classes (a real 3-level hierarchy, 9→70→219 leaf classes; 100 train + 30 test texts per class) with the three embedder families of the breadth panel (BGE/E5/GTE-base). Geometry: all three show genuine tree structure beyond their spectrum-matched nulls (excess −0.020..−0.028) and strong sibling alignment with the ground-truth level-2 hierarchy (triplet agreement 0.88–0.89 vs 0.5 chance). Tool: these embedders sit at δ̂ = 0.122–0.132 — *above* the paper's δ̂≲0.10 threshold — and the projection behaves exactly as the diagnostic predicts for that regime: NC slightly negative (−0.4..−0.8pp), FS marginally positive (+0.04..+0.08pp; CI95 excludes 0 for E5/GTE) at a ~99% ceiling. The diagnostic thus transfers to a new modality, correctly predicting *when hyperbolic distances help little* — which the paper frames as half of its claim.

(+ cambio "four directions; two negative, two positive" → "five directions; two negative, three sharpen", + media línea en AC §4.)

## ⚠️ Nota obligatoria decida lo que se decida
La línea actual de Xbn5 — "A text-based downstream evaluation **was not run** in this period" — ya es falsa (lo hemos corrido). Antes de publicar hay que: (a) meter la propuesta, o (b) reescribir esa línea de otra forma honesta. No puede publicarse tal cual.
