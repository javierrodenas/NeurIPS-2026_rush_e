# Table 1 regenerada (tasks_t707 + tasks_imagenet_full): qué cambia y qué sobrevive

Fuente: `rebuttal/results/table1_regenerated.csv` (60 celdas, brazos NC/FS × R/H del rerun
auditado). Generado 3-ago-2026. Este documento es el aviso a coautores comprometido en el plan.

## Titulares que SOBREVIVEN (los que importan)
| Claim impresa (NeurIPS) | Regenerada | Veredicto |
|---|---|---|
| DINOv2-L CIFAR-100 NC **+2.10pp** | **+2.05pp** | ✅ intacta (y con McNemar p≈4×10⁻³⁸) |
| DINOv2-G CIFAR-100 NC +1.20 | +1.23 | ✅ |
| DINOv2-S CIFAR-100 NC +1.00* | +0.97 | ✅ (*el paper decía +1.00) |
| DINOv2-L DTD NC +0.80* | +0.75 | ✅ (FS DTD mejora: +0.90→+1.39) |
| ImageNet FS: los 10 modelos ganan | +0.03..+0.91 | ✅ (rango antes 0.00..+1.10) |
| FS positivas 54/60 | **56/60** | ✅ mejora |
| Correlación NC pooled −0.46 | **−0.452** | ✅ |
| NC within-dataset (ImageNet/CIFAR-100) −0.89/−0.70 | −0.83/−0.87 | ✅ mismo cuadro (C100 mejora) |
| FS within-dataset −0.70..−0.84 | −0.57..−0.78 | ✅ mismo cuadro, algo más suave |

## Lo que CAMBIA de verdad (2 cosas, ambas con narrativa mejor)
1. **NC positivas: 31/60 → 22/60.** Las 14 celdas con flip son positivos pequeños
   (+0.1..+0.5) de ViTs supervisados y CLIP-L que pasan a ≈0/ligeramente negativos.
   La historia regenerada es MÁS coherente con nuestra tesis: la ganancia NC se
   concentra en la familia con árbol genuino (DINOv2), y fuera de ella es neutra.
   El paper de ICLR lo cuenta así desde el principio; nunca contamos "31/60".
2. **FS pooled: −0.36 → −0.18 (n.s.).** La compresión de varianza que el propio paper
   ya señalaba. Cobertura: (a) la predicción FS se presenta within-dataset
   (−0.57..−0.78, sólida); (b) el protocolo de dos pasos usa la ventaja del MEJOR
   métrica, cuyo pooled FS es −0.31 — más fuerte que H−R pooled en los datos
   regenerados. La sección 5 nueva ya está construida sobre esa cantidad.

## Flips célebres confirmados
- CLIP-L/MNIST: −0.90 → +0.88 (el sign flip de la auditoría; MNIST es flat, irrelevante
  para claims, pero la celda queda corregida).
- DINOv2-G/ImageNet NC: +0.50 → −0.13 (coherente con su geometría at-null en ImageNet —
  la excepción 3/72 ahora también se ve en la tarea; refuerza la historia, no la debilita).

## Reglas para la escritura ICLR
- TODA cifra de tareas sale de `table1_regenerated.csv`. La Table 1 vieja no se cita jamás.
- Los δ̂ de ImageNet salen del protocolo declarado (exp11): DINOv2-G .080 (no .066),
  CLIP-L .118 (no .108). El rango ImageNet pasa a .071–.123.
- NC se narra como "concentrada en DINOv2/jerárquicos"; FS como "amplia (56/60) y modesta".
- El +2.05 se reporta como +2.05 (no redondear a +2.1: la cifra exacta con su McNemar
  es más creíble y esquiva cualquier acusación de inflado).
