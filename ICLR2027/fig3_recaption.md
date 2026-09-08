# Figura 3: números full-D y recaption (4-ago-2026)

## Medición (CIFAR-100, centroides train, 20 superclases estándar)
| | DINOv2-S (d=384) | DINOv2-G (d=1536) |
|---|---|---|
| Ratio spoke/separación-de-hubs en dimensión completa | 0.743 | **1.040** |
| Varianza de hubs capturada por la PCA-2 del panel | 30.1% | 21.7% |
| Longitud de spoke conservada en el plano | 10.5% | **5.9%** |

## Lo que esto significa
1. En dimensión completa, los spokes de G son RELATIVAMENTE MÁS LARGOS que los de S
   (1.04 vs 0.74): la comparación visual del caption impreso ("G shows tighter and
   more compact spokes") no solo no está soportada, está INVERTIDA por esta medida.
2. La causa es exactamente el artefacto: el panel de G conserva solo el 5.9% de la
   longitud de spoke (vs 10.5% en S). G "parece" más compacto porque su proyección
   se come el doble de spoke.
3. La tree-likeness real de G no depende de esto: la miden δ̂-exceso, ξ (exp12) y ORC,
   todas full-D. La figura debe ilustrar, no testificar.

## Decisión de presentación (para la fase de escritura, con Javi)
- OPCIÓN RECOMENDADA: un solo panel (DINOv2-G) como ilustración del concepto
  hub-spoke, caption: "Illustrative 2-D projection (PCA fit on the 20 super-class
  hubs; 22% of hub variance, 6% of spoke length retained). Quantitative tree-likeness
  comparisons are made in the full feature space (Sec. 4)." Sin comparación S-vs-G.
- Alternativa: mantener ambos paneles PERO sin frase comparativa y con las fracciones
  anotadas en cada panel. Riesgo: invita a comparar visualmente lo incomparable.
- Nunca: mantener el caption impreso (un reviewer que haga esta cuenta nos pilla la
  inversión).
