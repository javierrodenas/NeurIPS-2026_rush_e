# Código del proyecto (backup)

Copia de seguridad del código de experimentos del paper — no es material de entrega.

- `scripts/` — extracción de features, geometría (delta, ORC), tareas downstream (NC/kNN/few-shot/retrieval), sweeps de radio/curvatura, verificación y plots. Corren sobre la cache de features en `results/practical_tasks_cache/` de la máquina local.
- `experiments/training/run_all_scaled_toy.py` — pipeline de entrenamiento con cabezas Hyperbolic/Cosine/Linear (experimento complementario, no incluido en el paper actual).
- El código de los experimentos del rebuttal está aparte, en `rebuttal/scripts/`.
