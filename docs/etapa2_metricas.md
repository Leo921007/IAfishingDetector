# Métricas — Detector de corcho v1 (Etapa 2)

Modelo: **YOLO11n** (Ultralytics 8.3.108) · checkpoint `best.pt` = **época 74** (selección por
*fitness* = 0.1·mAP50 + 0.9·mAP50-95).
Entrenamiento: 124 épocas corridas (early stopping, `patience=50`), **~4.7 min** en **RTX 4050
Laptop (6 GB)**, `seed=0`, `deterministic=True`.

## Resultados en validación (20 imágenes / 20 instancias, 1 clase `corcho`)

| Métrica | Valor |
|---|---|
| **mAP50** | **≈ 0.86** (0.862 train-final / 0.850 re-val) |
| **mAP50-95** | **≈ 0.42** (0.421 / 0.411) |
| Precision (P) | ~0.74–0.77 (punto de operación) |
| Recall (R) | ~0.75–0.90 (punto de operación) |
| Velocidad inferencia | **~1.3 ms/img** (GPU), preprocess 0.2 ms |

**Matriz de confusión @ conf=0.25** (filas=predicho, cols=real; clase `corcho` vs fondo):

|              | real corcho | real fondo |
|--------------|-------------|------------|
| pred corcho  | **TP = 17** | **FP = 8** |
| pred fondo   | **FN = 3**  | —          |

## Lectura honesta

- **mAP50 ≈ 0.86** es sólido **dentro de la misma distribución** (una zona/sesión, recortes
  pequeños ~430×257). En zonas/clima/iluminación/resolución distintas **caerá**.
- Val de **solo 20 imágenes** → P/R en un punto de operación son **ruidosas y optimistas**;
  `mAP` es la métrica fiable.
- A `conf=0.25` hay **8 falsos positivos** de fondo. En producción conviene **subir el umbral**
  (p. ej. 0.4–0.5) y/o **cruzar con la señal de audio** para descartar FP.
- `mAP50-95 ≈ 0.42` indica localización mejorable (bbox no siempre ajustado), normal con objeto
  pequeño y dataset reducido.

Artefactos visuales (locales, no versionados): `runs/corcho/v1/confusion_matrix.png`,
`results.png`, `PR_curve.png`, `val_batch0_pred.jpg`.

Reproducción: `./.venv/bin/python -m tools.train_corcho` (ver `ETAPA2_ENTRENAMIENTO.md`).
