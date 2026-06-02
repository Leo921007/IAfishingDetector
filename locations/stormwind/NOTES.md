# Ubicación: stormwind

Lugar de pesca: agua/canal de Ventormenta (la zona donde se entrenó y validó el bot).

## Modelo
- `detector.onnx` (gitignored): YOLO11n fine-tuneado para esta zona.
- Entrenado desde las sesiones `20260601_222727` y `20260601_221053` + capturas etiquetadas a mano.
- Métricas (Etapa 6B/8B): **mAP50 ≈ 0.99**; a **conf 0.25 no da falsos positivos** sobre la espuma.
- El `.onnx` no se versiona: colócalo en `locations/stormwind/detector.onnx` en cada máquina (o entrena).

## ROI
- `roi.yaml`: `left=586, top=126, width=748, height=387` (encuadre con el que se entrenó/opera).

## Dataset (local, gitignored)
- `locations/stormwind/dataset/` (raw etiquetado, images/labels train/val, análisis). Reentrenar con
  `build_zona_dataset.py` + `train_corcho_zona.py`.
