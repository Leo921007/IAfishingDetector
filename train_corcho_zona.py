"""Fine-tuning del detector de corcho para la zona del usuario (Etapa 6B). NO ejecutar en 6A.

Parte del modelo actual (models/corcho_detector/best.pt) y reentrena con data/corcho_zona.yaml
(dataset original + zona; val = zona nueva). Misma disciplina de augmentación que la Etapa 2
(sin flipud/rotación/shear, que deforman la apariencia del corcho).

Uso (Etapa 6B, después de build_zona_dataset.py):
    .venv/bin/python train_corcho_zona.py

Equivalente por CLI:
    .venv/bin/yolo detect train model=models/corcho_detector/best.pt data=data/corcho_zona.yaml \
        imgsz=640 batch=16 epochs=100 patience=40 seed=0 deterministic=True \
        project=runs/corcho name=zona_v1 exist_ok=True \
        hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 fliplr=0.5 flipud=0.0 \
        degrees=0.0 shear=0.0 perspective=0.0 mosaic=1.0 close_mosaic=10
"""
from pathlib import Path

from ultralytics import YOLO

REPO = Path(__file__).resolve().parent
DATA = REPO / "data" / "corcho_zona.yaml"
BASE = REPO / "models" / "corcho_detector" / "best.pt"  # fine-tune desde el modelo actual


def main():
    if not DATA.exists():
        raise SystemExit(f"Falta {DATA}. Ejecuta build_zona_dataset.py tras etiquetar.")
    model = YOLO(str(BASE))
    model.train(
        data=str(DATA),
        imgsz=640,
        batch=16,
        epochs=100,          # fine-tune: menos épocas que el entrenamiento desde cero
        patience=40,
        seed=0,
        deterministic=True,
        project=str(REPO / "runs" / "corcho"),
        name="zona_v1",
        exist_ok=True,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        fliplr=0.5, flipud=0.0,
        degrees=0.0, shear=0.0, perspective=0.0,
        mosaic=1.0, close_mosaic=10,
    )


if __name__ == "__main__":
    main()
