"""Entrenamiento del detector de corcho (Etapa 2) con Ultralytics YOLO11n.

Reproducible: semilla fija (seed=0) y deterministic=True.

Uso:
    .venv/bin/python train_corcho.py

Equivalente por CLI:
    .venv/bin/yolo detect train model=yolo11n.pt data=data/corcho.yaml \
        imgsz=640 batch=16 epochs=150 patience=50 seed=0 deterministic=True \
        project=runs/corcho name=v1 exist_ok=True \
        hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 fliplr=0.5 flipud=0.0 \
        degrees=0.0 shear=0.0 perspective=0.0 mosaic=1.0 close_mosaic=10
"""
from pathlib import Path
from ultralytics import YOLO

REPO = Path(__file__).resolve().parent
DATA = REPO / "data" / "corcho.yaml"


def main():
    model = YOLO("yolo11n.pt")  # base COCO; se descarga si no está presente
    model.train(
        data=str(DATA),
        imgsz=640,
        batch=16,            # reducir a 8 si hay OOM en GPU de 6 GB
        epochs=150,
        patience=50,         # early stopping
        seed=0,
        deterministic=True,
        project=str(REPO / "runs" / "corcho"),
        name="v1",
        exist_ok=True,
        # --- Augmentación adecuada a dataset pequeño y objeto pequeño ---
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,        # variación de luz/color del agua
        fliplr=0.5,                               # espejo horizontal: válido para el corcho
        flipud=0.0,                               # NO voltear en vertical (el corcho flota erguido)
        degrees=0.0, shear=0.0, perspective=0.0,  # sin deformar la forma/apariencia del corcho
        mosaic=1.0, close_mosaic=10,              # mosaic ayuda con pocos datos
    )


if __name__ == "__main__":
    main()
