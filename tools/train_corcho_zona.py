"""Fine-tuning del detector de corcho para la zona del usuario (Etapa 6B).

Parte del modelo actual (models/corcho_detector/best.pt) y reentrena con data/corcho_zona.yaml
(dataset original + zona; val = ciclos de zona reservados). Misma disciplina de augmentación que la
Etapa 2 (sin flipud/rotación/shear). Épocas modestas + early stopping: el dataset es chico y muy
correlacionado -> cuidado con el sobreajuste.

Exporta el modelo nuevo a un ONNX SEPARADO (models/corcho_detector/best_zona.onnx) SIN tocar best.onnx
(se necesita para comparar; la promoción la decide el usuario).

Uso (tras build_zona_dataset.py):
    .venv/bin/python train_corcho_zona.py
"""
import shutil
from pathlib import Path

from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "corcho_zona.yaml"
BASE = REPO / "models" / "corcho_detector" / "best.pt"  # fine-tune desde el modelo actual
RUN_DIR = REPO / "runs" / "corcho" / "zona_v1"


def main():
    if not DATA.exists():
        raise SystemExit(f"Falta {DATA}. Ejecuta build_zona_dataset.py tras etiquetar.")

    model = YOLO(str(BASE))
    model.train(
        data=str(DATA),
        imgsz=640,
        batch=16,
        epochs=80,           # fine-tune corto (dataset chico/correlacionado)
        patience=20,         # early stopping
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

    # Export ONNX a archivo SEPARADO (no sobrescribir best.onnx).
    best_pt = RUN_DIR / "weights" / "best.pt"
    onnx_src = YOLO(str(best_pt)).export(format="onnx", imgsz=640)
    dst = REPO / "locations" / "stormwind" / "detector.onnx"
    shutil.copy(onnx_src, dst)
    print(f"Modelo nuevo: {best_pt}")
    print(f"ONNX de zona (separado): {dst}")


if __name__ == "__main__":
    main()
