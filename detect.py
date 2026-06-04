"""Detección por lotes sobre data/game_screenshots usando el detector ONNX de la Etapa 3.

Anota cada imagen con las cajas detectadas y las guarda en detections/.
Reemplaza la antigua carga vía torch.hub/yolov5 (rota).
"""
from pathlib import Path

import cv2

from pesca.config import REPO_ROOT, load_config
from pesca.corcho_detector import CorchoDetector


def main() -> None:
    cfg = load_config()
    detector = CorchoDetector(
        cfg.model_onnx,
        conf_threshold=cfg.detector.conf_threshold,
        iou_threshold=cfg.detector.iou_threshold,
        imgsz=cfg.detector.imgsz,
    )

    source_folder = REPO_ROOT / "data" / "game_screenshots"
    output_folder = REPO_ROOT / "detections"
    output_folder.mkdir(exist_ok=True)

    for image_path in sorted(source_folder.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        dets = detector.detect(img)
        for d in dets:
            cv2.rectangle(img, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (0, 255, 0), 2)
            cv2.putText(
                img, f"{d.conf:.2f}", (int(d.x1), int(d.y1) - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )
        cv2.imwrite(str(output_folder / image_path.name), img)
        print(f"{image_path.name}: {len(dets)} detección(es)")

    print(f"\nDetecciones guardadas en: {output_folder}")


if __name__ == "__main__":
    main()
