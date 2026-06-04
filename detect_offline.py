"""Verificación OFFLINE del detector de corcho.

Corre el detector ONNX sobre imágenes guardadas (por defecto dataset/images/val) e imprime
las detecciones (bbox + confianza). No requiere el juego, ni display, ni audio: valida la
ruta de inferencia de forma reproducible en WSL2.

Uso:
    python detect_offline.py [--source RUTA] [--conf F] [--save]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from pesca.config import REPO_ROOT, load_config
from pesca.corcho_detector import CorchoDetector

EXTS = {".jpg", ".jpeg", ".png"}


def iter_images(source: Path):
    if source.is_file():
        yield source
    else:
        for p in sorted(source.iterdir()):
            if p.suffix.lower() in EXTS:
                yield p


def main() -> None:
    ap = argparse.ArgumentParser(description="Verificación offline del detector de corcho")
    ap.add_argument("--source", default=str(REPO_ROOT / "dataset" / "images" / "val"),
                    help="Imagen o carpeta a procesar")
    ap.add_argument("--conf", type=float, default=None, help="Umbral de confianza (override)")
    ap.add_argument("--save", action="store_true", help="Guardar imágenes anotadas en detections/")
    args = ap.parse_args()

    cfg = load_config()
    conf = args.conf if args.conf is not None else cfg.detector.conf_threshold
    detector = CorchoDetector(
        cfg.model_onnx,
        conf_threshold=conf,
        iou_threshold=cfg.detector.iou_threshold,
        imgsz=cfg.detector.imgsz,
    )

    out_dir = REPO_ROOT / "detections"
    total_dets = 0
    n_imgs = 0
    for path in iter_images(Path(args.source)):
        img = cv2.imread(str(path))
        if img is None:
            continue
        n_imgs += 1
        dets = detector.detect(img)
        total_dets += len(dets)
        if dets:
            for d in dets:
                print(f"{path.name}: bbox=({d.x1:.0f},{d.y1:.0f},{d.x2:.0f},{d.y2:.0f}) "
                      f"conf={d.conf:.3f}")
        else:
            print(f"{path.name}: sin detecciones")
        if args.save:
            out_dir.mkdir(exist_ok=True)
            for d in dets:
                cv2.rectangle(img, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (0, 255, 0), 2)
            cv2.imwrite(str(out_dir / path.name), img)

    print(f"\nResumen: {total_dets} detección(es) en {n_imgs} imágenes (conf>={conf}).")


if __name__ == "__main__":
    main()
