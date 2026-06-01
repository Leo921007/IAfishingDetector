"""Barrido de umbral de confianza sobre el set de validación (con GT labels).

Para cada umbral calcula precisión/recall emparejando detecciones con las cajas reales por IoU>0.5,
e imprime una tabla. Ayuda a elegir el punto de operación con datos. En el loop gated por audio
interesa **recall alto** (localizar el corcho cuando ya hubo mordida), aceptando más falsos
positivos. NO modifica la config: solo recomienda.

Uso:
    python tune_threshold.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from config import REPO_ROOT, load_config
from corcho_detector import CorchoDetector

SWEEP_MIN_CONF = 0.05  # se infiere una vez a conf baja y se filtra por umbral
IOU_MATCH = 0.5


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _load_gt(label_path: Path, w: int, h: int) -> List[Tuple[float, float, float, float]]:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, bw, bh = (float(p) for p in parts)
        boxes.append((( cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return boxes


def sweep(images_dir: Path, labels_dir: Path):
    cfg = load_config()
    detector = CorchoDetector(
        cfg.model_onnx, conf_threshold=SWEEP_MIN_CONF,
        iou_threshold=cfg.detector.iou_threshold, imgsz=cfg.detector.imgsz,
    )

    # Inferir una vez a conf baja; guardar (conf, bbox) y las GT por imagen.
    per_image = []
    for img_path in sorted(images_dir.glob("*.jpg")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        dets = [(d.conf, (d.x1, d.y1, d.x2, d.y2)) for d in detector.detect(img)]
        gt = _load_gt(labels_dir / f"{img_path.stem}.txt", w, h)
        per_image.append((dets, gt))

    thresholds = np.round(np.arange(0.10, 0.91, 0.05), 2)
    print(f"Barrido sobre {len(per_image)} imágenes (IoU match>{IOU_MATCH}).\n")
    print(f"{'conf':>5} | {'TP':>3} {'FP':>3} {'FN':>3} | {'precision':>9} {'recall':>7} {'F1':>6}")
    print("-" * 48)

    best = None
    for t in thresholds:
        tp = fp = fn = 0
        for dets, gt in per_image:
            cand = sorted([d for d in dets if d[0] >= t], key=lambda d: -d[0])
            used = [False] * len(gt)
            for _, box in cand:
                m = -1
                for i, g in enumerate(gt):
                    if not used[i] and _iou(box, g) > IOU_MATCH:
                        m = i
                        break
                if m >= 0:
                    used[m] = True
                    tp += 1
                else:
                    fp += 1
            fn += used.count(False)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        print(f"{t:>5.2f} | {tp:>3} {fp:>3} {fn:>3} | {precision:>9.3f} {recall:>7.3f} {f1:>6.3f}")
        if best is None or f1 > best[1]:
            best = (t, f1, precision, recall)

    print("-" * 48)
    print(f"Mejor F1: conf={best[0]:.2f} (F1={best[1]:.3f}, P={best[2]:.3f}, R={best[3]:.3f}).")
    print("Recomendación para el loop gated por audio (prioriza recall): conf ~0.30-0.35.")
    print(f"El valor activo en config.yaml es {load_config().detector.conf_threshold:.2f}; "
          "ajústalo a mano si procede (este script no lo modifica).")


def main():
    ap = argparse.ArgumentParser(description="Barrido de umbral de confianza")
    ap.add_argument("--images", default=str(REPO_ROOT / "dataset" / "images" / "val"))
    ap.add_argument("--labels", default=str(REPO_ROOT / "dataset" / "labels" / "val"))
    args = ap.parse_args()
    sweep(Path(args.images), Path(args.labels))


if __name__ == "__main__":
    main()
