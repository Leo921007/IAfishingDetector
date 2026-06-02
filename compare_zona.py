"""Compara el detector VIEJO (best.pt) vs el NUEVO (zona_v1) sobre el MISMO val de zona.

Mide:
  - mAP50, mAP50-95, precision, recall de cada modelo (Ultralytics .val()).
  - La métrica del problema real: FP a conf 0.25 sobre los frames NEGATIVOS del val (agua sin corcho)
    -> nº de imágenes con falsa detección y nº total de cajas espurias.

Uso (tras build_zona_dataset.py + train_corcho_zona.py):
    .venv/bin/python compare_zona.py
"""
from pathlib import Path

from ultralytics import YOLO

from config import REPO_ROOT

DATA = REPO_ROOT / "data" / "corcho_zona.yaml"
OLD = REPO_ROOT / "models" / "corcho_detector" / "best.pt"
NEW = REPO_ROOT / "runs" / "corcho" / "zona_v1" / "weights" / "best.pt"
VAL_IMG = REPO_ROOT / "dataset_zona" / "images" / "val"
VAL_LBL = REPO_ROOT / "dataset_zona" / "labels" / "val"
CONF_FP = 0.25


def negatives():
    negs = []
    for img in sorted(VAL_IMG.glob("*.jpg")):
        lbl = VAL_LBL / f"{img.stem}.txt"
        if (not lbl.exists()) or (lbl.read_text(encoding="utf-8").strip() == ""):
            negs.append(img)
    return negs


def eval_map(model_path: Path, tag: str):
    m = YOLO(str(model_path))
    r = m.val(data=str(DATA), iou=0.7, verbose=False, plots=False,
              project=str(REPO_ROOT / "runs" / "corcho"), name=f"cmp_{tag}", exist_ok=True)
    return r.box.map50, r.box.map, r.box.mp, r.box.mr


def count_fp(model_path: Path, negs):
    m = YOLO(str(model_path))
    imgs_with_fp = 0
    total_boxes = 0
    for img in negs:
        res = m.predict(str(img), conf=CONF_FP, verbose=False)
        n = len(res[0].boxes)
        total_boxes += n
        imgs_with_fp += 1 if n > 0 else 0
    return imgs_with_fp, total_boxes


def main():
    negs = negatives()
    print("=" * 64)
    print(f"Comparación VIEJO vs NUEVO  |  val de zona  |  negativos: {len(negs)} frames")
    print("=" * 64)
    print(f"{'modelo':6} | {'mAP50':>6} {'mAP50-95':>8} {'P':>6} {'R':>6} | "
          f"{'FP@.25 (imgs/cajas)':>20}")
    print("-" * 64)
    results = {}
    for tag, path in (("VIEJO", OLD), ("NUEVO", NEW)):
        map50, map5095, p, r = eval_map(path, tag)
        iw, tot = count_fp(path, negs)
        results[tag] = (map50, map5095, p, r, iw, tot)
        print(f"{tag:6} | {map50:6.3f} {map5095:8.3f} {p:6.3f} {r:6.3f} | "
              f"{iw:>3}/{len(negs)} imgs, {tot:>3} cajas")
    print("-" * 64)
    o, n = results["VIEJO"], results["NUEVO"]
    print(f"Δ mAP50: {n[0] - o[0]:+.3f} | Δ mAP50-95: {n[1] - o[1]:+.3f} | "
          f"FP imgs: {o[4]} -> {n[4]} | FP cajas: {o[5]} -> {n[5]}")
    print("=" * 64)


if __name__ == "__main__":
    main()
