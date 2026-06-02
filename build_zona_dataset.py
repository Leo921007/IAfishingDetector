"""Construye el dataset de zona reentrenable a partir de dataset_zona/raw/ (Etapa 6B).

Tras el ETIQUETADO HUMANO de dataset_zona/raw/ (corregir/borrar las propuestas, dejar los frames de
solo-agua/espuma SIN .txt como negativos), divide en train/val y escribe data/corcho_zona.yaml que
FUSIONA el dataset original (en train) con el de zona, dejando el VAL = zona nueva para medir la
generalización in-domain. NO toca el dataset original (additive).

NO reentrena. Ejecutar en la Etapa 6B, después de etiquetar:
    .venv/bin/python build_zona_dataset.py
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from config import REPO_ROOT

RAW = REPO_ROOT / "dataset_zona" / "raw"
ZONA = REPO_ROOT / "dataset_zona"


def main() -> None:
    ap = argparse.ArgumentParser(description="Arma el split de zona y el data yaml (tras etiquetar)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    images = sorted(RAW.glob("*.jpg"))
    if not images:
        raise SystemExit(f"No hay imágenes en {RAW}. Ejecuta extract_zona.py y etiqueta primero.")

    rng = random.Random(args.seed)
    rng.shuffle(images)
    n_val = max(1, int(len(images) * args.val_frac))
    splits = {"val": images[:n_val], "train": images[n_val:]}

    counts = {}
    for split, files in splits.items():
        img_dir = ZONA / "images" / split
        lbl_dir = ZONA / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        negatives = 0
        for img in files:
            shutil.copy(img, img_dir / img.name)
            txt = img.with_suffix(".txt")
            if txt.exists():
                shutil.copy(txt, lbl_dir / txt.name)
            else:
                negatives += 1  # imagen sin label = fondo (negativo)
        counts[split] = (len(files), negatives)

    # data yaml: fusiona original (train) + zona (train); val = zona nueva.
    yaml_path = REPO_ROOT / "data" / "corcho_zona.yaml"
    yaml_path.write_text(
        "# Dataset de zona (Etapa 6B): original + zona en train; val = zona nueva (generalización).\n"
        f"path: {REPO_ROOT}\n"
        "train:\n"
        "  - dataset/images/train\n"
        "  - dataset_zona/images/train\n"
        "val:\n"
        "  - dataset_zona/images/val\n"
        "nc: 1\n"
        "names:\n"
        "  0: corcho\n",
        encoding="utf-8",
    )

    print("Dataset de zona construido:")
    for split, (total, neg) in counts.items():
        print(f"  {split}: {total} imágenes ({neg} negativos sin label)")
    print(f"Config escrita: {yaml_path}")
    print("Siguiente: .venv/bin/python train_corcho_zona.py")


if __name__ == "__main__":
    main()
