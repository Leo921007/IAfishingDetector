"""Construye el dataset de zona reentrenable con split POR CICLO (Etapa 6B, anti-fuga).

Los frames de un mismo ciclo son casi idénticos: si caen en train Y val, las métricas se inflan. Usando
dataset_zona/raw/_manifest.csv (provenance frame -> sesión/ciclo, verificado por
`extract_zona.py --manifest-only`), agrupa por ciclo y reserva ~15-20% de los CICLOS para val,
estratificando para que val tenga ciclos con positivos y con negativos.

train = ciclos restantes + dataset original (additive); val = ciclos reservados (zona nueva). Negativo =
.txt vacío o ausente (= fondo). Escribe data/corcho_zona.yaml. NO toca el dataset original.

Uso (Etapa 6B, tras etiquetar y generar el manifest):
    .venv/bin/python extract_zona.py --manifest-only
    .venv/bin/python build_zona_dataset.py
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

from config import REPO_ROOT

RAW = REPO_ROOT / "dataset_zona" / "raw"
ZONA = REPO_ROOT / "dataset_zona"
MANIFEST = RAW / "_manifest.csv"


def is_positive(stem: str) -> bool:
    txt = RAW / f"{stem}.txt"
    return txt.exists() and txt.read_text(encoding="utf-8").strip() != ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Split por ciclo del dataset de zona (anti-fuga)")
    ap.add_argument("--val-frac", type=float, default=0.18)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not MANIFEST.exists():
        raise SystemExit(f"Falta {MANIFEST}. Ejecuta: .venv/bin/python extract_zona.py --manifest-only")

    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
    by_cycle = defaultdict(list)
    for r in rows:
        by_cycle[(r["session"], r["cycle"])].append(r["frame"])
    cycles = sorted(by_cycle.keys())

    pos_cycles = [c for c in cycles if any(is_positive(s) for s in by_cycle[c])]
    neg_cycles = [c for c in cycles if c not in set(pos_cycles)]

    rng = random.Random(args.seed)

    def reserve(group):
        g = sorted(group)
        rng.shuffle(g)
        k = max(1, round(len(g) * args.val_frac)) if g else 0
        return set(g[:k])

    val_cycles = reserve(pos_cycles) | reserve(neg_cycles)
    train_cycles = set(cycles) - val_cycles
    assert train_cycles.isdisjoint(val_cycles), "solape de ciclos train/val"

    # Reconstruir limpio (dataset_zona/images|labels son gitignored).
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        d = ZONA / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    counts = {"train": [0, 0], "val": [0, 0]}  # [positivos, negativos]
    for c in cycles:
        split = "val" if c in val_cycles else "train"
        for stem in by_cycle[c]:
            shutil.copy(RAW / f"{stem}.jpg", ZONA / "images" / split / f"{stem}.jpg")
            txt = RAW / f"{stem}.txt"
            if txt.exists():
                shutil.copy(txt, ZONA / "labels" / split / f"{stem}.txt")
            if is_positive(stem):
                counts[split][0] += 1
            else:
                counts[split][1] += 1

    yaml_path = REPO_ROOT / "data" / "corcho_zona.yaml"
    yaml_path.write_text(
        "# Dataset de zona (Etapa 6B): split POR CICLO (anti-fuga).\n"
        "# train = dataset original + zona (train); val = ciclos de zona reservados.\n"
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

    print("Split por ciclo:")
    print(f"  ciclos: {len(cycles)} total | train {len(train_cycles)} | val {len(val_cycles)}")
    for split in ("train", "val"):
        p, n = counts[split]
        print(f"  {split}: {p + n} frames ({p} positivos, {n} negativos)")
    assert counts["val"][0] > 0 and counts["val"][1] > 0, "val debe tener positivos y negativos"
    print(f"Config escrita: {yaml_path}")
    print("Siguiente: .venv/bin/python train_corcho_zona.py")


if __name__ == "__main__":
    main()
