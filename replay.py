"""Replay OFFLINE de una sesión grabada (solo DETECCIÓN; el bot es 100% visual desde la Etapa 8C).

Reproduce una sesión (frames + events.jsonl) a través de **la misma** lógica de detección
(corcho_detector), sin I/O en vivo. Permite validar la detección en WSL2 sin el juego.

Uso:
    python replay.py --make-sample sessions/_muestra   # sintetiza una sesión-muestra mínima
    python replay.py --session sessions/_muestra        # reproduce una sesión
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import cv2

from config import REPO_ROOT, load_config
from corcho_detector import CorchoDetector


def _read_events(session_dir: Path) -> List[dict]:
    path = session_dir / "events.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No hay events.jsonl en {session_dir}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def replay_session(session_dir: str | Path) -> List[dict]:
    """Reproduce la sesión y devuelve una lista de resultados de detección por ciclo."""
    session_dir = Path(session_dir)
    cfg = load_config()
    detector = CorchoDetector(
        cfg.model_onnx,
        conf_threshold=cfg.detector.conf_threshold,
        iou_threshold=cfg.detector.iou_threshold,
        imgsz=cfg.detector.imgsz,
    )

    results: List[dict] = []
    for ev in _read_events(session_dir):
        cycle = ev.get("cycle")
        detection = None
        if ev.get("frame"):
            img = cv2.imread(str(session_dir / ev["frame"]))
            dets = detector.detect(img) if img is not None else []
            if dets:
                best = max(dets, key=lambda d: d.conf)
                detection = {"bbox": [best.x1, best.y1, best.x2, best.y2], "conf": best.conf}

        results.append({"cycle": cycle, "detection": detection, "outcome_grabado": ev.get("outcome")})
        det = (f"bbox conf={detection['conf']:.3f}" if detection else "sin detección")
        print(f"ciclo {cycle}: {det} | grabado='{ev.get('outcome')}'")
    print(f"\nReplay completado: {len(results)} ciclo(s).")
    return results


def make_sample_session(out_dir: str | Path) -> Path:
    """Sintetiza una sesión-muestra mínima (1 ciclo) con una imagen de val."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    val_img = REPO_ROOT / "dataset" / "images" / "val" / "corcho_009.jpg"
    if not val_img.exists():
        val_img = sorted((REPO_ROOT / "dataset" / "images" / "val").glob("*.jpg"))[0]

    frame_name = "cycle_0001_roi.png"
    cv2.imwrite(str(out_dir / frame_name), cv2.imread(str(val_img)))
    event = {"cycle": 1, "frame": frame_name, "detection": None, "outcome": "muestra_sintetica"}
    (out_dir / "events.jsonl").write_text(json.dumps(event, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Sesión-muestra creada en: {out_dir}")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay offline de sesiones (solo detección)")
    ap.add_argument("--session", help="Carpeta de sesión a reproducir")
    ap.add_argument("--make-sample", help="Sintetizar una sesión-muestra en la carpeta indicada")
    args = ap.parse_args()

    if args.make_sample:
        make_sample_session(args.make_sample)
    if args.session:
        replay_session(args.session)
    if not args.session and not args.make_sample:
        ap.error("indica --session y/o --make-sample")


if __name__ == "__main__":
    main()
