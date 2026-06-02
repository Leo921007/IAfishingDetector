"""Extracción + selección + propuestas de etiqueta del dataset de la zona del usuario.

Recorre las sesiones grabadas (de la MISMA escala = mismas dimensiones de frame), selecciona por ciclo
unos pocos frames diversos (el frame del "dip" por máximo movimiento inter-frame + frames separados),
deduplica casi-duplicados (average-hash) y los guarda en dataset_zona/raw/ junto a propuestas de etiqueta
YOLO generadas por el detector ONNX a confianza baja. Genera un montaje para juzgar la calidad.

IMPORTANTE: las .txt son PROPUESTAS POCO FIABLES (acelerar el etiquetado), a corregir a mano.
Headless: cv2 / numpy / onnxruntime (sin mss/pyautogui/keyboard/sounddevice).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from config import REPO_ROOT, load_config
from corcho_detector import CorchoDetector

# Sesiones de la zona del usuario a 387x748 (misma escala). Otras (stub 50x80, sintética) se excluyen.
DEFAULT_SESSIONS = ["20260601_222727", "20260601_221053"]


def ahash(img: np.ndarray, size: int = 16) -> np.ndarray:
    # 16x16 (256 bits): suficientemente discriminativo para no confundir frames de agua distintos
    # con duplicados; solo los casi-idénticos quedan a pocos bits de distancia.
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    return (small > small.mean()).flatten()


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def load_events(session_dir: Path) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    p = session_dir / "events.jsonl"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line)
                out[e["cycle"]] = e
    return out


def select_indices(frame_paths: List[Path], per_cycle: int) -> List[int]:
    """Devuelve los índices de frames a conservar: el del dip (máx movimiento) + frames separados."""
    grays = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2GRAY).astype(np.int16) for p in frame_paths]
    motion = [0] + [int(np.abs(grays[i] - grays[i - 1]).sum()) for i in range(1, len(grays))]
    dip = int(np.argmax(motion))
    n = len(frame_paths)
    candidates = [dip, n // 4, 3 * n // 4, n // 2]
    idxs: List[int] = []
    for i in candidates:
        if 0 <= i < n and i not in idxs:
            idxs.append(i)
        if len(idxs) >= per_cycle:
            break
    return idxs, dip


def main() -> None:
    ap = argparse.ArgumentParser(description="Extrae el dataset de la zona desde las sesiones grabadas")
    ap.add_argument("--sessions", nargs="*", default=DEFAULT_SESSIONS)
    ap.add_argument("--out", default=str(REPO_ROOT / "dataset_zona" / "raw"))
    ap.add_argument("--per-cycle", type=int, default=3)
    ap.add_argument("--max", type=int, default=250)
    ap.add_argument("--conf", type=float, default=0.10, help="conf baja para PROPUESTAS")
    ap.add_argument("--hamming", type=int, default=10,
                    help="umbral dedup sobre aHash 16x16 (256 bits); menor = conserva más")
    args = ap.parse_args()

    cfg = load_config()
    detector = CorchoDetector(cfg.model_onnx, conf_threshold=args.conf,
                              iou_threshold=cfg.detector.iou_threshold, imgsz=cfg.detector.imgsz)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    montage_items: List[tuple] = []  # (thumb, label)
    n_saved = n_proposals = n_cycles = cycles_with_corcho = 0
    expected_dims = None

    for sname in args.sessions:
        sdir = REPO_ROOT / "sessions" / sname
        if not sdir.is_dir():
            print(f"[aviso] sesión no encontrada: {sdir}")
            continue
        events = load_events(sdir)
        for fdir in sorted(sdir.glob("cycle_*_frames")):
            if n_saved >= args.max:
                break
            cycle = int(fdir.name.split("_")[1])
            frames = sorted(fdir.glob("frame_*.png"))
            if not frames:
                continue
            n_cycles += 1
            outcome = events.get(cycle, {}).get("outcome", "?")

            # comprobar dimensiones (solo misma escala)
            shape = cv2.imread(str(frames[0])).shape[:2]
            if expected_dims is None:
                expected_dims = shape
            elif shape != expected_dims:
                print(f"[aviso] {sname}/{fdir.name} dims {shape} != {expected_dims}; se omite")
                continue

            idxs, dip = select_indices(frames, args.per_cycle)
            cycle_has_corcho = False
            cycle_hashes: List[np.ndarray] = []  # dedup SOLO dentro del ciclo
            for i in idxs:
                if n_saved >= args.max:
                    break
                img = cv2.imread(str(frames[i]))
                h = ahash(img)
                if any(hamming(h, kh) < args.hamming for kh in cycle_hashes):
                    continue  # frame casi-idéntico a otro ya elegido de ESTE ciclo (p.ej. agua estática)
                cycle_hashes.append(h)

                stem = f"frame_{n_saved:04d}"
                cv2.imwrite(str(out_dir / f"{stem}.jpg"), img)
                n_saved += 1

                # propuestas (poco fiables)
                dets = detector.detect(img)
                if dets:
                    H, W = img.shape[:2]
                    lines = []
                    best_conf = 0.0
                    for d in dets:
                        cx = ((d.x1 + d.x2) / 2) / W
                        cy = ((d.y1 + d.y2) / 2) / H
                        bw = (d.x2 - d.x1) / W
                        bh = (d.y2 - d.y1) / H
                        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                        best_conf = max(best_conf, d.conf)
                    (out_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
                    n_proposals += 1
                    if best_conf > 0.4:
                        cycle_has_corcho = True

                thumb = cv2.resize(img, (160, int(160 * img.shape[0] / img.shape[1])))
                tag = "DIP" if i == dip else "   "
                cv2.putText(thumb, f"{outcome[:6]} {tag}", (3, 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                montage_items.append(thumb)
            if cycle_has_corcho:
                cycles_with_corcho += 1

    # montaje
    if montage_items:
        cols = 10
        th = montage_items[0].shape[0]
        tw = montage_items[0].shape[1]
        rows = (len(montage_items) + cols - 1) // cols
        sheet = np.full((rows * th, cols * tw, 3), 40, dtype=np.uint8)
        for k, thumb in enumerate(montage_items):
            r, c = divmod(k, cols)
            sheet[r * th:r * th + thumb.shape[0], c * tw:c * tw + thumb.shape[1]] = thumb
        montage_path = Path(args.out).parent / "montage.jpg"
        cv2.imwrite(str(montage_path), sheet)
    else:
        montage_path = None

    print("=" * 56)
    print(f"Sesiones: {args.sessions}  (dims frame: {expected_dims})")
    print(f"Ciclos procesados:           {n_cycles}")
    print(f"Frames guardados (tras dedup): {n_saved}  -> {out_dir}")
    print(f"Frames con propuesta de caja:  {n_proposals}")
    print(f"Ciclos con corcho de alta conf (>0.4, proxy 'visible'): {cycles_with_corcho}")
    print(f"Montaje: {montage_path}")
    print("Recuerda: las .txt son PROPUESTAS poco fiables; corrige a mano (ver ETAPA6_DATASET.md).")
    print("=" * 56)


if __name__ == "__main__":
    main()
