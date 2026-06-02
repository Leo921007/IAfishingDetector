#!/usr/bin/env python3
"""Etiquetador manual mínimo para dataset_zona/raw/ (headless salvo la ventana de cv2).

Muestra cada frame .jpg con las cajas PROPUESTAS del .txt dibujadas tenues (gris fino, solo pista) y
permite:
  - arrastrar caja(s) con el ratón sobre el corcho real (zoom configurable con --scale)
  - [espacio]  guardar la(s) caja(s) y pasar al siguiente
  - [n]        sin corcho -> negativo: borra el .txt y pasa al siguiente
  - [u]        deshacer la última caja dibujada
  - [s]        saltar (sin marcar como hecho; reaparece en la próxima corrida)
  - [q]        salir guardando el progreso

Escribe labels YOLO (clase 0, normalizado) reemplazando el .txt in-place. Resumible: salta los frames
ya hechos (registrados en dataset_zona/raw/_progress.txt). Solo depende de cv2 (+ stdlib).

El zoom (--scale) agranda la ventana para un corcho pequeño y mapea las coordenadas del ratón de vuelta
al espacio de la imagen, así que las cajas no se descuadran aunque se trabaje ampliado.

Uso:
    .venv/bin/python label_zona.py [--dir dataset_zona/raw] [--scale 1.5]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

WIN_NAME = "label_zona"
PROGRESS_NAME = "_progress.txt"

# Estado compartido con el callback del ratón (coordenadas en espacio de IMAGEN).
state: dict = {"drawing": False, "x0": 0.0, "y0": 0.0, "cur": None, "boxes": [], "scale": 1.5}


def on_mouse(event, x, y, flags, param):  # firma de cv2
    s = state["scale"]
    ix, iy = x / s, y / s  # de coords de ventana a coords de imagen
    if event == cv2.EVENT_LBUTTONDOWN:
        state["drawing"] = True
        state["x0"], state["y0"] = ix, iy
        state["cur"] = (ix, iy, ix, iy)
    elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
        state["cur"] = (state["x0"], state["y0"], ix, iy)
    elif event == cv2.EVENT_LBUTTONUP and state["drawing"]:
        state["drawing"] = False
        if abs(ix - state["x0"]) > 2 and abs(iy - state["y0"]) > 2:
            state["boxes"].append((state["x0"], state["y0"], ix, iy))
        state["cur"] = None


def load_proposed_boxes(txt_path: Path, w: int, h: int):
    """Lee las cajas propuestas del .txt (YOLO normalizado) como píxeles."""
    if not txt_path.exists():
        return []
    out = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            _cls, cx, cy, bw, bh = (float(v) for v in parts)
        except ValueError:
            continue
        out.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def boxes_to_yolo(boxes_px, w: int, h: int):
    """Convierte cajas en píxeles a líneas YOLO (clase 0, normalizado)."""
    lines = []
    for (ax, ay, bx, by) in boxes_px:
        x0, x1 = sorted((ax, bx))
        y0, y1 = sorted((ay, by))
        cx = min(max(((x0 + x1) / 2) / w, 0.0), 1.0)
        cy = min(max(((y0 + y1) / 2) / h, 0.0), 1.0)
        bw = min(max((x1 - x0) / w, 0.0), 1.0)
        bh = min(max((y1 - y0) / h, 0.0), 1.0)
        if bw > 0 and bh > 0:
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def load_progress(progress_file: Path) -> set[str]:
    if not progress_file.exists():
        return set()
    return {ln.strip() for ln in progress_file.read_text(encoding="utf-8").splitlines() if ln.strip()}


def mark_done(progress_file: Path, basename: str) -> None:
    with progress_file.open("a", encoding="utf-8") as f:
        f.write(basename + "\n")


def render(img, scale, proposed, drawn, live):
    disp = cv2.resize(img, None, fx=scale, fy=scale)

    def r(box, color, thick):
        x0, y0, x1, y1 = box
        cv2.rectangle(disp, (int(x0 * scale), int(y0 * scale)),
                      (int(x1 * scale), int(y1 * scale)), color, thick)

    for b in proposed:  # propuestas: gris tenue
        r(b, (110, 110, 110), 1)
    for b in drawn:     # confirmadas: verde
        r(b, (0, 255, 0), 2)
    if live is not None:  # arrastre en curso: amarillo
        r(live, (0, 255, 255), 1)
    return disp


def draw_hud(disp, idx, total, basename, n_boxes, n_proposed):
    lines = [
        f"[{idx + 1}/{total}] {basename}  (propuestas: {n_proposed}, dibujadas: {n_boxes})",
        "drag=caja  [espacio]=guardar+next  [n]=negativo  [u]=undo  [s]=skip  [q]=salir",
    ]
    y = 20
    for line in lines:
        cv2.putText(disp, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(disp, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22


def main() -> int:
    ap = argparse.ArgumentParser(description="Etiquetador mínimo del dataset de zona")
    ap.add_argument("--dir", default=str(Path(__file__).parent / "locations" / "stormwind" / "dataset" / "raw"))
    ap.add_argument("--scale", type=float, default=1.5, help="zoom de la ventana (objeto pequeño)")
    args = ap.parse_args()

    raw_dir = Path(args.dir)
    if not raw_dir.exists():
        print(f"[label_zona] No existe {raw_dir}", file=sys.stderr)
        return 1
    jpgs = sorted(p for p in raw_dir.iterdir() if p.suffix.lower() == ".jpg")
    if not jpgs:
        print(f"[label_zona] No hay .jpg en {raw_dir}", file=sys.stderr)
        return 1

    state["scale"] = args.scale
    progress_file = raw_dir / PROGRESS_NAME
    done = load_progress(progress_file)
    total = len(jpgs)
    pending = [p for p in jpgs if p.stem not in done]
    if not pending:
        print(f"[label_zona] Los {total} frames ya están en {PROGRESS_NAME}. "
              "Borrá ese archivo para re-etiquetar.")
        return 0
    print(f"[label_zona] {len(pending)}/{total} pendientes (resumiendo desde {PROGRESS_NAME}).")

    cv2.namedWindow(WIN_NAME)
    cv2.setMouseCallback(WIN_NAME, on_mouse)

    quit_flag = False
    for idx, jpg in enumerate(pending):
        if quit_flag:
            break
        img = cv2.imread(str(jpg))
        if img is None:
            print(f"[label_zona] skip (no se pudo leer): {jpg.name}", file=sys.stderr)
            continue
        h, w = img.shape[:2]
        txt_path = jpg.with_suffix(".txt")
        proposed = load_proposed_boxes(txt_path, w, h)
        state["boxes"], state["cur"], state["drawing"] = [], None, False

        while True:
            disp = render(img, state["scale"], proposed, state["boxes"], state["cur"])
            draw_hud(disp, idx, len(pending), jpg.stem, len(state["boxes"]), len(proposed))
            cv2.imshow(WIN_NAME, disp)

            k = cv2.waitKey(20) & 0xFF
            if k == ord(" "):  # guardar caja(s)
                txt_path.write_text("\n".join(boxes_to_yolo(state["boxes"], w, h))
                                    + ("\n" if state["boxes"] else ""), encoding="utf-8")
                mark_done(progress_file, jpg.stem)
                break
            if k == ord("n"):  # negativo: borrar .txt
                if txt_path.exists():
                    txt_path.unlink()
                mark_done(progress_file, jpg.stem)
                break
            if k == ord("u"):  # deshacer
                if state["boxes"]:
                    state["boxes"].pop()
            elif k == ord("s"):  # saltar (no marca progreso)
                break
            elif k == ord("q"):  # salir guardando
                quit_flag = True
                break

            if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:  # ventana cerrada
                quit_flag = True
                break

    cv2.destroyAllWindows()
    remaining = sum(1 for p in jpgs if p.stem not in load_progress(progress_file))
    print(f"[label_zona] hecho. Pendientes restantes: {remaining}/{total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
