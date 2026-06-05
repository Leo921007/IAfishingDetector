"""Captura dedicada de la mordida a alta tasa (Etapa 8B) — SE CORRE EN WINDOWS.

Graba la ROI (de config) lo más rápido posible con mss, manteniendo un buffer rodante en RAM. Cuando el
usuario ve un chapuzón (mordida) y pulsa 'b', vuelca a disco la ventana de los últimos segundos ANTES del
keypress (y un poco después) como JPEGs + manifest.json. NO usa el detector, NO clickea ni castea: es
captura pura, sin sesgo de loot.

Por qué la ventana se extiende ANTES del keypress: la reacción humana llega ~0.3-0.5 s tarde, así que el
splash real está en los frames previos al 'b'. El análisis (analyze_splash.py) lo busca ahí.

`mss` y `keyboard` son dependencias de Windows y se importan de forma PEREZOSA: este módulo se puede
importar en WSL2 (para verificar que parsea) sin tenerlas instaladas; solo main() las necesita.

Uso (en Windows, con el juego en primer plano; keyboard puede requerir admin):
    .venv\\Scripts\\python -m tools.capture_bite --cond noche_lluvia [--loc stormwind --pre 4.0 --post 1.5]
Teclas: 'b' = marcar mordida y volcar la ventana | 'q' = salir.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from pesca.config import REPO_ROOT, load_config

MANIFEST_VERSION = 2
CONDITIONS = ("dia_claro", "dia_lluvia", "noche_claro", "noche_lluvia")


def build_manifest(frame_times, t_press, pre, post, roi, location, condition,
                   version: int = MANIFEST_VERSION) -> dict:
    """Construye el manifest (v2) a partir de los TIEMPOS de los frames de la ventana.

    Función PURA: no toca mss/keyboard/cv2/juego ni el disco — recibe tiempos y metadatos y devuelve el
    dict. Los nombres de archivo se derivan determinísticamente (`frame_NNNN.jpg`), iguales a los que
    escribe `_dump_window`, para que el manifest se pueda construir y testear headless.

    `bite_events` = la(s) marca(s) 'b' del usuario (aquí una por ventana) = GROUND-TRUTH de mordida.
    """
    n = len(frame_times)
    t0 = frame_times[0]
    span = frame_times[-1] - t0
    fps_real = (n - 1) / span if n > 1 and span > 0 else 0.0
    keypress_index = min(range(n), key=lambda i: abs(frame_times[i] - t_press))
    frames_meta = [
        {"file": f"frame_{i:04d}.jpg",
         "t_rel_seg": round(frame_times[i] - t0, 4),
         "dt_from_press": round(frame_times[i] - t_press, 4)}
        for i in range(n)
    ]
    bite_events = [{"t_rel_seg": round(t_press - t0, 4), "frame_index": keypress_index}]
    return {
        "version": version,
        "location": location,
        "condition": condition,
        "roi": roi,
        "fps_real": round(fps_real, 1),
        "pre_seconds": pre, "post_seconds": post,
        "n_frames": n, "keypress_index": keypress_index,
        "frames": frames_meta,
        "bite_events": bite_events,
    }


def _dump_window(buf, t_press, pre, post, out_root, quality, bite_idx, location, condition):
    """Vuelca [t_press-pre, t_press+post] a captures_bite/<loc>_<cond>_<ts>_<NN>/ con manifest v2."""
    lo, hi = t_press - pre, t_press + post
    window = [(t, f) for (t, f) in buf if lo <= t <= hi]
    if not window:
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(out_root) / f"{location}_{condition}_{ts}_{bite_idx:02d}"
    out.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest([t for t, _ in window], t_press, pre, post,
                              load_config().roi.as_mss(), location, condition)
    for i, (_, frame_bgra) in enumerate(window):
        bgr = np.ascontiguousarray(frame_bgra[:, :, :3])
        cv2.imwrite(str(out / f"frame_{i:04d}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out, len(window), manifest["fps_real"]


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Captura de la mordida a alta tasa (Windows)")
    ap.add_argument("--loc", default=cfg.location,
                    help=f"ubicación (default: {cfg.location}, de config.yaml)")
    ap.add_argument("--cond", required=True, choices=CONDITIONS,
                    help="condición de la sesión (día/noche × claro/lluvia)")
    ap.add_argument("--pre", type=float, default=4.0, help="segundos antes del keypress a guardar")
    ap.add_argument("--post", type=float, default=1.5, help="segundos después del keypress a guardar")
    ap.add_argument("--buffer", type=float, default=6.0, help="segundos del buffer rodante en RAM")
    ap.add_argument("--quality", type=int, default=90, help="calidad JPEG")
    ap.add_argument("--out", default=str(REPO_ROOT / "captures_bite"))
    args = ap.parse_args()

    import keyboard  # import perezoso (Windows)
    import mss  # import perezoso (Windows)

    roi = load_config().roi.as_mss()
    buf = deque()  # (t, frame_bgra)
    bite_count = 0
    pending_press = None  # t_press a la espera de completar 'post'
    b_down = False
    last_fps_log = 0.0
    n_since = 0
    t_fps0 = time.monotonic()

    print(f"loc={args.loc} cond={args.cond} | ROI={roi} | buffer={args.buffer}s "
          f"pre={args.pre}s post={args.post}s")
    print("Grabando... pulsa 'b' al ver el chapuzón, 'q' para salir.")

    with mss.mss() as sct:
        while True:
            now = time.monotonic()
            frame = np.asarray(sct.grab(roi))  # BGRA, sin procesar (máx fps)
            buf.append((now, frame))
            # recortar el buffer por tiempo
            while buf and now - buf[0][0] > args.buffer:
                buf.popleft()

            # medición de fps (cada ~2 s)
            n_since += 1
            if now - t_fps0 >= 2.0:
                fps = n_since / (now - t_fps0)
                print(f"  fps medido: {fps:.1f} | buffer: {len(buf)} frames | bites: {bite_count}")
                t_fps0, n_since = now, 0

            # teclas (flanco de subida)
            if keyboard.is_pressed("q"):
                break
            pressed_b = keyboard.is_pressed("b")
            if pressed_b and not b_down and pending_press is None:
                pending_press = now
                print("  'b' -> mordida marcada; completando ventana...")
            b_down = pressed_b

            # completar la ventana 'post' tras el keypress y volcar
            if pending_press is not None and now - pending_press >= args.post:
                res = _dump_window(buf, pending_press, args.pre, args.post, args.out, args.quality,
                                   bite_count, args.loc, args.cond)
                if res:
                    out, nframes, fps_real = res
                    bite_count += 1
                    print(f"  ventana guardada ({nframes} frames, {fps_real:.1f} fps) -> {out}")
                else:
                    print("  [aviso] buffer insuficiente para la ventana; nada guardado")
                pending_press = None

    print(f"Fin. Bites guardados: {bite_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
