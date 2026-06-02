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
    .venv\\Scripts\\python capture_bite.py [--pre 4.0 --post 0.5 --buffer 6.0 --quality 90]
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

from config import REPO_ROOT, load_config


def _dump_window(buf, t_press, pre, post, out_root, quality, bite_idx):
    """Vuelca [t_press-pre, t_press+post] a captures_bite/<ts>/ con manifest.json."""
    lo, hi = t_press - pre, t_press + post
    window = [(t, f) for (t, f) in buf if lo <= t <= hi]
    if not window:
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_") + f"{bite_idx:02d}"
    out = Path(out_root) / ts
    out.mkdir(parents=True, exist_ok=True)

    # índice del frame más cercano al keypress (dentro de la ventana guardada)
    keypress_index = min(range(len(window)), key=lambda i: abs(window[i][0] - t_press))
    fps_real = (len(window) - 1) / (window[-1][0] - window[0][0]) if len(window) > 1 else 0.0

    frames_meta = []
    for i, (t, frame_bgra) in enumerate(window):
        name = f"frame_{i:04d}.jpg"
        bgr = np.ascontiguousarray(frame_bgra[:, :, :3])
        cv2.imwrite(str(out / name), bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        frames_meta.append({"file": name, "t": round(t - window[0][0], 4),
                            "dt_from_press": round(t - t_press, 4)})

    manifest = {
        "roi": load_config().roi.as_mss(),
        "fps_real": round(fps_real, 1),
        "pre_seconds": pre, "post_seconds": post,
        "n_frames": len(window), "keypress_index": keypress_index,
        "frames": frames_meta,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out, len(window), fps_real


def main() -> int:
    ap = argparse.ArgumentParser(description="Captura de la mordida a alta tasa (Windows)")
    ap.add_argument("--pre", type=float, default=4.0, help="segundos antes del keypress a guardar")
    ap.add_argument("--post", type=float, default=0.5, help="segundos después del keypress a guardar")
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

    print(f"ROI={roi} | buffer={args.buffer}s pre={args.pre}s post={args.post}s")
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
                res = _dump_window(buf, pending_press, args.pre, args.post, args.out, args.quality, bite_count)
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
