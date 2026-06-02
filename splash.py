"""Cálculo de foam y parche del corcho — fuente ÚNICA compartida.

La mordida (Etapa 8B) se detecta por FOAM = fracción de píxeles casi-blancos (alto V, baja S en HSV) en
un parche ~1.5x alrededor del bbox del corcho. Tanto el análisis offline (analyze_splash.py) como el
trigger en vivo (bite_trigger.py / main.py) importan estas funciones, para que **el cálculo en vivo sea
idéntico al offline**.

Puro cv2 / numpy: importable headless (sin mss/pyautogui/keyboard/sounddevice).
"""
from __future__ import annotations

import cv2
import numpy as np

PATCH_SCALE = 1.5


def patch_box(bbox, w: int, h: int, scale: float = PATCH_SCALE):
    """Parche (X1,Y1,X2,Y2) de tamaño bbox*scale, clamp al frame. bbox=None -> central (40%)."""
    if bbox is None:
        return int(w * 0.3), int(h * 0.3), int(w * 0.7), int(h * 0.7)
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    pw, ph = (x2 - x1) * scale, (y2 - y1) * scale
    return (max(0, int(cx - pw / 2)), max(0, int(cy - ph / 2)),
            min(w, int(cx + pw / 2)), min(h, int(cy + ph / 2)))


def foam_value(img_bgr: np.ndarray, patch) -> float:
    """Fracción de píxeles casi-blancos (V>200, S<60 en HSV) en el parche."""
    X1, Y1, X2, Y2 = patch
    area = max(1, (Y2 - Y1) * (X2 - X1))
    hsv = cv2.cvtColor(img_bgr[Y1:Y2, X1:X2], cv2.COLOR_BGR2HSV)
    white = int(((hsv[:, :, 2] > 200) & (hsv[:, :, 1] < 60)).sum())
    return white / area


def frame_diff(prev_gray: np.ndarray, cur_gray: np.ndarray) -> float:
    """Media de |cur-prev| (respaldo del foam: movimiento del agua)."""
    return float(np.mean(np.abs(cur_gray.astype(np.int16) - prev_gray.astype(np.int16))))
