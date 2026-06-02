"""Detector de mordida por FOAM (stateful) — Etapa 8C.

Recibe el frame de la ROI y el bbox del corcho, calcula el foam del parche (reusa splash.foam_value /
patch_box, idéntico al análisis offline) y **dispara UNA vez** cuando el foam supera el umbral durante
>= min_frames frames consecutivos (el chapuzón de la mordida). `reset()` lo re-arma tras el loot.

Sin dependencias de audio ni de plataforma: importable y testeable headless.
"""
from __future__ import annotations

import numpy as np

from splash import foam_value, patch_box


class FoamBiteDetector:
    def __init__(self, threshold: float, min_frames: int) -> None:
        self.threshold = float(threshold)
        self.min_frames = max(1, int(min_frames))
        self._count = 0
        self._fired = False

    def reset(self) -> None:
        self._count = 0
        self._fired = False

    def update(self, frame_bgr: np.ndarray, bbox) -> tuple[float, bool]:
        """Devuelve (foam, fired). fired=True solo en el frame que arma el disparo (una vez)."""
        h, w = frame_bgr.shape[:2]
        foam = foam_value(frame_bgr, patch_box(bbox, w, h))
        self._count = self._count + 1 if foam > self.threshold else 0
        fired = False
        if self._count >= self.min_frames and not self._fired:
            fired = True
            self._fired = True
        return foam, fired
