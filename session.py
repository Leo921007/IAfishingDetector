"""Grabador de sesión: persiste por ciclo el frame de la ROI, el chunk de audio y un evento.

Captura datos reales durante una corrida en vivo para etapas futuras (mejora de la mordida,
ampliación del dataset) y permite reproducir la sesión offline con replay.py.

Estructura: sessions/<timestamp>/
    cycle_0001_roi.png      (frame de la ROI capturado al detectar mordida)
    cycle_0001_audio.wav    (chunk grabado en el momento de la decisión)
    events.jsonl            (una línea por ciclo: scores de audio, detección, tiempos, desenlace)

Headless (cv2 / scipy / json): no requiere display ni audio.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy.io.wavfile import write as wav_write


class SessionRecorder:
    def __init__(self, base_dir: Path, enabled: bool, fs: int) -> None:
        self.enabled = enabled
        self.fs = fs
        self.dir: Optional[Path] = None
        self._events: Optional[Path] = None
        if enabled:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.dir = Path(base_dir) / ts
            self.dir.mkdir(parents=True, exist_ok=True)
            self._events = self.dir / "events.jsonl"

    def record_cycle(
        self,
        idx: int,
        frame_bgr: Optional[np.ndarray] = None,
        audio_int16: Optional[np.ndarray] = None,
        event: Optional[dict] = None,
    ) -> None:
        if not self.enabled or self.dir is None:
            return

        record = {"cycle": idx, **(event or {})}
        if frame_bgr is not None:
            fname = f"cycle_{idx:04d}_roi.png"
            cv2.imwrite(str(self.dir / fname), frame_bgr)
            record["frame"] = fname
        if audio_int16 is not None:
            aname = f"cycle_{idx:04d}_audio.wav"
            wav_write(str(self.dir / aname), self.fs, np.asarray(audio_int16))
            record["audio"] = aname

        with open(self._events, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
