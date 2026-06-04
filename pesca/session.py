"""Grabador de sesión: persiste por ciclo el frame de la ROI, el chunk de audio, un evento y
(opcionalmente) una **secuencia de frames** alrededor de la decisión.

Captura datos reales durante una corrida en vivo para etapas futuras: la secuencia de frames es el
metraje de mordidas que necesitará la mordida VISUAL (dip del corcho) de la Etapa 6.

Estructura: sessions/<timestamp>/
    cycle_0001_roi.png        (frame de la ROI en el momento de la decisión)
    cycle_0001_audio.wav      (chunk grabado)
    cycle_0001_frames/        (secuencia: frame_000.png, frame_001.png, ... con cap de disco)
    events.jsonl              (una línea por ciclo: scores, detección, conteo de frames, desenlace)

Headless (cv2 / scipy / json): no requiere display ni audio.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
from scipy.io.wavfile import write as wav_write


class SessionRecorder:
    def __init__(self, base_dir: Path, enabled: bool, fs: int, frames_max: Optional[int] = None) -> None:
        self.enabled = enabled
        self.fs = fs
        self.frames_max = frames_max
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
        frames: Optional[Sequence[np.ndarray]] = None,
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
        if frames:
            n = len(frames) if self.frames_max is None else min(len(frames), self.frames_max)
            seq_dir = self.dir / f"cycle_{idx:04d}_frames"
            seq_dir.mkdir(exist_ok=True)
            for k, fr in enumerate(list(frames)[:n]):
                cv2.imwrite(str(seq_dir / f"frame_{k:03d}.png"), fr)
            record["frames_dir"] = seq_dir.name
            record["frames_count"] = n

        with open(self._events, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
