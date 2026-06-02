"""Adaptadores de plataforma: captura de pantalla e inyección de input.

Estas dependencias (mss, pyautogui, keyboard) son específicas del equipo de juego (Windows
con display/teclado) y NO funcionan en WSL2 headless. Por eso se aíslan aquí y se importan
de forma **perezosa** dentro de cada constructor: importar este módulo no exige display, y
la ruta de detección (corcho_detector) no lo importa nunca.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


class ScreenCapturer:
    """Captura una región de pantalla con mss y la devuelve en BGR (compatible con OpenCV)."""

    def __init__(self) -> None:
        import mss  # import perezoso

        self._sct = mss.mss()

    def grab(self, roi: dict) -> np.ndarray:
        shot = self._sct.grab(roi)  # roi = {left, top, width, height}
        # mss devuelve BGRA; nos quedamos con BGR
        return np.array(shot)[:, :, :3]


class InputController:
    """Mueve el ratón, hace clic y pulsa teclas en el equipo de juego."""

    def __init__(self) -> None:
        import keyboard  # import perezoso
        import pyautogui  # import perezoso

        self._pg = pyautogui
        self._kb = keyboard
        self._pg.FAILSAFE = True

    def move_and_click(self, x: int, y: int, button: str = "left") -> None:
        self._pg.moveTo(x, y)
        self._pg.click(button=button)

    def press_key(self, key: str) -> None:
        self._kb.press_and_release(key)


class AudioRecorder:
    """Graba audio del micrófono con sounddevice. Aísla la captura del match (audio_match)."""

    def __init__(self) -> None:
        import sounddevice as sd  # import perezoso

        self._sd = sd

    def record(self, duration: float, fs: int) -> np.ndarray:
        recording = self._sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
        self._sd.wait()
        return recording


class FrameGrabber:
    """Captura la ROI a ~fps en un hilo, manteniendo un ring buffer (ventana alrededor del candidato).

    Permite obtener metraje de la mordida sin bloquear el loop. Concurrencia y mss aislados aquí
    (import perezoso): importar el módulo no exige display.
    """

    def __init__(self, roi: dict, fps: float, max_frames: int) -> None:
        from collections import deque

        self._roi = roi
        self._interval = 1.0 / fps if fps > 0 else 0.1
        self._buf = deque(maxlen=max_frames)
        self._stop = None
        self._thread = None

    def start(self) -> None:
        import threading

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        import time

        import mss  # import perezoso

        with mss.mss() as sct:
            while self._stop is not None and not self._stop.is_set():
                shot = sct.grab(self._roi)
                self._buf.append(np.array(shot)[:, :, :3])
                time.sleep(self._interval)

    def snapshot(self) -> list:
        """Devuelve la ventana de frames más reciente (copia)."""
        return list(self._buf)

    def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
