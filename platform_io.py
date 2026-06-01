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
