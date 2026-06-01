"""Pruebas del match de audio (offline, sin sounddevice).

Valida que el algoritmo conservado distingue el sonido de referencia del ruido.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.io import wavfile

from audio_match import match_audio

REPO = Path(__file__).resolve().parents[1]
REF1 = REPO / "Fishing_1.wav"
REFS = [REPO / f"Fishing_{i}.wav" for i in (1, 2, 3)]


@dataclass
class _AudioCfg:
    gain: float = 6.0
    bandpass: Tuple[float, float] = (300.0, 3000.0)
    similarity_threshold: float = 0.5


def test_match_sin_sounddevice():
    import sys

    assert "sounddevice" not in sys.modules


def test_referencia_contra_si_misma_coincide():
    fs, data = wavfile.read(str(REF1))
    matched, scores = match_audio(data, fs, REFS, _AudioCfg())
    assert matched, f"la referencia debería coincidir consigo misma; scores={scores}"
    # el mayor score debe ser el de la propia referencia
    assert max(scores, key=scores.get) == REF1.name


def test_referencia_discrimina_del_ruido():
    # Nota: el algoritmo (coseno de magnitudes FFT) tiene especificidad baja — el ruido de
    # banda ancha puede superar el umbral absoluto. Lo que SÍ se cumple, y es lo que validamos,
    # es que la referencia genuina puntúa más alto que el ruido (discriminación). La mejora de
    # especificidad del audio es trabajo de la Etapa 5.
    fs, ref = wavfile.read(str(REF1))
    _, self_scores = match_audio(ref, fs, [REF1], _AudioCfg())

    rng = np.random.default_rng(0)
    noise = rng.normal(0, 80, fs).astype(np.int16)
    _, noise_scores = match_audio(noise, fs, [REF1], _AudioCfg())

    assert self_scores[REF1.name] > noise_scores[REF1.name]
