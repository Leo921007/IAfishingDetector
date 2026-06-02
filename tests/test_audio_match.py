"""Pruebas del match de mordida por audio (NCC de envolvente, offline sin sounddevice).

Valida que el nuevo discriminador separa el sonido genuino del ruido (lo que el coseno-FFT no hacía)
y que generaliza a una referencia evaluada contra las OTRAS (leave-one-out).
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.io import wavfile

from audio_match import match_audio

REPO = Path(__file__).resolve().parents[1]
REFS = [REPO / f"Fishing_{i}.wav" for i in (1, 2, 3)]


@dataclass
class _AudioCfg:
    gain: float = 6.0                      # no usado por el discriminador NCC (invariante a escala)
    bandpass: Tuple[float, float] = (300.0, 3000.0)
    similarity_threshold: float = 0.30     # calibrado desde bench_audio.py


def test_match_sin_sounddevice():
    import sys

    assert "sounddevice" not in sys.modules


def test_referencia_contra_si_misma_coincide():
    fs, data = wavfile.read(str(REFS[0]))
    matched, scores = match_audio(data, fs, REFS, _AudioCfg())
    assert matched, f"la referencia debería coincidir; scores={scores}"
    assert max(scores, key=scores.get) == REFS[0].name


def test_referencia_cruzada_coincide():
    # Fishing_1 evaluada contra las OTRAS dos (no contra sí misma): debe superar el umbral.
    fs, data = wavfile.read(str(REFS[0]))
    matched, scores = match_audio(data, fs, REFS[1:], _AudioCfg())
    assert matched, f"la mordida genuina debería coincidir con otras referencias; scores={scores}"


def test_ruido_blanco_no_coincide():
    fs = 44100
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 80, 3 * fs).astype(np.int16)
    matched, scores = match_audio(noise, fs, REFS, _AudioCfg())
    assert not matched, f"el ruido blanco NO debería coincidir; scores={scores}"
    assert max(scores.values()) < _AudioCfg().similarity_threshold
