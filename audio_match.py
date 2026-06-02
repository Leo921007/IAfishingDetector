"""Match de la mordida por audio: NCC de la ENVOLVENTE de amplitud.

La versión anterior (coseno de magnitudes FFT) tenía especificidad casi nula: el ruido de banda
ancha puntuaba igual que el sonido genuino (ver bench_audio.py). Este discriminador compara la
**envolvente de amplitud** (forma del transitorio: onset agudo + decaimiento del splash) mediante
correlación cruzada normalizada (NCC). La envolvente es consistente entre grabaciones distintas del
mismo sonido y plana en el ruido → margen de separabilidad amplio (~+0.44 vs ~0 del coseno).

Pocos parámetros (banda, submuestreo de envolvente, umbral) — discriminador principista, no entrenado.
Recibe la señal ya grabada (int16) y devuelve (matched, scores): importable/testeable sin sounddevice.
La interfaz match_audio(recording_int16, fs, references, audio_cfg) -> (matched, scores) NO cambia.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, correlate, hilbert, lfilter

ENVELOPE_HZ = 1000  # submuestreo de la envolvente


def apply_gain(audio_int16: np.ndarray, gain: float) -> np.ndarray:
    """Amplifica con saturación. (Utilidad; el match por NCC es invariante a escala y no la usa.)"""
    amplified = np.asarray(audio_int16).astype(np.float32) * gain
    amplified = np.clip(amplified, -32768, 32767)
    return amplified.astype(np.int16)


def bandpass_filter(signal: np.ndarray, fs: int, lowcut: float, highcut: float, order: int = 4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return lfilter(b, a, signal)


def to_mono_normalized(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples)
    if samples.ndim > 1:  # estéreo -> mono
        samples = samples.mean(axis=1)
    samples = samples.astype(np.float32)
    norm = np.max(np.abs(samples))
    return samples / norm if norm else samples


def load_reference(path: str | Path, fs: int) -> np.ndarray:
    sr, data = wavfile.read(str(path))
    if sr != fs:
        raise ValueError(f"Referencia {path} a {sr} Hz; se esperaba {fs} Hz")
    return to_mono_normalized(data)


def _preprocess(signal_int16: np.ndarray, fs: int, bandpass: Tuple[float, float]) -> np.ndarray:
    return bandpass_filter(to_mono_normalized(signal_int16), fs, bandpass[0], bandpass[1])


def envelope(signal: np.ndarray, fs: int, target_hz: int = ENVELOPE_HZ) -> np.ndarray:
    """Envolvente de amplitud (|Hilbert|) submuestreada a ~target_hz."""
    env = np.abs(hilbert(signal))
    step = max(1, fs // target_hz)
    return env[::step]


def ncc_peak(a: np.ndarray, b: np.ndarray) -> float:
    """Pico de la correlación cruzada normalizada entre dos señales (invariante a desfase/escala)."""
    a = a - np.mean(a)
    b = b - np.mean(b)
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.max(np.abs(correlate(a, b, mode="full", method="fft"))))


def match_audio(
    recording_int16: np.ndarray,
    fs: int,
    references: Sequence[str | Path],
    audio_cfg,
) -> Tuple[bool, Dict[str, float]]:
    """Compara la grabación contra las referencias por NCC de envolvente.

    Devuelve (matched, {nombre_referencia: ncc_envolvente}). matched = True si alguna referencia
    supera audio_cfg.similarity_threshold. Interfaz idéntica a la versión anterior.
    """
    rec_env = envelope(_preprocess(recording_int16, fs, audio_cfg.bandpass), fs)

    scores: Dict[str, float] = {}
    matched = False
    for ref_path in references:
        ref_env = envelope(_preprocess(load_reference(ref_path, fs), fs, audio_cfg.bandpass), fs)
        score = ncc_peak(rec_env, ref_env)
        scores[Path(ref_path).name] = score
        if score > audio_cfg.similarity_threshold:
            matched = True
    return matched, scores
