"""Match de audio por similitud FFT.

Algoritmo **conservado** de la versión original (ganancia → normalización → filtro pasa-banda
Butterworth → FFT → similitud coseno contra referencias). Separado de la captura: recibe la
señal ya grabada (int16) y devuelve (matched, scores), por lo que es importable y testeable en
WSL2 sin `sounddevice`.

Carga los WAV de referencia con `scipy.io.wavfile` — numéricamente equivalente al antiguo
`audio_to_np_mono` para PCM 16-bit mono (formato de Fishing_*.wav y de las grabaciones).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter


def apply_gain(audio_int16: np.ndarray, gain: float) -> np.ndarray:
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


def _fft_mag_norm(signal: np.ndarray, fs: int, bandpass: Tuple[float, float]) -> np.ndarray:
    filt = bandpass_filter(signal, fs, bandpass[0], bandpass[1], order=4)
    mag = np.abs(np.fft.rfft(filt))
    peak = np.max(mag)
    return mag / peak if peak else mag


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def match_audio(
    recording_int16: np.ndarray,
    fs: int,
    references: Sequence[str | Path],
    audio_cfg,
) -> Tuple[bool, Dict[str, float]]:
    """Compara la grabación contra las referencias.

    Devuelve (matched, {nombre_referencia: similitud_coseno}). La decisión (matched) es idéntica
    a la original: True si alguna referencia supera audio_cfg.similarity_threshold. Se calculan
    todos los scores (útil para logging) en lugar de cortar en la primera coincidencia.
    """
    amp = apply_gain(recording_int16, audio_cfg.gain)
    rec = to_mono_normalized(amp)
    fft_rec = _fft_mag_norm(rec, fs, audio_cfg.bandpass)

    scores: Dict[str, float] = {}
    matched = False
    for ref_path in references:
        ref = load_reference(ref_path, fs)
        fft_ref = _fft_mag_norm(ref, fs, audio_cfg.bandpass)
        score = _cosine(fft_rec, fft_ref)
        scores[Path(ref_path).name] = score
        if score > audio_cfg.similarity_threshold:
            matched = True
    return matched, scores
