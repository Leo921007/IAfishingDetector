"""Benchmark de especificidad del match de mordida por audio (offline, reproducible, headless).

Compara dos discriminadores sobre positivos (sonido de pesca) y negativos (ruido):
  - COSENO de magnitudes FFT  (método antiguo, especificidad baja)
  - NCC de la ENVOLVENTE de amplitud  (método nuevo)

El score de un candidato = máximo sobre las referencias-plantilla (igual que match_audio, que da
positivo si ALGUNA referencia supera el umbral). Reporta min/mediana/max de positivos y negativos y el
MARGEN = min(positivos) - max(negativos). Margen > 0 => separables.

Positivos/negativos sintéticos por defecto. Clips reales opcionales:
    python bench_audio.py --positives DIR --negativos DIR

Solo numpy/scipy: importable y ejecutable en WSL2 sin audio/display.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, List, Sequence

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, correlate, hilbert, lfilter

from config import REPO_ROOT, load_config

SEED = 0


# --- front-end común ---------------------------------------------------------
def _norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x).astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    m = np.max(np.abs(x))
    return x / m if m else x


def _bandpass(x: np.ndarray, fs: int, lo: float, hi: float) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(4, [lo / nyq, hi / nyq], btype="band")
    return lfilter(b, a, x)


def _pre(x: np.ndarray, fs: int, bp) -> np.ndarray:
    return _bandpass(_norm(x), fs, bp[0], bp[1])


# --- discriminadores ---------------------------------------------------------
def cosine_fft_score(rec: np.ndarray, tmpl: np.ndarray) -> float:
    fa, fb = np.abs(np.fft.rfft(rec)), np.abs(np.fft.rfft(tmpl))
    fa = fa / (np.max(fa) or 1.0)
    fb = fb / (np.max(fb) or 1.0)
    n = min(len(fa), len(fb))
    fa, fb = fa[:n], fb[:n]
    return float(np.dot(fa, fb) / (np.linalg.norm(fa) * np.linalg.norm(fb) + 1e-10))


def _envelope(x: np.ndarray, fs: int, target_hz: int = 1000) -> np.ndarray:
    env = np.abs(hilbert(x))
    step = max(1, fs // target_hz)
    return env[::step]


def envelope_ncc_score(rec: np.ndarray, tmpl: np.ndarray, fs: int) -> float:
    a, b = _envelope(rec, fs), _envelope(tmpl, fs)
    a = a - np.mean(a)
    b = b - np.mean(b)
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.max(np.abs(correlate(a, b, mode="full", method="fft"))))


# --- construcción de positivos/negativos ------------------------------------
def _load_wavs(folder: Path) -> List[np.ndarray]:
    if not folder or not Path(folder).is_dir():
        return []
    return [wavfile.read(str(p))[1] for p in sorted(Path(folder).glob("*.wav"))]


def _synth_negatives(fs: int, rng) -> List[np.ndarray]:
    white = rng.normal(0, 80, 3 * fs).astype(np.int16)
    band = rng.normal(0, 300, 3 * fs).astype(np.int16)  # se filtra a banda en el front-end
    return [white, band]


def _embed(ref: np.ndarray, fs: int, rng) -> np.ndarray:
    buf = rng.normal(0, 30, 3 * fs).astype(np.float32)
    off = int(0.27 * fs)
    seg = ref.astype(np.float32)[: len(buf) - off]
    buf[off : off + len(seg)] += seg * 3.0
    return buf.astype(np.int16)


def _max_over_templates(cand: np.ndarray, templates: Sequence[np.ndarray], fs: int, bp, method: str) -> float:
    cp = _pre(cand, fs, bp)
    best = -1.0
    for t in templates:
        tp = _pre(t, fs, bp)
        s = cosine_fft_score(cp, tp) if method == "cosine" else envelope_ncc_score(cp, tp, fs)
        best = max(best, s)
    return best


def _stats(name: str, pos: List[float], neg: List[float]) -> None:
    pos, neg = np.array(pos), np.array(neg)
    print(f"\n[{name}]")
    print(f"  positivos: min={pos.min():.3f} med={np.median(pos):.3f} max={pos.max():.3f}  (n={len(pos)})")
    print(f"  negativos: min={neg.min():.3f} med={np.median(neg):.3f} MAX={neg.max():.3f}  (n={len(neg)})")
    margin = pos.min() - neg.max()
    veredicto = "SEPARABLES" if margin > 0 else "SOLAPADOS"
    print(f"  MARGEN (min_pos - max_neg) = {margin:+.3f}  -> {veredicto}")
    if margin > 0:
        print(f"  umbral sugerido (punto medio) = {(pos.min() + neg.max()) / 2:.3f}")


def run(positives_dir: Path | None, negatives_dir: Path | None) -> None:
    cfg = load_config()
    fs = cfg.audio.fs
    bp = cfg.audio.bandpass
    refs = [wavfile.read(str(r))[1] for r in cfg.audio.references]
    rng = np.random.default_rng(SEED)

    # Positivos: cada ref embebida en ruido (realista) + leave-one-out (ref vs las OTRAS refs).
    pos_candidates = []  # (cand, templates)
    for i, ref in enumerate(refs):
        pos_candidates.append((_embed(ref, fs, rng), refs))                 # embebido vs todas
        others = [r for j, r in enumerate(refs) if j != i]
        if others:
            pos_candidates.append((ref, others))                            # cross (leave-one-out)
    for clip in _load_wavs(positives_dir):                                  # reales aportados
        pos_candidates.append((clip, refs))

    # Negativos: ruido sintético (+ reales aportados), siempre evaluados contra todas las refs.
    neg_candidates = [(n, refs) for n in _synth_negatives(fs, rng)]
    for clip in _load_wavs(negatives_dir):
        neg_candidates.append((clip, refs))

    print("=" * 60)
    print(f"Benchmark de especificidad de audio  (positivos={len(pos_candidates)}, "
          f"negativos={len(neg_candidates)})")
    if positives_dir or negatives_dir:
        print(f"  clips reales: positives={positives_dir} negatives={negatives_dir}")
    else:
        print("  (solo sintéticos: resultado PROVISIONAL hasta tener negativos de gameplay)")
    print("=" * 60)

    for method, label in (("cosine", "COSENO-FFT (antiguo)"), ("envelope_ncc", "NCC-ENVOLVENTE (nuevo)")):
        pos = [_max_over_templates(c, t, fs, bp, method) for c, t in pos_candidates]
        neg = [_max_over_templates(c, t, fs, bp, method) for c, t in neg_candidates]
        _stats(label, pos, neg)


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark de especificidad del match de audio")
    ap.add_argument("--positives", default=None, help="Carpeta con WAV positivos reales (opcional)")
    ap.add_argument("--negativos", "--negatives", dest="negatives", default=None,
                    help="Carpeta con WAV negativos reales (opcional)")
    args = ap.parse_args()
    run(Path(args.positives) if args.positives else None,
        Path(args.negatives) if args.negatives else None)


if __name__ == "__main__":
    main()
