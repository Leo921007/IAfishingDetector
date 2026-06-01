"""Prueba del replay offline (headless): sintetiza una sesión-muestra y la reproduce."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MODEL = REPO / "models" / "corcho_detector" / "best.onnx"

needs_model = pytest.mark.skipif(not MODEL.exists(), reason="modelo ONNX ausente (gitignored)")


@needs_model
def test_replay_muestra_headless(tmp_path):
    from replay import make_sample_session, replay_session

    sess = make_sample_session(tmp_path / "muestra")
    results = replay_session(sess)

    assert len(results) == 1
    r = results[0]
    assert r["audio_matched"] is True  # el chunk es una referencia -> coincide
    assert r["detection"] is not None  # el frame de val contiene corcho

    # el replay no debe requerir I/O de plataforma
    for mod in ("mss", "pyautogui", "keyboard", "sounddevice"):
        assert mod not in sys.modules
