"""Prueba de la captura de secuencia de frames del grabador (headless, con cap de disco)."""
import json

import numpy as np

from pesca.session import SessionRecorder


def test_record_cycle_guarda_secuencia_con_cap(tmp_path):
    rec = SessionRecorder(tmp_path, enabled=True, fs=44100, frames_max=5)
    frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(8)]

    rec.record_cycle(1, frames=frames, event={"outcome": "stub"})

    seq_dir = rec.dir / "cycle_0001_frames"
    pngs = sorted(seq_dir.glob("*.png"))
    assert len(pngs) == 5, "debe respetar frames_max (cap de disco)"

    event = json.loads((rec.dir / "events.jsonl").read_text(encoding="utf-8").strip())
    assert event["frames_count"] == 5
    assert event["frames_dir"] == "cycle_0001_frames"


def test_record_cycle_sin_frames_no_crea_carpeta(tmp_path):
    rec = SessionRecorder(tmp_path, enabled=True, fs=44100, frames_max=5)
    rec.record_cycle(2, audio_int16=np.zeros(100, dtype=np.int16), event={"outcome": "stub"})
    assert not (rec.dir / "cycle_0002_frames").exists()
