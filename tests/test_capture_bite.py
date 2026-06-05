"""Pruebas de la construcción PURA del manifest v2 de capture_bite (headless, sin mss/keyboard/juego)."""
from tools.capture_bite import CONDITIONS, MANIFEST_VERSION, build_manifest


def _times(n, fps, t0=1000.0):
    """n tiempos monótonos a `fps` arrancando en t0 (simula el buffer de captura)."""
    return [t0 + i / fps for i in range(n)]


def test_build_manifest_estructura_v2():
    times = _times(30, fps=30.0)            # 30 frames a 30 fps -> ventana de ~0.97 s
    t_press = times[20]                      # 'b' en el frame 20
    roi = {"left": 586, "top": 126, "width": 748, "height": 387}

    man = build_manifest(times, t_press, pre=4.0, post=1.5, roi=roi,
                         location="stormwind", condition="noche_lluvia")

    assert man["version"] == MANIFEST_VERSION == 2
    assert man["location"] == "stormwind"
    assert man["condition"] == "noche_lluvia" and man["condition"] in CONDITIONS
    assert man["roi"] == roi
    assert man["n_frames"] == 30
    assert man["pre_seconds"] == 4.0 and man["post_seconds"] == 1.5
    assert abs(man["fps_real"] - 30.0) < 0.5


def test_bite_event_es_ground_truth_en_el_keypress():
    times = _times(30, fps=30.0)
    t_press = times[20]
    man = build_manifest(times, t_press, 4.0, 1.5, {}, "stormwind", "dia_claro")

    # el ground-truth apunta al frame del keypress y a su tiempo relativo
    assert man["keypress_index"] == 20
    assert len(man["bite_events"]) == 1
    ev = man["bite_events"][0]
    assert ev["frame_index"] == 20
    assert abs(ev["t_rel_seg"] - man["frames"][20]["t_rel_seg"]) < 1e-6
    # el keypress index coincide con dt_from_press ~ 0
    assert abs(man["frames"][20]["dt_from_press"]) < 1e-6


def test_keypress_index_es_el_frame_mas_cercano():
    times = _times(40, fps=20.0)
    t_press = times[15] + 0.51 / 20.0       # entre frame 15 y 16, más cerca del 16
    man = build_manifest(times, t_press, 4.0, 1.5, {}, "stormwind", "dia_lluvia")
    assert man["keypress_index"] == 16


def test_frames_con_tiempo_relativo_monotono():
    times = _times(25, fps=24.0)
    man = build_manifest(times, times[10], 4.0, 1.5, {}, "stormwind", "noche_claro")
    rels = [f["t_rel_seg"] for f in man["frames"]]
    assert rels[0] == 0.0
    assert all(b >= a for a, b in zip(rels, rels[1:]))      # monótono no decreciente
    assert all(f["file"] == f"frame_{i:04d}.jpg" for i, f in enumerate(man["frames"]))


def test_fps_real_cero_con_un_solo_frame():
    man = build_manifest([1000.0], 1000.0, 4.0, 1.5, {}, "stormwind", "dia_claro")
    assert man["n_frames"] == 1
    assert man["fps_real"] == 0.0
    assert man["bite_events"][0]["frame_index"] == 0
