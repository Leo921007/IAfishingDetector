"""Pruebas del arnés de validación de la mordida a nivel de evento (headless, FakeTrigger + manifest sintético)."""
import json

import cv2
import numpy as np

from tools.capture_bite import build_manifest
from tools.eval_bite import (aggregate, classify_session, evaluate, firing_times,
                             load_session)

WIN, PRE = 4.0, 1.0          # ventana [t_b - 1.0, t_b + 4.0]
T_B = 5.0                     # evento en t=5.0 -> ventana [4.0, 9.0]


# --------------------------------------------------------------------- regla de disparo
def test_firing_times_dispara_tras_min_frames():
    scores = [0, 0.1, 0.1, 0.1, 0, 0.1, 0.1]
    times = list(range(len(scores)))
    # threshold 0.05, min_frames 2 -> 1er run dispara en idx 2 (2 consecutivos), re-arma, 2º run en idx 6
    assert firing_times(scores, times, 0.05, 2) == [2, 6]


def test_firing_times_min_frames_no_alcanzado():
    scores = [0, 0.1, 0, 0.1, 0]
    times = list(range(5))
    assert firing_times(scores, times, 0.05, 2) == []   # nunca 2 consecutivos


def test_sweep_monotonia_mas_umbral_menos_o_igual_disparos():
    scores = [0.002, 0.006, 0.006, 0.02, 0.02, 0.001, 0.05, 0.05]
    times = list(range(len(scores)))
    counts = [len(firing_times(scores, times, th, 2)) for th in (0.001, 0.005, 0.01, 0.03, 0.1)]
    assert all(b <= a for a, b in zip(counts, counts[1:])), counts


# --------------------------------------------------------------------- clasificación de evento
def test_catch_disparo_en_ventana_latencia_positiva():
    r = classify_session([T_B], firings=[5.5], win=WIN, pre_margin=PRE, duration=12.0)
    ev = r["per_event"][0]
    assert ev["catch"] is True
    assert abs(ev["latency"] - 0.5) < 1e-9
    assert r["false_fires"] == 0


def test_catch_latencia_negativa_el_bot_le_gana_al_humano():
    r = classify_session([T_B], firings=[4.5], win=WIN, pre_margin=PRE, duration=12.0)
    ev = r["per_event"][0]
    assert ev["catch"] is True
    assert ev["latency"] < 0 and abs(ev["latency"] + 0.5) < 1e-9


def test_false_fire_fuera_de_ventana():
    r = classify_session([T_B], firings=[10.0], win=WIN, pre_margin=PRE, duration=12.0)
    assert r["per_event"][0]["catch"] is False
    assert r["false_fires"] == 1


def test_miss_sin_disparo():
    r = classify_session([T_B], firings=[], win=WIN, pre_margin=PRE, duration=12.0)
    assert r["per_event"][0]["catch"] is False
    assert r["per_event"][0]["sensitivity"] is False
    assert r["false_fires"] == 0


def test_primer_disparo_antes_de_ventana_es_miss_false_cast_no_rescatable():
    # disparo en flotando (2.0, antes de [4,9]) ANTES que uno en ventana (5.5):
    # el primero decide -> MISS + false-cast; el 2º NO lo rescata, pero sensitivity=True
    r = classify_session([T_B], firings=[2.0, 5.5], win=WIN, pre_margin=PRE, duration=12.0)
    ev = r["per_event"][0]
    assert ev["catch"] is False          # MISS por false-cast
    assert ev["sensitivity"] is True     # diagnóstico: SÍ hubo señal en la ventana
    assert r["false_fires"] == 1         # el disparo de 2.0 cuenta como false-fire en flotando


def test_false_fires_por_min_usa_segundos_flotando():
    # 1 evento (ventana de 5 s) en una sesión de 65 s -> flotando = 60 s -> 1 ff = 1.0/min
    r = classify_session([T_B], firings=[2.0], win=WIN, pre_margin=PRE, duration=65.0)
    m = aggregate([r])
    assert abs(m["floating"] - 60.0) < 1e-6
    assert abs(m["false_fires_min"] - 1.0) < 1e-6


# --------------------------------------------------------------------- agregación / por condición
def test_aggregate_catch_rate_y_agrupacion_por_condicion():
    s_catch = classify_session([T_B], [5.2], WIN, PRE, 12.0); s_catch["condition"] = "dia_claro"
    s_miss = classify_session([T_B], [], WIN, PRE, 12.0); s_miss["condition"] = "noche_lluvia"
    m = aggregate([s_catch, s_miss])
    assert m["n_eventos"] == 2 and m["n_sesiones"] == 2
    assert abs(m["catch_rate"] - 0.5) < 1e-9
    # por condición: dia_claro 100%, noche_lluvia 0%
    by = {s["condition"]: aggregate([s]) for s in (s_catch, s_miss)}
    assert by["dia_claro"]["catch_rate"] == 1.0
    assert by["noche_lluvia"]["catch_rate"] == 0.0


# --------------------------------------------------------------------- integración: FakeTrigger + manifest
class FakeTrigger:
    """Devuelve una secuencia de scores conocida, ignorando el frame (test del camino completo)."""
    def __init__(self, scores):
        self.scores = scores
        self.i = 0

    def reset(self):
        self.i = 0

    def score(self, frame, bbox=None, t=0.0):
        v = self.scores[self.i]
        self.i += 1
        return v


def _make_session(tmp, scores, fps=10.0, keypress_index=20):
    tmp.mkdir(parents=True, exist_ok=True)
    times = [i / fps for i in range(len(scores))]
    man = build_manifest(times, times[keypress_index], 4.0, 1.5, {}, "stormwind", "dia_claro")
    for fm in man["frames"]:
        cv2.imwrite(str(tmp / fm["file"]), np.zeros((8, 8, 3), dtype=np.uint8))
    (tmp / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    return tmp


def test_integracion_load_session_y_evaluate_catch(tmp_path):
    n, kp = 40, 20
    scores = [0.0] * n
    scores[21] = scores[22] = 0.5     # splash justo tras el keypress (t = 2.1, 2.2 s; evento en 2.0)
    d = _make_session(tmp_path / "stormwind_dia_claro_x_00", scores, fps=10.0, keypress_index=kp)

    ft = FakeTrigger(scores)
    s = load_session(d, ft)
    assert s["condition"] == "dia_claro"
    assert len(s["events"]) == 1 and abs(s["events"][0] - 2.0) < 1e-6

    m, _ = evaluate([s], threshold=0.05, min_frames=2, win=WIN, pre_margin=PRE)
    assert m["catch_rate"] == 1.0
    assert m["latencia_med"] is not None and m["latencia_med"] >= 0


def test_integracion_v1_sin_condicion_es_desconocida(tmp_path):
    # manifest v1 (sin version/condition/bite_events) -> condición "desconocida", evento del keypress_index
    d = tmp_path / "vieja_00"
    d.mkdir(parents=True)
    frames = [{"file": f"frame_{i:04d}.jpg", "t": i / 10.0, "dt_from_press": (i - 15) / 10.0}
              for i in range(30)]
    for fm in frames:
        cv2.imwrite(str(d / fm["file"]), np.zeros((8, 8, 3), dtype=np.uint8))
    (d / "manifest.json").write_text(json.dumps(
        {"fps_real": 10.0, "keypress_index": 15, "frames": frames}), encoding="utf-8")

    s = load_session(d, FakeTrigger([0.0] * 30))
    assert s["condition"] == "desconocida"
    assert abs(s["events"][0] - 1.5) < 1e-6
