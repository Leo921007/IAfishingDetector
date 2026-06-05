"""Arnés de validación de la mordida a nivel de EVENTO (Fase 0, §7 del plan) — headless, WSL2.

Corre un *trigger* sobre las sesiones de `captures_bite/` (replay offline determinista) y mide, contra el
ground-truth (las marcas `b` del manifest), si el bot **atraparía la picadura** — no exactitud por frame.

Principio (§7): por sesión se cachea el score por frame (una sola pasada del modelo) y sobre esos scores se
aplica la regla de disparo (threshold + min_frames). El **primer disparo de la sesión** decide el evento:

  - primer disparo en [t_b - pre_margin, t_b + win]  -> CATCH (latencia = disparo - t_b, puede ser negativa)
  - primer disparo ANTES de la ventana (en flotando)  -> MISS + false-cast (en vivo recastearía antes)
  - después de la ventana, o sin disparo               -> MISS

`false_fires_por_min` cuenta TODOS los disparos en flotando (no solo el primero). `sensitivity` (columna
secundaria de diagnóstico) = hubo ALGÚN disparo en la ventana, desacoplado del orden: separa
miss-por-no-señal de miss-por-false-cast.

Trigger por defecto = **FoamTrigger** (baseline §7.5): reusa `pesca.corcho_detector` (localiza el corcho a
cadencia `relocate_seconds`, igual que `main.run_cycle`) + `pesca.splash` (foam del parche). No modifica el
runtime: lo usa read-only. Los triggers aprendidos (clase mordida / clasificador) se enchufan luego con la
misma interfaz `score(frame, bbox=None) -> float`.

Uso:
    python -m tools.eval_bite                         # baseline foam sobre captures_bite/, umbral de config
    python -m tools.eval_bite --sweep                 # barrido de umbral (§7.3) + punto de operación
    python -m tools.eval_bite --sessions captures_bite/stormwind_* --threshold 0.008 --min-frames 2
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from pesca.config import REPO_ROOT, load_config
from pesca.splash import foam_value, patch_box

RESULTS_CSV = REPO_ROOT / "results" / "eval_bite.csv"
CSV_HEADER = ["fecha", "loc", "condiciones", "trigger", "threshold", "min_frames",
              "catch_rate", "false_fires_min", "latencia_med", "n_eventos", "n_sesiones", "sensitivity"]
FALSE_FIRE_TARGET = 0.2  # §8: false-fire <= 0.2/min en el punto de operación
DEFAULT_SWEEP = [0.001, 0.002, 0.003, 0.005, 0.008, 0.012, 0.02, 0.03, 0.05]


# --------------------------------------------------------------------------- triggers
class FoamTrigger:
    """Baseline §7.5: localiza el corcho (detector, cadencia relocate_seconds) y devuelve el foam del parche.

    score(frame, bbox=None): si bbox=None auto-localiza (réplica fiel de run_cycle en vivo); si se sirve un
    bbox lo usa (modo diagnóstico: aísla la señal de mordida de los fallos de localización). Estado por
    sesión -> llamar reset() entre sesiones.
    """

    def __init__(self, detector, relocate_seconds: float):
        self.detector = detector
        self.relocate_seconds = float(relocate_seconds)
        self.reset()

    def reset(self) -> None:
        self._bbox = None
        self._last_relocate_t = None

    def _relocate(self, frame, t) -> None:
        due = self._last_relocate_t is None or (t - self._last_relocate_t) >= self.relocate_seconds
        if not due:
            return
        self._last_relocate_t = t
        dets = self.detector.detect(frame)
        if dets:  # 'keep' la última bbox si este relocate falla (igual que el loop vivo)
            b = max(dets, key=lambda d: d.conf)
            self._bbox = (b.x1, b.y1, b.x2, b.y2)

    def score(self, frame, bbox=None, t: float = 0.0) -> float:
        if bbox is None:
            self._relocate(frame, t)
            bbox = self._bbox
        h, w = frame.shape[:2]
        return foam_value(frame, patch_box(bbox, w, h))


# --------------------------------------------------------------------------- lógica pura (testeable)
def firing_times(scores, times, threshold: float, min_frames: int):
    """Tiempos de disparo: run de `score > threshold` durante >= min_frames -> un disparo al completar el run.

    Re-arma al bajar del umbral (permite varios disparos por sesión). Mismo criterio que
    pesca.bite_trigger.FoamBiteDetector, generalizado a múltiples disparos para contar false-fires.
    """
    fires, count, armed = [], 0, True
    for s, t in zip(scores, times):
        if s > threshold:
            count += 1
            if count >= min_frames and armed:
                fires.append(t)
                armed = False
        else:
            count, armed = 0, True
    return fires


def _in_window(f, t_b, pre_margin, win):
    return (t_b - pre_margin) <= f <= (t_b + win)


def classify_session(events, firings, win: float, pre_margin: float, duration: float):
    """Clasifica una sesión: por evento {catch, latency, sensitivity} + false_fires + segundos flotando.

    El primer disparo de cada evento (tras la ventana del evento anterior) decide catch/miss (Q2). Todos
    los disparos fuera de toda ventana cuentan como false-fires.
    """
    events = sorted(events)
    per_event, prev_hi = [], float("-inf")
    for t_b in events:
        lo, hi = t_b - pre_margin, t_b + win
        relevant = [f for f in firings if f >= prev_hi]
        first = relevant[0] if relevant else None
        catch = first is not None and _in_window(first, t_b, pre_margin, win)
        per_event.append({
            "catch": catch,
            "latency": (first - t_b) if catch else None,
            "sensitivity": any(_in_window(f, t_b, pre_margin, win) for f in firings),
        })
        prev_hi = hi

    windows = [(t_b - pre_margin, t_b + win) for t_b in events]
    false_fires = sum(1 for f in firings if not any(lo <= f <= hi for lo, hi in windows))
    # segundos flotando = duración total - cobertura de las ventanas de evento (clamp al [0, duración])
    covered = _covered_seconds(windows, duration)
    floating = max(0.0, duration - covered)
    return {"per_event": per_event, "false_fires": false_fires, "floating": floating}


def _covered_seconds(windows, duration):
    """Unión (sin doble conteo) de las ventanas recortadas a [0, duration]."""
    clamped = sorted((max(0.0, lo), min(duration, hi)) for lo, hi in windows if hi > 0 and lo < duration)
    total, cur_lo, cur_hi = 0.0, None, None
    for lo, hi in clamped:
        if cur_hi is None or lo > cur_hi:
            if cur_hi is not None:
                total += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
        else:
            cur_hi = max(cur_hi, hi)
    if cur_hi is not None:
        total += cur_hi - cur_lo
    return total


def aggregate(session_results):
    """Agrega métricas sobre varias sesiones ya clasificadas."""
    n_sesiones = len(session_results)
    events = [ev for s in session_results for ev in s["per_event"]]
    n_eventos = len(events)
    catches = [ev for ev in events if ev["catch"]]
    latencies = [ev["latency"] for ev in catches]
    false_fires = sum(s["false_fires"] for s in session_results)
    floating = sum(s["floating"] for s in session_results)
    return {
        "n_sesiones": n_sesiones,
        "n_eventos": n_eventos,
        "catch_rate": (len(catches) / n_eventos) if n_eventos else 0.0,
        "false_fires": false_fires,
        "false_fires_min": (false_fires / (floating / 60.0)) if floating > 0 else 0.0,
        "latencia_med": statistics.median(latencies) if latencies else None,
        "latencia_mean": statistics.fmean(latencies) if latencies else None,
        "sensitivity": (sum(1 for ev in events if ev["sensitivity"]) / n_eventos) if n_eventos else 0.0,
        "floating": floating,
    }


def evaluate(sessions, threshold, min_frames, win, pre_margin):
    """De [{events, scores, times, duration, condition}] -> métricas, por sesión clasificadas."""
    classified = []
    for s in sessions:
        fires = firing_times(s["scores"], s["times"], threshold, min_frames)
        r = classify_session(s["events"], fires, win, pre_margin, s["duration"])
        r["condition"] = s["condition"]
        classified.append(r)
    return aggregate(classified), classified


# --------------------------------------------------------------------------- carga de sesiones (replay)
def _frame_time(fm):
    return float(fm.get("t_rel_seg", fm.get("t", 0.0)))


def load_session(d: Path, trigger):
    """Lee una sesión, corre el trigger por frame (en orden) y cachea scores/tiempos. Maneja v1 y v2."""
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    trigger.reset()
    scores, times = [], []
    for fm in man["frames"]:
        img = cv2.imread(str(d / fm["file"]))
        if img is None:
            continue
        t = _frame_time(fm)
        scores.append(trigger.score(img, t=t))
        times.append(t)
    if not times:
        return None
    # eventos: v2 trae bite_events; v1 -> derivar del keypress_index
    if man.get("bite_events"):
        events = [float(ev["t_rel_seg"]) for ev in man["bite_events"]]
    else:
        kp = man.get("keypress_index", 0)
        events = [_frame_time(man["frames"][kp])] if man.get("frames") else []
    return {
        "name": d.name,
        "condition": man.get("condition", "desconocida"),
        "location": man.get("location", "desconocida"),
        "events": events,
        "scores": scores,
        "times": times,
        "duration": times[-1] - times[0],
    }


def find_sessions(patterns):
    dirs = []
    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            pd = Path(p)
            if pd.is_dir() and (pd / "manifest.json").exists():
                dirs.append(pd)
    return dirs


# --------------------------------------------------------------------------- salida
def _fmt(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"


def print_report(metrics, classified, threshold, min_frames):
    print("=" * 64)
    print(f"Arnés de mordida | trigger=foam threshold={threshold} min_frames={min_frames}")
    print("=" * 64)
    print(f"  sesiones={metrics['n_sesiones']} eventos={metrics['n_eventos']} "
          f"flotando={metrics['floating']:.1f}s")
    print(f"  catch_rate     = {_fmt(metrics['catch_rate'])}  "
          f"(sensitivity={_fmt(metrics['sensitivity'])})")
    print(f"  false_fires    = {metrics['false_fires']}  -> {_fmt(metrics['false_fires_min'])}/min")
    print(f"  latencia       = med {_fmt(metrics['latencia_med'])}s  mean {_fmt(metrics['latencia_mean'])}s")

    by_cond = {}
    for c in classified:
        by_cond.setdefault(c["condition"], []).append(c)
    if len(by_cond) > 1:
        print("\n  por condición:")
        for cond in sorted(by_cond):
            m = aggregate(by_cond[cond])
            print(f"    {cond:14s} catch={_fmt(m['catch_rate'])} "
                  f"ff/min={_fmt(m['false_fires_min'])} lat_med={_fmt(m['latencia_med'])} "
                  f"n_ev={m['n_eventos']}")


def run_sweep(sessions, thresholds, min_frames, win, pre_margin):
    print("\nBarrido de umbral (§7.3) — punto de operación: máx catch con "
          f"false-fire <= {FALSE_FIRE_TARGET}/min")
    print(f"  {'threshold':>10} {'catch_rate':>11} {'ff/min':>8} {'sensitivity':>12}")
    rows = []
    for th in thresholds:
        m, _ = evaluate(sessions, th, min_frames, win, pre_margin)
        rows.append((th, m))
    feasible = [(th, m) for th, m in rows if m["false_fires_min"] <= FALSE_FIRE_TARGET]
    best_th = max(feasible, key=lambda x: x[1]["catch_rate"])[0] if feasible else None
    for th, m in rows:
        mark = "  <-- punto de operación" if th == best_th else ""
        print(f"  {th:>10.4f} {m['catch_rate']:>11.3f} {m['false_fires_min']:>8.3f} "
              f"{m['sensitivity']:>12.3f}{mark}")
    if best_th is None:
        print("  (ningún umbral cumple el objetivo de false-fire; revisar datos/condiciones)")
    return best_th


def append_csv(metrics, loc, conditions, trigger, threshold, min_frames, fecha):
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    new = not RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(CSV_HEADER)
        w.writerow([
            fecha, loc, "+".join(conditions), trigger,
            threshold, min_frames,
            round(metrics["catch_rate"], 4),
            round(metrics["false_fires_min"], 4),
            "" if metrics["latencia_med"] is None else round(metrics["latencia_med"], 4),
            metrics["n_eventos"], metrics["n_sesiones"],
            round(metrics["sensitivity"], 4),
        ])


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Arnés de validación de la mordida a nivel de evento")
    ap.add_argument("--sessions", nargs="+", default=[str(REPO_ROOT / "captures_bite" / "*")],
                    help="glob(s)/dir(s) de sesiones de captures_bite")
    ap.add_argument("--model", default=str(cfg.model_onnx), help="modelo ONNX (default: el de la ubicación)")
    ap.add_argument("--trigger", default="foam", choices=["foam"], help="trigger a evaluar")
    ap.add_argument("--threshold", type=float, default=cfg.bite.foam_threshold)
    ap.add_argument("--min-frames", type=int, default=cfg.bite.foam_min_frames)
    ap.add_argument("--win", type=float, default=4.0, help="ventana de captura desde t_b (s)")
    ap.add_argument("--pre-margin", type=float, default=1.0, help="margen previo a t_b (reacción humana, s)")
    ap.add_argument("--relocate-seconds", type=float, default=cfg.bite.relocate_seconds)
    ap.add_argument("--sweep", action="store_true", help="barrido de umbral (§7.3)")
    ap.add_argument("--loc", default=cfg.location, help="etiqueta de ubicación para el CSV")
    ap.add_argument("--fecha", default=date.today().isoformat(), help="fecha para el CSV (YYYY-MM-DD)")
    args = ap.parse_args()

    dirs = find_sessions(args.sessions)
    if not dirs:
        print(f"No hay sesiones en {args.sessions}. Corre tools.capture_bite en Windows y trae las carpetas.")
        return 0

    from pesca.corcho_detector import CorchoDetector  # import perezoso (necesita el .onnx)
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Modelo ausente: {model_path}. Colocá el detector.onnx de la ubicación (gitignored).")
        return 0
    detector = CorchoDetector(model_path, conf_threshold=cfg.detector.conf_threshold,
                              iou_threshold=cfg.detector.iou_threshold, imgsz=cfg.detector.imgsz)
    trigger = FoamTrigger(detector, args.relocate_seconds)

    print(f"Cargando {len(dirs)} sesiones y corriendo el trigger foam...")
    sessions = [s for d in dirs if (s := load_session(d, trigger))]
    if not sessions:
        print("Las sesiones no tenían frames legibles.")
        return 0

    metrics, classified = evaluate(sessions, args.threshold, args.min_frames, args.win, args.pre_margin)
    print_report(metrics, classified, args.threshold, args.min_frames)

    conditions = sorted({s["condition"] for s in sessions})
    append_csv(metrics, args.loc, conditions, args.trigger, args.threshold, args.min_frames, args.fecha)
    print(f"\nFila anexada a {RESULTS_CSV.relative_to(REPO_ROOT)}")

    if args.sweep:
        run_sweep(sessions, DEFAULT_SWEEP, args.min_frames, args.win, args.pre_margin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
