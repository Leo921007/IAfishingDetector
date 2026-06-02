"""Análisis offline de la "mordida visual" (dip del corcho) — Etapa 8A.

Corre el detector NUEVO (best_zona.onnx) frame a frame sobre las secuencias de cada ciclo y caracteriza
la firma del dip: caída de center_y (el corcho se hunde -> y aumenta hacia abajo en la imagen) y/o
desaparición del corcho (salpicón). Compara ciclos con mordida (outcome=recogido) vs control (esperando).

Solo ANÁLISIS: no toca el loop, el detector, el dataset ni el audio. Headless (matplotlib Agg).

Uso:
    .venv/bin/python analyze_dip.py
Salidas (gitignored, dentro de dataset_zona/):
    dataset_zona/dip_analysis/per_frame.csv, per_cycle.csv, cy_<sesion>_c<NN>_<outcome>.png
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from config import REPO_ROOT
from corcho_detector import CorchoDetector

DEFAULT_SESSIONS = ["20260602_100757", "20260601_222727", "20260601_221053"]
MODEL = REPO_ROOT / "models" / "corcho_detector" / "best_zona.onnx"
OUT = REPO_ROOT / "dataset_zona" / "dip_analysis"
CONF = 0.25
BASELINE_WIN = 6


def load_events(sdir: Path) -> dict:
    out = {}
    p = sdir / "events.jsonl"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line)
                out[e["cycle"]] = e
    return out


def trajectory(detector, frame_paths):
    traj = []
    for i, fp in enumerate(frame_paths):
        dets = detector.detect(cv2.imread(str(fp)))
        if dets:
            b = max(dets, key=lambda x: x.conf)
            traj.append({"idx": i, "det": 1, "cy": (b.y1 + b.y2) / 2, "conf": float(b.conf)})
        else:
            traj.append({"idx": i, "det": 0, "cy": None, "conf": 0.0})
    return traj


def characterize(traj):
    det = [t["det"] for t in traj]
    n = len(traj)

    # Magnitud del dip: máx (center_y - mediana móvil reciente) hacia abajo.
    max_dip, dip_idx, seen = 0.0, None, []
    for t in traj:
        if t["det"]:
            if len(seen) >= 3:
                base = statistics.median(seen[-BASELINE_WIN:])
                drop = t["cy"] - base
                if drop > max_dip:
                    max_dip, dip_idx = drop, t["idx"]
            seen.append(t["cy"])

    # Rachas sin detección (vanish): (inicio, longitud).
    runs, run, start = [], 0, 0
    for i, d in enumerate(det):
        if d == 0:
            if run == 0:
                start = i
            run += 1
        elif run:
            runs.append((start, run))
            run = 0
    if run:
        runs.append((start, run))

    longest = max((r for _, r in runs), default=0)
    trailing = runs[-1][1] if runs and runs[-1][0] + runs[-1][1] == n else 0
    mid_vanish = int(any(s > 0 and s + r < n for s, r in runs))  # desaparece y reaparece
    return {
        "max_dip_px": round(max_dip, 1), "dip_idx": dip_idx if dip_idx is not None else -1,
        "longest_vanish": longest, "trailing_vanish": trailing, "mid_vanish": mid_vanish,
        "detect_frac": round(sum(det) / n, 2),
    }


def _summ(label, values):
    if not values:
        print(f"  {label:16}: (sin datos)")
        return
    vs = sorted(values)
    q = lambda p: vs[min(len(vs) - 1, int(p * (len(vs) - 1)))]
    print(f"  {label:16}: n={len(vs):3} med={statistics.median(vs):6.1f} "
          f"p25={q(0.25):6.1f} p75={q(0.75):6.1f} max={max(vs):6.1f}")


def make_png(traj, info, session, cycle, outcome):
    xs = [t["idx"] for t in traj]
    ys = [t["cy"] if t["det"] else None for t in traj]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(xs, ys, "-o", ms=3, color="tab:blue", label="center_y")
    for t in traj:  # no-detecciones marcadas abajo
        if not t["det"]:
            ax.axvline(t["idx"], color="tab:red", alpha=0.25, lw=4)
    if info["dip_idx"] >= 0:
        ax.axvline(info["dip_idx"], color="tab:green", ls="--", label=f"dip {info['max_dip_px']}px")
    ax.invert_yaxis()  # y crece hacia abajo en imagen
    ax.set_title(f"{session} c{cycle} [{outcome}]  vanish_trail={info['trailing_vanish']} "
                 f"detect_frac={info['detect_frac']}")
    ax.set_xlabel("frame (10 fps)")
    ax.set_ylabel("center_y (px, abajo=mayor)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"cy_{session}_c{cycle:03d}_{outcome}.png", dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Caracteriza el dip de la mordida (offline)")
    ap.add_argument("--sessions", nargs="*", default=DEFAULT_SESSIONS)
    ap.add_argument("--conf", type=float, default=CONF)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    detector = CorchoDetector(MODEL, conf_threshold=args.conf, iou_threshold=0.45, imgsz=640)

    per_frame = open(OUT / "per_frame.csv", "w", newline="", encoding="utf-8")
    fw = csv.writer(per_frame); fw.writerow(["session", "cycle", "outcome", "idx", "det", "cy", "conf"])
    per_cycle = open(OUT / "per_cycle.csv", "w", newline="", encoding="utf-8")
    cw = csv.writer(per_cycle)
    cw.writerow(["session", "cycle", "outcome", "max_dip_px", "dip_idx", "longest_vanish",
                 "trailing_vanish", "mid_vanish", "detect_frac"])

    by_outcome = {}  # outcome -> list of info dicts
    png_quota = {"recogido": 3, "esperando": 3}
    pngs_done = {"recogido": 0, "esperando": 0}

    for sname in args.sessions:
        sdir = REPO_ROOT / "sessions" / sname
        if not sdir.is_dir():
            print(f"[aviso] sesión no encontrada: {sname}")
            continue
        events = load_events(sdir)
        for fdir in sorted(sdir.glob("cycle_*_frames")):
            cycle = int(fdir.name.split("_")[1])
            frames = sorted(fdir.glob("frame_*.png"))
            if not frames:
                continue
            outcome = events.get(cycle, {}).get("outcome", "?")
            traj = trajectory(detector, frames)
            info = characterize(traj)
            by_outcome.setdefault(outcome, []).append(info)

            for t in traj:
                fw.writerow([sname, cycle, outcome, t["idx"], t["det"],
                             "" if t["cy"] is None else round(t["cy"], 1), round(t["conf"], 3)])
            cw.writerow([sname, cycle, outcome, info["max_dip_px"], info["dip_idx"],
                         info["longest_vanish"], info["trailing_vanish"], info["mid_vanish"],
                         info["detect_frac"]])

            if sname == "20260602_100757" and pngs_done.get(outcome, 0) < png_quota.get(outcome, 0):
                make_png(traj, info, sname, cycle, outcome)
                pngs_done[outcome] += 1

    per_frame.close(); per_cycle.close()

    print("=" * 64)
    print(f"Análisis del dip  |  modelo: {MODEL.name}  |  conf={args.conf}")
    print(f"Salidas: {OUT}")
    print("=" * 64)
    for outcome in ("recogido", "esperando", "corcho_no_detectado", "recast_sin_corcho", "sin_sonido"):
        infos = by_outcome.get(outcome)
        if not infos:
            continue
        print(f"\n[{outcome}]  (n={len(infos)} ciclos)")
        _summ("max_dip_px", [i["max_dip_px"] for i in infos])
        _summ("longest_vanish", [i["longest_vanish"] for i in infos])
        _summ("trailing_vanish", [i["trailing_vanish"] for i in infos])
        _summ("detect_frac", [i["detect_frac"] for i in infos])
        print(f"  mid_vanish (desaparece-reaparece): {sum(i['mid_vanish'] for i in infos)}/{len(infos)}")
    print("\n(PNGs de QA en", OUT, ")")


if __name__ == "__main__":
    main()
