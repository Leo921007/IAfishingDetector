"""Análisis del splash de la mordida (Etapa 8B) — headless, WSL2.

Consume captures_bite/<ts>/ (frames + manifest.json de capture_bite.py). Por ventana:
  - localiza el corcho con el detector (best_zona.onnx) en los frames PREVIOS al splash;
  - define un parche alrededor del bbox (x1.5) y calcula por frame dos métricas de splash:
      (a) frame-diff: media de |gray[t]-gray[t-k]| en el parche (movimiento del agua);
      (b) foam: fracción de píxeles casi-blancos (alto V, baja S en HSV) en el parche (agua blanca);
  - baseline = métrica durante el flote (estable); spike = pico en la ventana previa al keypress
    (el usuario marca ~0.3-0.5 s tarde). Reporta cuál separa mejor, su duración y el fps mínimo.

Reusa CorchoDetector. No toca loop/detector/dataset/audio. Salidas en captures_bite/analysis/ (gitignored).

Uso:
    .venv/bin/python analyze_splash.py            # analiza captures_bite/*/
    .venv/bin/python analyze_splash.py --selftest # prueba el pipeline con una ventana sintética
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import cv2
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from config import REPO_ROOT
from corcho_detector import CorchoDetector
from splash import PATCH_SCALE, foam_value, patch_box  # fuente única (vivo == offline)

ROOT = REPO_ROOT / "captures_bite"
OUT = ROOT / "analysis"
MODEL = REPO_ROOT / "models" / "corcho_detector" / "best_zona.onnx"
DIFF_K = 1
REACTION_S = 0.4  # lag humano: el splash está ~0.4 s antes del keypress


def load_window(d: Path):
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    frames = [(fm, cv2.imread(str(d / fm["file"]))) for fm in man["frames"]]
    frames = [(fm, img) for fm, img in frames if img is not None]
    return man, frames


def locate_bbox(frames, detector, keypress_index):
    """Última detección confiable del corcho en los frames previos al keypress."""
    bbox = None
    for fm, img in frames[:max(1, keypress_index + 1)]:
        dets = detector.detect(img)
        if dets:
            b = max(dets, key=lambda x: x.conf)
            bbox = (b.x1, b.y1, b.x2, b.y2)
    return bbox


def metrics(frames, patch, k=DIFF_K):
    X1, Y1, X2, Y2 = patch
    grays = [cv2.cvtColor(img[Y1:Y2, X1:X2], cv2.COLOR_BGR2GRAY).astype(np.int16) for _, img in frames]
    fdiff = [0.0] * len(frames)
    for i in range(k, len(frames)):
        fdiff[i] = float(np.mean(np.abs(grays[i] - grays[i - k])))
    foam = [foam_value(img, patch) for _, img in frames]
    return fdiff, foam


def _sep(values, keypress_index, fps):
    """baseline (flote) vs spike (ventana previa al keypress por el lag humano)."""
    reaction = max(1, int(REACTION_S * fps))
    bite_lo = max(0, keypress_index - reaction - max(2, int(0.6 * fps)))
    bite_hi = keypress_index
    float_vals = values[:bite_lo] if bite_lo > 2 else values[:max(1, keypress_index // 2)]
    base = statistics.median(float_vals) if float_vals else 0.0
    bite_vals = values[bite_lo:bite_hi + 1] or values
    spike = max(bite_vals)
    spike_idx = bite_lo + bite_vals.index(spike)
    half = base + 0.5 * (spike - base)
    dur = sum(1 for v in bite_vals if v >= half)
    return {"baseline": base, "spike": spike, "spike_idx": spike_idx,
            "ratio": (spike / base) if base > 1e-9 else float("inf"), "peak_frames": dur}


def make_png(name, fdiff, foam, man, sep_foam):
    kp = man["keypress_index"]
    fig, ax = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
    ax[0].plot(foam, "-o", ms=2, color="tab:orange"); ax[0].set_ylabel("foam")
    ax[1].plot(fdiff, "-o", ms=2, color="tab:blue"); ax[1].set_ylabel("frame-diff")
    for a in ax:
        a.axvline(kp, color="k", ls=":", label="keypress")
        a.axvline(sep_foam["spike_idx"], color="tab:green", ls="--", label="pico foam")
    ax[0].legend(fontsize=7, loc="upper left")
    ax[1].set_xlabel(f"frame ({man['fps_real']} fps)")
    ax[0].set_title(name)
    fig.tight_layout(); fig.savefig(OUT / f"{name}.png", dpi=90); plt.close(fig)


def fps_min(values, keypress_index, fps):
    """¿hasta qué submuestreo (fps/step) el pico foam sigue separable (ratio>=2)?"""
    best_step = 1
    for step in (1, 2, 3, 4, 6):
        sub = values[::step]
        kp = keypress_index // step
        s = _sep(sub, kp, fps / step)
        if s["ratio"] >= 2.0:
            best_step = step
    return fps / best_step, best_step


def analyze_dir(d: Path, detector, csv_writer):
    man, frames = load_window(d)
    if not frames:
        return None
    h, w = frames[0][1].shape[:2]
    bbox = locate_bbox(frames, detector, man["keypress_index"])
    patch = patch_box(bbox, w, h)
    fdiff, foam = metrics(frames, patch)
    sf = _sep(foam, man["keypress_index"], man["fps_real"])
    sd = _sep(fdiff, man["keypress_index"], man["fps_real"])
    fmin, step = fps_min(foam, man["keypress_index"], man["fps_real"])
    for i, (fm, _) in enumerate(frames):
        csv_writer.writerow([d.name, i, fm.get("dt_from_press"), round(fdiff[i], 3), round(foam[i], 5)])
    make_png(d.name, fdiff, foam, man, sf)
    return {"name": d.name, "fps": man["fps_real"], "bbox_found": bbox is not None,
            "foam": sf, "fdiff": sd, "fps_min_separable": round(fmin, 1), "step": step}


def _make_selftest(dirpath: Path, fps=30, n=60, keypress=45):
    dirpath.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    meta = []
    for i in range(n):
        img = rng.integers(20, 40, (80, 80, 3), dtype=np.uint8)  # agua oscura
        cv2.circle(img, (40, 40), 4, (20, 30, 120), -1)          # corcho
        if keypress - 8 <= i <= keypress - 3:                    # splash antes del keypress
            cv2.circle(img, (40, 40), 13, (240, 240, 240), -1)
        cv2.imwrite(str(dirpath / f"frame_{i:04d}.jpg"), img)
        meta.append({"file": f"frame_{i:04d}.jpg", "t": round(i / fps, 4),
                     "dt_from_press": round((i - keypress) / fps, 4)})
    (dirpath / "manifest.json").write_text(json.dumps(
        {"roi": {}, "fps_real": fps, "pre_seconds": keypress / fps, "post_seconds": (n - keypress) / fps,
         "n_frames": n, "keypress_index": keypress, "frames": meta}), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Análisis del splash de la mordida")
    ap.add_argument("--selftest", action="store_true", help="prueba el pipeline con una ventana sintética")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    detector = CorchoDetector(MODEL, conf_threshold=0.25, iou_threshold=0.45, imgsz=640)

    if args.selftest:
        d = OUT / "_selftest"
        _make_selftest(d)
        csv_path = OUT / "selftest.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            res = analyze_dir(d, detector, csv.writer(f))
        print(f"[selftest] foam baseline={res['foam']['baseline']:.4f} spike={res['foam']['spike']:.4f} "
              f"ratio={res['foam']['ratio']:.1f} peak_frames={res['foam']['peak_frames']}")
        passed = res["foam"]["ratio"] >= 2.0
        print("[selftest]", "PASS" if passed else "FAIL")
        return 0 if passed else 1

    dirs = [d for d in sorted(ROOT.iterdir()) if d.is_dir() and d.name != "analysis"
            and (d / "manifest.json").exists()] if ROOT.is_dir() else []
    if not dirs:
        print(f"No hay capturas en {ROOT}. Corre capture_bite.py en Windows y trae las carpetas aquí.")
        return 0

    rows = []
    with (OUT / "per_frame.csv").open("w", newline="", encoding="utf-8") as f:
        cw = csv.writer(f)
        cw.writerow(["window", "idx", "dt_from_press", "frame_diff", "foam"])
        for d in dirs:
            r = analyze_dir(d, detector, cw)
            if r:
                rows.append(r)

    print("=" * 60)
    print(f"Análisis del splash | {len(rows)} ventanas | salidas: {OUT}")
    print("=" * 60)
    for r in rows:
        print(f"\n[{r['name']}] fps={r['fps']} bbox={'sí' if r['bbox_found'] else 'fallback'}")
        print(f"  foam : base={r['foam']['baseline']:.4f} spike={r['foam']['spike']:.4f} "
              f"ratio={r['foam']['ratio']:.1f} dur={r['foam']['peak_frames']}f")
        print(f"  fdiff: base={r['fdiff']['baseline']:.2f} spike={r['fdiff']['spike']:.2f} "
              f"ratio={r['fdiff']['ratio']:.1f} dur={r['fdiff']['peak_frames']}f")
        print(f"  fps mínimo separable (foam): ~{r['fps_min_separable']} (step {r['step']})")
    if rows:
        med_foam = statistics.median(r["foam"]["ratio"] for r in rows if r["foam"]["ratio"] != float("inf"))
        print(f"\nResumen: ratio foam mediano={med_foam:.1f}. "
              "Regla candidata: foam_patch > baseline*K durante >=N frames (ajustar con estos datos).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
