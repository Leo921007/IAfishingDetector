"""Carga de la configuración central (config.yaml).

Resuelve todas las rutas relativas a la raíz del repositorio (sin rutas absolutas de Windows).
Importable sin display ni audio.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DetectorCfg:
    conf_threshold: float
    iou_threshold: float
    imgsz: int


@dataclass(frozen=True)
class ROICfg:
    left: int
    top: int
    width: int
    height: int

    def as_mss(self) -> dict:
        """Diccionario en el formato que espera mss.grab()."""
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}


@dataclass(frozen=True)
class InputCfg:
    cast_key: str
    loot_button: str
    delay_after_click: float
    move_settle: float
    click_hold: float
    loot_settle: float
    watchdog_interval: float
    watchdog_warn_after: int
    mouse_park: Tuple[int, int]


@dataclass(frozen=True)
class BiteCfg:
    foam_threshold: float
    foam_min_frames: int
    poll_fps: float
    relocate_seconds: float
    relocate_tolerance: int
    max_wait_seconds: float
    locate_timeout: float


@dataclass(frozen=True)
class LoggingCfg:
    level: str
    dir: Path
    file: str
    max_bytes: int
    backups: int


@dataclass(frozen=True)
class FramesCfg:
    enabled: bool
    fps: float
    max_frames: int


@dataclass(frozen=True)
class SessionCfg:
    enabled: bool
    dir: Path
    frames: FramesCfg


@dataclass(frozen=True)
class Config:
    model_onnx: Path
    detector: DetectorCfg
    roi: ROICfg
    input: InputCfg
    bite: BiteCfg
    logging: LoggingCfg
    session: SessionCfg


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path is not None else REPO_ROOT / "config.yaml"
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    det = data["detector"]
    roi = data["roi"]
    inp = data["input"]
    log = data["logging"]
    ses = data["session"]

    return Config(
        model_onnx=REPO_ROOT / data["model"]["onnx"],
        detector=DetectorCfg(
            conf_threshold=float(det["conf_threshold"]),
            iou_threshold=float(det["iou_threshold"]),
            imgsz=int(det["imgsz"]),
        ),
        roi=ROICfg(int(roi["left"]), int(roi["top"]), int(roi["width"]), int(roi["height"])),
        bite=BiteCfg(
            foam_threshold=float(data.get("bite", {}).get("foam_threshold", 0.005)),
            foam_min_frames=int(data.get("bite", {}).get("foam_min_frames", 2)),
            poll_fps=float(data.get("bite", {}).get("poll_fps", 30)),
            relocate_seconds=float(data.get("bite", {}).get("relocate_seconds", 0.5)),
            relocate_tolerance=int(data.get("bite", {}).get("relocate_tolerance", 3)),
            max_wait_seconds=float(data.get("bite", {}).get("max_wait_seconds", 25)),
            locate_timeout=float(data.get("bite", {}).get("locate_timeout", 3.0)),
        ),
        input=InputCfg(
            cast_key=str(inp["cast_key"]),
            loot_button=str(inp["loot_button"]),
            delay_after_click=float(inp["delay_after_click"]),
            move_settle=float(inp.get("move_settle", 0.08)),
            click_hold=float(inp.get("click_hold", 0.06)),
            loot_settle=float(inp.get("loot_settle", 0.2)),
            watchdog_interval=float(inp.get("watchdog_interval", 3.0)),
            watchdog_warn_after=int(inp.get("watchdog_warn_after", 5)),
            mouse_park=(int(inp.get("mouse_park", {}).get("x", 300)),
                        int(inp.get("mouse_park", {}).get("y", 700))),
        ),
        logging=LoggingCfg(
            level=str(log["level"]),
            dir=REPO_ROOT / log["dir"],
            file=str(log["file"]),
            max_bytes=int(log["max_bytes"]),
            backups=int(log["backups"]),
        ),
        session=SessionCfg(
            enabled=bool(ses["enabled"]),
            dir=REPO_ROOT / ses["dir"],
            frames=FramesCfg(
                enabled=bool(ses["frames"]["enabled"]),
                fps=float(ses["frames"]["fps"]),
                max_frames=int(ses["frames"]["max_frames"]),
            ),
        ),
    )
