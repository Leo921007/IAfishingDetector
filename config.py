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


@dataclass(frozen=True)
class AudioCfg:
    references: List[Path]
    fs: int
    duration: int
    gain: float
    bandpass: Tuple[float, float]
    similarity_threshold: float
    listen_window: int


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
    audio: AudioCfg
    logging: LoggingCfg
    session: SessionCfg


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path is not None else REPO_ROOT / "config.yaml"
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    det = data["detector"]
    roi = data["roi"]
    inp = data["input"]
    aud = data["audio"]
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
        input=InputCfg(
            cast_key=str(inp["cast_key"]),
            loot_button=str(inp["loot_button"]),
            delay_after_click=float(inp["delay_after_click"]),
        ),
        audio=AudioCfg(
            references=[REPO_ROOT / r for r in aud["references"]],
            fs=int(aud["fs"]),
            duration=int(aud["duration"]),
            gain=float(aud["gain"]),
            bandpass=(float(aud["bandpass"][0]), float(aud["bandpass"][1])),
            similarity_threshold=float(aud["similarity_threshold"]),
            listen_window=int(aud["listen_window"]),
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
