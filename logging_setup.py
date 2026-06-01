"""Logging estructurado: consola + archivo rotativo en logs/ (gitignored).

Una corrida real deja un log con timestamp analizable por ciclo. Importable headless.
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

LOGGER_NAME = "pesca"


def setup_logging(logging_cfg, level_override: str | None = None) -> logging.Logger:
    level_name = (level_override or logging_cfg.level).upper()
    level = getattr(logging, level_name, logging.INFO)

    logging_cfg.dir.mkdir(parents=True, exist_ok=True)
    log_path = logging_cfg.dir / logging_cfg.file

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    fileh = RotatingFileHandler(
        log_path, maxBytes=logging_cfg.max_bytes, backupCount=logging_cfg.backups, encoding="utf-8"
    )
    fileh.setFormatter(fmt)
    logger.addHandler(fileh)

    logger.info("Logging inicializado (nivel=%s, archivo=%s)", level_name, log_path)
    return logger
