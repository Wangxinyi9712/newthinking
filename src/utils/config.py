from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    data: dict[str, Any]
    model: dict[str, Any]
    train: dict[str, Any]
    loss: dict[str, Any]
    inference: dict[str, Any]
    log: dict[str, Any]


def load_config(path: str | Path) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    required = ["data", "model", "train", "loss", "inference", "log"]
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"Missing config sections: {missing}")
    return Config(**{k: raw[k] for k in required})
