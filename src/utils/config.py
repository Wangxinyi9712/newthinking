from dataclasses import dataclass, field
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

    ablation_switches: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


def load_config(path: str | Path):
    path = Path(path)

    if not path.exists():
        root = Path(__file__).resolve().parents[1]
        alt = root / path
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(f"config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return raw