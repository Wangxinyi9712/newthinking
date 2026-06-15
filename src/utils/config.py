from pathlib import Path
from typing import Any
import yaml


class Config(dict):
    """
    dict-based config (TMI stable version)
    supports cfg["train"] style only
    """

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)

    def __setattr__(self, key, value):
        self[key] = value

    def get(self, key, default=None):
        return super().get(key, default)


def load_config(path: str | Path) -> Config:
    path = Path(path)

    if not path.exists():
        root = Path(__file__).resolve().parents[2]
        alt = root / "src" / "configs" / path.name

        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    required = ["data", "model", "train", "loss", "inference", "log"]

    for k in required:
        if k not in raw:
            raise ValueError(f"Missing config section: {k}")

    cfg = Config(raw)
    return cfg