from pathlib import Path
import yaml


class Config(dict):
    """
    TMI-safe config:
    - supports cfg["train"]
    - supports cfg.get()
    - avoids attribute confusion
    """

    def __init__(self, raw: dict):
        super().__init__(raw)
        self.__dict__ = self

    def __getattr__(self, item):
        if item in self:
            return self[item]
        raise AttributeError(item)


def load_config(path: str | Path) -> Config:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Config(raw)