from __future__ import annotations

import argparse
import copy
import subprocess
from pathlib import Path

import yaml


def deep_update(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True, help="e.g. src/configs/brats_group_e.yaml")
    parser.add_argument("--suite", default="src/configs/method_suite.yaml")
    parser.add_argument("--out", default="method/generated")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--train-script", default="src/scripts/train.py")
    args = parser.parse_args()

    with open(args.base_config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    with open(args.suite, "r", encoding="utf-8") as f:
        suite = yaml.safe_load(f)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods: dict = suite["methods"]
    for method_name, spec in methods.items():
        overrides = spec.get("overrides", {}) or {}
        cfg = deep_update(base_cfg, overrides)

        cfg["log"]["out_dir"] = f"runs/method_suite/{method_name}"

        cfg_path = out_dir / f"{method_name}.yaml"
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

        print(f"[GEN] {method_name}: {cfg_path}")

        if args.run:
            seeds = cfg.get("train", {}).get("seed", [0, 1, 2])
            for s in seeds:
                cmd = [
                    "python",
                    args.train_script,
                    "--config",
                    str(cfg_path),
                    "--seed",
                    str(s),
                ]
                print("[RUN]", " ".join(cmd))
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()