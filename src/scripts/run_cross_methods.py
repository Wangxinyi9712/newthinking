from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_CONFIGS = [
    "src/configs/experiments/brats_group_k_unimatch_sdf.yaml",
    "src/configs/experiments/brats_group_l_unimatch_pde.yaml",
    "src/configs/experiments/brats_group_m_unimatch_ot.yaml",
    "src/configs/experiments/brats_group_n_pde_sdf_ot_uniteacher.yaml",
    "src/configs/experiments/brats_group_o_pde_sdf_ot_proto_uniteacher.yaml",
]


def run_cmd(cmd: list[str]) -> None:
    print("\n" + "=" * 80)
    print(" ".join(cmd))
    print("=" * 80)
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--ckpt", type=str, default="best")
    parser.add_argument("--source", type=str, default="teacher", choices=["teacher", "student"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for cfg in args.configs:
        cfg_path = Path(cfg)

        if not cfg_path.is_absolute():
            cfg_path = PROJECT_ROOT / cfg_path

        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")

        for seed in args.seeds:
            if not args.skip_train:
                cmd = [
                    sys.executable,
                    "src/scripts/train.py",
                    "--config",
                    str(cfg_path),
                    "--seed",
                    str(seed),
                ]

                if args.smoke:
                    cmd.append("--smoke")

                run_cmd(cmd)

            if not args.skip_eval:
                cmd = [
                    sys.executable,
                    "-m",
                    "src.scripts.evaluate",
                    "--config",
                    str(cfg_path),
                    "--ckpt",
                    args.ckpt,
                    "--seed",
                    str(seed),
                    "--source",
                    args.source,
                ]

                if args.smoke:
                    cmd.extend(["--max-val-batches", "5"])

                run_cmd(cmd)


if __name__ == "__main__":
    main()