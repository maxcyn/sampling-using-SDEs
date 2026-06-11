"""Run all configured M2R experiments and write CSV outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from run_experiment import run_config_file

DEFAULT_CONFIGS = [
    "configs/gaussian_2d.yaml",
    "configs/linear_regression.yaml",
    "configs/logistic_regression.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        nargs="*",
        default=DEFAULT_CONFIGS,
        help="Config files to run. Defaults to the full configured suite.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for config in args.configs:
        print(f"running {config}")
        written = run_config_file(Path(config))
        for key, path in written.items():
            print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
