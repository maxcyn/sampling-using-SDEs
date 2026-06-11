"""Run one configured M2R experiment and write CSV outputs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from m2r_langevin.experiments import (
    dimension_scaling,
    gaussian_sweep,
    klmc_friction,
    linear_regression,
    logistic_regression,
    precision_scaling,
)
from m2r_langevin.results import write_outputs

EXPERIMENTS = {
    "gaussian_sweep": gaussian_sweep,
    "linear_regression": linear_regression,
    "logistic_regression": logistic_regression,
    "dimension_scaling": dimension_scaling,
    "precision_scaling": precision_scaling,
    "klmc_friction": klmc_friction,
}


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("config file must contain a mapping")
    return config


def run_config(config: Mapping[str, Any]) -> dict[str, Path]:
    experiment_name = str(config.get("experiment", ""))
    if experiment_name not in EXPERIMENTS:
        known = ", ".join(sorted(EXPERIMENTS))
        raise ValueError(
            f"unknown experiment {experiment_name!r}; expected one of {known}"
        )

    module = EXPERIMENTS[experiment_name]
    outputs = module.run(config)
    output_paths = module.output_paths(config)
    return write_outputs(outputs, output_paths)


def run_config_file(path: str | Path) -> dict[str, Path]:
    return run_config(load_config(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a YAML experiment config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = run_config_file(args.config)
    for key, path in written.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
