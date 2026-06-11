"""Generate PNG report figures from saved CSV summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from m2r_langevin.plotting import (
    plot_cost_aware_sweep,
    plot_gaussian_ula_mala_sweep,
    plot_trace_acf,
)
from m2r_langevin.samplers import mala
from m2r_langevin.targets import make_correlated_gaussian

DEFAULT_FIGURE_DIR = Path("fig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure-dir",
        default=DEFAULT_FIGURE_DIR,
        type=Path,
        help="Directory for generated PNG figures.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip figures whose source CSVs do not exist.",
    )
    return parser.parse_args()


def make_gaussian_trace_acf(output_path: Path) -> Path:
    """Regenerate the Gaussian MALA trace/ACF diagnostic figure."""

    target = make_correlated_gaussian(d=2, rho=0.9)
    result = mala(
        target.log_prob,
        target.grad_log_prob,
        np.zeros(2),
        step_size=0.25,
        n_samples=20_000,
        rng=0,
    )
    return plot_trace_acf(
        result.samples[2_000:],
        output_path,
        coordinate=0,
        max_lag=200,
        title="Gaussian MALA diagnostic",
    )


def maybe_plot(
    source_path: str,
    plotter,
    output_path: Path,
    allow_missing: bool,
    *args,
    **kwargs,
) -> Path | None:
    """Run one CSV-backed plot, optionally skipping missing inputs."""

    path = Path(source_path)
    if not path.exists():
        if allow_missing:
            print(f"skipping missing source: {path}")
            return None
        raise FileNotFoundError(
            f"{path} does not exist; run the corresponding experiment first"
        )
    return plotter(path, output_path, *args, **kwargs)


def main() -> None:
    args = parse_args()
    figure_dir = args.figure_dir

    outputs: list[Path | None] = [
        make_gaussian_trace_acf(figure_dir / "gaussian_trace_acf.png"),
        maybe_plot(
            "results/summary/gaussian_2d_sweep.csv",
            plot_gaussian_ula_mala_sweep,
            figure_dir / "gaussian_ula_mala_sweep.png",
            args.allow_missing,
        ),
        maybe_plot(
            "results/summary/linear_regression_cost_aware.csv",
            plot_cost_aware_sweep,
            figure_dir / "linear_regression_cost_aware.png",
            args.allow_missing,
            title="Bayesian linear regression",
        ),
        maybe_plot(
            "results/summary/logistic_regression_cost_aware.csv",
            plot_cost_aware_sweep,
            figure_dir / "logistic_regression_cost_aware.png",
            args.allow_missing,
            title="Bayesian logistic regression",
        ),
    ]

    for path in outputs:
        if path is not None:
            print(path)


if __name__ == "__main__":
    main()
