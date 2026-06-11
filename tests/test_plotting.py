from pathlib import Path

import numpy as np
import pandas as pd

from m2r_langevin.plotting import (
    plot_cost_aware_sweep,
    plot_gaussian_ula_mala_sweep,
    plot_trace_acf,
)


def assert_png(path: Path) -> None:
    assert path.exists()
    assert path.suffix == ".png"
    assert path.stat().st_size > 0


def sweep_frame() -> pd.DataFrame:
    rows = []
    for sampler in ["ULA", "MALA", "KLMC"]:
        for h in [0.01, 0.02]:
            rows.append(
                {
                    "sampler": sampler,
                    "h": h,
                    "mean_error": 0.1 + h,
                    "cov_error": 0.2 + h,
                    "min_ess": 20.0 / h,
                    "work_units": 1000,
                    "ess_per_work": 0.01 / h,
                }
            )
    return pd.DataFrame(rows)


def test_plot_trace_acf_creates_png(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    samples = rng.standard_normal((200, 2))

    path = plot_trace_acf(samples, tmp_path / "trace.png", max_lag=50)

    assert_png(path)


def test_plot_gaussian_ula_mala_sweep_creates_png(tmp_path: Path) -> None:
    path = plot_gaussian_ula_mala_sweep(sweep_frame(), tmp_path / "gaussian.png")

    assert_png(path)


def test_plot_cost_aware_sweep_creates_png(tmp_path: Path) -> None:
    path = plot_cost_aware_sweep(sweep_frame(), tmp_path / "cost.png", title="Cost")

    assert_png(path)
