"""Plotting helpers for saved experiment outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from m2r_langevin.diagnostics import autocorr_fft, effective_sample_size

DataFrameOrPath = pd.DataFrame | str | Path
SAMPLER_ORDER = ["ULA", "MALA", "KLMC"]


def _load_frame(data: DataFrameOrPath) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.read_csv(data)


def _prepare_output(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() != ".png":
        output_path = output_path.with_suffix(".png")
    return output_path


def _save(fig: plt.Figure, output_path: str | Path) -> Path:
    path = _prepare_output(output_path)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include="number").columns
    return (
        df.groupby(group_cols, as_index=False)[list(numeric_cols)]
        .mean(numeric_only=True)
        .sort_values(group_cols)
    )


def _plot_metric_by_sampler(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    ylabel: str,
) -> None:
    for sampler in SAMPLER_ORDER:
        subset = df[df["sampler"] == sampler]
        if subset.empty:
            continue
        subset = subset.sort_values(x_col)
        ax.plot(subset[x_col], subset[y_col], marker="o", label=sampler)
    ax.set_xlabel(x_col.replace("_", " "))
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def plot_trace_acf(
    samples: ArrayLike,
    output_path: str | Path,
    coordinate: int = 0,
    max_lag: int = 200,
    title: str | None = None,
) -> Path:
    """Plot a coordinate trace and ACF for any sampled chain."""

    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    if samples.ndim != 2:
        raise ValueError("samples must be one- or two-dimensional")
    if not 0 <= coordinate < samples.shape[1]:
        raise ValueError("coordinate is out of bounds")

    chain = samples[:, coordinate]
    acf = autocorr_fft(chain, max_lag=min(max_lag, chain.size))
    ess = effective_sample_size(chain, max_lag=min(max_lag, chain.size))

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].plot(chain, linewidth=0.7)
    axes[0].set_title(f"Trace, coordinate {coordinate}")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("value")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(np.arange(acf.size), acf, linewidth=1.0)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_title(f"ACF, ESS={ess:.0f}")
    axes[1].set_xlabel("lag")
    axes[1].set_ylabel("autocorrelation")
    axes[1].grid(True, alpha=0.3)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, output_path)


def plot_gaussian_ula_mala_sweep(
    data: DataFrameOrPath,
    output_path: str | Path,
) -> Path:
    """Plot the Gaussian ULA-vs-MALA sweep used in the report."""

    df = _load_frame(data)
    df = df[df["sampler"].isin(["ULA", "MALA"])]
    df = _aggregate(df, ["sampler", "h"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    _plot_metric_by_sampler(axes[0], df, "h", "mean_error", "mean error")
    _plot_metric_by_sampler(axes[1], df, "h", "cov_error", "covariance error")
    _plot_metric_by_sampler(axes[2], df, "h", "min_ess", "minimum ESS")
    axes[0].set_title("Mean error")
    axes[1].set_title("Covariance error")
    axes[2].set_title("Minimum coordinate ESS")
    axes[0].legend()
    fig.tight_layout()
    return _save(fig, output_path)


def plot_cost_aware_sweep(
    data: DataFrameOrPath,
    output_path: str | Path,
    title: str,
) -> Path:
    """Plot mean error, covariance error, and ESS per work unit vs step size."""

    df = _aggregate(_load_frame(data), ["sampler", "h"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    _plot_metric_by_sampler(axes[0], df, "h", "mean_error", "mean error")
    _plot_metric_by_sampler(axes[1], df, "h", "cov_error", "covariance error")
    _plot_metric_by_sampler(axes[2], df, "h", "ess_per_work", "ESS per work unit")
    axes[0].set_title("Mean error")
    axes[1].set_title("Covariance error")
    axes[2].set_title("Cost-normalised ESS")
    axes[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, output_path)


def plot_dimension_scaling(data: DataFrameOrPath, output_path: str | Path) -> Path:
    """Plot diagnostics as target dimension changes."""

    df = _aggregate(_load_frame(data), ["sampler", "dimension"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    _plot_metric_by_sampler(axes[0], df, "dimension", "mean_error", "mean error")
    _plot_metric_by_sampler(axes[1], df, "dimension", "cov_error", "covariance error")
    _plot_metric_by_sampler(
        axes[2], df, "dimension", "ess_per_work", "ESS per work unit"
    )
    axes[0].set_title("Mean error")
    axes[1].set_title("Covariance error")
    axes[2].set_title("Cost-normalised ESS")
    axes[0].legend()
    fig.tight_layout()
    return _save(fig, output_path)


def plot_precision_scaling(data: DataFrameOrPath, output_path: str | Path) -> Path:
    """Plot diagnostics against computational work budget."""

    df = _aggregate(_load_frame(data), ["sampler", "work_units"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    _plot_metric_by_sampler(axes[0], df, "work_units", "mean_error", "mean error")
    _plot_metric_by_sampler(axes[1], df, "work_units", "cov_error", "covariance error")
    _plot_metric_by_sampler(axes[2], df, "work_units", "min_ess", "minimum ESS")
    for ax in axes:
        ax.set_xscale("log")
    axes[0].set_title("Mean error")
    axes[1].set_title("Covariance error")
    axes[2].set_title("Minimum coordinate ESS")
    axes[0].legend()
    fig.tight_layout()
    return _save(fig, output_path)


def plot_klmc_friction_sweep(data: DataFrameOrPath, output_path: str | Path) -> Path:
    """Plot KLMC diagnostics over the friction parameter."""

    df = _aggregate(_load_frame(data), ["sampler", "gamma"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    _plot_metric_by_sampler(axes[0], df, "gamma", "mean_error", "mean error")
    _plot_metric_by_sampler(axes[1], df, "gamma", "cov_error", "covariance error")
    _plot_metric_by_sampler(axes[2], df, "gamma", "ess_per_work", "ESS per work unit")
    for ax in axes:
        ax.set_xscale("log", base=2)
    axes[0].set_title("Mean error")
    axes[1].set_title("Covariance error")
    axes[2].set_title("Cost-normalised ESS")
    fig.tight_layout()
    return _save(fig, output_path)


__all__ = [
    "plot_cost_aware_sweep",
    "plot_dimension_scaling",
    "plot_gaussian_ula_mala_sweep",
    "plot_klmc_friction_sweep",
    "plot_precision_scaling",
    "plot_trace_acf",
]
