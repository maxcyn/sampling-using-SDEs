from pathlib import Path

import numpy as np
import pandas as pd

import m2r_langevin.plotting as plotting
from m2r_langevin.plotting import (
    plot_cost_aware_sweep,
    plot_gaussian_sampler_sweep,
    plot_trace_acf,
)

DIAGNOSTIC_PANEL_TITLES = [
    "Mean error",
    "Covariance error",
    "Minimum-coordinate ESS",
    "Cost-normalised minimum-coordinate ESS",
]


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


def capture_plot_figure(plotter, tmp_path: Path, monkeypatch):
    captured = {}

    def capture_figure(fig, output_path):
        captured["figure"] = fig
        return Path(output_path).with_suffix(".png")

    monkeypatch.setattr(plotting, "_save", capture_figure)
    plotter(sweep_frame(), tmp_path / "plot.png")
    return captured["figure"]


def assert_four_diagnostic_panels(fig) -> None:
    assert len(fig.axes) == 4
    assert [ax.get_title() for ax in fig.axes] == DIAGNOSTIC_PANEL_TITLES
    assert fig.axes[2].get_ylabel() == "minimum ESS"
    assert fig.axes[3].get_ylabel() == "ESS per work unit"
    assert np.allclose(fig.axes[2].lines[0].get_ydata(), [2000.0, 1000.0])
    assert np.allclose(fig.axes[3].lines[0].get_ydata(), [1.0, 0.5])


def test_plot_trace_acf_creates_png(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    samples = rng.standard_normal((200, 2))

    path = plot_trace_acf(samples, tmp_path / "trace.png", max_lag=50)

    assert_png(path)


def test_plot_gaussian_sampler_sweep_creates_png(tmp_path: Path) -> None:
    path = plot_gaussian_sampler_sweep(sweep_frame(), tmp_path / "gaussian.png")

    assert_png(path)


def test_plot_gaussian_sweep_includes_klmc(tmp_path: Path, monkeypatch) -> None:
    plotted_labels: list[str] = []

    def capture_labels(fig, output_path):
        plotted_labels.extend(line.get_label() for line in fig.axes[0].lines)
        return Path(output_path).with_suffix(".png")

    monkeypatch.setattr(plotting, "_save", capture_labels)

    plot_gaussian_sampler_sweep(sweep_frame(), tmp_path / "gaussian.png")

    assert plotted_labels == ["ULA", "MALA", "KLMC"]


def test_plot_gaussian_sweep_uses_four_diagnostic_panels(
    tmp_path: Path, monkeypatch
) -> None:
    fig = capture_plot_figure(plot_gaussian_sampler_sweep, tmp_path, monkeypatch)

    assert_four_diagnostic_panels(fig)


def test_plot_cost_aware_sweep_creates_png(tmp_path: Path) -> None:
    path = plot_cost_aware_sweep(sweep_frame(), tmp_path / "cost.png")

    assert_png(path)


def test_plot_cost_aware_sweep_uses_four_diagnostic_panels(
    tmp_path: Path, monkeypatch
) -> None:
    fig = capture_plot_figure(plot_cost_aware_sweep, tmp_path, monkeypatch)

    assert_four_diagnostic_panels(fig)


def test_plot_cost_aware_sweep_has_no_figure_title(tmp_path: Path, monkeypatch) -> None:
    figure_title = object()

    def capture_title(fig, output_path):
        nonlocal figure_title
        figure_title = fig._suptitle
        return Path(output_path).with_suffix(".png")

    monkeypatch.setattr(plotting, "_save", capture_title)

    plot_cost_aware_sweep(sweep_frame(), tmp_path / "cost.png")

    assert figure_title is None
