"""Result containers and CSV helpers for experiment outputs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd


def rows_to_dataframe(rows: list[Mapping[str, object]]) -> pd.DataFrame:
    """Convert row dictionaries to a stable DataFrame."""

    return pd.DataFrame.from_records(rows)


def write_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame to CSV, creating parent directories as needed."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def resolve_output_paths(
    config: Mapping[str, object],
    defaults: Mapping[str, str | Path],
) -> dict[str, Path]:
    """Resolve output paths from config, supporting one or many outputs."""

    if "output_paths" in config:
        configured = dict(config["output_paths"])  # type: ignore[arg-type]
        return {
            key: Path(str(configured.get(key, default_path)))
            for key, default_path in defaults.items()
        }

    if "output_path" in config:
        if len(defaults) != 1:
            raise ValueError("output_path is only valid for single-output experiments")
        key = next(iter(defaults))
        return {key: Path(str(config["output_path"]))}

    return {key: Path(path) for key, path in defaults.items()}


def write_outputs(
    outputs: Mapping[str, pd.DataFrame],
    output_paths: Mapping[str, str | Path],
) -> dict[str, Path]:
    """Write named DataFrames to their configured CSV paths."""

    written: dict[str, Path] = {}
    for key, df in outputs.items():
        if key not in output_paths:
            raise KeyError(f"no output path configured for {key!r}")
        written[key] = write_dataframe(df, output_paths[key])
    return written


__all__ = [
    "resolve_output_paths",
    "rows_to_dataframe",
    "write_dataframe",
    "write_outputs",
]
