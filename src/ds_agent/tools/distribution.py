from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Tool


def _describe_distribution(col: str, skew: float, kurt: float, n: int) -> str:
    if abs(skew) < 0.5:
        skew_desc = "approximately symmetric"
    elif skew >= 1.5:
        skew_desc = "strongly right-skewed"
    elif skew >= 0.5:
        skew_desc = "moderately right-skewed"
    elif skew <= -1.5:
        skew_desc = "strongly left-skewed"
    else:
        skew_desc = "moderately left-skewed"

    if kurt > 1.0:
        kurt_desc = "with heavy tails (leptokurtic)"
    elif kurt < -1.0:
        kurt_desc = "with light tails (platykurtic)"
    else:
        kurt_desc = "with near-normal tail behavior"

    return f"**{col}** (n={n}) is {skew_desc} {kurt_desc}."


class DistributionTool(Tool):
    @property
    def name(self) -> str:
        return "distribution_analysis"

    @property
    def description(self) -> str:
        return (
            "Compute histograms, skew, and kurtosis for numeric columns. "
            "Returns aggregate statistics only — bin counts/edges, skew, kurtosis per column (ADR-0002). "
            "Call after schema_inference to summarize numeric distributions."
        )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Numeric column names to analyze. If omitted, all numeric columns.",
                }
            },
            "required": [],
        }

    def run(self, df: pd.DataFrame, columns: list[str] | None = None, **kwargs) -> dict:
        if columns is None:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
        else:
            numeric_cols = [
                c for c in columns
                if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
            ]

        results = []
        for col_name in numeric_cols:
            series = df[col_name].dropna()
            if len(series) == 0:
                continue

            counts, edges = np.histogram(series.values, bins="auto")
            skew = float(series.skew())
            kurt = float(series.kurtosis())  # excess kurtosis (Fisher definition)
            narrative = _describe_distribution(col_name, skew, kurt, len(series))

            results.append({
                "column": col_name,
                "bin_counts": counts.tolist(),
                "bin_edges": [round(e, 6) for e in edges.tolist()],
                "skew": round(skew, 4),
                "kurtosis": round(kurt, 4),
                "sample_count": len(series),
                "narrative": narrative,
            })

        return {"columns": results}
