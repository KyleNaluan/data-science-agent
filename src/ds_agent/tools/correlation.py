from __future__ import annotations

import math

import pandas as pd

from .base import Tool


def _describe_correlation(col_a: str, col_b: str, r: float) -> str:
    abs_r = abs(r)
    direction = "positively" if r > 0 else "negatively"

    if abs_r >= 0.9:
        strength = "very strong"
    elif abs_r >= 0.7:
        strength = "strong"
    elif abs_r >= 0.5:
        strength = "moderate"
    else:
        strength = "weak"

    return f"**{col_a}** and **{col_b}** are {strength}ly {direction} correlated (r={r:.3f})."


class CorrelationTool(Tool):
    @property
    def name(self) -> str:
        return "correlation_analysis"

    @property
    def description(self) -> str:
        return (
            "Compute Pearson correlation matrix and identify notable pairwise relationships. "
            "Returns the full matrix and ranked pairs only — no row-level data (ADR-0002). "
            "Populates the Correlations report section with a heatmap and narrative."
        )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "min_abs_correlation": {
                    "type": "number",
                    "description": "Minimum |r| to include in ranked pairs list. Default 0.3.",
                },
                "top_n": {
                    "type": "integer",
                    "description": "Maximum number of notable pairs to surface. Default 10.",
                },
            },
            "required": [],
        }

    def run(
        self,
        df: pd.DataFrame,
        min_abs_correlation: float = 0.3,
        top_n: int = 10,
        **kwargs,
    ) -> dict:
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.shape[1] < 2:
            return {"columns": [], "matrix": {}, "ranked_pairs": [], "narrative_pairs": []}

        corr = numeric_df.corr(method="pearson")
        columns = corr.columns.tolist()

        matrix: dict[str, dict[str, float]] = {
            col: {other: round(float(corr.loc[col, other]), 4) for other in columns}
            for col in columns
        }

        pairs = []
        for i, col_a in enumerate(columns):
            for j, col_b in enumerate(columns):
                if j <= i:
                    continue
                r = float(corr.loc[col_a, col_b])
                if not math.isfinite(r):
                    continue
                abs_r = abs(r)
                if abs_r >= min_abs_correlation:
                    pairs.append({
                        "col_a": col_a,
                        "col_b": col_b,
                        "correlation": round(r, 4),
                        "abs_correlation": round(abs_r, 4),
                    })

        pairs.sort(key=lambda p: -p["abs_correlation"])
        top_pairs = pairs[:top_n]

        narrative_pairs = [
            _describe_correlation(p["col_a"], p["col_b"], p["correlation"])
            for p in top_pairs
        ]

        return {
            "columns": columns,
            "matrix": matrix,
            "ranked_pairs": top_pairs,
            "narrative_pairs": narrative_pairs,
        }
