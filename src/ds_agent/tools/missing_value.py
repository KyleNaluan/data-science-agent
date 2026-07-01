from __future__ import annotations

import pandas as pd

from .base import Tool


def _imputation_recommendation(dtype_kind: str, null_rate: float, skew: float | None) -> str:
    if null_rate == 0.0:
        return "No missing values"
    if null_rate > 0.9:
        return "Critically high missingness (>90%) — consider dropping this column"
    if null_rate > 0.5:
        return "High missingness (>50%) — indicator variable + fill or advanced imputation recommended"
    if dtype_kind in ("i", "u", "f"):
        if skew is not None and abs(skew) > 1.0:
            return "Numeric with skewed distribution — median imputation recommended"
        return "Numeric — mean or median imputation recommended"
    if dtype_kind == "M":
        return "Datetime — forward-fill or interpolation recommended"
    if dtype_kind == "b":
        return "Boolean — mode imputation or indicator variable recommended"
    return "Categorical/text — mode imputation or 'Unknown' indicator recommended"


class MissingValueTool(Tool):
    @property
    def name(self) -> str:
        return "missing_value_analysis"

    @property
    def description(self) -> str:
        return (
            "Compute per-column missing-value counts, rates, and imputation recommendations. "
            "Returns aggregate statistics only — no raw row data (ADR-0002). "
            "Call after schema_inference to enrich the Data Quality Scorecard."
        )

    @property
    def input_schema(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    def run(self, df: pd.DataFrame, **kwargs) -> dict:
        row_count = len(df)
        columns: list[dict] = []
        total_missing = 0

        for col_name in df.columns:
            series = df[col_name]
            null_count = int(series.isna().sum())
            null_rate = round(null_count / max(row_count, 1), 4)
            total_missing += null_count

            skew = None
            dtype_kind = series.dtype.kind
            if dtype_kind in ("i", "u", "f"):
                non_null = series.dropna()
                if len(non_null) > 2:
                    try:
                        skew = float(non_null.skew())
                    except Exception:
                        pass

            recommendation = _imputation_recommendation(dtype_kind, null_rate, skew)

            columns.append({
                "column": col_name,
                "missing_count": null_count,
                "missing_rate": null_rate,
                "dtype": str(series.dtype),
                "imputation_recommendation": recommendation,
            })

        total_cells = row_count * max(len(df.columns), 1)
        overall_rate = round(total_missing / max(total_cells, 1), 4)

        return {
            "row_count": row_count,
            "total_missing_cells": total_missing,
            "overall_missing_rate": overall_rate,
            "columns": columns,
        }
